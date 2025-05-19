# eval_restructured.py
import os
import json
import torch
import argparse
import pandas as pd
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
from transformers import T5Tokenizer
from torchvision import transforms
from torch.utils.data import DataLoader

from TSVLM_dataset import Dataset
from TSVLM import TSVLM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save-dir', type=str, default='./results')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--freeze-lm', action='store_true')
    parser.add_argument('--lm', type=str, default='T5-Tiny', choices=['T5-Small', 'T5-Mini', 'T5-Tiny', 'T5-Large'])
    parser.add_argument('--lora', action='store_true')
    parser.add_argument('--lora-dim', type=int, default=64)
    parser.add_argument('--lora-alpha', type=int, default=32)
    parser.add_argument('--lora-dropout', type=float, default=0.05)
    parser.add_argument('--max-len', type=int, default=512)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--model-name', type=str, default='T5-Tiny')
    parser.add_argument('--sorting-type', type=str, default='trainable_softsort',
                        choices=['trainable_softsort', 'simplesoftmax', 'hardtop1', 'topksoft', 'uniform',
                                 'trainable_sinkhorn'])
    parser.add_argument('--tau', type=float, default=0.1)
    parser.add_argument('--topk', type=int, default=3)
    return parser.parse_args()


def load_tokenizer(name):
    model_map = {
        'T5-Small': 'google-t5/t5-small',
        'T5-Mini': 'google/t5-efficient-mini',
        'T5-Tiny': 'google/t5-efficient-tiny',
        'T5-Large': 'google-t5/t5-large'
    }
    tokenizer = T5Tokenizer.from_pretrained(model_map[name])
    tokenizer.add_tokens('<')
    return tokenizer


def run_inference(model, processor, dataloader, config, image_id_dict):
    model.eval()
    predictions = []
    visited_ids = set()

    with torch.no_grad():
        for idx, (questions, encodings, imgs, labels, img_paths) in tqdm(enumerate(dataloader), total=len(dataloader)):
            outputs = model.generate(encodings, imgs)
            answers = [processor.decode(o, skip_special_tokens=True) for o in outputs]

            if idx % 100 == 0:
                print(questions)
                print(answers)

            for path, q, a in zip(img_paths, questions, answers):
                key = path[0] + ' ' + q
                img_id = image_id_dict.get(key, [None])[0]
                if img_id is not None and img_id not in visited_ids and len(a) <= config.max_len:
                    visited_ids.add(img_id)
                    predictions.append({'image_id': img_id, 'caption': a})

    out_file = os.path.join(config.save_dir, config.model_name, 'predictions.json')
    with open(out_file, 'w') as f:
        json.dump(predictions, f)


def save_metrics(coco_eval, config):
    metrics = {metric: [score] for metric, score in coco_eval.eval.items()}
    df = pd.DataFrame(metrics)
    df.to_csv(os.path.join(config.save_dir, config.model_name, 'metrics.csv'), index=False)


def main():
    config = get_config()
    processor = load_tokenizer(config.lm)

    model = TSVLM(config).to(device)
    model.model.resize_token_embeddings(len(processor))

    ckpt_path = os.path.join(config.save_dir, config.model_name, 'best_epoch.pth')
    model.load_state_dict(torch.load(ckpt_path))

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))
    ])
    test_dataset = Dataset(
        input_file=os.path.join('data', 'multi_frame', 'multi_frame_test.json'),
        tokenizer=processor,
        transform=transform
    )
    test_loader = DataLoader(test_dataset, shuffle=True, batch_size=config.batch_size,
                             collate_fn=test_dataset.test_collate_fn, drop_last=True)

    with open(os.path.join('data', 'multi_frame', 'image_id.json')) as f:
        image_id_dict = json.load(f)

    run_inference(model, processor, test_loader, config, image_id_dict)

    coco = COCO(os.path.join('data', 'multi_frame', 'multi_frame_test_coco.json'))
    results = coco.loadRes(os.path.join(config.save_dir, config.model_name, 'predictions.json'))
    evaluator = COCOEvalCap(coco, results)
    evaluator.params['image_id'] = results.getImgIds()
    evaluator.evaluate()
    save_metrics(evaluator, config)


if __name__ == '__main__':
    main()
