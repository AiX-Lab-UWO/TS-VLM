# train_restructured.py
import os
import json
import time
import torch
import random
import argparse
import datetime
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import T5Tokenizer
import torch.multiprocessing as mp

from TSVLM_dataset import Dataset
from TSVLM import TSVLM


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--save-dir', type=str, default='./results')
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--freeze-lm', action='store_true')
    p.add_argument('--lm', type=str, default='T5-Tiny',
                   choices=['T5-Small', 'T5-Mini', 'T5-Tiny', 'T5-Large'])
    p.add_argument('--checkpoint-frequency', type=int, default=500)
    p.add_argument('--lora', action='store_true')
    p.add_argument('--lora-dim', type=int, default=64)
    p.add_argument('--lora-alpha', type=int, default=32)
    p.add_argument('--lora-dropout', type=float, default=0.05)
    p.add_argument('--num-workers', type=int, default=6)
    p.add_argument('--load-checkpoint', action='store_true')
    p.add_argument('--checkpoint-file', type=str, default='T5-Tiny')
    p.add_argument('--data-path', type=str, default='.')
    p.add_argument('--sorting-type', type=str, default='trainable_softsort',
                   choices=['trainable_softsort', 'simplesoftmax', 'hardtop1', 'topksoft', 'uniform',
                            'trainable_sinkhorn'])
    p.add_argument('--tau', type=float, default=0.1)
    p.add_argument('--topk', type=int, default=3)
    return p.parse_args()


def build_tokenizer(name):
    model_map = {
        'T5-Small': 'google-t5/t5-small',
        'T5-Mini': 'google/t5-efficient-mini',
        'T5-Tiny': 'google/t5-efficient-tiny',
        'T5-Large': 'google-t5/t5-large'
    }
    tokenizer = T5Tokenizer.from_pretrained(model_map[name])
    tokenizer.add_tokens('<')
    return tokenizer


def setup_dataloaders(config, processor):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))
    ])

    def load(split):
        return Dataset(
            input_file=os.path.join(config.data_path, 'data', 'multi_frame', f'multi_frame_{split}.json'),
            tokenizer=processor,
            transform=transform,
            data_root=config.data_path
        )

    loaders = {
        split: DataLoader(load(split), shuffle=True, batch_size=config.batch_size,
                          num_workers=config.num_workers, collate_fn=load(split).collate_fn,
                          persistent_workers=True)
        for split in ['train', 'val', 'test']
    }
    return loaders['train'], loaders['val'], loaders['test']


def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    for inputs, imgs, labels in tqdm(dataloader):
        with torch.no_grad():
            loss = model(inputs, imgs, labels).loss
        total_loss += loss.item()
    return total_loss / len(dataloader)


def visualize_loss(train_loss, val_loss, out_path):
    x = range(1, len(train_loss) + 1)
    plt.plot(x, train_loss, label='Training Loss')
    plt.plot(x, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(out_path, 'loss.png'))


def record_stats(train_loss, val_loss, epoch, lr, out_path):
    with open(os.path.join(out_path, 'stats.json'), 'w') as f:
        json.dump({
            'losses': train_loss,
            'val losses': val_loss,
            'min train loss': min(train_loss),
            'min val loss': min(val_loss),
            'epochs': epoch,
            'learning rate': lr,
            'LM': 'T5-Tiny',
            'Image Embedding': 'Patch'
        }, f)


def train_model(model, processor, train_loader, val_loader, config, save_dir):
    opt = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=0.9)

    train_loss_log, val_loss_log = [], []
    best_weights, best_train_loss, best_val_loss = None, float('inf'), float('inf')

    for ep in range(config.epochs):
        model.train()
        ep_loss = 0
        print(f'--- Epoch {ep + 1}/{config.epochs} ---')

        for step, (inp, img, tgt) in enumerate(tqdm(train_loader)):
            output = model(inp, img, tgt)
            loss = output.loss
            ep_loss += loss.item()

            if step % config.checkpoint_frequency == 0:
                pred = torch.argmax(output.logits, dim=-1)
                print('\nSample predictions:')
                print('Questions:', [processor.decode(i, skip_special_tokens=True) for i in inp])
                print('Answers:', [processor.decode(p, skip_special_tokens=True) for p in pred])
                print('Ground Truth:', [processor.decode(l, skip_special_tokens=True) for l in tgt])

            loss.backward()
            opt.step()
            opt.zero_grad()

        avg_train_loss = ep_loss / len(train_loader)
        avg_val_loss = evaluate(model, val_loader)

        train_loss_log.append(avg_train_loss)
        val_loss_log.append(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_weights = deepcopy(model.state_dict())

        scheduler.step()
        record_stats(train_loss_log, val_loss_log, ep + 1, scheduler.get_last_lr()[0], save_dir)
        torch.save(best_weights, os.path.join(save_dir, f'epoch_{ep + 1}.pth'))
        print(f'Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')

    visualize_loss(train_loss_log, val_loss_log, save_dir)
    return best_weights, min(train_loss_log), min(val_loss_log)


def summarize_experiment(config, min_train, min_val, test, timestr):
    df = pd.DataFrame({
        'Model name': [timestr],
        'Learning rate': [config.learning_rate],
        'Weight decay': [config.weight_decay],
        'Batch size': [config.batch_size],
        'Epochs': [config.epochs],
        'LoRA finetuning': [config.lora],
        'LoRA Dimension': [config.lora_dim],
        'LoRA Alpha': [config.lora_alpha],
        'LoRA Dropout': [config.lora_dropout],
        'Freeze T5': [config.freeze_lm],
        'Min Training Loss': [min_train],
        'Min Validation Loss': [min_val],
        'Min Testing Loss': [test],
    })
    df.to_csv(os.path.join(config.save_dir, timestr, 'multi_frame_results.csv'), index=False)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    config = get_args()
    run_id = datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + f"-{random.randint(1000, 9999)}"
    out_path = os.path.join(config.save_dir, run_id)
    os.makedirs(out_path, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = build_tokenizer(config.lm)
    model = TSVLM(config).to(device)
    model.model.resize_token_embeddings(len(tokenizer))

    train_dl, val_dl, test_dl = setup_dataloaders(config, tokenizer)

    if config.load_checkpoint:
        ckpt = os.path.join(config.save_dir, config.checkpoint_file, 'latest_model.pth')
        print(f'Loading checkpoint from {ckpt}')
        model.load_state_dict(torch.load(ckpt))
        run_id = config.checkpoint_file

    weights, best_train, best_val = train_model(model, tokenizer, train_dl, val_dl, config, out_path)

    model.load_state_dict(weights)
    model.to(device)
    final_test_loss = evaluate(model, test_dl)

    summarize_experiment(config, best_train, best_val, final_test_loss, run_id)
    print('Training complete.')
