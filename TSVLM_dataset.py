from torch.utils.data import Dataset
from torchvision.io import read_image
import os
import json
import torch


class Dataset(Dataset):
    def __init__(self, input_file, tokenizer, transform=None, data_root=""):
        with open(input_file) as f:
            self.data = json.load(f)
        # if len(self.data) > 10:
        #     self.data = self.data[:10]
        self.tokenizer = tokenizer
        self.transform = transform
        self.data_root = data_root
        if torch.cuda.is_available() and torch.utils.data.get_worker_info() is None:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        qa, img_path = self.data[idx]
        img_path = [os.path.join(self.data_root, p) for p in img_path.values()]
        q_text, a_text = qa['Q'], qa['A']
        q_text = f"Question: {q_text} Answer:"
        imgs = [self.transform(read_image(p).float()).to(self.device) for p in img_path]
        imgs = torch.stack(imgs, dim=0)
        return q_text, imgs, a_text, sorted(list(img_path))

    def collate_fn(self, batch):
        q_texts, imgs, a_texts, _ = zip(*batch)
        imgs = torch.stack(list(imgs), dim=0)
        encodings = self.tokenizer(
            q_texts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).input_ids.to(self.device)
        labels = self.tokenizer(
            a_texts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).input_ids.to(self.device)
        return encodings, imgs, labels

    def test_collate_fn(self, batch):
        q_texts, imgs, a_texts, img_path = zip(*batch)
        imgs = torch.stack(list(imgs), dim=0)
        encodings = self.tokenizer(
            q_texts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).input_ids.to(self.device)
        labels = self.tokenizer(
            a_texts,
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).input_ids.to(self.device)
        return list(q_texts), encodings, imgs, labels, img_path
