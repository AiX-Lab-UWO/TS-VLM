from transformers import T5ForConditionalGeneration, AutoImageProcessor, MobileViTV2Model
import torch.nn as nn
import torch
from peft import LoraConfig, get_peft_model, LoftQConfig
import torch.nn.functional as F
import numpy as np

MOBILEVIT_HIDDEN_STATE = 512
MOBILEVIT_SEQ_LENGTH = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class SimpleSoftmax(nn.Module):
    def forward(self, scores):
        return torch.softmax(scores, dim=-1)


class HardTop1(nn.Module):
    def forward(self, scores):
        max_idx = scores.argmax(dim=-1)
        one_hot = F.one_hot(max_idx, num_classes=scores.size(-1)).float()
        return one_hot


class TopKSoft(nn.Module):
    def __init__(self, k=3):
        super().__init__()
        self.k = k

    def forward(self, scores):
        topk_scores, topk_indices = torch.topk(scores, self.k, dim=-1)
        mask = torch.zeros_like(scores)
        mask.scatter_(dim=-1, index=topk_indices, value=1.0)
        masked_scores = scores * mask + (-1e9) * (1 - mask)
        return torch.softmax(masked_scores, dim=-1)


class UniformPooling(nn.Module):
    def forward(self, scores):
        return torch.ones_like(scores) / scores.size(-1)


class TrainableSoftSort(nn.Module):
    def __init__(self, init_tau=0.1):
        super().__init__()
        self.tau = nn.Parameter(torch.tensor(init_tau))

    def forward(self, scores):
        return torch.softmax(scores / self.tau, dim=-1)


class TrainableSinkhornSort(nn.Module):
    def __init__(self, n_iters=5, init_tau=0.1):
        super().__init__()
        self.n_iters = n_iters
        self.tau = nn.Parameter(torch.tensor(init_tau))

    def forward(self, scores):
        P = torch.exp(scores / self.tau)
        for _ in range(self.n_iters):
            P = P / (P.sum(dim=-1, keepdim=True) + 1e-8)
            P = P / (P.sum(dim=-2, keepdim=True) + 1e-8)
        return P.sum(dim=-2)


class TextGuidedSoftSortPooling(nn.Module):
    def __init__(self, hidden_dim, sorting_type='softsort', tau=0.1, topk=3):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.sorting_type_name = sorting_type
        if sorting_type == 'trainable_softsort':
            self.sorting = TrainableSoftSort(init_tau=tau)
        elif sorting_type == 'trainable_sinkhorn':
            self.sorting = TrainableSinkhornSort(init_tau=tau)
        elif sorting_type == 'simplesoftmax':
            self.sorting = SimpleSoftmax()
        elif sorting_type == 'hardtop1':
            self.sorting = HardTop1()
        elif sorting_type == 'topksoft':
            self.sorting = TopKSoft(k=topk)
        elif sorting_type == 'uniform':
            self.sorting = UniformPooling()
        else:
            raise ValueError(f"Unknown sorting type: {sorting_type}")

    def forward(self, img_embeddings, text_embedding):
        img_norm = F.normalize(img_embeddings, p=2, dim=-1)
        text_norm = F.normalize(text_embedding, p=2, dim=-1)
        sim_scores = torch.bmm(img_norm, text_norm.unsqueeze(2)).squeeze(2)
        sort_weights = self.sorting(sim_scores)
        fused_embedding = torch.sum(img_embeddings * sort_weights.unsqueeze(-1), dim=1)
        fused_embedding = self.proj(fused_embedding)
        return fused_embedding


class TSVLM(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.lm == 'T5-Tiny':
            self.model = T5ForConditionalGeneration.from_pretrained("google/t5-efficient-tiny", trust_remote_code=True)
        elif config.lm == 'T5-Mini':
            self.model = T5ForConditionalGeneration.from_pretrained("google/t5-efficient-mini", trust_remote_code=True)
        elif config.lm == 'T5-Small':
            self.model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-small')
        else:
            self.model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-large')
            # loftq_config = LoftQConfig(loftq_bits=8)
            # lora_config = LoraConfig(
            #     r=config.lora_dim,
            #     lora_alpha=config.lora_alpha,
            #     loftq_config=loftq_config,
            #     lora_dropout=config.lora_dropout,
            #     bias='none',
            #     target_modules=['q', 'v']
            # )
            # self.model = get_peft_model(self.model, lora_config)
        hidden_size = self.model.config.d_model
        self.mvp = self.MultiViewProcessor(hidden_size, config, freeze=True)

    class MultiViewProcessor(nn.Module):

        def __init__(self, hidden_size, config, freeze=False):
            super().__init__()
            self.img_model = MobileViTV2Model.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256")
            self.image_processor = AutoImageProcessor.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256")
            self.lm = config.lm
            self.modal_embeddings = nn.Embedding(2, hidden_size)
            self.modal_embeddings.weight.data.normal_(mean=0.0, std=0.02)
            if freeze:
                for param in self.img_model.parameters():
                    param.requires_grad = False
            self.tg_softsort_pooling = TextGuidedSoftSortPooling(
                hidden_dim=hidden_size,
                sorting_type=config.sorting_type,
                tau=getattr(config, 'tau', 0.1),
                topk=getattr(config, 'topk', 3)
            )
            self.img_projection_layer = nn.Linear(MOBILEVIT_HIDDEN_STATE, hidden_size)

        def get_img_embedding(self, imgs, text_embedding):
            N, num_frames, C, H, W = imgs.shape
            frame_features = []
            for i in range(num_frames):
                img = imgs[:, i]
                img = img.clamp(0, 1)
                inputs = self.image_processor(images=img, return_tensors="pt")
                inputs = {k: v.to(device) for k, v in inputs.items()}
                img_features = self.img_model(**inputs).last_hidden_state
                img_features = torch.nn.functional.adaptive_avg_pool2d(img_features, (7, 7))
                batch_size, hidden_dim, height, width = img_features.shape
                seq_length = height * width
                img_features = img_features.view(batch_size, hidden_dim, seq_length).permute(0, 2, 1)

                img_features = self.img_projection_layer(img_features)
                frame_features.append(img_features)
            img_features = torch.stack(frame_features, dim=1)
            fused_tokens = []
            for token_idx in range(img_features.shape[2]):
                token_views = img_features[:, :, token_idx, :]
                fused_token = self.tg_softsort_pooling(token_views, text_embedding)
                fused_tokens.append(fused_token.unsqueeze(1))
            fused_img_embedding = torch.cat(fused_tokens, dim=1)
            return fused_img_embedding

        def forward(self, text_enc, imgs, text_model):
            text_embeddings = text_model.get_input_embeddings()(text_enc)
            text_embeddings += self.modal_embeddings(
                torch.zeros((1, text_embeddings.shape[1]), dtype=torch.int, device=device))
            imgs_embedding = self.get_img_embedding(imgs, text_embeddings.mean(dim=1))
            merged_embedding = torch.cat([text_embeddings, imgs_embedding], dim=1)
            return merged_embedding

    def forward(self, text_enc, imgs, labels=None):
        merged_embedding = self.mvp(text_enc, imgs, self.model)
        return self.model(inputs_embeds=merged_embedding, labels=labels)

    def generate(self, text_enc, imgs, lidar=None):
        merged_embedding = self.mvp(text_enc, imgs, self.model)
        attention_mask = torch.ones(merged_embedding.shape[:2], dtype=torch.long, device=device)
        decoder_input_ids = torch.ones((merged_embedding.shape[0], 1), dtype=torch.long,
                                       device=device) * self.model.config.decoder_start_token_id
        output_ids = self.model.generate(attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                                         inputs_embeds=merged_embedding, max_length=512, early_stopping=True)
        return output_ids
