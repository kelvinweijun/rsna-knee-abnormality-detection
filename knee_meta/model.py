"""Backbone plus slot-attention head: one bag of slot images to twelve logits."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, ViTConfig, ViTModel

from .config import (N_LABELS, N_SLOT, POOL, POOL_PARTS, SLOT_PRIOR_STRENGTH,
                     SLOT_PRIOR_TABLE, TARGETS, UNFREEZE_LAST)


class SlotHead(nn.Module):
    def __init__(self, dim, n_slot, n_out, hidden=256, p=0.2, prior=True):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(hidden, n_out)
        self.hidden = hidden
        p_ = torch.zeros(n_out, n_slot)
        if prior and n_slot == N_SLOT and n_out == N_LABELS:
            for label, slots in SLOT_PRIOR_TABLE.items():
                if label in TARGETS:
                    p_[TARGETS.index(label), list(slots)] = SLOT_PRIOR_STRENGTH
        self.prior = prior
        if prior:
            self.register_buffer("slot_prior", p_)

    def forward(self, x, mask):
        h = self.proj(x) + self.slot_emb
        att = torch.einsum("bsh,oh->bos", h, self.query) / self.hidden ** 0.5
        if self.prior:
            att = att + self.slot_prior.unsqueeze(0)
        att = att.masked_fill(mask.unsqueeze(1) < 0.5, -1e4).softmax(-1)
        ctx = self.drop(torch.einsum("bos,bsh->boh", att, h))
        return (ctx * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias


class Model(nn.Module):
    def __init__(self, backbone, dim, pool=POOL, prior=True, img_size=None):
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        self.img_size = img_size
        self.head = SlotHead(dim * POOL_PARTS[pool], N_SLOT, N_LABELS, prior=prior)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def encode(self, x, img_size=None):
        if img_size is not None and img_size != x.shape[-1]:
            x = F.interpolate(x, size=(img_size, img_size), mode="bilinear",
                              align_corners=False)
        x = (x - self.mean) / self.std
        out = self.backbone(pixel_values=x).last_hidden_state
        patch = out[:, 1:]
        parts = [out[:, 0], patch.mean(1)]
        if self.pool == "cls_mean_focal":
            k = max(1, patch.shape[1] // 8)
            parts.append(patch.topk(k, dim=1).values.mean(1))
        return torch.cat(parts, dim=1)

    def forward(self, imgs, mask, img_size=None):
        if imgs.dtype == torch.uint8:
            raise ValueError(
                "Model.forward expects float images in [0,1], got uint8; "
                "divide by 255 before calling the model")
        lo, hi = float(imgs.min()), float(imgs.max())
        if hi > 1.5 or lo < -0.5:
            raise ValueError(
                f"Model.forward input range [{lo:.3f}, {hi:.3f}] outside "
                f"[-0.5, 1.5]; expected float images in [0,1]")
        b, s = imgs.shape[:2]
        x = imgs.reshape(b * s, *imgs.shape[2:]).float()
        feat = self.encode(x, img_size if img_size is not None else self.img_size)
        return self.head(feat.reshape(b, s, -1), mask)


def _tiny_backbone(img_size=64):
    cfg = ViTConfig(hidden_size=64, num_hidden_layers=2, num_attention_heads=2,
                    intermediate_size=128, image_size=img_size, patch_size=16,
                    num_channels=3)
    return ViTModel(cfg)


def build_model(unfreeze_last=UNFREEZE_LAST, source=None, pool=POOL, prior=True,
                tiny=False, img_size=None):
    if tiny:
        bb = _tiny_backbone(img_size or 64)
        dim = bb.config.hidden_size
        return Model(bb, dim, pool=pool, prior=prior, img_size=img_size)
    if source is None:
        from .config import find_backbone
        source = find_backbone()
    if source is None:
        raise FileNotFoundError("DINOv2 weights not attached")
    bb = AutoModel.from_pretrained(str(source))
    n_layer = len(bb.encoder.layer)
    for prm in bb.parameters():
        prm.requires_grad = False
    for blk in bb.encoder.layer[max(0, n_layer - unfreeze_last):]:
        for prm in blk.parameters():
            prm.requires_grad = True
    for prm in bb.layernorm.parameters():
        prm.requires_grad = True
    return Model(bb, bb.config.hidden_size, pool=pool, prior=prior, img_size=img_size)


def param_groups(model, lr_head, lr_backbone, weight_decay):
    head = list(model.head.parameters())
    backbone = [p for p in model.backbone.parameters() if p.requires_grad]
    return [
        {"params": head, "lr": lr_head},
        {"params": backbone, "lr": lr_backbone},
    ], {"weight_decay": weight_decay}
