"""Study bags from the uint8 cache, with rigid jitter and no laterality flips."""
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .config import (AUG_INTENSITY, AUG_ROT_DEG, AUG_SCALE, AUG_SHIFT, N_SLOT,
                     SLICES_PER_SLOT, TARGETS)
from .labels import targets_from_table


def slice_groups(k=SLICES_PER_SLOT):
    return max(1, k // 3)


def pick_group(k, rng, group=None):
    n = slice_groups(k)
    if group is None:
        group = int(rng.integers(n)) if n > 1 else 0
    start = group * 3
    return start, min(start + 3, k)


def augment(x, rng):
    if AUG_ROT_DEG <= 0 and AUG_SCALE <= 0 and AUG_SHIFT <= 0 and AUG_INTENSITY <= 0:
        return x
    h, w = x.shape[-2:]
    angle = float(rng.uniform(-AUG_ROT_DEG, AUG_ROT_DEG)) if AUG_ROT_DEG else 0.0
    scale = float(rng.uniform(1.0 - AUG_SCALE, 1.0 + AUG_SCALE)) if AUG_SCALE else 1.0
    tx = float(rng.uniform(-AUG_SHIFT, AUG_SHIFT)) if AUG_SHIFT else 0.0
    ty = float(rng.uniform(-AUG_SHIFT, AUG_SHIFT)) if AUG_SHIFT else 0.0
    rad = np.deg2rad(angle)
    cos, sin = np.cos(rad) / scale, np.sin(rad) / scale
    theta = torch.tensor([[cos, -sin, tx], [sin, cos, ty]], dtype=x.dtype)
    theta = theta.unsqueeze(0).repeat(x.shape[0], 1, 1)
    grid = F.affine_grid(theta, size=x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border",
                      align_corners=False)
    if AUG_INTENSITY:
        gain = float(rng.uniform(1.0 - AUG_INTENSITY, 1.0 + AUG_INTENSITY))
        bias = float(rng.uniform(-AUG_INTENSITY, AUG_INTENSITY)) * 0.1
        x = x * gain + bias
    return x


class StudyDataset(Dataset):
    def __init__(self, cache, mask, studies, table, train=True, groups=None,
                 img_size=None, study_ids=None):
        self.cache = cache
        self.mask = torch.as_tensor(mask, dtype=torch.float32)
        self.studies = list(studies)
        self.train = train
        self.groups = groups
        self.img_size = img_size
        if study_ids is None:
            order = {s: i for i, s in enumerate(self.studies)}
            table = table[table.StudyInstanceUID.isin(order)]
            rows = table.set_index("StudyInstanceUID").loc[
                [s for s in self.studies if s in set(table.StudyInstanceUID)]]
            values, conf, gold = targets_from_table(rows.reset_index())
            self.uids = list(rows.index)
            self.idx = [order[u] for u in self.uids]
        else:
            pos = {s: i for i, s in enumerate(self.studies)}
            self.uids = [s for s in study_ids if s in pos]
            self.idx = [pos[s] for s in self.uids]
            sub = table.set_index("StudyInstanceUID").loc[self.uids].reset_index()
            values, conf, gold = targets_from_table(sub)
        self.values = torch.as_tensor(values, dtype=torch.float32)
        self.conf = torch.as_tensor(conf, dtype=torch.float32)
        self.gold = torch.as_tensor(gold, dtype=torch.float32)
        self.n_groups = slice_groups(cache.shape[2])

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, i):
        row = self.cache[self.idx[i]]
        mask = self.mask[self.idx[i]]
        if self.groups is None:
            group = None
        elif isinstance(self.groups, int):
            group = self.groups
        else:
            group = self.groups[i]
        start, _ = pick_group(row.shape[1], np.random.default_rng(None), group)
        imgs = torch.from_numpy(np.ascontiguousarray(row[:, start:start + 3]))
        if imgs.shape[1] < 3:
            pad = imgs[:, -1:].repeat(1, 3 - imgs.shape[1], 1, 1)
            imgs = torch.cat([imgs, pad], dim=1)
        imgs = imgs.float() / 255.0
        if self.train:
            imgs = augment(imgs, np.random.default_rng(None))
        return {
            "imgs": imgs,
            "mask": mask.clone(),
            "target": self.values[i],
            "conf": self.conf[i],
            "gold": self.gold[i],
        }


def make_dataset(cache, mask, studies, table, train=True, **kw):
    return StudyDataset(cache, mask, studies, table, train=train, **kw)


def make_loader(dataset, batch_size, shuffle, num_workers=0):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                       shuffle=shuffle, num_workers=num_workers,
                                       drop_last=shuffle and len(dataset) > batch_size)


def empty_batch(batch_size):
    return {
        "imgs": torch.zeros(batch_size, N_SLOT, 3, 0, 0),
        "mask": torch.zeros(batch_size, N_SLOT),
        "target": torch.zeros(batch_size, len(TARGETS)),
        "conf": torch.zeros(batch_size, len(TARGETS)),
        "gold": torch.zeros(batch_size),
    }
