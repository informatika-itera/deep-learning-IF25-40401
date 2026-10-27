"""Data: Imagenette 160 px -> DataLoader train & val.

    Imagenette train (9.469 citra) --> train   update bobot
    Imagenette val   (3.925 citra) --> val     early stopping, checkpoint terbaik, membandingkan run

Unduhan otomatis (±94 MB) ke data_root saat pertama kali dijalankan.
"""
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, default_collate
from torchvision import datasets
from torchvision.transforms import v2

NUM_CLASSES = 10


def build_transforms(img_size, augment, mean, std):
    """augment: "none" | "basic" | "strong". Citra val TIDAK PERNAH diaugmentasi."""
    to_tensor = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)]
    val_tf = v2.Compose([v2.Resize(int(img_size / 0.875)), v2.CenterCrop(img_size), *to_tensor])

    if augment == "none":
        train_tf = val_tf
    elif augment == "basic":
        train_tf = v2.Compose([v2.RandomResizedCrop(img_size, scale=(0.35, 1.0)), v2.RandomHorizontalFlip(),
                               *to_tensor])
    elif augment == "strong":
        train_tf = v2.Compose([v2.RandomResizedCrop(img_size, scale=(0.35, 1.0)), v2.RandomHorizontalFlip(),
                               v2.TrivialAugmentWide(), *to_tensor, v2.RandomErasing(p=0.25)])
    else:
        raise ValueError(f"augment tidak dikenal: {augment}")
    return train_tf, val_tf


class MixUpCutMix:
    """collate_fn: gabungkan batch, lalu campur citra & label -> label menjadi "soft" (distribusi probabilitas).

    Berupa class (bukan lambda) agar bisa dipakai DataLoader dengan num_workers > 0.
    """

    def __init__(self, mixup_alpha, cutmix_alpha):
        mix = []
        if mixup_alpha > 0:
            mix.append(v2.MixUp(alpha=mixup_alpha, num_classes=NUM_CLASSES))
        if cutmix_alpha > 0:
            mix.append(v2.CutMix(alpha=cutmix_alpha, num_classes=NUM_CLASSES))
        self.mix = v2.RandomChoice(mix)  # tiap batch: pilih salah satu

    def __call__(self, batch):
        return self.mix(*default_collate(batch))


def build_loaders(c, mean, std, device):
    root = Path(c["data_root"])
    download = not (root / "imagenette2-160").exists()
    train_tf, val_tf = build_transforms(c["img_size"], c["augment"], mean, std)

    train_ds = datasets.Imagenette(root, split="train", size="160px", download=download, transform=train_tf)
    val_ds = datasets.Imagenette(root, split="val", size="160px", transform=val_tf)
    class_names = [names[0] for names in val_ds.classes]

    if c["train_fraction"] < 1.0:  # simulasi "data sedikit": ambil sebagian data train secara acak
        g = torch.Generator().manual_seed(c["seed"])
        idx = torch.randperm(len(train_ds), generator=g)[: int(len(train_ds) * c["train_fraction"])]
        train_ds = Subset(train_ds, idx.tolist())

    if c["mixup_alpha"] > 0 or c["cutmix_alpha"] > 0:
        collate = MixUpCutMix(c["mixup_alpha"], c["cutmix_alpha"])
    else:
        collate = default_collate

    kw = dict(batch_size=c["batch_size"], num_workers=c["num_workers"], pin_memory=device == "cuda",
              persistent_workers=c["num_workers"] > 0)
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=True, collate_fn=collate, **kw)
    val_dl = DataLoader(val_ds, shuffle=False, **kw)
    return train_dl, val_dl, class_names
