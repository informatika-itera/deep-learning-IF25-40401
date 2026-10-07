"""Baseline sederhana Studio 2: fine-tuning ResNet-18 (pretrained ImageNet).

Titik awal saja — kelompok diharapkan melakukan eksperimen sendiri
(backbone lain, augmentasi, freezing, scheduler, data tambahan, dll.).

    python train_baseline.py --data ./train --epochs 8
    python train_baseline.py --data ./train --epochs 8 --scratch   # tanpa bobot pretrained

Hasil: submission/model.pth + folder my_val/ (split validasi lokal 15%)
       yang bisa langsung diuji dengan:
    python evaluate.py --submission ./submission --data ./my_val
"""
import argparse
import random
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from submission.model import get_model, get_transform


def make_local_split(src, dst, frac, seed):
    """Salin 15% gambar per kelas dari train ke my_val/ (sekali saja), sisanya ke my_train/."""
    dst_tr, dst_va = Path(dst) / "my_train", Path(dst) / "my_val"
    if dst_va.exists():
        return dst_tr, dst_va
    rng = random.Random(seed)
    for cls_dir in sorted(Path(src).iterdir()):
        if not cls_dir.is_dir():
            continue
        files = sorted(cls_dir.glob("*.jpg"))
        rng.shuffle(files)
        n_va = int(len(files) * frac)
        for i, f in enumerate(files):
            out = (dst_va if i < n_va else dst_tr) / cls_dir.name
            out.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, out / f.name)
    return dst_tr, dst_va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder train/ dari dataset")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--scratch", action="store_true", help="latih tanpa bobot pretrained")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    tr_dir, va_dir = make_local_split(a.data, ".", 0.15, a.seed)

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tr_ds = datasets.ImageFolder(tr_dir, train_tf)
    va_ds = datasets.ImageFolder(va_dir, get_transform())
    print("kelas:", tr_ds.classes, "| train:", len(tr_ds), "| val:", len(va_ds))
    tr_dl = DataLoader(tr_ds, a.batch_size, shuffle=True, num_workers=2)
    va_dl = DataLoader(va_ds, a.batch_size, num_workers=2)

    model = get_model()
    if not a.scratch:  # salin bobot ImageNet ke semua layer kecuali head
        pre = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).state_dict()
        pre = {k: v for k, v in pre.items() if not k.startswith("fc.")}
        model.load_state_dict(pre, strict=False)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    best = 0.0
    for ep in range(1, a.epochs + 1):
        model.train()
        for x, y in tr_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss_fn(model(x), y).backward()
            opt.step()
        sched.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in va_dl:
                correct += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
        acc = correct / len(va_ds)
        print(f"epoch {ep:2d}  val_acc={acc:.4f}")
        if acc > best:
            best = acc
            torch.save(model.state_dict(), "submission/model.pth")
    print(f"best val_acc={best:.4f} -> submission/model.pth")


if __name__ == "__main__":
    main()
