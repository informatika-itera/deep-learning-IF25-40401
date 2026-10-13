"""Pertemuan 6 — Eksperimen Hyperparameter: CNN & ViT pada Imagenette (+ Weights & Biases).

Satu file untuk demo di kelas: ubah nilai di blok CONFIG, jalankan, lalu bandingkan
run-run Anda di dashboard W&B. Tidak ada argparse — semua diganti manual di bawah.

    uv pip install -r requirements.txt      # torch, torchvision, timm, wandb
    wandb login                             # sekali saja (atau set wandb_mode="offline")
    python train.py

Alur data (penting!):
    Imagenette train (9.469 citra) --> 85% train  +  15% valid  (memilih model & early stopping)
    Imagenette val   (3.925 citra) --> test        (hanya disentuh SEKALI di akhir)

Model (timm, varian teringan tiap keluarga — semuanya pretrained ImageNet):
    resnet18         11,2 M param   CNN klasik + residual            (P5)
    mobilenetv3      1,5 M param    depthwise separable + SE         (P5)
    efficientnet_b0  4,0 M param    compound scaling                 (P5)
    convnext_atto    3,4 M param    CNN "modern" ala Transformer     (P5)
    vit_tiny         5,5 M param    Vision Transformer (patch 16)
    (jumlah parameter setelah head diganti menjadi 10 kelas)
"""
import csv
import math
import random
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader, Subset, default_collate
from torchvision import datasets
from torchvision.transforms import v2

# =====================================================================================
# CONFIG — ganti nilai di sini untuk setiap eksperimen (satu perubahan per run!)
# =====================================================================================
CONFIG = dict(
    # --- Data ---------------------------------------------------------------------
    data_root="./data",
    img_size=160,              # 128 | 160 | 224   (lebih besar = lebih akurat, lebih lambat)
    train_fraction=1.0,        # 0.2 = pakai 20% data train (demo cepat / simulasi data sedikit)
    valid_split=0.15,          # porsi train yang disisihkan sebagai validation set
    augment="basic",           # "none" | "basic" | "strong"
    mixup_alpha=0.0,           # 0 = mati; 0.2 umum dipakai
    cutmix_alpha=0.0,          # 0 = mati; 1.0 umum dipakai

    # --- Model --------------------------------------------------------------------
    model="resnet18",          # "resnet18" | "mobilenetv3" | "efficientnet_b0" | "convnext_atto" | "vit_tiny"
    pretrained=True,           # False = latih dari nol (bandingkan!)
    freeze_backbone=False,     # True = linear probing (hanya head yang dilatih)
    dropout=0.0,               # dropout sebelum classifier head (0.0 – 0.5)
    drop_path=0.0,             # stochastic depth: buang cabang residual secara acak (0.0 – 0.2)

    # --- Optimizer ----------------------------------------------------------------
    optimizer="adamw",         # "sgd" | "adam" | "adamw" | "rmsprop"
    lr=1e-3,                   # learning rate puncak untuk head
    backbone_lr_mult=0.1,      # lr backbone = lr * mult (1.0 = sama; 0.1 = backbone lebih pelan)
    weight_decay=0.05,         # L2 (adam/sgd/rmsprop) atau decoupled (adamw)
    momentum=0.9,              # dipakai sgd & rmsprop
    nesterov=True,             # dipakai sgd
    batch_size=64,
    epochs=30,                 # RTX 5070: ±2,5–4,5 menit untuk 30 epoch (bottleneck: loading data di CPU)
    grad_clip=None,            # None = tanpa clipping; mis. 1.0
    label_smoothing=0.0,       # 0.0 – 0.2

    # --- LR scheduler -------------------------------------------------------------
    scheduler="cosine",        # "none" | "step" | "cosine" | "onecycle" | "plateau"
    warmup_epochs=1,           # warmup linear (tidak dipakai onecycle & plateau)
    step_size=10,              # scheduler "step": turunkan lr tiap N epoch ...
    gamma=0.1,                 # ... dikali gamma
    plateau_patience=2,        # scheduler "plateau": turunkan lr jika val_loss tak membaik N epoch
    min_lr=1e-6,

    # --- Early stopping -----------------------------------------------------------
    early_stopping=True,
    patience=5,                # berhenti jika metrik tak membaik selama N epoch
    min_delta=1e-3,
    monitor="val_loss",        # "val_loss" (min) | "val_acc" (max) | "val_f1" (max)

    # --- Lain-lain ----------------------------------------------------------------
    seed=42,
    amp=True,                  # mixed precision (hanya aktif di GPU CUDA)
    num_workers=8,             # proses loading data paralel; sesuaikan jumlah core CPU (Colab: 2)
    log_images=True,           # tabel contoh prediksi test set di W&B

    # --- Weights & Biases ---------------------------------------------------------
    wandb_project="if25-p6-hyperparameter",
    wandb_entity=None,         # None = akun pribadi; isi nama team W&B kelas bila ada
    wandb_mode="online",       # "online" | "offline" (sinkron nanti: wandb sync) | "disabled"
    wandb_group=None,          # kelompokkan run satu seri eksperimen, mis. "sweep-lr"
    run_name=None,             # None = dibuat otomatis dari hyperparameter
    notes="",                  # tulis HIPOTESIS Anda sebelum menjalankan, mis. "lr 1e-2 akan divergen"
)

# Nama pendek -> checkpoint timm (pretrained ImageNet-1k; vit_tiny: in21k -> in1k)
MODEL_ZOO = {
    "resnet18": "resnet18.a1_in1k",
    "mobilenetv3": "mobilenetv3_small_100.lamb_in1k",
    "efficientnet_b0": "efficientnet_b0.ra_in1k",
    "convnext_atto": "convnext_atto.d2_in1k",
    "vit_tiny": "vit_tiny_patch16_224.augreg_in21k_ft_in1k",
}


# =====================================================================================
# Utilitas
# =====================================================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_run_name(c):
    name = f"{c['model']}-{c['optimizer']}-lr{c['lr']:g}-{c['scheduler']}"
    if not c["pretrained"]:
        name += "-scratch"
    if c["freeze_backbone"]:
        name += "-frozen"
    if c["dropout"] > 0:
        name += f"-do{c['dropout']:g}"
    if c["augment"] != "basic":
        name += f"-aug{c['augment']}"
    return name


# =====================================================================================
# Data
# =====================================================================================
def build_transforms(c, mean, std):
    s = c["img_size"]
    normalize = [v2.ToImage(), v2.ToDtype(torch.float32, scale=True), v2.Normalize(mean, std)]
    eval_tf = v2.Compose([v2.Resize(int(s / 0.875)), v2.CenterCrop(s), *normalize])

    if c["augment"] == "none":
        return v2.Compose([v2.Resize(int(s / 0.875)), v2.CenterCrop(s), *normalize]), eval_tf
    aug = [v2.RandomResizedCrop(s, scale=(0.35, 1.0)), v2.RandomHorizontalFlip()]
    if c["augment"] == "strong":
        aug += [v2.TrivialAugmentWide()]
        normalize += [v2.RandomErasing(p=0.25)]
    return v2.Compose(aug + normalize), eval_tf


class MixCollate:
    """Gabungkan batch lalu terapkan MixUp/CutMix (class, bukan lambda, agar bisa dipakai num_workers > 0)."""

    def __init__(self, mix):
        self.mix = v2.RandomChoice(mix)

    def __call__(self, batch):
        return self.mix(*default_collate(batch))


def build_loaders(c, mean, std, device):
    root = Path(c["data_root"])
    download = not (root / "imagenette2-160").exists()
    train_tf, eval_tf = build_transforms(c, mean, std)

    # Dua objek dataset dari folder yang sama: satu dengan augmentasi (train), satu tanpa (valid)
    full_train = datasets.Imagenette(root, split="train", size="160px", download=download, transform=train_tf)
    full_valid = datasets.Imagenette(root, split="train", size="160px", transform=eval_tf)
    test_ds = datasets.Imagenette(root, split="val", size="160px", transform=eval_tf)

    idx = torch.randperm(len(full_train), generator=torch.Generator().manual_seed(c["seed"])).tolist()
    n_valid = int(len(idx) * c["valid_split"])
    valid_idx, train_idx = idx[:n_valid], idx[n_valid:]
    train_idx = train_idx[: int(len(train_idx) * c["train_fraction"])]

    # MixUp / CutMix dilakukan per batch -> label menjadi "soft" (distribusi probabilitas)
    mix = []
    if c["mixup_alpha"] > 0:
        mix.append(v2.MixUp(alpha=c["mixup_alpha"], num_classes=10))
    if c["cutmix_alpha"] > 0:
        mix.append(v2.CutMix(alpha=c["cutmix_alpha"], num_classes=10))
    collate = MixCollate(mix) if mix else default_collate

    kw = dict(batch_size=c["batch_size"], num_workers=c["num_workers"],
              pin_memory=device == "cuda", persistent_workers=c["num_workers"] > 0)
    train_dl = DataLoader(Subset(full_train, train_idx), shuffle=True, drop_last=True, collate_fn=collate, **kw)
    valid_dl = DataLoader(Subset(full_valid, valid_idx), shuffle=False, **kw)
    test_dl = DataLoader(test_ds, shuffle=False, **kw)
    class_names = [names[0] for names in full_train.classes]
    return train_dl, valid_dl, test_dl, class_names


# =====================================================================================
# Model, optimizer, scheduler
# =====================================================================================
def build_model(c):
    kwargs = dict(pretrained=c["pretrained"], num_classes=10, drop_rate=c["dropout"])
    if c["drop_path"] > 0:
        kwargs["drop_path_rate"] = c["drop_path"]
    if c["model"] == "vit_tiny":
        kwargs["img_size"] = c["img_size"]  # positional embedding diinterpolasi otomatis oleh timm
    model = timm.create_model(MODEL_ZOO[c["model"]], **kwargs)

    head_params = {id(p) for p in model.get_classifier().parameters()}
    if c["freeze_backbone"]:
        for p in model.parameters():
            p.requires_grad = id(p) in head_params
    return model, head_params


def set_train_mode(model, c):
    model.train()
    if c["freeze_backbone"]:  # backbone beku -> statistik BatchNorm juga jangan ikut berubah
        for m in model.modules():
            if isinstance(m, nn.modules.batchnorm._BatchNorm):
                m.eval()


def build_optimizer(model, head_params, c):
    head = [p for p in model.parameters() if p.requires_grad and id(p) in head_params]
    backbone = [p for p in model.parameters() if p.requires_grad and id(p) not in head_params]
    groups = [{"params": head, "lr": c["lr"]}]
    if backbone:
        groups.append({"params": backbone, "lr": c["lr"] * c["backbone_lr_mult"]})

    name, wd = c["optimizer"], c["weight_decay"]
    if name == "sgd":
        return torch.optim.SGD(groups, momentum=c["momentum"], nesterov=c["nesterov"], weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(groups, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(groups, weight_decay=wd)
    if name == "rmsprop":
        return torch.optim.RMSprop(groups, momentum=c["momentum"], weight_decay=wd)
    raise ValueError(f"optimizer tidak dikenal: {name}")


def build_scheduler(opt, c, steps_per_epoch):
    """Semua scheduler di-step per BATCH, kecuali 'plateau' (per epoch, butuh val_loss)."""
    S = torch.optim.lr_scheduler
    name = c["scheduler"]
    total = c["epochs"] * steps_per_epoch
    warmup = c["warmup_epochs"] * steps_per_epoch

    if name == "plateau":
        return S.ReduceLROnPlateau(opt, mode="min", factor=c["gamma"], patience=c["plateau_patience"], min_lr=c["min_lr"])
    if name == "onecycle":
        return S.OneCycleLR(opt, max_lr=[g["lr"] for g in opt.param_groups], total_steps=total, pct_start=0.25)
    if name == "none":
        main = S.ConstantLR(opt, factor=1.0)
    elif name == "step":
        main = S.StepLR(opt, step_size=c["step_size"] * steps_per_epoch, gamma=c["gamma"])
    elif name == "cosine":
        main = S.CosineAnnealingLR(opt, T_max=max(1, total - warmup), eta_min=c["min_lr"])
    else:
        raise ValueError(f"scheduler tidak dikenal: {name}")
    if warmup == 0:
        return main
    warm = S.LinearLR(opt, start_factor=0.01, total_iters=warmup)
    return S.SequentialLR(opt, [warm, main], milestones=[warmup])


class EarlyStopping:
    def __init__(self, monitor, patience, min_delta):
        self.mode = "min" if monitor == "val_loss" else "max"
        self.patience, self.min_delta = patience, min_delta
        self.best, self.bad_epochs = None, 0

    def step(self, value):
        """Return True jika value adalah yang terbaik sejauh ini."""
        improved = (self.best is None
                    or (self.mode == "min" and value < self.best - self.min_delta)
                    or (self.mode == "max" and value > self.best + self.min_delta))
        if improved:
            self.best, self.bad_epochs = value, 0
        else:
            self.bad_epochs += 1
        return improved

    @property
    def should_stop(self):
        return self.bad_epochs >= self.patience


# =====================================================================================
# Training & evaluasi
# =====================================================================================
def train_one_epoch(model, loader, opt, sched, loss_fn, scaler, c, device, global_step):
    set_train_mode(model, c)
    use_amp = c["amp"] and device == "cuda"
    params = [p for p in model.parameters() if p.requires_grad]
    total_loss, correct, n = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(x)
            loss = loss_fn(logits, y)

        if not torch.isfinite(loss):  # divergen -> hentikan epoch, main() akan menghentikan training
            return math.nan, math.nan, global_step

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, c["grad_clip"] or float("inf"))
        scaler.step(opt)
        scaler.update()
        if c["scheduler"] != "plateau":
            sched.step()

        hard_y = y.argmax(1) if y.ndim == 2 else y  # MixUp/CutMix -> label soft
        total_loss += loss.item() * x.size(0)
        correct += (logits.argmax(1) == hard_y).sum().item()
        n += x.size(0)
        global_step += 1
        if global_step % 10 == 0:
            wandb.log({"train/loss": loss.item(), "train/grad_norm": grad_norm.item(),
                       "train/lr": opt.param_groups[0]["lr"]}, step=global_step)
    return total_loss / n, correct / n, global_step


@torch.no_grad()
def evaluate(model, loader, loss_fn, device, num_classes=10):
    model.eval()
    total_loss, preds, targets = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += loss_fn(logits, y).item() * x.size(0)
        preds.append(logits.argmax(1).cpu())
        targets.append(y.cpu())
    preds, targets = torch.cat(preds), torch.cat(targets)

    cm = torch.zeros(num_classes, num_classes, dtype=torch.long)
    cm.index_put_((targets, preds), torch.ones_like(preds), accumulate=True)
    tp = cm.diag().float()
    precision = tp / cm.sum(0).clamp(min=1)
    recall = tp / cm.sum(1).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
    return {"loss": total_loss / len(targets), "acc": (preds == targets).float().mean().item(),
            "f1": f1.mean().item(), "preds": preds, "targets": targets}


def log_prediction_table(model, dataset, class_names, mean, std, device, n=40):
    """Tabel W&B: citra test + label asli + prediksi + confidence."""
    model.eval()
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(len(dataset), generator=g)[:n].tolist()
    mean, std = torch.tensor(mean).view(3, 1, 1), torch.tensor(std).view(3, 1, 1)
    table = wandb.Table(columns=["image", "label", "prediksi", "confidence", "benar"])
    with torch.no_grad():
        for i in idx:
            x, y = dataset[i]
            prob = model(x.unsqueeze(0).to(device)).softmax(1)[0].cpu()
            p = prob.argmax().item()
            img = (x * std + mean).clamp(0, 1)
            table.add_data(wandb.Image(img), class_names[y], class_names[p], round(prob[p].item(), 3), p == y)
    wandb.log({"test/predictions": table})


def main(config=None):
    """Jalankan satu eksperimen. config=None -> pakai CONFIG di atas. Mengembalikan ringkasan + riwayat per epoch."""
    c = dict(config or CONFIG)
    c["run_name"] = c["run_name"] or make_run_name(c)
    seed_everything(c["seed"])
    device = get_device()

    model, head_params = build_model(c)
    data_cfg = timm.data.resolve_data_config(model=model)  # mean/std sesuai bobot pretrained
    mean, std = data_cfg["mean"], data_cfg["std"]
    train_dl, valid_dl, test_dl, class_names = build_loaders(c, mean, std, device)
    model.to(device)

    opt = build_optimizer(model, head_params, c)
    sched = build_scheduler(opt, c, len(train_dl))
    train_loss_fn = nn.CrossEntropyLoss(label_smoothing=c["label_smoothing"])
    eval_loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=c["amp"] and device == "cuda")
    stopper = EarlyStopping(c["monitor"], c["patience"], c["min_delta"])

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    run = wandb.init(project=c["wandb_project"], entity=c["wandb_entity"], name=c["run_name"],
                     group=c["wandb_group"], mode=c["wandb_mode"], config=c, notes=c["notes"],
                     tags=[c["model"], c["optimizer"], c["scheduler"]])
    run.config.update({"device": device, "n_params": n_params, "n_trainable": n_trainable,
                       "n_train": len(train_dl.dataset), "n_valid": len(valid_dl.dataset)})
    wandb.define_metric("epoch")
    for key in ("epoch/*", "valid/*"):
        wandb.define_metric(key, step_metric="epoch")

    print(f"[{c['run_name']}] device={device} | params={n_params / 1e6:.2f}M (trainable {n_trainable / 1e6:.2f}M)"
          f" | train={len(train_dl.dataset)} valid={len(valid_dl.dataset)} test={len(test_dl.dataset)}")

    ckpt_path = Path("checkpoints") / f"{c['run_name']}.pth"
    ckpt_path.parent.mkdir(exist_ok=True)
    global_step, best_epoch, stopped_epoch, history = 0, 0, c["epochs"], []
    t_start = time.time()

    for epoch in range(1, c["epochs"] + 1):
        t0 = time.time()
        tr_loss, tr_acc, global_step = train_one_epoch(model, train_dl, opt, sched, train_loss_fn,
                                                       scaler, c, device, global_step)
        va = evaluate(model, valid_dl, eval_loss_fn, device) if math.isfinite(tr_loss) else {"loss": math.nan}
        if not math.isfinite(va["loss"]):
            stopped_epoch = epoch
            run.summary["diverged"] = True
            print(f"epoch {epoch:2d} | loss NaN/Inf -> DIVERGEN, training dihentikan (learning rate terlalu besar?)")
            break
        if c["scheduler"] == "plateau":
            sched.step(va["loss"])

        is_best = stopper.step(va[c["monitor"].removeprefix("val_")])
        if is_best:
            best_epoch = epoch
            torch.save(model.state_dict(), ckpt_path)

        log = {"epoch": epoch, "epoch/train_loss": tr_loss, "epoch/train_acc": tr_acc,
               "valid/loss": va["loss"], "valid/acc": va["acc"], "valid/f1": va["f1"],
               "epoch/lr": opt.param_groups[0]["lr"], "epoch/gap_acc": tr_acc - va["acc"],
               "epoch/time_sec": time.time() - t0}
        wandb.log(log, step=global_step)
        history.append(log)
        print(f"epoch {epoch:2d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | valid loss {va['loss']:.4f}"
              f" acc {va['acc']:.4f} f1 {va['f1']:.4f} | lr {opt.param_groups[0]['lr']:.2e}"
              f" | {time.time() - t0:.0f}s{'  *' if is_best else ''}")

        if c["early_stopping"] and stopper.should_stop:
            stopped_epoch = epoch
            print(f"Early stopping: {c['monitor']} tidak membaik selama {c['patience']} epoch.")
            break

    # --- Evaluasi akhir pada test set dengan bobot TERBAIK (bukan bobot epoch terakhir) ---
    if best_epoch == 0:  # divergen sebelum ada checkpoint yang valid
        wandb.finish()
        return {"run_name": c["run_name"], "best_epoch": 0, "stopped_epoch": stopped_epoch,
                "test_acc": math.nan, "test_f1": math.nan, "history": history}
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    te = evaluate(model, test_dl, eval_loss_fn, device)
    total_min = (time.time() - t_start) / 60
    print(f"TEST (bobot epoch {best_epoch}) | loss {te['loss']:.4f} acc {te['acc']:.4f} macro-F1 {te['f1']:.4f}"
          f" | total {total_min:.1f} menit")

    run.summary.update({"test/loss": te["loss"], "test/acc": te["acc"], "test/f1": te["f1"],
                        "best_epoch": best_epoch, "stopped_epoch": stopped_epoch,
                        "best_" + c["monitor"]: stopper.best, "train_time_min": total_min})
    wandb.log({"test/confusion_matrix": wandb.plot.confusion_matrix(
        y_true=te["targets"].tolist(), preds=te["preds"].tolist(), class_names=class_names)}, step=global_step)
    if c["log_images"]:
        log_prediction_table(model, test_dl.dataset, class_names, mean, std, device)

    # Cadangan lokal (berguna jika internet kelas bermasalah): satu baris per run
    results = Path("results.csv")
    row = {**{k: c[k] for k in ("run_name", "model", "pretrained", "freeze_backbone", "optimizer", "lr",
                                "scheduler", "dropout", "weight_decay", "augment", "epochs")},
           "best_epoch": best_epoch, "stopped_epoch": stopped_epoch,
           "test_acc": round(te["acc"], 4), "test_f1": round(te["f1"], 4), "minutes": round(total_min, 1),
           "params_m": round(n_params / 1e6, 2)}
    write_header = not results.exists()
    with results.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)
    wandb.finish()
    return {**row, "history": history}


if __name__ == "__main__":
    main()
