"""Pertemuan 6 — Eksperimen Hyperparameter: CNN & ViT pada Imagenette (+ Weights & Biases).

Cara pakai di kelas: tulis nilai-nilai yang ingin dicoba sebagai LIST di CONFIG, lalu jalankan

    python train.py

train.py mencoba SEMUA kombinasi list (grid search), satu run W&B per kombinasi, lalu mencetak
ringkasan. Contoh: optimizer=["adam", "adamw", "nadam"] dan model=["resnet18", "vit_tiny"]
-> 3 x 2 = 6 run. Bandingkan hasilnya di dashboard W&B (filter per group).

Tiga file, dibaca dari atas ke bawah:
    train.py       CONFIG, loop eksperimen, training, evaluasi, logging W&B   (file ini)
    datareader.py  Imagenette -> DataLoader train & val (+ augmentasi, MixUp/CutMix)
    models.py      memilih arsitektur dari folder architectures/ + strategi fine-tuning

Data: Imagenette train (9.469 citra) untuk training, Imagenette val (3.925 citra) untuk
early stopping, memilih checkpoint, dan membandingkan run.
"""
import csv
import itertools
import math
import random
import statistics
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import wandb

from datareader import NUM_CLASSES, build_loaders
from models import build_model, parameter_groups, set_train_mode

# =====================================================================================
# CONFIG
#   Nilai berbentuk LIST  -> dicoba satu per satu (semua kombinasi antar-list).
#   Nilai tunggal (bukan list) -> tetap untuk semua run.
#   Tips: ubah SATU kunci menjadi list berisi beberapa nilai, biarkan kunci lain 1 nilai.
#   Jumlah run = perkalian panjang semua list -> hati-hati, cepat membengkak!
# =====================================================================================
CONFIG = dict(
    # --- Data ---------------------------------------------------------------------
    img_size=[160],              # 128 | 160 | 224   (lebih besar = lebih akurat, lebih lambat)
    train_fraction=[1.0],        # 0.1 = pakai 10% data train (simulasi data sedikit)
    augment=["basic"],           # "none" | "basic" | "strong"
    mixup_alpha=[0.0],           # 0 = mati; 0.2 umum dipakai
    cutmix_alpha=[0.0],          # 0 = mati; 1.0 umum dipakai

    # --- Model --------------------------------------------------------------------
    model=["resnet18", "vit_tiny"],  # "resnet18" | "mobilenetv3" | "efficientnet_b0" | "convnext_atto" | "vit_tiny"
    finetune=["full_ft"],        # "scratch" (bobot acak) | "linear_probe" (backbone beku) | "full_ft"
                                 # linear_probe & full_ft = bobot pretrained ImageNet
    init=["default"],            # hanya untuk scratch: "default" (timm) | "kaiming" | "xavier" | "normal"
    backbone_lr_mult=[0.1],      # lr backbone = lr * mult (1.0 = sama dengan head)
    dropout=[0.0],               # dropout sebelum classifier head (0.0 – 0.5)
    drop_path=[0.0],             # stochastic depth (0.0 – 0.2)

    # --- Optimizer ----------------------------------------------------------------
    optimizer=["adamw"],         # "sgd" | "adam" | "adamw" | "nadam" | "rmsprop"
    lr=[1e-3],                   # learning rate puncak untuk head
    weight_decay=[0.05],         # L2 (sgd/adam/nadam/rmsprop) atau decoupled (adamw)
    batch_size=[64],
    epochs=[10],                 # RTX 5070: ±1 menit per run untuk 10 epoch
    label_smoothing=[0.0],       # 0.0 – 0.2

    # --- LR scheduler -------------------------------------------------------------
    scheduler=["cosine"],        # "none" | "step" | "cosine" | "onecycle" | "plateau"
    warmup_epochs=[1],           # warmup linear (tidak dipakai onecycle & plateau)

    # --- Early stopping -----------------------------------------------------------
    patience=[3],                # berhenti jika val tak membaik N epoch; None = tanpa early stopping

    # --- Presisi numerik ----------------------------------------------------------
    precision=["fp16_amp"],      # "fp16_amp" (mixed precision, hanya CUDA) | "fp32" | "fp64" (sangat lambat)

    # --- Pengaturan tetap (jarang diubah) -----------------------------------------
    seed=42,                     # satu seed untuk semua run (demi waktu; selisih kecil bisa saja noise)
    momentum=0.9,                # sgd & rmsprop
    step_size=4,                 # scheduler "step": lr dikali gamma tiap step_size epoch
    gamma=0.1,                   # dipakai "step" & "plateau"
    plateau_patience=2,          # scheduler "plateau": turunkan lr jika val_loss tak membaik N epoch
    min_lr=1e-6,
    monitor="val_loss",          # metrik early stopping & checkpoint: "val_loss" | "val_acc" | "val_f1"
    min_delta=1e-3,
    grad_clip=None,              # None = tanpa clipping; mis. 1.0
    data_root="./data",
    num_workers=8,               # proses loading data paralel; sesuaikan jumlah core CPU (Colab: 2)

    # --- Weights & Biases ---------------------------------------------------------
    wandb_project="if25-p6-hyperparameter",
    wandb_entity=None,           # None = akun pribadi; isi nama team W&B kelas bila ada
    wandb_mode="online",         # "online" | "offline" (sinkron nanti: wandb sync) | "disabled"
    wandb_group=None,            # None = otomatis dari kunci yang divariasikan, mis. "optimizer"
    notes="",                    # tulis HIPOTESIS sebelum menjalankan, mis. "nadam ≈ adam"
)


# =====================================================================================
# Daftar eksperimen
# =====================================================================================
def expand_grid(config):
    """CONFIG -> daftar dict (satu per run) + daftar kunci yang nilainya lebih dari satu."""
    list_keys = [k for k, v in config.items() if isinstance(v, list)]
    varied = [k for k in list_keys if len(config[k]) > 1]
    runs = []
    for values in itertools.product(*(config[k] for k in list_keys)):
        runs.append({**config, **dict(zip(list_keys, values))})
    return runs, varied


def make_run_name(c, varied):
    """Contoh: "vit_tiny-optimizer=nadam" (hanya kunci yang divariasikan yang ditulis)."""
    parts = [f"{k}={c[k]}" for k in varied if k != "model"]
    return "-".join([c["model"], *parts]) if parts else f"{c['model']}-baseline"


def make_group(c, varied):
    if c["wandb_group"]:
        return c["wandb_group"]
    keys = [k for k in varied if k != "model"]
    return "-".join(keys) if keys else "baseline"


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


def get_dtype(c, device):
    """Tipe data bobot & input. fp16_amp tetap menyimpan bobot di fp32; autocast menurunkan sebagian operasi ke fp16."""
    if c["precision"] == "fp64":
        if device == "mps":
            raise ValueError("fp64 tidak didukung MPS (Apple Silicon); pakai GPU CUDA atau CPU")
        return torch.float64
    if c["precision"] not in ("fp16_amp", "fp32"):
        raise ValueError(f"precision tidak dikenal: {c['precision']}")
    return torch.float32


def build_optimizer(groups, c):
    name, wd = c["optimizer"], c["weight_decay"]
    if name == "sgd":
        return torch.optim.SGD(groups, momentum=c["momentum"], nesterov=True, weight_decay=wd)
    if name == "adam":
        return torch.optim.Adam(groups, weight_decay=wd)
    if name == "adamw":
        return torch.optim.AdamW(groups, weight_decay=wd)
    if name == "nadam":
        return torch.optim.NAdam(groups, weight_decay=wd)
    if name == "rmsprop":
        return torch.optim.RMSprop(groups, momentum=c["momentum"], weight_decay=wd)
    raise ValueError(f"optimizer tidak dikenal: {name}")


def build_scheduler(opt, c, steps_per_epoch):
    """Semua scheduler di-step per BATCH, kecuali "plateau" (per epoch, butuh val_loss)."""
    S = torch.optim.lr_scheduler
    name = c["scheduler"]
    total = c["epochs"] * steps_per_epoch
    warmup = c["warmup_epochs"] * steps_per_epoch

    if name == "plateau":
        return S.ReduceLROnPlateau(opt, mode="min", factor=c["gamma"], patience=c["plateau_patience"],
                                   min_lr=c["min_lr"])
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
    warm = S.LinearLR(opt, start_factor=0.01, total_iters=warmup)  # lr naik dari 1% ke 100%
    return S.SequentialLR(opt, [warm, main], milestones=[warmup])


class EarlyStopping:
    """Mencatat nilai terbaik; should_stop = True jika tidak membaik selama `patience` epoch."""

    def __init__(self, monitor, patience, min_delta):
        self.mode = "min" if monitor == "val_loss" else "max"
        self.patience, self.min_delta = patience, min_delta
        self.best, self.bad_epochs = None, 0

    def update(self, value):
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
        return self.patience is not None and self.bad_epochs >= self.patience


# =====================================================================================
# Training & evaluasi
# =====================================================================================
def train_one_epoch(model, loader, opt, sched, loss_fn, scaler, c, device, step):
    set_train_mode(model, c)
    use_amp = c["precision"] == "fp16_amp" and device == "cuda"
    dtype = get_dtype(c, device)
    params = [p for p in model.parameters() if p.requires_grad]
    total_loss, correct, n = 0.0, 0, 0

    for x, y in loader:
        x = x.to(device, dtype)
        y = y.to(device, dtype) if y.is_floating_point() else y.to(device)  # label soft (MixUp) ikut dtype
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(x)
            loss = loss_fn(logits, y)

        if not torch.isfinite(loss):  # divergen (lr terlalu besar?) -> run dihentikan
            return math.nan, math.nan, step

        opt.zero_grad()
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
        step += 1
        if step % 10 == 0:
            wandb.log({"train/loss": loss.item(), "train/grad_norm": grad_norm.item(),
                       "train/lr": opt.param_groups[0]["lr"]}, step=step)
    return total_loss / n, correct / n, step


@torch.no_grad()
def evaluate(model, loader, loss_fn, c, device):
    model.eval()
    use_amp = c["precision"] == "fp16_amp" and device == "cuda"
    dtype = get_dtype(c, device)
    total_loss, preds, targets = 0.0, [], []
    for x, y in loader:
        x, y = x.to(device, dtype), y.to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(x)
            total_loss += loss_fn(logits, y).item() * x.size(0)
        preds.append(logits.argmax(1).cpu())
        targets.append(y.cpu())
    preds, targets = torch.cat(preds), torch.cat(targets)

    # macro-F1 dari confusion matrix: rata-rata F1 tiap kelas
    cm = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    cm.index_put_((targets, preds), torch.ones_like(preds), accumulate=True)
    tp = cm.diag().float()
    precision = tp / cm.sum(0).clamp(min=1)
    recall = tp / cm.sum(1).clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
    return {"loss": total_loss / len(targets), "acc": (preds == targets).float().mean().item(),
            "f1": f1.mean().item(), "preds": preds, "targets": targets}


def run_experiment(c, varied):
    """Satu run: training + validasi tiap epoch + logging W&B. Mengembalikan ringkasan hasil."""
    name = make_run_name(c, varied)
    seed_everything(c["seed"])
    device = get_device()

    model = build_model(c, NUM_CLASSES)
    data_cfg = timm.data.resolve_data_config(model=model)  # mean/std sesuai bobot pretrained
    train_dl, val_dl, class_names = build_loaders(c, data_cfg["mean"], data_cfg["std"], device)
    model.to(device, get_dtype(c, device))  # fp64: semua bobot & buffer BatchNorm jadi float64
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    opt = build_optimizer(parameter_groups(model, c), c)
    sched = build_scheduler(opt, c, len(train_dl))
    train_loss_fn = nn.CrossEntropyLoss(label_smoothing=c["label_smoothing"])
    val_loss_fn = nn.CrossEntropyLoss()
    # GradScaler: kalikan loss sebelum backward agar gradien fp16 yang kecil tidak menjadi 0 (underflow)
    scaler = torch.amp.GradScaler("cuda", enabled=c["precision"] == "fp16_amp" and device == "cuda")
    stopper = EarlyStopping(c["monitor"], c["patience"], c["min_delta"])

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Semua hyperparameter run ini (termasuk nilai tetap) tercatat di config W&B
    run = wandb.init(project=c["wandb_project"], entity=c["wandb_entity"], mode=c["wandb_mode"],
                     name=name, group=make_group(c, varied), notes=c["notes"], tags=[c["model"], *varied],
                     config={**c, "varied": varied, "device": device, "n_params": n_params,
                             "n_trainable": n_trainable, "n_train": len(train_dl.dataset),
                             "n_val": len(val_dl.dataset)})
    wandb.define_metric("epoch")
    wandb.define_metric("epoch/*", step_metric="epoch")
    wandb.define_metric("val/*", step_metric="epoch")

    print(f"[{name}] device={device} | params={n_params / 1e6:.2f}M (trainable {n_trainable / 1e6:.2f}M)"
          f" | train={len(train_dl.dataset)} val={len(val_dl.dataset)}")

    ckpt_path = Path("checkpoints") / f"{name}.pth"
    ckpt_path.parent.mkdir(exist_ok=True)
    step, best, best_epoch, stopped_epoch, epoch_times = 0, None, 0, c["epochs"], []
    t_start = time.time()

    for epoch in range(1, c["epochs"] + 1):
        t0 = time.time()
        train_loss, train_acc, step = train_one_epoch(model, train_dl, opt, sched, train_loss_fn,
                                                      scaler, c, device, step)
        if not math.isfinite(train_loss):
            stopped_epoch = epoch
            run.summary["diverged"] = True
            print(f"epoch {epoch:2d} | loss NaN/Inf -> DIVERGEN, run dihentikan (learning rate terlalu besar?)")
            break

        val = evaluate(model, val_dl, val_loss_fn, c, device)
        epoch_times.append(time.time() - t0)
        if c["scheduler"] == "plateau":
            sched.step(val["loss"])

        is_best = stopper.update(val[c["monitor"].removeprefix("val_")])
        if is_best:  # simpan bobot terbaik (bukan bobot epoch terakhir)
            best, best_epoch = val, epoch
            torch.save(model.state_dict(), ckpt_path)

        lr = opt.param_groups[0]["lr"]
        wandb.log({"epoch": epoch, "epoch/train_loss": train_loss, "epoch/train_acc": train_acc,
                   "val/loss": val["loss"], "val/acc": val["acc"], "val/f1": val["f1"],
                   "epoch/gap_acc": train_acc - val["acc"], "epoch/lr": lr,
                   "epoch/time_sec": epoch_times[-1]}, step=step)
        print(f"epoch {epoch:2d} | train loss {train_loss:.4f} acc {train_acc:.4f} | val loss {val['loss']:.4f}"
              f" acc {val['acc']:.4f} f1 {val['f1']:.4f} | lr {lr:.2e} | {epoch_times[-1]:.0f}s"
              f"{'  *' if is_best else ''}")

        if stopper.should_stop:
            stopped_epoch = epoch
            print(f"Early stopping: {c['monitor']} tidak membaik selama {c['patience']} epoch.")
            break

    minutes = (time.time() - t_start) / 60
    sec_per_epoch = statistics.median(epoch_times) if epoch_times else math.nan  # median: abaikan epoch 1 yang lambat
    gpu_mem_gb = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else math.nan
    result = {"run_name": name, "best_epoch": best_epoch, "stopped_epoch": stopped_epoch,
              "val_acc": round(best["acc"], 4) if best else math.nan,
              "val_f1": round(best["f1"], 4) if best else math.nan, "minutes": round(minutes, 1),
              "sec_per_epoch": round(sec_per_epoch, 1), "gpu_mem_gb": round(gpu_mem_gb, 2)}
    print(f"TERBAIK (epoch {best_epoch}) | val acc {result['val_acc']} macro-F1 {result['val_f1']}"
          f" | {minutes:.1f} menit, {sec_per_epoch:.1f} s/epoch, memori GPU puncak {gpu_mem_gb:.2f} GB")

    run.summary.update({"best/val_acc": result["val_acc"], "best/val_f1": result["val_f1"],
                        "best_epoch": best_epoch, "stopped_epoch": stopped_epoch, "train_time_min": minutes,
                        "sec_per_epoch": sec_per_epoch, "gpu_mem_gb": gpu_mem_gb})
    if best:
        wandb.log({"val/confusion_matrix": wandb.plot.confusion_matrix(
            y_true=best["targets"].tolist(), preds=best["preds"].tolist(), class_names=class_names)}, step=step)
    wandb.finish()
    return result


def save_results(rows):
    """Cadangan lokal (berguna jika internet kelas bermasalah): satu baris per run di results.csv."""
    path = Path("results.csv")
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main():
    runs, varied = expand_grid(CONFIG)
    print(f"{len(runs)} run akan dijalankan | divariasikan: {', '.join(varied) or '(tidak ada)'}")

    hyperparams = [k for k, v in CONFIG.items() if isinstance(v, list)]
    results = []
    for i, c in enumerate(runs, 1):
        print(f"\n===== Run {i}/{len(runs)} =====")
        result = run_experiment(c, varied)
        results.append({**result, "group": make_group(c, varied), **{k: c[k] for k in hyperparams}})
    save_results(results)

    print("\n===== RINGKASAN (urut val acc) =====")
    for r in sorted(results, key=lambda r: -r["val_acc"] if math.isfinite(r["val_acc"]) else 1):
        print(f"{r['run_name']:<45} val acc {r['val_acc']:<7} f1 {r['val_f1']:<7}"
              f" best ep {r['best_epoch']:>2}/{r['stopped_epoch']:<2} {r['sec_per_epoch']} s/epoch"
              f" {r['minutes']} menit")


if __name__ == "__main__":
    main()
