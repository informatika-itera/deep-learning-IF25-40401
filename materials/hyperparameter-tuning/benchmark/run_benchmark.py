"""Suite eksperimen untuk slide Pertemuan 6 (dijalankan sekali di mesin GPU).

Setiap eksperimen = CONFIG di train.py + beberapa perubahan. Hasil ditulis ke:
    benchmark/results/runs.csv      satu baris per run (konfigurasi + metrik test)
    benchmark/results/history.csv   satu baris per (run, epoch) — untuk kurva di slide

Jalankan dari folder materials/hyperparameter-tuning/:
    python benchmark/run_benchmark.py              # run yang sudah ada di runs.csv dilewati
    python benchmark/run_benchmark.py lr opt       # hanya grup tertentu
"""
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import train  # noqa: E402

OUT = Path(__file__).resolve().parent / "results"

# Baseline semua grup: CONFIG di train.py apa adanya (ResNet-18, AdamW 1e-3, cosine + warmup, 10 epoch)
BASE = dict(num_workers=8, wandb_mode="offline", log_images=False)

# Rezim data sedikit (10% train ~ 800 citra), 25 epoch tanpa early stopping, tanpa regularisasi apa pun
LOWDATA = dict(train_fraction=0.1, epochs=25, early_stopping=False)
NOREG = dict(augment="none", dropout=0.0, weight_decay=0.0, label_smoothing=0.0)

EXPERIMENTS = {
    "lr": {f"lr-{lr:g}": dict(lr=lr) for lr in (1e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 1e-1)},
    "opt": {
        "opt-sgd-1e-3": dict(optimizer="sgd", lr=1e-3),
        "opt-sgd-1e-2": dict(optimizer="sgd", lr=1e-2),
        "opt-adam-wd0.05": dict(optimizer="adam"),
        "opt-adam-wd0": dict(optimizer="adam", weight_decay=0.0),
        "opt-rmsprop": dict(optimizer="rmsprop"),
        "opt-adamw": dict(optimizer="adamw"),
    },
    "sched": {
        "sched-none": dict(scheduler="none", warmup_epochs=0),
        "sched-step": dict(scheduler="step", warmup_epochs=0),
        "sched-cosine-nowarmup": dict(scheduler="cosine", warmup_epochs=0),
        "sched-cosine-warmup": dict(scheduler="cosine", warmup_epochs=1),
        "sched-onecycle": dict(scheduler="onecycle"),
        "sched-plateau": dict(scheduler="plateau", warmup_epochs=0),
    },
    "ft": {
        "ft-scratch": dict(pretrained=False, backbone_lr_mult=1.0),
        "ft-frozen-1e-3": dict(freeze_backbone=True),
        "ft-frozen-1e-2": dict(freeze_backbone=True, lr=1e-2),
        "ft-full-mult1": dict(backbone_lr_mult=1.0),
        "ft-full-mult0.1": dict(backbone_lr_mult=0.1),
    },
    "reg": {
        "reg-none": {**LOWDATA, **NOREG},
        "reg-earlystop": {**LOWDATA, **NOREG, "early_stopping": True},
        "reg-aug-basic": {**LOWDATA, **NOREG, "augment": "basic"},
        "reg-aug-strong": {**LOWDATA, **NOREG, "augment": "strong"},
        "reg-dropout0.5": {**LOWDATA, **NOREG, "dropout": 0.5},
        "reg-wd0.05": {**LOWDATA, **NOREG, "weight_decay": 0.05},
        "reg-ls0.1": {**LOWDATA, **NOREG, "label_smoothing": 0.1},
        "reg-mixcut": {**LOWDATA, **NOREG, "mixup_alpha": 0.2, "cutmix_alpha": 1.0},
        "reg-all": {**LOWDATA, "augment": "strong", "dropout": 0.3, "weight_decay": 0.05,
                    "label_smoothing": 0.1, "mixup_alpha": 0.2, "cutmix_alpha": 1.0},
    },
    "model": {f"model-{m}": dict(model=m) for m in train.MODEL_ZOO},
    "model10": {f"model10-{m}": dict(model=m, train_fraction=0.1) for m in train.MODEL_ZOO},
    "bs": {
        "bs-32": dict(batch_size=32),
        "bs-64": dict(batch_size=64),
        "bs-256": dict(batch_size=256),
        "bs-256-lr4e-3": dict(batch_size=256, lr=4e-3),
    },
}

RUN_COLS = ["group", "run_name", "model", "pretrained", "freeze_backbone", "optimizer", "lr", "backbone_lr_mult",
            "weight_decay", "batch_size", "scheduler", "warmup_epochs", "augment", "dropout", "label_smoothing",
            "mixup_alpha", "cutmix_alpha", "train_fraction", "epochs", "early_stopping",
            "best_epoch", "stopped_epoch", "test_acc", "test_f1", "minutes", "params_m"]
HIST_COLS = ["group", "run_name", "epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "lr", "time_sec"]


def append(path, cols, rows):
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def main():
    OUT.mkdir(exist_ok=True)
    runs_csv, hist_csv = OUT / "runs.csv", OUT / "history.csv"
    done = set()
    if runs_csv.exists():
        with runs_csv.open() as f:
            done = {r["run_name"] for r in csv.DictReader(f)}

    groups = sys.argv[1:] or list(EXPERIMENTS)
    for group in groups:
        for name, override in EXPERIMENTS[group].items():
            if name in done:
                print(f"[skip] {name}")
                continue
            cfg = {**train.CONFIG, **BASE, **override, "run_name": name, "wandb_group": group}
            t0 = time.time()
            res = train.main(cfg)
            append(runs_csv, RUN_COLS, [{**cfg, **res, "group": group, "minutes": round((time.time() - t0) / 60, 2)}])
            append(hist_csv, HIST_COLS, [
                {"group": group, "run_name": name, "epoch": h["epoch"],
                 "train_loss": round(h["epoch/train_loss"], 5), "train_acc": round(h["epoch/train_acc"], 5),
                 "val_loss": round(h["valid/loss"], 5), "val_acc": round(h["valid/acc"], 5),
                 "val_f1": round(h["valid/f1"], 5), "lr": h["epoch/lr"], "time_sec": round(h["epoch/time_sec"], 2)}
                for h in res["history"]])


if __name__ == "__main__":
    main()
