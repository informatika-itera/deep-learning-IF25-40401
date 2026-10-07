"""Skrip evaluasi resmi Studio 2 — Klasifikasi Sampah (IF25-40401).

Skrip INI JUGA yang dipakai asisten dosen untuk menghitung skor pada validation set
tersembunyi. Pastikan submission kelompok Anda lolos skrip ini sebelum tenggat.

Pemakaian (uji pada split validasi buatan Anda sendiri dari data train):
    python evaluate.py --submission ./submission --data ./my_val

    --submission  folder berisi model.py dan model.pth
    --data        folder berstruktur <data>/<nama_kelas>/*.jpg
    --out         (opsional) simpan hasil ke file JSON

Kontrak submission (folder submission/):
    model.py   wajib mendefinisikan:
                 get_model()     -> torch.nn.Module (arsitektur saja, TANPA mengunduh bobot)
                 get_transform() -> callable: PIL.Image (RGB) -> torch.Tensor [3, H, W]
    model.pth  state_dict hasil torch.save(model.state_dict(), "model.pth")
               Output model: logits berukuran [batch, 5] dengan urutan kelas = CLASSES di bawah.
"""
import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

CLASSES = ["botol_plastik", "gelas_plastik", "kertas_kardus", "saset_kemasan", "styrofoam"]
EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def load_submission(sub_dir, device):
    sub_dir = Path(sub_dir).resolve()
    spec = importlib.util.spec_from_file_location("submission_model", sub_dir / "model.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(sub_dir))  # izinkan model.py meng-import file pendukung di folder yang sama
    spec.loader.exec_module(mod)
    for fn in ("get_model", "get_transform"):
        if not hasattr(mod, fn):
            raise SystemExit(f"[GAGAL] model.py tidak mendefinisikan {fn}()")
    model = mod.get_model()
    state = torch.load(sub_dir / "model.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, mod.get_transform()


def list_images(data_dir):
    items = []
    for ci, cls in enumerate(CLASSES):
        d = Path(data_dir) / cls
        if not d.is_dir():
            raise SystemExit(f"[GAGAL] folder kelas tidak ditemukan: {d}")
        items += [(p, ci) for p in sorted(d.iterdir()) if p.suffix.lower() in EXTS]
    return items


def metrics(y_true, y_pred, k=len(CLASSES)):
    cm = [[0] * k for _ in range(k)]
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    f1s = []
    for c in range(k):
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(k)) - tp
        fn = sum(cm[c]) - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    acc = sum(cm[c][c] for c in range(k)) / max(1, len(y_true))
    return {"macro_f1": sum(f1s) / k, "accuracy": acc,
            "per_class_f1": dict(zip(CLASSES, f1s)), "confusion_matrix": cm}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submission", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    model, tf = load_submission(a.submission, device)
    items = list_images(a.data)
    print(f"device={device}  gambar={len(items)}")

    y_true, y_pred, t0 = [], [], time.time()
    for i in range(0, len(items), a.batch_size):
        chunk = items[i:i + a.batch_size]
        x = torch.stack([tf(Image.open(p).convert("RGB")) for p, _ in chunk]).to(device)
        logits = model(x)
        if logits.ndim != 2 or logits.shape[1] != len(CLASSES):
            raise SystemExit(f"[GAGAL] output model harus [batch, {len(CLASSES)}], didapat {tuple(logits.shape)}")
        y_pred += logits.argmax(1).cpu().tolist()
        y_true += [c for _, c in chunk]

    m = metrics(y_true, y_pred)
    m["n_images"] = len(items)
    m["seconds"] = round(time.time() - t0, 1)
    print(f"\nMacro-F1 : {m['macro_f1']:.4f}\nAkurasi  : {m['accuracy']:.4f}")
    for c, f in m["per_class_f1"].items():
        print(f"  F1 {c:<15} {f:.4f}")
    print("Confusion matrix (baris = label asli, kolom = prediksi):")
    for c, row in zip(CLASSES, m["confusion_matrix"]):
        print(f"  {c:<15} {row}")
    if a.out:
        Path(a.out).write_text(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
