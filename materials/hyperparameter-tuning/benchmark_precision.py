"""Benchmark presisi numerik: fp16 AMP vs fp32 vs fp64, murni kecepatan GPU (tanpa DataLoader).

Memakai batch acak (sintetis) agar yang terukur hanya komputasi forward + backward + update,
bukan loading data. Untuk akurasi & waktu per epoch sungguhan, jalankan train.py dengan
precision=["fp16_amp", "fp32", "fp64"].

    python benchmark_precision.py
"""
import time

import torch
import torch.nn as nn

from models import ARCHITECTURES

MODELS = ["resnet18", "vit_tiny"]
PRECISIONS = ["fp16_amp", "fp32", "fp64"]
BATCH_SIZE, IMG_SIZE = 64, 160
STEPS = {"fp16_amp": 30, "fp32": 30, "fp64": 5}  # fp64 sangat lambat di GPU consumer


def benchmark(model_name, precision):
    dtype = torch.float64 if precision == "fp64" else torch.float32
    use_amp = precision == "fp16_amp"
    model = ARCHITECTURES[model_name](10, False, 0.0, 0.0, IMG_SIZE).to("cuda", dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    x = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE, device="cuda", dtype=dtype)
    y = torch.randint(0, 10, (BATCH_SIZE,), device="cuda")

    def train_step():
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            loss = nn.functional.cross_entropy(model(x), y)
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    for _ in range(3):  # pemanasan: alokasi memori & pemilihan kernel cuDNN
        train_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.time()
    for _ in range(STEPS[precision]):
        train_step()
    torch.cuda.synchronize()  # GPU bekerja asinkron: tunggu selesai sebelum stop timer
    seconds = time.time() - t0
    return BATCH_SIZE * STEPS[precision] / seconds, torch.cuda.max_memory_allocated() / 1e9


if __name__ == "__main__":
    print(f"GPU: {torch.cuda.get_device_name()} | batch {BATCH_SIZE}, {IMG_SIZE} px, forward + backward + update")
    for model_name in MODELS:
        base = None
        for precision in PRECISIONS:
            img_per_sec, mem_gb = benchmark(model_name, precision)
            base = base or img_per_sec
            print(f"{model_name:<10} {precision:<9} {img_per_sec:8.0f} citra/s  ({img_per_sec / base:5.2f}x)"
                  f"  memori puncak {mem_gb:.2f} GB")
