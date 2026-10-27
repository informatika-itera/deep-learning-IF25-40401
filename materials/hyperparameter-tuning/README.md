# Pertemuan 6 — Eksperimen Hyperparameter (CNN & ViT)

Kode pendamping slide [`pertemuan-06-hyperparameter-tuning.pdf`](../pertemuan-06-hyperparameter-tuning.pdf).
Melatih & mengevaluasi 5 backbone ringan pada **Imagenette** dan mencatat setiap run ke **Weights & Biases**.
Tidak ada argparse: tulis nilai yang ingin dicoba sebagai **list** di blok `CONFIG` (`train.py`), lalu jalankan.

```python
optimizer=["adam", "adamw", "nadam"],
model=["resnet18", "vit_tiny"],
# -> 3 x 2 = 6 run, satu run W&B per kombinasi, ringkasan dicetak di akhir
```

## Struktur

```text
hyperparameter-tuning/
├── train.py          # CONFIG, loop semua kombinasi, training, evaluasi, logging W&B
├── datareader.py     # Imagenette -> DataLoader train & val (augmentasi, MixUp/CutMix)
├── models.py         # memilih arsitektur + strategi fine-tuning (scratch / linear_probe / full_ft) + init bobot
├── benchmark_precision.py  # kecepatan GPU murni: fp16 AMP vs fp32 vs fp64 (data sintetis)
└── architectures/    # satu file per arsitektur, masing-masing berisi build(...)
    ├── resnet18.py
    ├── mobilenetv3.py
    ├── efficientnet_b0.py
    ├── convnext_atto.py
    └── vit_tiny.py
```

Folder arsitektur tidak dinamai `models/` karena akan bentrok dengan `models.py` saat di-import.
Menambah arsitektur: buat `architectures/<nama>.py` berisi `build(...)`, lalu daftarkan di `ARCHITECTURES` (`models.py`).

## Menjalankan

**Lokal (uv)**

```bash
cd materials/hyperparameter-tuning
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
wandb login            # sekali saja; tanpa akun: set wandb_mode="offline" atau "disabled"
python train.py
```

**Google Colab / Kaggle** (Runtime → GPU)

```python
!git clone --depth 1 https://github.com/informatika-itera/deep-learning-IF25-40401.git
%cd deep-learning-IF25-40401/materials/hyperparameter-tuning
!pip install -q timm wandb
import wandb; wandb.login()
# buka train.py dari panel Files, ubah CONFIG, simpan, lalu:
!python train.py
```

Imagenette versi 160 px (±94 MB) diunduh otomatis ke `./data` saat pertama kali dijalankan.

## Data & model

| Split | Sumber | Jumlah | Dipakai untuk |
|---|---|---:|---|
| train | Imagenette `train` | 9.469 | update bobot (diaugmentasi) |
| val | Imagenette `val` | 3.925 | dievaluasi tiap epoch: early stopping, checkpoint terbaik, membandingkan run |

Demi waktu kelas, semua run memakai split train/val tetap dan satu seed (`seed=42`). Harganya: skor val terbaik
sedikit optimistis (val juga dipakai memilih), dan selisih kecil antar-run bisa saja kebetulan. Untuk angka yang
lebih bisa dipercaya: train/val/test, stratified k-fold, dan/atau beberapa seed (lihat slide "Protokol Evaluasi").
Di Studio 2, validation set tersembunyi milik asisten berperan sebagai test set.

| `model` | Checkpoint timm | Param (10 kelas) |
|---|---|---:|
| `resnet18` | `resnet18.a1_in1k` | 11,2 M |
| `mobilenetv3` | `mobilenetv3_small_100.lamb_in1k` | 1,5 M |
| `efficientnet_b0` | `efficientnet_b0.ra_in1k` | 4,0 M |
| `convnext_atto` | `convnext_atto.d2_in1k` | 3,4 M |
| `vit_tiny` | `vit_tiny_patch16_224.augreg_in21k_ft_in1k` | 5,5 M |

## Hyperparameter di `CONFIG`

Nilai berbentuk list dicoba satu per satu (semua kombinasi antar-list = grid search).
Nilai tunggal (bukan list) tetap untuk semua run. Jumlah run = perkalian panjang semua list.

| Kelompok | Kunci (list) | Pilihan / rentang wajar |
|---|---|---|
| Data | `img_size`, `train_fraction`, `augment`, `mixup_alpha`, `cutmix_alpha` | 128–224 · 0,1–1,0 · `none`/`basic`/`strong` · 0,2 · 1,0 |
| Model | `model`, `finetune`, `init`, `backbone_lr_mult`, `dropout`, `drop_path` | lihat tabel di atas · `scratch`/`linear_probe`/`full_ft` · `default`/`kaiming`/`xavier`/`normal` (hanya scratch) · 0,1–1 · 0–0,5 · 0–0,2 |
| Optimizer | `optimizer`, `lr`, `weight_decay`, `batch_size`, `epochs`, `label_smoothing` | `sgd`/`adam`/`adamw`/`nadam`/`rmsprop` · 1e-5–1e-1 · 0–0,1 · 32–256 · 10 · 0–0,2 |
| Scheduler | `scheduler`, `warmup_epochs` | `none`/`step`/`cosine`/`onecycle`/`plateau` · 0–1 |
| Early stopping | `patience` | `None` (mati) / 2–5 |
| Presisi | `precision` | `fp16_amp` (mixed precision, hanya CUDA) / `fp32` / `fp64` (sangat lambat, tidak didukung MPS) |

Pengaturan tetap (bukan list): `seed`, `momentum`, `step_size`, `gamma`, `plateau_patience`, `min_lr`, `monitor`
(`val_loss`/`val_acc`/`val_f1`), `min_delta`, `grad_clip`, `data_root`, `num_workers`, dan pengaturan W&B
(`wandb_project`, `wandb_entity`, `wandb_mode`, `wandb_group`, `notes`).

Aturan main: **ubah satu kunci menjadi list per eksperimen** (boleh ditambah `model`), tulis hipotesis di `notes`,
lalu bandingkan di W&B. Nama run dan group dibuat otomatis dari kunci yang divariasikan, mis. run
`vit_tiny-optimizer=nadam` di group `optimizer`.

## Yang dicatat ke W&B

- `config`: **seluruh** hyperparameter run tersebut + daftar kunci yang divariasikan, jumlah parameter, device, ukuran split
- per 10 step: `train/loss`, `train/lr`, `train/grad_norm`
- per epoch: `epoch/train_loss`, `epoch/train_acc`, `val/loss`, `val/acc`, `val/f1`, `epoch/gap_acc`, `epoch/lr`, `epoch/time_sec`
- akhir run: `best/val_acc`, `best/val_f1`, `best_epoch`, `stopped_epoch`, `train_time_min`, `sec_per_epoch` (median),
  `gpu_mem_gb` (memori GPU puncak), confusion matrix val (epoch terbaik)

Setiap run juga ditambahkan ke `results.csv` lokal (berguna jika internet bermasalah); checkpoint terbaik
disimpan di `checkpoints/<nama run>.pth`.

## Estimasi waktu (10 epoch, 160 px, batch 64)

Terukur di RTX 5070 (AMP, `num_workers=8`), termasuk validasi 3.925 citra tiap epoch; epoch pertama ±5 s lebih lambat.
Bottleneck-nya loading data di CPU, bukan ukuran model.

| `model` | s/epoch | 10 epoch |
|---|---:|---:|
| `resnet18` | 4 | ±0,8 menit |
| `mobilenetv3` | 5 | ±0,9 menit |
| `efficientnet_b0` | 7 | ±1,3 menit |
| `convnext_atto` | 5 | ±1,0 menit |
| `vit_tiny` | 5 | ±0,9 menit |

MacBook (MPS) jauh lebih lambat: untuk demo di laptop pakai `model=["mobilenetv3"]`, `train_fraction=[0.3]`, `epochs=[5]`.

## Kartu eksperimen di slide

Slide berisi 11 kartu (learning rate, optimizer, batch size, scheduler, overfitting, regularisasi, early stopping,
strategi fine-tuning, CNN vs ViT, inisialisasi bobot, presisi numerik). Setiap kartu menyebut list yang diisi di `CONFIG`,
hipotesis, dan tabel kosong yang diisi dari hasil run di kelas. Kembalikan list ke satu nilai (baseline) sebelum
pindah kartu.
