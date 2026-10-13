# Pertemuan 6 — Eksperimen Hyperparameter (CNN & ViT)

Kode pendamping slide [`pertemuan-06-hyperparameter-tuning.pdf`](../pertemuan-06-hyperparameter-tuning.pdf).
Satu file, [`train.py`](./train.py), untuk melatih & mengevaluasi 5 backbone ringan pada **Imagenette**
dan mencatat setiap eksperimen ke **Weights & Biases**. Tidak ada argparse: ubah nilai di blok `CONFIG`, lalu jalankan.

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
| train | 85% Imagenette `train` | 8.049 | update bobot |
| valid | 15% Imagenette `train` | 1.420 | early stopping, memilih checkpoint, membandingkan run |
| test | Imagenette `val` | 3.925 | **sekali** di akhir, dengan checkpoint terbaik |

| `model` | Checkpoint timm | Param (10 kelas) |
|---|---|---:|
| `resnet18` | `resnet18.a1_in1k` | 11,2 M |
| `mobilenetv3` | `mobilenetv3_small_100.lamb_in1k` | 1,5 M |
| `efficientnet_b0` | `efficientnet_b0.ra_in1k` | 4,0 M |
| `convnext_atto` | `convnext_atto.d2_in1k` | 3,4 M |
| `vit_tiny` | `vit_tiny_patch16_224.augreg_in21k_ft_in1k` | 5,5 M |

## Hyperparameter di `CONFIG`

| Kelompok | Kunci | Pilihan / rentang wajar |
|---|---|---|
| Data | `img_size`, `train_fraction`, `augment`, `mixup_alpha`, `cutmix_alpha` | 128–224 · 0,1–1,0 · `none`/`basic`/`strong` · 0,2 · 1,0 |
| Model | `model`, `pretrained`, `freeze_backbone`, `dropout`, `drop_path` | lihat tabel di atas · `True`/`False` · 0–0,5 · 0–0,2 |
| Optimizer | `optimizer`, `lr`, `backbone_lr_mult`, `weight_decay`, `momentum`, `batch_size`, `grad_clip`, `label_smoothing` | `sgd`/`adam`/`adamw`/`rmsprop` · 1e-5–1e-1 · 0,1–1 · 0–0,1 · 0,9 · 32–256 · `None`/1,0 · 0–0,2 |
| Scheduler | `scheduler`, `warmup_epochs`, `step_size`, `gamma`, `plateau_patience` | `none`/`step`/`cosine`/`onecycle`/`plateau` |
| Early stopping | `early_stopping`, `patience`, `min_delta`, `monitor` | `val_loss`/`val_acc`/`val_f1` |
| W&B | `wandb_project`, `wandb_entity`, `wandb_group`, `wandb_mode`, `notes` | tulis hipotesis di `notes` |

Aturan main: **ubah satu hyperparameter per run**, tulis hipotesis di `notes`, lalu bandingkan di W&B
(filter per `group`/tag, panel *Parallel coordinates*, dan tabel run diurutkan menurut `valid/acc`).

## Yang dicatat ke W&B

- per 10 step: `train/loss`, `train/lr`, `train/grad_norm`
- per epoch: `epoch/train_loss`, `epoch/train_acc`, `valid/loss`, `valid/acc`, `valid/f1`, `epoch/gap_acc`, `epoch/lr`, `epoch/time_sec`
- akhir run: `test/acc`, `test/f1`, confusion matrix, tabel contoh prediksi, `best_epoch`, `stopped_epoch`
- seluruh `CONFIG` + jumlah parameter, device, ukuran split

Salinan ringkas setiap run juga ditambahkan ke `results.csv` lokal (berguna jika internet bermasalah).

## Estimasi waktu (30 epoch, 160 px, batch 64)

Terukur di RTX 5070 (AMP, `num_workers=8`), termasuk validasi tiap epoch, saat CPU juga dipakai proses lain
(ResNet-18 & EfficientNet-B0: run penuh 30 epoch; lainnya: estimasi dari 3 epoch). Bottleneck-nya loading data di CPU,
bukan ukuran model — MobileNetV3 (1,5 M) tidak lebih cepat dari ResNet-18 (11,2 M).

| `model` | s/epoch | 30 epoch |
|---|---:|---:|
| `resnet18` | 4,8 | ±2,5 menit |
| `mobilenetv3` | 5,7 | ±3,0 menit |
| `efficientnet_b0` | 8,6 | ±4,4 menit |
| `convnext_atto` | 5,4 | ±3,0 menit |
| `vit_tiny` | 6,3 | ±3,3 menit |

Early stopping (`patience=5`) biasanya menghentikan run lebih awal. `train_fraction=0.1` ±1,5 s/epoch.
MacBook M1 Pro (MPS): ResNet-18 ±40 s/epoch — untuk demo di laptop pakai `train_fraction=0.3`, `epochs=5`,
atau `model="mobilenetv3"`.

## Kartu eksperimen di slide

Slide berisi 10 kartu (noise antar-seed, learning rate, optimizer, batch size, scheduler, overfitting,
regularisasi, early stopping, strategi fine-tuning, CNN vs ViT). Setiap kartu menyebut kunci `CONFIG` yang
diubah, hipotesis, dan tabel kosong yang diisi dari hasil run di kelas. Kembalikan nilai ke baseline sebelum
pindah kartu, dan pakai `wandb_group` yang sama untuk satu kartu agar mudah dibandingkan.
