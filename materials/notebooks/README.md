# Modul Praktikum Hands-on: Convolutional Neural Networks (CNN) From Scratch

Selamat datang di repositori modul *hands-on* perkuliahan **Deep Learning (IF25-40401)**, Program Studi Teknik Informatika, Institut Teknologi Sumatera (ITERA). Modul ini dirancang untuk mendampingi materi **Pertemuan 04 — Deep Computer Vision (1): CNN From Scratch** sesuai dengan Rencana Pembelajaran Semester (RPS).

Ketiga notebook ini dirancang untuk dieksekusi secara *live-coding* oleh dosen di depan kelas pada sesi Tatap Muka (TM: 3×50′) sekaligus sebagai materi eksplorasi mandiri dan Tugas Terstruktur mahasiswa (TT: 3×60′).

---

## 🗺️ Peta Navigasi Materi & Pemetaan Capaian Pembelajaran

| No | Berkas Notebook | Topik & Cakupan Materi | Frame Slide Terkait | Sub-CPMK RPS | Estimasi Waktu | Tautan Langsung |
|:---:|:---|:---|:---:|:---:|:---:|:---:|
| **01** | [`01-konvolusi-pooling-from-scratch.ipynb`](./01-konvolusi-pooling-from-scratch.ipynb) | **Mekanika Konvolusi & Pooling:** Representasi citra matriks, konvolusi 2D nested loop NumPy, validasi formula $W_{out}$ dan parameter, Max/Avg pooling, invariansi translasi, bank filter citra nyata. | Slide Frame 33–49 (Bagian 04) | Sub-CPMK 1 & 2 | ~40 Menit | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/informatika-itera/deep-learning-IF25-40401/blob/main/materials/notebooks/01-konvolusi-pooling-from-scratch.ipynb) |
| **02** | [`02-lenet5-pytorch.ipynb`](./02-lenet5-pytorch.ipynb) | **LeNet-5 from Scratch (PyTorch):** Dataset MNIST (32×32), arsitektur LeNet-5 (`Tanh` + `AvgPool`), ringkasan `torchinfo`, trace tensor shape hook, parameter breakdown (61.706 bobot), alur 5 langkah training loop, demo bahaya lupa `zero_grad()`, confusion matrix NumPy. | Slide Frame 50–71 (Bagian 05) | Sub-CPMK 3 & 4 | ~60 Menit | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/informatika-itera/deep-learning-IF25-40401/blob/main/materials/notebooks/02-lenet5-pytorch.ipynb) |
| **03** | [`03-analisis-eksperimen-cnn.ipynb`](./03-analisis-eksperimen-cnn.ipynb) | **Analisis Eksperimen & Representasi:** Head-to-head MLP vs CNN (1,06M vs 61K param), uji ketahanan translasi spasial, visualisasi filter C1, visualisasi *hierarchical feature maps* (C1 & C3 via forward hook), analisis Receptive Field (2×3×3 vs 1×5×5), studi ablasi terkontrol, dan adaptasi RGB CIFAR-10. | Pemenuhan Janji Pengalaman Belajar RPS Minggu 4 & Jembatan P05 | Sub-CPMK 1, 2, 3, 4 | ~45 Menit | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/informatika-itera/deep-learning-IF25-40401/blob/main/materials/notebooks/03-analisis-eksperimen-cnn.ipynb) |

---

## 🎯 Capaian Pembelajaran (Sub-CPMK Mingguan)
- **Sub-CPMK 1:** Menjelaskan operasi konvolusi 2D, fungsi kernel/filter, konsep *stride*, *padding* (*valid* vs *same*), dan operasi *pooling* (*max* vs *average*).
- **Sub-CPMK 2:** Menghitung ukuran *feature map* keluaran ($W_{out}$) dan jumlah parameter konvolusi secara matematis.
- **Sub-CPMK 3:** Membangun dan melatih model CNN dari awal menggunakan modul `torch.nn.Module`, mengimplementasikan alur 5 langkah *training loop* PyTorch, serta mengevaluasi performanya.
- **Sub-CPMK 4:** Menganalisis peran *weight sharing* dan *local connectivity* dalam mereduksi parameter serta membentuk invariansi translasi dibandingkan MLP.

---

## 🚀 Tiga Cara Menjalankan Notebook

### 1. Google Colab (Paling Direkomendasikan untuk Mahasiswa)
Klik lencana **Open In Colab** pada tabel di atas. Di Google Colab:
- Seluruh pustaka dasar (`torch`, `torchvision`, `matplotlib`, `numpy`) sudah terpasang bawaan Google.
- Perintah `%pip install torchinfo -q` pada sel pertama akan otomatis memasang pustaka pelengkap summary arsitektur.
- Akselerator T4 GPU dapat diaktifkan melalui menu `Runtime > Change runtime type > T4 GPU`.

### 2. Kaggle Notebooks
1. Unduh berkas `.ipynb` yang diinginkan dari repositori ini.
2. Buka dashboard [Kaggle](https://www.kaggle.com/code) dan pilih **New Notebook**.
3. Pilih menu `File > Import Notebook`, lalu unggah berkas `.ipynb`.
4. Pilih accelerator `CPU` atau `GPU P100`.

### 3. Lingkungan Lokal (JupyterLab / VS Code)
Pastikan Anda telah memasang Python 3.9+ (disarankan melalui virtual environment):
```bash
# 1. Kloning repositori
git clone https://github.com/informatika-itera/deep-learning-IF25-40401.git
cd deep-learning-IF25-40401/materials/notebooks

# 2. Buat dan aktifkan virtual environment (opsional namun disarankan)
python3 -m venv .venv
source .venv/bin/activate  # macOS / Linux
# atau: .venv\Scripts\activate  # Windows

# 3. Pasang pustaka pendukung
pip install torch torchvision torchinfo matplotlib jupyterlab

# 4. Jalankan JupyterLab
jupyter lab
```

---

## 📦 Ketergantungan Paket & Versi Minimum
- `python >= 3.9`
- `torch >= 2.0.0`
- `torchvision >= 0.15.0`
- `torchinfo >= 1.8.0`
- `matplotlib >= 3.5.0`
- `numpy >= 1.22.0`

---

## 💡 Catatan Unduhan Dataset & Mirror Kampus
- Pada **Notebook 02 & 03**, dataset MNIST dan CIFAR-10 akan diunduh secara otomatis ke direktori `./data`.
- Jika jaringan internet kampus mengalami *timeout* atau pemblokiran ke mirror HTTP Yann LeCun (`yann.lecun.com`), skrip telah dilengkapi mekanisme penanganan otomatis (*fallback*) menuju dataset citra berukuran sama (`FashionMNIST`).
- Untuk **Notebook 01**, seluruh eksperimen menggunakan citra sintetis NumPy dan citra bawaan Matplotlib (`grace_hopper.jpg`), sehingga dapat dijalankan **100% luring (*offline*)** tanpa sambungan internet.

---

## ⚖️ Lisensi & Hak Cipta
Modul ini merupakan bagian dari materi ajar resmi mata kuliah **Deep Learning (IF25-40401)** Program Studi Teknik Informatika, Institut Teknologi Sumatera (ITERA). Digunakan untuk kepentingan akademik dan pendidikan.
