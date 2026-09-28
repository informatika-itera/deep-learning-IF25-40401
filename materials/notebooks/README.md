# Modul Praktikum Hands-on: Deep Computer Vision dengan CNN

Selamat datang di repositori modul *hands-on* perkuliahan **Deep Learning (IF25-40401)**, Program Studi Teknik Informatika, Institut Teknologi Sumatera (ITERA). Modul ini mendampingi materi berikut sesuai dengan Rencana Pembelajaran Semester (RPS):
- **Pertemuan 04 — Deep Computer Vision (1): CNN From Scratch** (Notebook 01–03)
- **Pertemuan 05 — Deep Computer Vision (2): CNN dan Variasinya** (Notebook 04–06)

Setiap notebook dirancang untuk dieksekusi secara *live-coding* oleh dosen di depan kelas pada sesi Tatap Muka (TM: 3×50′) sekaligus sebagai materi eksplorasi mandiri dan Tugas Terstruktur mahasiswa (TT: 3×60′).

---

## 🗺️ Peta Navigasi Materi & Pemetaan Capaian Pembelajaran

### Pertemuan 04 — CNN From Scratch

| No | Berkas Notebook | Topik & Cakupan Materi | Frame Slide Terkait | Sub-CPMK RPS | Estimasi Waktu | Tautan Langsung |
|:---:|:---|:---|:---:|:---:|:---:|:---:|
| **01** | [`01-konvolusi-pooling-from-scratch.ipynb`](./01-konvolusi-pooling-from-scratch.ipynb) | **Mekanika Konvolusi & Pooling:** Representasi citra matriks, konvolusi 2D nested loop NumPy, validasi formula $W_{out}$ dan parameter, Max/Avg pooling, invariansi translasi, bank filter citra nyata. | Slide Frame 33–49 (Bagian 04) | Sub-CPMK 1 & 2 | ~40 Menit | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/informatika-itera/deep-learning-IF25-40401/blob/main/materials/notebooks/01-konvolusi-pooling-from-scratch.ipynb) |
| **02** | [`02-lenet5-pytorch.ipynb`](./02-lenet5-pytorch.ipynb) | **LeNet-5 from Scratch (PyTorch):** Dataset MNIST (32×32), arsitektur LeNet-5 (`Tanh` + `AvgPool`), ringkasan `torchinfo`, trace tensor shape hook, parameter breakdown (61.706 bobot), alur 5 langkah training loop, demo bahaya lupa `zero_grad()`, confusion matrix NumPy. | Slide Frame 50–71 (Bagian 05) | Sub-CPMK 3 & 4 | ~60 Menit | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/informatika-itera/deep-learning-IF25-40401/blob/main/materials/notebooks/02-lenet5-pytorch.ipynb) |
| **03** | [`03-analisis-eksperimen-cnn.ipynb`](./03-analisis-eksperimen-cnn.ipynb) | **Analisis Eksperimen & Representasi:** Head-to-head MLP vs CNN (1,06M vs 61K param), uji ketahanan translasi spasial, visualisasi filter C1, visualisasi *hierarchical feature maps* (C1 & C3 via forward hook), analisis Receptive Field (2×3×3 vs 1×5×5), studi ablasi terkontrol, dan adaptasi RGB CIFAR-10. | Pemenuhan Janji Pengalaman Belajar RPS Minggu 4 & Jembatan P05 | Sub-CPMK 1, 2, 3, 4 | ~45 Menit | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/informatika-itera/deep-learning-IF25-40401/blob/main/materials/notebooks/03-analisis-eksperimen-cnn.ipynb) |


### Pertemuan 05 — CNN dan Variasinya

| No | Berkas Notebook | Topik & Cakupan Materi | Slide Terkait | Sub-CPMK RPS | Estimasi Waktu | Tautan Langsung |
|:---:|:---|:---|:---:|:---:|:---:|:---:|
| **04** | [`04-alexnet-vgg-resnet.ipynb`](./04-alexnet-vgg-resnet.ipynb) | **Dari AlexNet ke ResNet:** penelusuran dimensi & parameter AlexNet (96% di FC, 92% komputasi di konvolusi), tumpukan $3\times3$ VGG dan *receptive field* empiris via gradien, VGG-16 dari `vgg_block` (138.357.544 parameter), konvolusi $1\times1$ & modul Inception, recap BatchNorm, eksperimen *degradation problem* Plain vs ResNet 20/56 layer di CIFAR-10, Basic vs Bottleneck block. | Slide 6–26 (Bagian 01–02) | Sub-CPMK 1, 2, 3 | ~50 Menit | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/informatika-itera/deep-learning-IF25-40401/blob/main/materials/notebooks/04-alexnet-vgg-resnet.ipynb) |
| **05** | [`05-mobilenet-convnext-perbandingan.ipynb`](./05-mobilenet-convnext-perbandingan.ipynb) | **CNN Efisien & Modern:** depthwise separable (8,7× lebih hemat MAC), argumen `groups`, MAC vs latensi, MobileNetV1 utuh dengan $\alpha$ & $\rho$, inverted residual & eksperimen *linear bottleneck* (spiral), ConvNeXt block, tabel & grafik 10 backbone `torchvision`, inferensi pretrained dan penggantian *head* (jembatan P6). | Slide 28–41 (Bagian 03–05) | Sub-CPMK 4, 5 | ~45 Menit | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/informatika-itera/deep-learning-IF25-40401/blob/main/materials/notebooks/05-mobilenet-convnext-perbandingan.ipynb) |
| **06** | [`06-bbox-iou-nms-deteksi-segmentasi.ipynb`](./06-bbox-iou-nms-deteksi-segmentasi.ipynb) | **Deteksi & Segmentasi:** konversi bounding box Pascal VOC ↔ COCO ↔ YOLO, parsing berkas anotasi, jebakan salah format & resize, IoU / NMS / mAP dari nol (diverifikasi dengan `torchvision.ops`), target grid YOLO $7\times7\times30$, Faster R-CNN & DeepLabV3 pretrained, mini U-Net dengan vs tanpa *skip connection*. | Slide 43–52 (Bagian 06) | Sub-CPMK 6 | ~55 Menit | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/informatika-itera/deep-learning-IF25-40401/blob/main/materials/notebooks/06-bbox-iou-nms-deteksi-segmentasi.ipynb) |

---

## 🎯 Capaian Pembelajaran (Sub-CPMK Mingguan)

### Pertemuan 04
- **Sub-CPMK 1:** Menjelaskan operasi konvolusi 2D, fungsi kernel/filter, konsep *stride*, *padding* (*valid* vs *same*), dan operasi *pooling* (*max* vs *average*).
- **Sub-CPMK 2:** Menghitung ukuran *feature map* keluaran ($W_{out}$) dan jumlah parameter konvolusi secara matematis.
- **Sub-CPMK 3:** Membangun dan melatih model CNN dari awal menggunakan modul `torch.nn.Module`, mengimplementasikan alur 5 langkah *training loop* PyTorch, serta mengevaluasi performanya.
- **Sub-CPMK 4:** Menganalisis peran *weight sharing* dan *local connectivity* dalam mereduksi parameter serta membentuk invariansi translasi dibandingkan MLP.


### Pertemuan 05
- **Sub-CPMK 1:** Menjelaskan evolusi CNN dari AlexNet menuju VGG.
- **Sub-CPMK 2:** Menjelaskan desain blok konvolusi berlapis ($3\times3$ bertumpuk) pada VGG.
- **Sub-CPMK 3:** Menjelaskan *residual connection* dan alasan ResNet bisa jauh lebih dalam.
- **Sub-CPMK 4:** Membandingkan efisiensi MobileNet (*depthwise separable*) dengan CNN standar.
- **Sub-CPMK 5:** Mengidentifikasi elemen modern ConvNeXt yang diadopsi dari Transformer.
- **Sub-CPMK 6:** Menjelaskan prinsip *object detection* dan segmentasi sebagai perluasan CNN.

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
# 1. Kloning repositori & masuk ke direktori
git clone https://github.com/informatika-itera/deep-learning-IF25-40401.git
cd deep-learning-IF25-40401

# 2. Buat virtual environment
# Menggunakan uv (direkomendasikan, sangat cepat):
uv venv --python 3.11 .venv
source .venv/bin/activate

# Atau menggunakan python standar venv:
python3 -m venv .venv
source .venv/bin/activate  # macOS / Linux
# atau: .venv\Scripts\activate  # Windows

# 3. Pasang pustaka pendukung dari requirements.txt
uv pip install -r requirements.txt
# Atau: pip install -r requirements.txt

# 4. Daftarkan kernel ke Jupyter / VS Code (opsional)
python -m ipykernel install --user --name dl_web --display-name "Python 3.11 (dl_web .venv)"

# 5. Jalankan JupyterLab atau buka di VS Code
jupyter lab
```

---

## 📦 Ketergantungan Paket & Versi Minimum
Daftar lengkap ada di [`requirements.txt`](./requirements.txt) (isinya sama dengan `requirements.txt` di root repositori).
- `python >= 3.9`
- `torch >= 2.1.0` (dibutuhkan `torch.utils.flop_counter` pada Notebook 04–05)
- `torchvision >= 0.16.0`
- `torchinfo >= 1.8.0`
- `matplotlib >= 3.5.0`
- `numpy >= 1.22.0`
- `pillow >= 9.0.0`

Seluruh notebook di repositori ini dijalankan dan disimpan beserta output-nya menggunakan environment `uv` (`.venv`, Python 3.11). Untuk menjalankan ulang sebuah notebook secara otomatis:
```bash
uv pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace 04-alexnet-vgg-resnet.ipynb
```

---

## 💡 Catatan Unduhan Dataset & Mirror Kampus
- Pada **Notebook 02 & 03**, dataset MNIST dan CIFAR-10 akan diunduh secara otomatis ke direktori `./data`.
- Jika jaringan internet kampus mengalami *timeout* atau pemblokiran ke mirror HTTP Yann LeCun (`yann.lecun.com`), skrip telah dilengkapi mekanisme penanganan otomatis (*fallback*) menuju dataset citra berukuran sama (`FashionMNIST`).
- Untuk **Notebook 01**, seluruh eksperimen menggunakan citra sintetis NumPy dan citra bawaan Matplotlib (`grace_hopper.jpg`), sehingga dapat dijalankan **100% luring (*offline*)** tanpa sambungan internet.
- **Notebook 04** memakai CIFAR-10 (diunduh otomatis ke `./data`, ±170 MB). Saklar `MODE_CEPAT = True` melatih 4 model pada 10.000 citra selama 5 epoch (±2 menit di GPU/MPS).
- **Notebook 05–06** membangun model `torchvision` tanpa bobot (`weights=None`) sehingga sebagian besar berjalan luring. Bagian inferensi pretrained dapat dimatikan dengan `UNDUH_PRETRAINED = False`. Bila aktif, bobot yang diunduh adalah MobileNetV3-L (~22 MB), ResNet-50 (~98 MB), Faster R-CNN MobileNetV3 (~74 MB), dan DeepLabV3 MobileNetV3 (~42 MB), ditambah satu citra COCO val2017.
- **Catatan latensi (Notebook 05):** pada build PyTorch tanpa oneDNN (mis. macOS ARM), konvolusi depthwise di CPU jauh lebih lambat sehingga MobileNet/EfficientNet/ConvNeXt terlihat lebih lambat daripada VGG. Di Colab (x86) atau GPU urutannya berbeda.

---

## ⚖️ Lisensi & Hak Cipta
Modul ini merupakan bagian dari materi ajar resmi mata kuliah **Deep Learning (IF25-40401)** Program Studi Teknik Informatika, Institut Teknologi Sumatera (ITERA). Digunakan untuk kepentingan akademik dan pendidikan.
