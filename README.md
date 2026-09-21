# Deep Learning (IF25-40401) — Portal Perkuliahan & RPS

Portal pembelajaran dan silabus Rencana Pembelajaran Semester (RPS) mata kuliah **Deep Learning (IF25-40401)** pada Program Studi Teknik Informatika, Institut Teknologi Sumatera (ITERA).

Dibangun dengan pendekatan desain minimalis berkontras tinggi terinspirasi oleh sistem tipografi dan *grid* pada [DESIGN.md](./DESIGN.md) (monochrome campaign hero, pill action buttons, and flat card rhythm).

---

## 🌐 Struktur URL & Navigasi Publik

Website ini di-hosting menggunakan GitHub Pages pada domain organisasi:

| Rute URL | Deskripsi Halaman | Konten Utama |
|---|---|---|
| [`/`](https://informatika-itera.github.io/deep-learning-IF25-40401/) | **Beranda / Modul Materi Kuliah** | Sistem tab dan kartu (card) materi perkuliahan terpilih: **Pertemuan 4, 5, dan 6** (Computer Vision, CNN From Scratch, Variasi Arsitektur, dan Studio ImageNet) dengan tombol aksi unduh PDF LaTeX dan tautan rincian RPS. |
| [`/rps/`](https://informatika-itera.github.io/deep-learning-IF25-40401/rps/) | **Silabus Lengkap (RPS)** | Rencana Pembelajaran Semester 16 minggu, CPMK, distribusi nilai, komponen evaluasi, studio proyek, dan tugas besar. |
| [`/rps-benchmark/`](https://informatika-itera.github.io/deep-learning-IF25-40401/rps-benchmark/) | **Benchmark Kurikulum** | Analisis komparasi RPS terhadap 8 universitas top dunia (Stanford CS230/CS231n, MIT 6.S191, CMU 11-785, UC Berkeley, Oxford, ETH Zurich, Princeton). |
| [`/materials/`](https://informatika-itera.github.io/deep-learning-IF25-40401/materials/) | **Berkas Slide & Panduan PDF** | Berkas PDF LaTeX Beamer untuk Pertemuan 4 (`pertemuan-04-cnn-from-scratch.pdf`), Pertemuan 5 (`pertemuan-05-cnn-variations.pdf`), dan Pertemuan 6 (`pertemuan-06-finetuning-imagenet.pdf`). |

---

## 📁 Struktur Repositori

```text
dl_web/
├── index.html                   # Halaman beranda modul kartu materi (Pertemuan 4, 5, 6)
├── styles.css                   # Sistem CSS token & komponen kartu (DESIGN.md specification)
├── script.js                    # Skrip interaksi ringan (smooth scroll & tabel responsif)
├── rps/
│   └── index.html               # Halaman publik Silabus / RPS (16 pekan pembelajaran)
├── rps-benchmark/
│   └── index.html               # Halaman evaluasi benchmark kurikulum internasional
├── materials/                   # Naskah Beamer LaTeX & berkas PDF materi perkuliahan
│   ├── pertemuan-04-cnn-from-scratch.tex  # Naskah Beamer LaTeX Pertemuan 4
│   ├── pertemuan-04-cnn-from-scratch.pdf  # PDF Slide Resmi Pertemuan 4 (15 slide)
│   ├── pertemuan-05-cnn-variations.tex    # Naskah Beamer LaTeX Pertemuan 5 (Placeholder)
│   ├── pertemuan-05-cnn-variations.pdf    # PDF Slide Pertemuan 5 (Placeholder)
│   ├── pertemuan-06-finetuning-imagenet.tex # Naskah Beamer LaTeX Pertemuan 6 (Placeholder)
│   ├── pertemuan-06-finetuning-imagenet.pdf # PDF Panduan Studio Pertemuan 6 (Placeholder)
│   ├── slide_preview-01.png               # Thumbnail slide 1
│   └── slide_preview-02.png               # Thumbnail slide 2
├── vite.config.js               # Konfigurasi bundler Vite (multi-page entry & rewrite URL)
├── package.json                 # Konfigurasi dependensi & npm scripts
├── DESIGN.md                    # Pedoman token desain, tipografi, dan komponen
└── README.md                    # Dokumentasi repositori
```

---

## 🚀 Panduan Pengembangan Lokal

### 1. Prasyarat
- Node.js (v18 atau lebih baru)
- TeX Live / MacTeX (opsional, jika ingin mengompilasi ulang slide `.tex` ke `.pdf` secara lokal)

### 2. Instalasi Dependensi
```bash
npm install
```

### 3. Menjalankan Development Server
```bash
npm run dev
```
Buka browser pada `http://localhost:5173/` untuk melihat beranda materi pertemuan 4, atau navigasi ke `/rps` dan `/rps-benchmark`.

### 4. Kompilasi Slide LaTeX ke PDF
Jika Anda melakukan pembaruan pada [materials/pertemuan-04-cnn-from-scratch.tex](./materials/pertemuan-04-cnn-from-scratch.tex), jalankan perintah berikut untuk mengompilasi ulang PDF:
```bash
npm run build:pdf
```

### 5. Membangun Aset Produksi (Build)
```bash
npm run build
```
Hasil kompilasi HTML, CSS, JS, dan seluruh berkas materi PDF akan diletakkan di direktori `dist/`.

---

## 🎨 Ringkasan Desain (DESIGN.md)

- **Palet Warna Netral & Berkontras Tinggi**:
  - Ink (`#111111`) dan Soft Cloud (`#F5F5F5`) sebagai latar primer dan sekunder.
  - Hairline (`#CACACB` / `#E5E5E5`) untuk pemisah grid dan batas kartu.
  - Aksen fungsional (`#0A7281` Teal dan `#1151FF` Blue).
- **Tipografi Bold & Terstruktur**: Judul *display uppercase* yang tegas dipadukan dengan tipografi isi sans-serif bersih.
- **Bentuk Tombol (Pill CTAs)**: Seluruh tombol aksi utama menggunakan kontur kapsul (`border-radius: 9999px`) tanpa drop-shadow buatan.
- **Grid Datar Datar (Flat Cards)**: Kartu diletakkan rata pada kontainer tanpa elevasi bayangan berlebih.

---

## 📚 Informasi Akademik

- **Mata Kuliah**: Deep Learning (IF25-40401)
- **Bobot**: 3 SKS
- **Program Studi**: Teknik Informatika
- **Institusi**: Institut Teknologi Sumatera (ITERA)
- **Repositori**: [informatika-itera/deep-learning-IF25-40401](https://github.com/informatika-itera/deep-learning-IF25-40401)
