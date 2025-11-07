# Whisper-CPP: Audio Transcription Tool

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Whisper.cpp](https://img.shields.io/badge/Whisper-cpp-green.svg)](https://github.com/ggerganov/whisper.cpp)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](#)

*Alat transkrip audio yang kuat menggunakan Whisper.cpp dengan dukungan Bahasa Indonesia*

</div>

---

## 📋 Daftar Isi

- [Tentang Proyek](#-tentang-proyek)
- [Fitur Utama](#-fitur-utama)
- [Prasyarat](#-prasyarat)
- [Instalasi](#-instalasi)
- [Cara Penggunaan](#-cara-penggunaan)
- [Model yang Tersedia](#-model-yang-tersedia)
- [Contoh Penggunaan](#-contoh-penggunaan)
- [Struktur Proyek](#-struktur-proyek)
- [Troubleshooting](#-troubleshooting)
- [Kontribusi](#-kontribusi)

---

## 🎯 Tentang Proyek

**Whisper-CPP** adalah wrapper Python untuk [whisper.cpp](https://github.com/ggerganov/whisper.cpp) yang memudahkan proses transkrip audio menjadi teks. Proyek ini dirancang khusus untuk menangani audio panjang dengan cara memecahnya menjadi potongan-potongan kecil, mentranskrip setiap bagian, lalu menggabungkannya kembali.

### Mengapa Whisper-CPP?

- ✅ **Efisien**: Implementasi C++ dari OpenAI Whisper yang lebih ringan dan cepat
- ✅ **Fleksibel**: Mendukung file lokal maupun URL audio
- ✅ **Otomatis**: Mengunduh model secara otomatis jika belum tersedia
- ✅ **Bahasa Indonesia**: Dukungan penuh untuk transkrip bahasa Indonesia
- ✅ **Audio Panjang**: Dapat menangani audio berdurasi panjang dengan chunking otomatis

### Teknologi yang Digunakan

- **Whisper.cpp**: Port C/C++ dari model Whisper OpenAI untuk performa tinggi
- **Python**: Scripting untuk automasi dan processing
- **PyDub**: Library untuk manipulasi audio
- **FFmpeg**: Backend untuk processing audio

---

## ✨ Fitur Utama

### 🎵 Processing Audio
- Mendukung berbagai format audio (MP3, WAV, M4A, dll)
- Otomatis memecah audio panjang menjadi chunk 10 menit
- Download audio dari URL atau gunakan file lokal

### 🤖 Model AI
- Akses ke berbagai ukuran model Whisper (tiny, base, small, medium, large)
- Auto-download model dari HuggingFace
- Dukungan untuk model terbaru (large-v3-turbo)

### 📝 Transkrip
- Transkrip otomatis dengan timestamp
- Dukungan multi-bahasa (fokus pada Bahasa Indonesia)
- Hasil transkrip digabung dalam satu file final

### 🛠️ Build Tools
- Script build otomatis untuk whisper.cpp
- Build static (tidak perlu shared libraries)
- Support multi-core compilation

---

## 📦 Prasyarat

Sebelum menggunakan proyek ini, pastikan sistem Anda memiliki:

### Software yang Diperlukan

```bash
# Python 3.8 atau lebih baru
python3 --version

# pip (Python package manager)
pip3 --version

# FFmpeg (untuk processing audio)
ffmpeg -version

# curl (untuk download model/audio)
curl --version

# CMake dan build tools (jika ingin rebuild)
cmake --version
make --version
```

### Instalasi Dependencies di Ubuntu/Debian

```bash
sudo apt update
sudo apt install -y python3 python3-pip ffmpeg curl cmake build-essential
```

### Instalasi Dependencies di macOS

```bash
# Menggunakan Homebrew
brew install python3 ffmpeg curl cmake
```

### Instalasi Dependencies di Windows

1. Install [Python 3.8+](https://www.python.org/downloads/)
2. Install [FFmpeg](https://ffmpeg.org/download.html)
3. Install [CMake](https://cmake.org/download/) (jika perlu rebuild)

---

## 🚀 Instalasi

### 1. Clone Repository

```bash
git clone https://github.com/Sparkplugx1904/Whisper-CPP.git
cd Whisper-CPP
```

### 2. Install Python Dependencies

```bash
pip3 install pydub
```

### 3. Verifikasi Binary

Binary whisper.cpp sudah tersedia di folder `build/bin/`. Verifikasi dengan:

```bash
./build/bin/whisper-cli --help
```

### 4. (Opsional) Rebuild Whisper.cpp

Jika ingin rebuild dari source:

```bash
# Build dengan semua CPU cores
./build_whisper_static.sh

# Atau tentukan jumlah jobs
./build_whisper_static.sh 4
```

---

## 💻 Cara Penggunaan

### Syntax Dasar

```bash
python3 transcriptor_cpp.py <sumber_audio> <nama_model>
```

### Parameter

- `<sumber_audio>`: Path ke file audio lokal atau URL
- `<nama_model>`: Model whisper yang akan digunakan (lihat daftar model di bawah)

### Contoh Penggunaan Sederhana

#### 1. Transkrip File Lokal

```bash
python3 transcriptor_cpp.py ./samples/jfk.mp3 base
```

#### 2. Transkrip dari URL

```bash
python3 transcriptor_cpp.py "https://example.com/audio.mp3" medium
```

#### 3. Menggunakan Model Berbeda

```bash
# Model kecil (cepat tapi kurang akurat)
python3 transcriptor_cpp.py audio.mp3 tiny

# Model besar (lambat tapi lebih akurat)
python3 transcriptor_cpp.py audio.mp3 large-v3
```

---

## 🎭 Model yang Tersedia

Whisper menyediakan berbagai ukuran model dengan trade-off antara kecepatan dan akurasi:

| Model | Ukuran | Parameter | Memori | Kecepatan | Akurasi |
|-------|--------|-----------|---------|-----------|---------|
| `tiny` | ~75 MB | 39 M | ~1 GB | ⚡⚡⚡⚡⚡ | ⭐⭐ |
| `base` | ~142 MB | 74 M | ~1 GB | ⚡⚡⚡⚡ | ⭐⭐⭐ |
| `small` | ~466 MB | 244 M | ~2 GB | ⚡⚡⚡ | ⭐⭐⭐⭐ |
| `medium` | ~1.5 GB | 769 M | ~5 GB | ⚡⚡ | ⭐⭐⭐⭐⭐ |
| `large-v1` | ~2.9 GB | 1550 M | ~10 GB | ⚡ | ⭐⭐⭐⭐⭐ |
| `large-v2` | ~2.9 GB | 1550 M | ~10 GB | ⚡ | ⭐⭐⭐⭐⭐ |
| `large-v3` | ~2.9 GB | 1550 M | ~10 GB | ⚡ | ⭐⭐⭐⭐⭐ |
| `large-v3-turbo` | ~1.6 GB | 809 M | ~6 GB | ⚡⚡ | ⭐⭐⭐⭐⭐ |

### Rekomendasi Model

- **Untuk testing cepat**: `tiny` atau `base`
- **Untuk penggunaan umum**: `small` atau `medium`
- **Untuk akurasi maksimal**: `large-v3` atau `large-v3-turbo`
- **Untuk production (balanced)**: `medium` atau `large-v3-turbo`

---

## 📚 Contoh Penggunaan

### Skenario 1: Transkrip Podcast Bahasa Indonesia

```bash
# Download dan transkrip podcast
python3 transcriptor_cpp.py "https://example.com/podcast-indonesia.mp3" medium
```

**Output:**
```
[+] Mengunduh: https://example.com/podcast-indonesia.mp3
[✓] Unduhan selesai: audio.mp3
[+] Memecah audio menjadi potongan 10 menit...
    → chunks/part_1.mp3
    → chunks/part_2.mp3
[✓] Total 2 potongan audio dibuat.
[+] Transcribing potongan 1/2: chunks/part_1.mp3
    → Waktu mulai: 00:00:00
    [✓] Disimpan: ./transcripts/part_1.mp3.txt
[+] Transcribing potongan 2/2: chunks/part_2.mp3
    → Waktu mulai: 00:10:00
    [✓] Disimpan: ./transcripts/part_2.mp3.txt
[+] Menggabungkan seluruh hasil transkripsi...
[✓] Transkrip akhir disimpan di: ./transcripts/final_transcript.txt
[✓] Final transcript: ./transcripts/audio.mp3.txt
```

### Skenario 2: Transkrip Meeting Recording

```bash
# Transkrip file meeting yang sudah ada
python3 transcriptor_cpp.py ./recordings/meeting-2025-01-15.mp3 large-v3-turbo
```

### Skenario 3: Batch Processing

Untuk memproses beberapa file sekaligus, buat script bash sederhana:

```bash
#!/bin/bash
# batch_transcribe.sh

MODEL="medium"

for audio in ./audio_files/*.mp3; do
    echo "Processing: $audio"
    python3 transcriptor_cpp.py "$audio" "$MODEL"
done
```

Jalankan dengan:
```bash
chmod +x batch_transcribe.sh
./batch_transcribe.sh
```

---

## 📁 Struktur Proyek

```
Whisper-CPP/
│
├── 📄 README.md                    # Dokumentasi proyek (file ini)
├── 🐍 transcriptor_cpp.py          # Script utama untuk transkrip
├── 🔧 build_whisper_static.sh      # Script untuk build whisper.cpp
│
├── 📦 build/                       # Binary hasil build
│   └── bin/
│       ├── whisper-cli             # CLI utama whisper
│       ├── whisper-server          # Server mode whisper
│       ├── whisper-bench           # Benchmarking tool
│       └── ...
│
├── 🎵 samples/                     # Contoh file audio
│   └── jfk.mp3
│
├── 🤖 models/                      # Direktori untuk model AI
│   ├── README.md                   # Info tentang model
│   ├── download-ggml-model.sh     # Script download model
│   └── *.bin                       # File model (akan diunduh otomatis)
│
├── 📝 grammars/                    # Grammar files untuk constrained generation
│   ├── assistant.gbnf
│   ├── chess.gbnf
│   └── colors.gbnf
│
├── 🐳 .devops/                     # Docker files
│   ├── main.Dockerfile
│   ├── cublas.Dockerfile
│   └── ...
│
└── 📤 transcripts/                 # Output transkrip (dibuat otomatis)
    └── *.txt
```

---

## 🔧 Troubleshooting

### Problem: "Model not found"

**Solusi:**
Script akan otomatis mengunduh model dari HuggingFace. Pastikan koneksi internet aktif.

Manual download:
```bash
cd models
./download-ggml-model.sh medium
```

### Problem: "FFmpeg not found"

**Solusi:**
Install FFmpeg:
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
# Download dari https://ffmpeg.org/download.html
```

### Problem: "Permission denied" saat menjalankan script

**Solusi:**
```bash
chmod +x build_whisper_static.sh
chmod +x transcriptor_cpp.py
```

### Problem: Audio terlalu panjang, proses lambat

**Solusi:**
- Gunakan model yang lebih kecil (`tiny` atau `base`)
- Adjust chunk size di `transcriptor_cpp.py` (default: 10 menit)
- Pastikan sistem memiliki cukup RAM

### Problem: Binary whisper-cli tidak ada

**Solusi:**
Rebuild whisper.cpp:
```bash
./build_whisper_static.sh
```

Atau download pre-built binary dari releases.

### Problem: Transkrip tidak akurat

**Solusi:**
1. Gunakan model yang lebih besar (`medium` atau `large-v3`)
2. Pastikan audio berkualitas baik (tidak terlalu banyak noise)
3. Untuk bahasa Indonesia, pastikan parameter `-l id` di script aktif

### Problem: "ModuleNotFoundError: No module named 'pydub'"

**Solusi:**
```bash
pip3 install pydub
```

---

## 🎯 Advanced Usage

### Menggunakan Whisper CLI Langsung

Untuk kontrol lebih detail, gunakan whisper-cli langsung:

```bash
./build/bin/whisper-cli \
    -m ./models/ggml-medium.bin \
    -f ./samples/jfk.mp3 \
    -l id \
    -otxt \
    -pp
```

**Parameter penting:**
- `-m`: Path ke model
- `-f`: Input audio file
- `-l`: Bahasa (id = Indonesia, en = English)
- `-otxt`: Output format text
- `-pp`: Pretty print
- `-nt`: No timestamps
- `-osrt`: Output SRT subtitle format
- `-ovtt`: Output VTT subtitle format

### Menggunakan Whisper Server

Jalankan sebagai server HTTP:

```bash
./build/bin/whisper-server \
    -m ./models/ggml-medium.bin \
    --port 8080 \
    --host 0.0.0.0
```

Kirim request:
```bash
curl -X POST http://localhost:8080/inference \
    -F file="@audio.mp3" \
    -F language="id"
```

### Grammar-Constrained Generation

Gunakan grammar files untuk membatasi output:

```bash
./build/bin/whisper-cli \
    -m ./models/ggml-medium.bin \
    -f audio.mp3 \
    -gr ./grammars/colors.gbnf
```

---

## 🤝 Kontribusi

Kontribusi sangat diterima! Berikut cara berkontribusi:

1. **Fork** repository ini
2. **Clone** fork Anda
3. Buat **branch** fitur baru (`git checkout -b fitur-baru`)
4. **Commit** perubahan (`git commit -m 'Menambah fitur baru'`)
5. **Push** ke branch (`git push origin fitur-baru`)
6. Buat **Pull Request**

### Area yang Bisa Dikontribusi

- 🐛 Bug fixes
- 📝 Dokumentasi
- ✨ Fitur baru
- 🌐 Terjemahan
- 🎨 UI/UX improvements
- 🧪 Testing
- 📊 Benchmarking

---

## 📜 Lisensi

Proyek ini menggunakan komponen dari [whisper.cpp](https://github.com/ggerganov/whisper.cpp) yang dilisensikan oleh pengelolanya.

Model Whisper dari OpenAI tersedia dengan lisensi MIT.

---

## 🙏 Acknowledgments

- **OpenAI** - Untuk model Whisper yang luar biasa
- **ggerganov** - Untuk implementasi whisper.cpp yang efisien
- **Komunitas Open Source** - Untuk berbagai tools dan libraries yang digunakan

---

## 📧 Kontak

Untuk pertanyaan, bug reports, atau feedback:

- **GitHub Issues**: [Create an issue](https://github.com/Sparkplugx1904/Whisper-CPP/issues)
- **Email**: anandapradnyana68@gmail.com

---

## 📊 Statistik & Performance

### Benchmark (pada sistem standar)

| Model | Audio Duration | Processing Time | Real-time Factor |
|-------|---------------|-----------------|------------------|
| tiny | 60 min | ~3 min | 0.05x |
| base | 60 min | ~6 min | 0.1x |
| small | 60 min | ~15 min | 0.25x |
| medium | 60 min | ~30 min | 0.5x |
| large-v3 | 60 min | ~60 min | 1.0x |

*Real-time factor: 0.1x berarti 10x lebih cepat dari durasi audio*

---

## 🗺️ Roadmap

- [ ] Web interface untuk transkrip
- [ ] Support untuk streaming audio
- [ ] Integration dengan cloud storage (Google Drive, Dropbox)
- [ ] Automatic punctuation restoration
- [ ] Speaker diarization (identifikasi pembicara)
- [ ] Real-time transcription
- [ ] Mobile app (Android/iOS)
- [ ] Docker container untuk deployment mudah

---

<div align="center">

**⭐ Jika proyek ini berguna, jangan lupa beri bintang! ⭐**

**Made with ❤️ by Sparkplugx1904**

</div>