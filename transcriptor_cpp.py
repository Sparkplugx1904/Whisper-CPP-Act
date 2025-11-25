#!/usr/bin/env python3
import sys
import os
import subprocess
import re
import traceback
from pathlib import Path
import numpy as np
from scipy.io import wavfile
import noisereduce as nr
from typing import List, Tuple

# --- Sistem Logging Kustom ---

def log_info(msg):
    """Mencatat pesan informasi."""
    print(f"[+] {msg}")

def log_success(msg):
    """Mencatat pesan sukses."""
    print(f"[✓] {msg}")

def log_warn(msg):
    """Mencatat pesan peringatan."""
    print(f"[!] {msg}")

def log_error(msg, exit_app=False):
    """Mencatat pesan error. Jika exit_app=True, hentikan skrip."""
    print(f"[✗] ERROR: {msg}", file=sys.stderr)
    if exit_app:
        sys.exit(1)

# --- Daftar Model yang Valid ---
VALID_MODELS = ["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3", "large-v3-turbo"]

# --- Fungsi Inti dengan Penanganan Error ---

def check_dependencies():
    """Memeriksa semua dependensi eksternal sebelum memulai."""
    log_info("Memeriksa dependensi...")
    dependencies_ok = True
    
    # 1. Cek 'curl'
    if subprocess.run(['which', 'curl'], capture_output=True).returncode != 0:
        log_error("'curl' tidak ditemukan. Harap instal 'curl' untuk mengunduh file.")
        dependencies_ok = False
        
    # 2. Cek 'whisper-cli'
    whisper_cli_path = Path("./build/bin/whisper-cli")
    if not whisper_cli_path.exists():
        log_error(f"'{whisper_cli_path}' tidak ditemukan. Pastikan Anda telah mengompilasi whisper.cpp.")
        dependencies_ok = False
        
    # 3. Cek pustaka Python (noisereduce, scipy, numpy)
    try:
        import noisereduce as nr
        import numpy as np
        from scipy.io import wavfile
        log_info("Pustaka Python (noisereduce, numpy, scipy) ditemukan.")
    except ImportError as e:
        log_error(f"Pustaka Python yang diperlukan tidak ditemukan: {e}. Harap instal dengan 'pip install noisereduce scipy numpy'.")
        dependencies_ok = False
        
    # Hapus pemeriksaan FFmpeg karena tidak digunakan

    if not dependencies_ok:
        log_error("Dependensi tidak lengkap. Keluar.", exit_app=True)
        
    # Catatan: FFmpeg DIHAPUS dari daftar yang dicek
    log_success("Semua dependensi (curl, whisper-cli, pustaka Python) ditemukan.")
    return whisper_cli_path
    
# --- Fungsi download_file, ensure_model_exists, download_audio tetap sama ---
def download_file(url, dest):
    """Mengunduh file menggunakan curl dengan penanganan error yang kuat."""
    log_info(f"Mengunduh: {url} → {dest}")
    try:
        subprocess.run(
            ["curl", "-L", "-o", str(dest), "-m", "300", url], 
            check=True
        )
        print()
        log_success(f"Unduhan selesai: {dest}")
        return True
    except subprocess.CalledProcessError as e:
        print()
        log_error(f"Gagal mengunduh file (curl return code: {e.returncode}). Lihat pesan error di atas.", exit_app=False)
        if dest.exists():
            dest.unlink()
        return False
    except Exception as e:
        log_error(f"Terjadi error tak terduga saat mengunduh: {e}", exit_app=False)
        return False

def ensure_model_exists(model_name):
    """Memastikan model ada, memvalidasi nama, dan mengunduh jika perlu."""
    if model_name not in VALID_MODELS:
        log_error(f"Nama model tidak valid: '{model_name}'. Pilihan: {', '.join(VALID_MODELS)}", exit_app=True)

    os.makedirs("models", exist_ok=True)
    model_path = Path(f"./models/ggml-{model_name}.bin")
    
    if model_path.exists():
        log_success(f"Model ditemukan: {model_path}")
        return model_path
    
    log_warn(f"Model '{model_name}' belum ada, mengunduh dari HuggingFace...")
    url = f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{model_name}.bin"
    
    if not download_file(url, model_path):
        log_error("Gagal mengunduh model. Membatalkan.", exit_app=True)
        
    log_success(f"Model '{model_name}' berhasil diunduh.")
    return model_path

def download_audio(url, output_path):
    """Wrapper untuk mengunduh file audio."""
    log_info(f"Mengunduh audio dari: {url}")
    if not download_file(url, output_path):
        log_error("Gagal mengunduh audio. Membatalkan.", exit_app=True)
    log_success(f"Audio berhasil diunduh ke {output_path}")

# -----------------------------------------------------
# FUNGSI BARU: PEMROSESAN AUDIO TANPA FFMPEG
# -----------------------------------------------------

def read_and_prepare_audio(input_path: Path) -> Tuple[int, np.ndarray]:
    """
    Membaca file audio menggunakan scipy.io.wavfile dan 
    memastikan formatnya float64 untuk noisereduce.
    CATATAN: File input harus berformat WAV atau format lain yang didukung 
    oleh scipy.io.wavfile (umumnya WAV) dan mono/stereo.
    Karena FFmpeg dihapus, kami tidak bisa memaksa konversi MP3 ke WAV 16kHz Mono.
    """
    log_info(f"Membaca file audio: {input_path.name}")
    try:
        # PENTING: Jika input_path adalah MP3, ini akan gagal karena scipy.io.wavfile 
        # hanya mendukung WAV, bukan MP3. Asumsi: input harus WAV/format yang didukung SciPy.
        rate, data = wavfile.read(input_path)
        log_success("Pembacaan audio selesai.")
        
        # Konversi ke Mono jika stereo (dengan mengambil rata-rata channel)
        if data.ndim > 1:
            log_warn("Audio terdeteksi stereo/multichannel. Mengonversi ke Mono (rata-rata channel).")
            data = np.mean(data, axis=1)
        
        # Konversi data ke float64 (-1.0 hingga 1.0), diperlukan untuk noisereduce
        # Hanya dilakukan jika tipe data bukan float
        if data.dtype.kind in ('i', 'u'): # Jika integer/unsigned integer
            data = data.astype(np.float64) / 32768.0
            
        return rate, data
    except Exception as e:
        log_error(f"Gagal membaca atau memproses file audio. Pastikan file input adalah format WAV yang valid: {e}", exit_app=True)

def process_and_normalize_audio(input_data: np.ndarray, rate: int, output_base_path: Path) -> Path:
    """
    1. Melakukan Denoising (noisereduce).
    2. Melakukan Normalisasi Puncak (Peak Normalization) statis pada seluruh file.
    3. Menyimpan hasil dalam format WAV.
    """
    
    # 1. Denoising dengan noisereduce
    log_info("1. Melakukan Denoising pada seluruh file audio (noisereduce)...")
    # Tentukan sampel noise (Asumsi 1 detik pertama adalah noise murni).
    noise_len_sec = min(1.0, len(input_data) / rate) 
    noise_sample = input_data[:int(rate * noise_len_sec)]
    
    denoised_data = nr.reduce_noise(
        audio_clip=input_data, 
        noise_clip=noise_sample, 
        verbose=False,
        sr=rate
    )
    log_success("Denoising selesai.")

    # 2. Normalisasi Puncak Statis (Pengganti Loudnorm FFmpeg)
    log_info("2. Melakukan Normalisasi Puncak Statis (mengurangi volume tertinggi ke 0 dBFS)...")
    
    # Cari nilai puncak (absolut) dalam data
    peak_value = np.max(np.abs(denoised_data))
    
    if peak_value > 0:
        # Hitung faktor scaling untuk membawa puncak ke 1.0 (0 dBFS)
        scaling_factor = 1.0 / peak_value
        normalized_data = denoised_data * scaling_factor
        log_info(f"Volume ditingkatkan sebesar faktor: {scaling_factor:.2f}")
    else:
        normalized_data = denoised_data
        log_warn("Nilai puncak audio nol, tidak ada normalisasi yang dilakukan.")

    # 3. Menyimpan Hasil
    final_processed_path = output_base_path.with_suffix(".wav")
    
    log_info(f"3. Menyimpan hasil denoising dan normalisasi ke {final_processed_path.name}...")
    
    # Konversi data kembali ke format integer 16-bit
    output_int16 = (normalized_data * 32767).astype(np.int16)
    
    try:
        wavfile.write(final_processed_path, rate, output_int16)
        log_success(f"Pemrosesan Selesai dan disimpan: {final_processed_path.name}")
        return final_processed_path
    except Exception as e:
        log_error(f"Gagal menyimpan file audio akhir: {e}", exit_app=True)

def cleanup_segments(temp_dir: Path):
    """Fungsi cleanup dipertahankan tetapi tidak melakukan apa-apa karena tidak ada segmen."""
    pass 
    
# -----------------------------------------------------
# FUNGSI transcribe_single_audio (Tidak Berubah)
# -----------------------------------------------------

def transcribe_single_audio(audio_path, model_path, whisper_cli_path):
    """Mentranskripsi seluruh file audio tunggal menggunakan whisper.cpp CLI."""
    # (Kode fungsi ini tetap sama seperti sebelumnya, menggunakan audio_path)
    os.makedirs("transcripts", exist_ok=True)

    final_txt = Path("transcripts/transcript.txt")
    final_srt = Path("transcripts/transcript.srt")
    
    try:
        final_txt.write_text("", encoding="utf-8")
        final_srt.write_text("", encoding="utf-8")
    except IOError as e:
        log_error(f"Gagal membuat file transkrip akhir di ./transcripts/. Periksa izin folder. Error: {e}", exit_app=True)

    log_info(f"Mentranskripsi file tunggal: {audio_path.name}")
    
    output_base_path_temp = Path(audio_path.stem)
    
    cmd = [
        str(whisper_cli_path),
        "-m", str(model_path),
        "-f", str(audio_path),
        "--temperature", "0.6",
        "-of", str(output_base_path_temp), # Output sementara di root
        "-otxt",
        "-osrt",
        "-l", "id",
        "-pp"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print()
    except subprocess.CalledProcessError as e:
        print()
        log_error(f"whisper-cli gagal pada file {audio_path.name} (return code: {e.returncode}). Lihat pesan error di atas.", exit_app=False)
        log_error("Gagal melakukan transkripsi. Proses dihentikan.", exit_app=True)
    except Exception as e:
        log_error(f"Error tak terduga saat menjalankan whisper-cli pada {audio_path.name}: {e}", exit_app=True)

    temp_txt_file = output_base_path_temp.with_suffix(".txt")
    try:
        if temp_txt_file.exists():
            content = temp_txt_file.read_text(encoding="utf-8").strip()
            final_txt.write_text(content, encoding="utf-8")
            temp_txt_file.unlink()
            log_success(f"TXT berhasil disimpan ke {final_txt}.")
        else:
            log_warn(f"File TXT output tidak ditemukan: {temp_txt_file}")
    except Exception as e:
        log_error(f"Gagal memproses file TXT {temp_txt_file}: {e}")

    temp_srt_file = output_base_path_temp.with_suffix(".srt")
    try:
        if temp_srt_file.exists():
            content = temp_srt_file.read_text(encoding="utf-8").strip()
            final_srt.write_text(content, encoding="utf-8")
            temp_srt_file.unlink()
            log_success(f"SRT berhasil disimpan ke {final_srt}.")
        else:
            log_warn(f"File SRT output tidak ditemukan: {temp_srt_file}")
    except Exception as e:
        log_error(f"Gagal memproses file SRT {temp_srt_file}: {e}")

    log_success("Transkripsi file tunggal selesai.")

# -----------------------------------------------------
# FUNGSI MAIN BARU
# -----------------------------------------------------

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 transcriptor_cpp.py <url_or_file> <model>")
        print(f"Model: {', '.join(VALID_MODELS)}")
        sys.exit(1)

    # Variabel ini hanya dipertahankan untuk membersihkan file lama, tidak digunakan dalam logika proses
    temp_segment_dir = Path("./temp_segments")

    try:
        whisper_cli_path = check_dependencies()
        source = sys.argv[1]
        model_name = sys.argv[2]
        model_path = ensure_model_exists(model_name)

        original_audio_path = Path("original_audio.mp3") 
        
        # Penentuan Path Audio Input
        if os.path.exists(source):
            audio_path_to_process = Path(source)
        else:
            # PENTING: Jika input adalah URL MP3, curl akan mengunduhnya.
            # Namun, langkah selanjutnya (read_and_prepare_audio) TIDAK mendukung MP3.
            # Jadi, kita harus berasumsi audio_path_to_process adalah WAV.
            download_audio(source, original_audio_path)
            audio_path_to_process = original_audio_path
            log_warn("PERINGATAN: Karena FFmpeg dihapus, file yang diunduh (MP3/format lain) mungkin GAGAL dibaca oleh SciPy. Hanya format WAV yang didukung.")


        # TAHAP 1: MEMBACA DAN MEMPERSIAPKAN (Pengganti convert_to_wav)
        # Gunakan path output yang berbeda untuk menghindari penulisan di atas file asli jika itu WAV
        temp_wav_path = Path(f"denoised_normalized_{audio_path_to_process.stem}.wav")
        rate, data = read_and_prepare_audio(audio_path_to_process)
        
        # TAHAP 2: DENOISING & NORMALISASI (Memproduksi file akhir)
        final_processed_audio_path = process_and_normalize_audio(data, rate, temp_wav_path)

        # TAHAP 3: TRANSKRIPSI
        transcribe_single_audio(final_processed_audio_path, model_path, whisper_cli_path)

    except Exception as e:
        log_error(f"Terjadi error fatal yang tidak terduga: {e}", exit_app=False)
        print("------ STACK TRACE LENGKAP ------")
        traceback.print_exc()
        print("---------------------------------")
        sys.exit(1)
        
    finally:
        # TAHAP 4: PEMBERSIHAN
        cleanup_segments(temp_segment_dir) # Dibiarkan untuk kompatibilitas, meski tidak melakukan apa-apa
        if 'temp_wav_path' in locals() and temp_wav_path.exists():
            temp_wav_path.unlink() # Menghapus file WAV yang telah diproses/dinormalisasi
        if 'original_audio_path' in locals() and original_audio_path.exists():
            original_audio_path.unlink() # Menghapus file yang diunduh dari URL
        
        log_success("====== PROSES SELESAI TOTAL ======")
        log_info("Output akhir ada di folder ./transcripts/")

if __name__ == "__main__":
    main()