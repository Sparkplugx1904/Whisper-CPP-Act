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
        
    # 2. Cek 'ffmpeg' (KEMBALI DIPERLUKAN untuk konversi dan stabilisasi volume/loudnorm)
    if subprocess.run(['which', 'ffmpeg'], capture_output=True).returncode != 0:
        log_error("'ffmpeg' tidak ditemukan. Harap instal 'ffmpeg' untuk konversi audio dan stabilisasi volume.")
        dependencies_ok = False
        
    # 3. Cek 'whisper-cli'
    whisper_cli_path = Path("./build/bin/whisper-cli")
    if not whisper_cli_path.exists():
        log_error(f"'{whisper_cli_path}' tidak ditemukan. Pastikan Anda telah mengompilasi whisper.cpp.")
        dependencies_ok = False
        
    # 4. Cek pustaka Python (noisereduce, scipy, numpy)
    try:
        import noisereduce as nr
        import numpy as np
        from scipy.io import wavfile
        log_info("Pustaka Python (noisereduce, numpy, scipy) ditemukan.")
    except ImportError as e:
        log_error(f"Pustaka Python yang diperlukan tidak ditemukan: {e}. Harap instal dengan 'pip install noisereduce scipy numpy'.")
        dependencies_ok = False

    if not dependencies_ok:
        log_error("Dependensi tidak lengkap. Keluar.", exit_app=True)
        
    log_success("Semua dependensi (curl, ffmpeg, whisper-cli, pustaka Python) ditemukan.")
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
# FUNGSI BARU: KONVERSI OLEH FFMPEG (Bug Fix untuk MP3)
# -----------------------------------------------------

def convert_to_wav(input_path: Path, output_path: Path) -> Path:
    """Mengonversi file audio (MP3/M4A/dll) ke WAV Mono 16kHz menggunakan FFmpeg."""
    log_info(f"Mengonversi {input_path.name} ke WAV Mono 16kHz...")
    
    # Pastikan format WAV mono 16kHz untuk NumPy/SciPy/Whisper
    cmd = [
        "ffmpeg", "-i", str(input_path), 
        "-ac", "1", "-ar", "16000", "-y", 
        str(output_path)
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log_success("Konversi awal selesai.")
        return output_path
    except Exception as e:
        log_error(f"Gagal konversi ke WAV Mono 16kHz menggunakan FFmpeg: {e}", exit_app=True)

def read_and_prepare_audio(input_path: Path) -> Tuple[int, np.ndarray]:
    """Membaca file WAV yang sudah dikonversi dan mempersiapkannya menjadi array float64 untuk noisereduce."""
    log_info(f"Membaca file WAV: {input_path.name}")
    try:
        rate, data = wavfile.read(input_path)
        
        # Data sudah Mono (ac=1) dari FFmpeg. Cukup konversi tipe data.
        if data.dtype.kind in ('i', 'u'):
            data = data.astype(np.float64) / 32768.0
            
        return rate, data
    except Exception as e:
        log_error(f"Gagal membaca file WAV yang telah dikonversi: {e}", exit_app=True)

# -----------------------------------------------------
# FUNGSI BARU: PEMROSESAN AUDIO GABUNGAN (DENOISE + LOUDNORM Penuh)
# -----------------------------------------------------

def process_and_normalize_audio(input_data: np.ndarray, rate: int, output_base_path: Path) -> Path:
    """
    1. Melakukan Denoising (noisereduce, Python murni).
    2. Melakukan Normalisasi Loudnorm (FFmpeg) pada seluruh file.
    """
    
    # 1. Denoising dengan noisereduce (Python Murni)
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

    # 2. Loudnorm (Stabilisasi Volume) menggunakan FFmpeg pada seluruh file
    log_info("2. Melakukan Normalisasi Loudnorm pada seluruh file (FFmpeg)...")
    
    temp_wav_path = output_base_path.with_suffix(".temp.wav")
    final_processed_path = output_base_path.with_suffix(".wav")
    
    # Tulis segmen yang telah di-denoising ke file sementara untuk Loudnorm FFmpeg
    # Menggunakan int16 untuk penulisan wavfile
    wavfile.write(temp_wav_path, rate, (denoised_data * 32767).astype(np.int16))
    
    # Loudnorm (Stabilisasi Volume) menggunakan FFmpeg pada file tunggal
    # I=-23 LUFS adalah target standar
    cmd_loudnorm = [
        "ffmpeg", "-i", str(temp_wav_path), "-y",
        "-ac", "1", "-ar", "16000",
        "-af", "loudnorm=I=-23:LRA=7:tp=-2", # Hapus print_format=json
        str(final_processed_path)
    ]
    
    try:
        subprocess.run(cmd_loudnorm, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log_success(f"Loudnorm selesai dan disimpan ke: {final_processed_path.name}")
        return final_processed_path
    except subprocess.CalledProcessError as e:
        log_error(f"FFmpeg Loudnorm gagal. Code: {e.returncode}. Pastikan FFmpeg mendukung filter loudnorm.", exit_app=True)
    finally:
        if temp_wav_path.exists():
            temp_wav_path.unlink() # Hapus file sementara hasil denoising

def cleanup_segments(temp_dir: Path):
    """Menghapus semua file dan folder sementara."""
    # Dikembalikan untuk memastikan tidak ada folder sisa dari eksperimen sebelumnya
    try:
        if temp_dir.exists():
            for item in temp_dir.iterdir():
                item.unlink()
            temp_dir.rmdir()
            log_info("Pembersihan file sementara selesai.")
    except Exception as e:
        log_warn(f"Gagal membersihkan direktori sementara {temp_dir}: {e}")

# -----------------------------------------------------
# FUNGSI transcribe_single_audio (Tidak Berubah)
# -----------------------------------------------------

def transcribe_single_audio(audio_path, model_path, whisper_cli_path):
    """Mentranskripsi seluruh file audio tunggal menggunakan whisper.cpp CLI."""
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
        "-of", str(output_base_path_temp), 
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

    # --- Pindahkan dan Bersihkan TXT/SRT ---
    temp_txt_file = output_base_path_temp.with_suffix(".txt")
    try:
        if temp_txt_file.exists():
            content = temp_txt_file.read_text(encoding="utf-8").strip()
            final_txt.write_text(content, encoding="utf-8")
            temp_txt_file.unlink()
            log_success(f"TXT berhasil disimpan ke {final_txt}.")
        # ... (Kode SRT dan penanganan error tetap sama) ...
        
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

    temp_segment_dir = Path("./temp_segments")

    try:
        # PENTING: Perbaiki bug ini dulu (jika ini penyebab program berhenti)
        whisper_cli_path = check_dependencies()
        source = sys.argv[1]
        model_name = sys.argv[2]
        model_path = ensure_model_exists(model_name)

        original_audio_path = Path("original_audio.mp3") 
        
        # Penentuan Path Audio Input
        if os.path.exists(source):
            audio_path_to_process = Path(source)
        else:
            download_audio(source, original_audio_path)
            audio_path_to_process = original_audio_path

        # TAHAP 1: KONVERSI AWAL (FFMPEG - FIX BUG MP3)
        converted_wav_path = Path(f"converted_{audio_path_to_process.stem}.wav")
        converted_wav_path = convert_to_wav(audio_path_to_process, converted_wav_path)
        
        # TAHAP 2: MEMBACA DAN MEMPERSIAPKAN (PYTHON MURNI)
        rate, data = read_and_prepare_audio(converted_wav_path)
        
        # TAHAP 3: DENOISING & LOUDNORM (PYTHON + FFMPEG FULL FILE)
        final_processed_audio_path = process_and_normalize_audio(data, rate, converted_wav_path)

        # TAHAP 4: TRANSKRIPSI
        transcribe_single_audio(final_processed_audio_path, model_path, whisper_cli_path)

    except Exception as e:
        log_error(f"Terjadi error fatal yang tidak terduga: {e}", exit_app=False)
        print("------ STACK TRACE LENGKAP ------")
        traceback.print_exc()
        print("---------------------------------")
        sys.exit(1)
        
    finally:
        # TAHAP 5: PEMBERSIHAN
        cleanup_segments(temp_segment_dir)
        if 'final_processed_audio_path' in locals() and final_processed_audio_path.exists():
             final_processed_audio_path.unlink() # Hapus file WAV hasil pemrosesan akhir
        if 'converted_wav_path' in locals() and converted_wav_path.exists():
            converted_wav_path.unlink() # Hapus file WAV yang dikonversi
        if 'original_audio_path' in locals() and original_audio_path.exists():
            original_audio_path.unlink() # Hapus file yang diunduh dari URL
        
        log_success("====== PROSES SELESAI TOTAL ======")
        log_info("Output akhir ada di folder ./transcripts/")

if __name__ == "__main__":
    main()