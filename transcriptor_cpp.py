#!/usr/bin/env python3

# -----------------------------------------------------
# LOG DEBUG LEVEL ATAS (MENGECEK APAKAH SKRIP BERJALAN)
# -----------------------------------------------------
print("--- [DEBUG] SCRIPT EXECUTOR BAWAH SADAR DIMULAI ---")

import sys
print("--- [DEBUG] Pustaka 'sys' berhasil diimpor.")
import os
print("--- [DEBUG] Pustaka 'os' berhasil diimpor.")
import subprocess
print("--- [DEBUG] Pustaka 'subprocess' berhasil diimpor.")
import re
print("--- [DEBUG] Pustaka 're' berhasil diimpor.")
import traceback
print("--- [DEBUG] Pustaka 'traceback' berhasil diimpor.")
from pathlib import Path
print("--- [DEBUG] Pustaka 'pathlib' berhasil diimpor.")
from typing import List, Tuple
print("--- [DEBUG] Pustaka 'typing' berhasil diimpor.")

try:
    import numpy as np
    print("--- [DEBUG] Pustaka 'numpy' berhasil diimpor.")
    from scipy.io import wavfile
    print("--- [DEBUG] Pustaka 'scipy.io.wavfile' berhasil diimpor.")
    import noisereduce as nr
    print("--- [DEBUG] Pustaka 'noisereduce' berhasil diimpor.")
except ImportError as e:
    print(f"[✗✗✗] FATAL: GAGAL MENGIMPOR PUSTAKA PENTING: {e}", file=sys.stderr)
    print("[✗✗✗] FATAL: Pastikan Anda telah menjalankan 'pip install -r requirements.txt'", file=sys.stderr)
    sys.exit(1)

print("--- [DEBUG] SEMUA PUSTAKA BERHASIL DIIMPOR ---")

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
        print("[✗] ERROR: Keluar dari program karena error fatal.", file=sys.stderr)
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
    else:
        log_info("Dependensi 'curl' ditemukan.")
        
    # 2. Cek 'ffmpeg' (DIPERLUKAN untuk konversi format awal MP3->WAV)
    if subprocess.run(['which', 'ffmpeg'], capture_output=True).returncode != 0:
        log_error("'ffmpeg' tidak ditemukan. Harap instal 'ffmpeg' untuk konversi audio.")
        dependencies_ok = False
    else:
        log_info("Dependensi 'ffmpeg' ditemukan.")
        
    # 3. Cek 'whisper-cli'
    whisper_cli_path = Path("./build/bin/whisper-cli")
    if not whisper_cli_path.exists():
        log_error(f"'{whisper_cli_path}' tidak ditemukan. Pastikan Anda telah mengompilasi whisper.cpp.")
        dependencies_ok = False
    else:
        log_info(f"Dependensi 'whisper-cli' ditemukan di {whisper_cli_path}.")
        
    if not dependencies_ok:
        log_error("Dependensi tidak lengkap. Keluar.", exit_app=True)
        
    log_success("Semua dependensi (curl, ffmpeg, whisper-cli, pustaka Python) ditemukan.")
    return whisper_cli_path
    
# --- Fungsi download_file, ensure_model_exists, download_audio (Tidak Berubah) ---
def download_file(url, dest):
    """Mengunduh file menggunakan curl dengan penanganan error yang kuat."""
    log_info(f"Mengunduh: {url} → {dest}")
    try:
        subprocess.run(
            ["curl", "-L", "-o", str(dest), "-m", "300", url], 
            check=True
        )
        print() # Newline setelah output curl
        log_success(f"Unduhan selesai: {dest}")
        return True
    except subprocess.CalledProcessError as e:
        print() # Newline setelah output curl
        log_error(f"Gagal mengunduh file (curl return code: {e.returncode}). Lihat pesan error di atas.", exit_app=False)
        if dest.exists():
            dest.unlink()
        return False
    except Exception as e:
        log_error(f"Terjadi error tak terduga saat mengunduh: {e}", exit_app=False)
        return False

def ensure_model_exists(model_name):
    """Memastikan model ada, memvalidasi nama, dan mengunduh jika perlu."""
    log_info(f"Memeriksa keberadaan model: {model_name}")
    if model_name not in VALID_MODELS:
        log_error(f"Nama model tidak valid: '{model_name}'. Pilihan: {', '.join(VALID_MODELS)}", exit_app=True)

    os.makedirs("models", exist_ok=True)
    model_path = Path(f"./models/ggml-{model_name}.bin")
    
    if model_path.exists():
        log_success(f"Model ditemukan: {model_path}")
        return model_path
    
    log_warn(f"Model '{model_name}' belum ada, mengunduh dari HuggingFace...")
    url = f"https.huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{model_name}.bin"
    
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
# TAHAP 1: KONVERSI OLEH FFMPEG (Hanya Konversi Format)
# -----------------------------------------------------

def convert_to_wav(input_path: Path, output_path: Path) -> Path:
    """Mengonversi file audio (MP3/M4A/dll) ke WAV Mono 16kHz menggunakan FFmpeg."""
    log_info(f"Mengonversi {input_path.name} ke WAV Mono 16kHz (format yang dibutuhkan SciPy/Whisper)...")
    
    cmd = [
        "ffmpeg", "-i", str(input_path), 
        "-ac", "1", "-ar", "16000", "-y", 
        str(output_path)
    ]
    log_info(f"Menjalankan FFmpeg: {' '.join(cmd)}")
    try:
        # Menampilkan output stderr dari FFmpeg jika terjadi error
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        log_success(f"Konversi awal selesai. Output FFmpeg: {result.stderr[:200]}...")
        return output_path
    except subprocess.CalledProcessError as e:
        log_error(f"FFmpeg GAGAL konversi ke WAV. Error: {e.stderr}", exit_app=True)
    except Exception as e:
        log_error(f"Gagal konversi ke WAV Mono 16kHz menggunakan FFmpeg: {e}", exit_app=True)

def read_and_prepare_audio(input_path: Path) -> Tuple[int, np.ndarray]:
    """Membaca file WAV yang sudah dikonversi dan mempersiapkannya menjadi array float64 untuk noisereduce."""
    log_info(f"Membaca file WAV: {input_path.name} (dengan SciPy)")
    try:
        rate, data = wavfile.read(input_path)
        log_info(f"File WAV berhasil dibaca. Rate: {rate}, Tipe Data: {data.dtype}, Shape: {data.shape}")
        
        # Data sudah Mono (ac=1) dari FFmpeg. Cukup konversi tipe data.
        if data.dtype.kind in ('i', 'u'):
            # Konversi ke float antara -1.0 dan 1.0
            data = data.astype(np.float64) / 32768.0
            log_info(f"Data dikonversi ke float64.")
        else:
            log_info("Data sudah dalam format float.")
            
        return rate, data
    except Exception as e:
        log_error(f"Gagal membaca file WAV yang telah dikonversi (SciPy Error): {e}", exit_app=True)

# -----------------------------------------------------
# TAHAP 2: PEMROSESAN AUDIO (DENOISE + NORMALISASI PYTHON)
# -----------------------------------------------------

def process_and_normalize_audio(input_data: np.ndarray, rate: int, output_base_path: Path) -> Path:
    """
    1. Melakukan Denoising (noisereduce, Python murni).
    2. Melakukan Normalisasi Puncak (NumPy, Python murni).
    """
    
    # 1. Denoising dengan noisereduce (PERBAIKAN API)
    log_info("1. Melakukan Denoising (noisereduce) pada seluruh file audio...")
    
    try:
        # Menggunakan argumen 'y' dan 'sr' sesuai dokumentasi
        denoised_data = nr.reduce_noise(
            y=input_data,
            sr=rate
        )
        log_success("Denoising selesai.")
    except Exception as e:
        log_error(f"Gagal menjalankan noisereduce: {e}", exit_app=True)


    # 2. Normalisasi Puncak Statis (NumPy) - Menstabilkan Volume
    log_info("2. Melakukan Normalisasi Puncak Statis (NumPy) pada seluruh file...")
    
    peak_value = np.max(np.abs(denoised_data))
    log_info(f"Nilai puncak audio (absolut) setelah denoising: {peak_value}")
    
    if peak_value > 0.001: # Menghindari pembagian dengan nol jika audio senyap
        # Target puncak di -1.0 dBFS (0.89) untuk memberi sedikit ruang/headroom
        scaling_factor = 0.89 / peak_value
        normalized_data = denoised_data * scaling_factor
        log_info(f"Volume disesuaikan dengan faktor: {scaling_factor:.2f}")
    else:
        normalized_data = denoised_data
        log_warn("Nilai puncak audio sangat rendah, tidak ada normalisasi yang dilakukan.")

    # 3. Menyimpan Hasil
    final_processed_path = output_base_path.with_suffix(".wav")
    log_info(f"3. Menyimpan hasil denoising dan normalisasi ke {final_processed_path.name}...")
    
    # Konversi data kembali ke format integer 16-bit untuk disimpan
    output_int16 = (normalized_data * 32767).astype(np.int16)
    
    try:
        wavfile.write(final_processed_path, rate, output_int16)
        log_success(f"Pemrosesan Selesai dan disimpan: {final_processed_path.name}")
        return final_processed_path
    except Exception as e:
        log_error(f"Gagal menyimpan file audio akhir (SciPy Error): {e}", exit_app=True)

# -----------------------------------------------------
# TAHAP 3: FUNGSI TRANSKRIPSI (Tidak Berubah)
# -----------------------------------------------------

def transcribe_single_audio(audio_path, model_path, whisper_cli_path):
    """Mentranskripsi seluruh file audio tunggal menggunakan whisper.cpp CLI."""
    os.makedirs("transcripts", exist_ok=True)
    log_info("Memastikan folder 'transcripts' ada.")

    final_txt = Path("transcripts/transcript.txt")
    final_srt = Path("transcripts/transcript.srt")
    
    try:
        final_txt.write_text("", encoding="utf-8")
        final_srt.write_text("", encoding="utf-8")
        log_info(f"File output '{final_txt}' dan '{final_srt}' berhasil dibuat/dikosongkan.")
    except IOError as e:
        log_error(f"Gagal membuat file transkrip akhir di ./transcripts/. Periksa izin folder. Error: {e}", exit_app=True)

    log_info(f"Mentranskripsi file tunggal: {audio_path.name}")
    
    # Gunakan nama file input yang diproses sebagai dasar output
    output_base_path_temp = Path(audio_path.stem)
    log_info(f"Nama file output sementara: {output_base_path_temp}")
    
    cmd = [
        str(whisper_cli_path),
        "-m", str(model_path),
        "-f", str(audio_path),
        "--temperature", "0.6",
        "-of", str(output_base_path_temp), 
        "-otxt",
        "-osrt",
        "-l", "id", # Menggunakan Bahasa Indonesia
        "-pp" # Mengaktifkan post-processor (misal: kapitalisasi)
    ]
    
    log_info(f"Menjalankan perintah whisper-cli: {' '.join(cmd)}")
    
    try:
        # Menampilkan output stderr dari whisper-cli jika terjadi error
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        print() # Newline setelah output whisper
        if result.stdout:
            log_info(f"Output whisper-cli (stdout): {result.stdout}")
        if result.stderr:
            log_info(f"Output whisper-cli (stderr): {result.stderr}")
            
    except subprocess.CalledProcessError as e:
        print() # Newline setelah output whisper
        log_error(f"whisper-cli GAGAL (return code: {e.returncode}).", exit_app=False)
        log_error(f"whisper-cli STDOUT: {e.stdout}")
        log_error(f"whisper-cli STDERR: {e.stderr}")
        log_error("Gagal melakukan transkripsi. Proses dihentikan.", exit_app=True)
    except Exception as e:
        log_error(f"Error tak terduga saat menjalankan whisper-cli pada {audio_path.name}: {e}", exit_app=True)

    # --- Pindahkan dan Bersihkan TXT/SRT ---
    log_info("Memindahkan file output sementara ke folder 'transcripts'...")
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
# FUNGSI MAIN BARU (Alur Hibrida)
# -----------------------------------------------------

def main():
    print("--- [DEBUG] MEMULAI FUNGSI main() ---")
    if len(sys.argv) < 3:
        print("Usage: python3 transcriptor_cpp.py <url_or_file> <model>")
        print(f"Model: {', '.join(VALID_MODELS)}")
        sys.exit(1)

    # Path file sementara
    original_audio_path = Path("original_audio_download") # Nama unik
    converted_wav_path = Path("processed_audio.wav") # Nama unik
    
    # Menggunakan try...except di dalam main()
    try:
        print("--- [DEBUG] TAHAP 0: Inisialisasi ---")
        whisper_cli_path = check_dependencies()
        source = sys.argv[1]
        model_name = sys.argv[2]
        log_info(f"Input Source: {source}")
        log_info(f"Input Model: {model_name}")
        
        model_path = ensure_model_exists(model_name)
        
        # Penentuan Path Audio Input
        if os.path.exists(source):
            log_info(f"Menggunakan file lokal: {source}")
            audio_path_to_process = Path(source)
            # Salin ke nama yang konsisten untuk pembersihan
            original_audio_path = audio_path_to_process
        else:
            log_info("Input adalah URL, mengunduh...")
            download_audio(source, original_audio_path)
            audio_path_to_process = original_audio_path

        print("--- [DEBUG] TAHAP 1: KONVERSI AWAL (FFMPEG - FIX BUG MP3) ---")
        converted_wav_path = convert_to_wav(audio_path_to_process, converted_wav_path)
        
        print("--- [DEBUG] TAHAP 2: MEMBACA DAN MEMPERSIAPKAN (PYTHON MURNI) ---")
        rate, data = read_and_prepare_audio(converted_wav_path)
        
        print("--- [DEBUG] TAHAP 3: DENOISING & NORMALISASI (PYTHON MURNI) ---")
        final_processed_audio_path = process_and_normalize_audio(data, rate, converted_wav_path)

        print("--- [DEBUG] TAHAP 4: TRANSKRIPSI ---")
        transcribe_single_audio(final_processed_audio_path, model_path, whisper_cli_path)

    except Exception as e:
        # Blok except ini sekarang berada di dalam main(), sebelum finally
        log_error(f"Terjadi error fatal yang tidak terduga: {e}", exit_app=False)
        print("------ STACK TRACE LENGKAP ------")
        traceback.print_exc()
        print("---------------------------------")
        sys.exit(1) # Keluar secara eksplisit jika terjadi error
        
    finally:
        # TAHAP 5: PEMBERSIHAN
        print("--- [DEBUG] TAHAP 5: Memulai pembersihan file sementara... ---")
        
        # Hapus file WAV yang diproses (converted_wav_path DAN final_processed_audio_path menunjuk ke file yang sama)
        if 'final_processed_audio_path' in locals() and final_processed_audio_path.exists():
            try:
                final_processed_audio_path.unlink()
                log_info(f"Berhasil menghapus: {final_processed_audio_path}")
            except Exception as e:
                log_warn(f"Gagal menghapus {final_processed_audio_path}: {e}")
        
        # Hapus file audio asli yang diunduh (jika diunduh)
        if 'original_audio_path' in locals() and 'source' in locals():
             if (not os.path.exists(source)) and original_audio_path.exists():
                try:
                    original_audio_path.unlink()
                    log_info(f"Berhasil menghapus: {original_audio_path}")
                except Exception as e:
                    log_warn(f"Gagal menghapus {original_audio_path}: {e}")
        
        log_success("====== PROSES SELESAI TOTAL ======")
        log_info("Output akhir ada di folder ./transcripts/")

# -----------------------------------------------------
# BLOK EKSEKUSI UTAMA (GLOBAL)
# -----------------------------------------------------
if __name__ == "__main__":
    print("--- [DEBUG] SCRIPT DIMULAI (if __name__ == '__main__') ---")
    try:
        main()
    except Exception as e:
        # Penjaga terakhir jika 'main()' gagal sebelum 'try' di dalamnya
        print(f"[✗✗✗] ERROR GLOBAL TIDAK TERDUGA: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)