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
from typing import List, Tuple, Optional
print("--- [DEBUG] Pustaka 'typing' berhasil diimpor.")
import argparse
print("--- [DEBUG] Pustaka 'argparse' berhasil diimpor.")

try:
    # Hanya impor pustaka yang tersisa
    import numpy as np
    print("--- [DEBUG] Pustaka 'numpy' berhasil diimpor.")
except ImportError as e:
    print(f"[✗✗✗] FATAL: GAGAL MENGIMPOR PUSTAKA PENTING: {e}", file=sys.stderr)
    print("[✗✗✗] FATAL: Pastikan Anda telah menjalankan 'pip install -r requirements.txt' (minimal numpy)", file=sys.stderr)
    sys.exit(1)

print("--- [DEBUG] SEMUA PUSTAKA INTI BERHASIL DIIMPOR ---")

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
DEFAULT_MODEL_NAME = "medium"

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
        
    # 2. Cek 'whisper-cli'
    whisper_cli_path = Path("./build/bin/whisper-cli")
    if not whisper_cli_path.exists():
        log_error(f"'{whisper_cli_path}' tidak ditemukan. Pastikan Anda telah mengompilasi whisper.cpp.")
        dependencies_ok = False
    else:
        log_info(f"Dependensi 'whisper-cli' ditemukan di {whisper_cli_path}.")
        
    if not dependencies_ok:
        log_error("Dependensi tidak lengkap. Keluar.", exit_app=True)
        
    log_success("Semua dependensi (curl, whisper-cli) ditemukan. (Catatan: whisper.cpp harus dikompilasi dengan dukungan FFmpeg untuk audio non-WAV)")
    return whisper_cli_path
    
def download_file(url: str, dest: Path) -> bool:
    """Mengunduh file menggunakan curl dengan penanganan error yang kuat."""
    log_info(f"Mengunduh: {url} → {dest}")
    os.makedirs(dest.parent, exist_ok=True) # Pastikan folder tujuan ada
    try:
        subprocess.run(
            ["curl", "-f", "-L", "-o", str(dest), "-m", "600", url], # Batas waktu 10 menit untuk model besar
            check=True
        )
        print() # Newline setelah output curl
        log_success(f"Unduhan selesai: {dest}")
        return True
    except subprocess.CalledProcessError as e:
        print() # Newline setelah output curl
        log_error(f"Gagal mengunduh file (curl return code: {e.returncode}). URL: {url}", exit_app=False)
        if dest.exists():
            dest.unlink() # Hapus file yang rusak
        return False
    except Exception as e:
        log_error(f"Terjadi error tak terduga saat mengunduh: {e}", exit_app=False)
        return False

def ensure_model_exists(model_name: str, custom_model_url: Optional[str]) -> Path:
    """Memastikan model ada, memvalidasi nama, dan mengunduh jika perlu (standar atau kustom)."""
    
    if custom_model_url:
        # --- LOGIKA MODEL KUSTOM ---
        model_filename = Path(custom_model_url).name
        if not model_filename or '.' not in model_filename:
            log_error("URL model kustom tidak valid atau tidak memiliki nama file yang jelas.", exit_app=True)
            
        model_path = Path(f"./models/{model_filename}")
        log_info(f"Memeriksa keberadaan model kustom: {model_path}")
        
        if model_path.exists():
            log_success(f"Model kustom ditemukan: {model_path}")
            return model_path
        
        log_warn(f"Model kustom belum ada, mengunduh dari URL: {custom_model_url}")
        if not download_file(custom_model_url, model_path):
            log_error("Gagal mengunduh model kustom. Membatalkan.", exit_app=True)
        log_success(f"Model kustom berhasil diunduh.")
        return model_path
        
    else:
        # --- LOGIKA MODEL STANDAR ---
        log_info(f"Memeriksa keberadaan model standar: {model_name}")
        if model_name not in VALID_MODELS:
            log_error(f"Nama model standar tidak valid: '{model_name}'. Pilihan: {', '.join(VALID_MODELS)}", exit_app=True)

        os.makedirs("models", exist_ok=True)
        model_path = Path(f"./models/ggml-{model_name}.bin")
        
        if model_path.exists():
            log_success(f"Model ditemukan: {model_path}")
            return model_path
        
        log_warn(f"Model standar '{model_name}' belum ada, mengunduh dari HuggingFace...")
        url = f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{model_name}.bin" 
        if not download_file(url, model_path):
            log_error("Gagal mengunduh model standar. Membatalkan.", exit_app=True)
            
        log_success(f"Model '{model_name}' berhasil diunduh.")
        return model_path

def download_audio(url: str, output_path: Path):
    """Wrapper untuk mengunduh file audio."""
    log_info(f"Mengunduh audio dari: {url}")
    if not download_file(url, output_path):
        log_error("Gagal mengunduh audio. Membatalkan.", exit_app=True)
    log_success(f"Audio berhasil diunduh ke {output_path}")

# -----------------------------------------------------
# FUNGSI KONVERSI FFmpeg (DIHAPUS SESUAI PERMINTAAN)
# -----------------------------------------------------


# -----------------------------------------------------
# TAHAP 2: FUNGSI TRANSKRIPSI
# -----------------------------------------------------

def transcribe_single_audio(audio_path: Path, model_path: Path, whisper_cli_path: Path):
    """
    Mentranskripsi seluruh file audio tunggal menggunakan whisper.cpp CLI.
    Sekarang, file audio akan diteruskan langsung ke whisper-cli.
    """
    os.makedirs("transcripts", exist_ok=True)
    log_info("Memastikan folder 'transcripts' ada.")

    # Tentukan nama file output akhir
    final_txt = Path("transcripts/transcript.txt")
    final_srt = Path("transcripts/transcript.srt")
    
    # Tentukan nama file output sementara yang akan dibuat oleh whisper-cli di CWD
    # Ini harus sesuai dengan stem dari audio_path yang diteruskan ke CLI.
    output_base_path_temp = audio_path.stem 
    temp_txt_file = Path(output_base_path_temp).with_suffix(".txt")
    temp_srt_file = Path(output_base_path_temp).with_suffix(".srt")

    # Bersihkan file output sebelumnya
    try:
        final_txt.write_text("", encoding="utf-8")
        final_srt.write_text("", encoding="utf-8")
        if temp_txt_file.exists(): temp_txt_file.unlink()
        if temp_srt_file.exists(): temp_srt_file.unlink()
        log_info("File output lama berhasil dibersihkan/dikosongkan.")
    except IOError as e:
        log_error(f"Gagal membersihkan/membuat file transkrip. Periksa izin folder. Error: {e}", exit_app=True)

    log_info(f"Mentranskripsi file tunggal: {audio_path.name}")
    
    cmd = [
        str(whisper_cli_path),
        "-m", str(model_path),
        "-f", str(audio_path),
        "--temperature", "0.6",
        # Gunakan output_base_path_temp sebagai dasar nama file output 
        "-of", str(output_base_path_temp), 
        "-otxt",
        "-osrt",
        "-l", "id", # Menggunakan Bahasa Indonesia
        "-pp" # Mengaktifkan post-processor (misal: kapitalisasi)
    ]
    
    log_info(f"Menjalankan perintah whisper-cli: {' '.join(cmd)}")
    
    try:
        # Menjalankan whisper-cli
        # PERUBAHAN UTAMA: capture_output=False agar output stdout/stderr langsung ke konsol.
        result = subprocess.run(cmd, check=True, capture_output=False)
        print() # Newline setelah output whisper
            
    except subprocess.CalledProcessError as e:
        print() # Newline setelah output whisper
        # Karena capture_output=False, e.stdout dan e.stderr mungkin kosong.
        # Output error harus sudah terlihat di konsol.
        log_error(f"whisper-cli GAGAL (return code: {e.returncode}).", exit_app=False)
        # log_error(f"whisper-cli STDOUT: {e.stdout}") # Dihapus
        # log_error(f"whisper-cli STDERR: {e.stderr}") # Dihapus
        log_error("Gagal melakukan transkripsi. Proses dihentikan.", exit_app=True)
    except Exception as e:
        log_error(f"Error tak terduga saat menjalankan whisper-cli pada {audio_path.name}: {e}", exit_app=True)

    # --- Pindahkan dan Bersihkan TXT/SRT ---
    log_info("Memindahkan file output sementara ke folder 'transcripts'...")
    
    # Pindah TXT
    try:
        if temp_txt_file.exists():
            content = temp_txt_file.read_text(encoding="utf-8").strip()
            final_txt.write_text(content, encoding="utf-8")
            temp_txt_file.unlink()
            log_success(f"TXT berhasil disimpan ke {final_txt}.")
        else:
            log_warn(f"File TXT output tidak ditemukan: {temp_txt_file}. Transkripsi mungkin gagal.")
    except Exception as e:
        log_error(f"Gagal memproses file TXT {temp_txt_file}: {e}")

    # Pindah SRT
    try:
        if temp_srt_file.exists():
            content = temp_srt_file.read_text(encoding="utf-8").strip()
            final_srt.write_text(content, encoding="utf-8")
            temp_srt_file.unlink()
            log_success(f"SRT berhasil disimpan ke {final_srt}.")
        else:
            log_warn(f"File SRT output tidak ditemukan: {temp_srt_file}. Transkripsi mungkin gagal.")
    except Exception as e:
        log_error(f"Gagal memproses file SRT {temp_srt_file}: {e}")

    log_success("Transkripsi file tunggal selesai.")

# -----------------------------------------------------
# FUNGSI MAIN BARU (Alur Sederhana)
# -----------------------------------------------------

def main():
    print("--- [DEBUG] MEMULAI FUNGSI main() ---")
    
    # --- 1. Parsing Argumen ---
    parser = argparse.ArgumentParser(
        description="Skrip transkripsi audio menggunakan whisper.cpp.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("source", help="URL audio atau path file lokal.")
    parser_model_group = parser.add_mutually_exclusive_group(required=False)
    parser_model_group.add_argument(
        "model", 
        nargs='?', 
        default=DEFAULT_MODEL_NAME,
        help=f"Nama model standar ({', '.join(VALID_MODELS)}). Default: {DEFAULT_MODEL_NAME}"
    )
    parser_model_group.add_argument(
        "-cm", "--custom-model", 
        help="URL lengkap ke file model GGML/GGUF kustom (.bin/.gguf). Jika digunakan, argumen 'model' standar diabaikan."
    )
    
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)
        
    args = parser.parse_args()

    # Path file sementara
    original_audio_path = Path("original_audio_download") 
    
    # Variabel untuk pembersihan
    audio_path_to_process = None
    is_source_url = not os.path.exists(args.source)
    
    try:
        print("--- [DEBUG] TAHAP 0: Inisialisasi ---")
        whisper_cli_path = check_dependencies()
        log_info(f"Input Source: {args.source}")
        log_info(f"Model Pilihan: {args.model} (Standar) atau {args.custom_model} (Kustom)")
        
        # 1. Pastikan Model Tersedia
        model_path = ensure_model_exists(args.model, args.custom_model)
        
        # 2. Penentuan Path Audio Input (Download jika URL)
        if is_source_url:
            log_info("Input adalah URL, mengunduh...")
            download_audio(args.source, original_audio_path)
            audio_path_to_process = original_audio_path
        else:
            log_info(f"Menggunakan file lokal: {args.source}")
            audio_path_to_process = Path(args.source)

        print("--- [DEBUG] TAHAP 1: PERSIAPAN AUDIO (KONVERSI FFmpeg DIHAPUS) ---")
        log_info("Langsung meneruskan file audio asli ke whisper-cli.")
        
        print("--- [DEBUG] TAHAP 3: TRANSKRIPSI ---")
        # Meneruskan audio_path_to_process (file asli/yang diunduh) langsung
        transcribe_single_audio(audio_path_to_process, model_path, whisper_cli_path)
        
    except Exception as e:
        log_error(f"Terjadi error fatal yang tidak terduga: {e}", exit_app=False)
        print("------ STACK TRACE LENGKAP ------")
        traceback.print_exc()
        print("---------------------------------")
        sys.exit(1)
        
    finally:
        # TAHAP 4: PEMBERSIHAN
        print("--- [DEBUG] TAHAP 4: Memulai pembersihan file sementara... ---")
        
        # Hapus file audio asli yang diunduh (jika source adalah URL)
        if is_source_url and original_audio_path.exists():
            try:
                original_audio_path.unlink()
                log_info(f"Berhasil menghapus: {original_audio_path}")
            except Exception as e:
                log_warn(f"Gagal menghapus {original_audio_path}: {e}")
        
        log_success("====== PROSES SELESAI TOTAL ======")
        log_info("Output akhir ada di folder ./transcripts/ (transcript.txt & transcript.srt)")

# -----------------------------------------------------
# BLOK EKSEKUSI UTAMA (GLOBAL)
# -----------------------------------------------------
if __name__ == "__main__":
    print("--- [DEBUG] SCRIPT DIMULAI (if __name__ == '__main__') ---")
    try:
        main()
    except Exception as e:
        print(f"[✗✗✗] ERROR GLOBAL TIDAK TERDUGA: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)