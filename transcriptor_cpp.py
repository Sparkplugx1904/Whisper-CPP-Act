#!/usr/bin/env python3
import sys
import os
import subprocess
import re
import traceback
from pathlib import Path

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
        
    if not dependencies_ok:
        log_error("Dependensi tidak lengkap. Keluar.", exit_app=True)
        
    log_success("Semua dependensi (curl, whisper-cli) ditemukan.")
    return whisper_cli_path
    
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
    # VALIDASI MODEL (TIDAK BERUBAH)
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

# Hapus fungsi split_audio
# Hapus fungsi shift_srt_time

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
    
    # Nama file output sementara di direktori saat ini
    output_base_path_temp = Path(audio_path.stem)
    
    # CATATAN: Menggunakan --temperature 0.6 seperti sebelumnya
    cmd = [
        str(whisper_cli_path),
        "-m", str(model_path),
        "-f", str(audio_path),
        "--temperature", "0.2",
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

    # --- Pindahkan dan Bersihkan TXT ---
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

    # --- Pindahkan dan Bersihkan SRT ---
    temp_srt_file = output_base_path_temp.with_suffix(".srt")
    try:
        if temp_srt_file.exists():
            # Cukup salin kontennya, tidak perlu ada re-indexing atau shifting karena ini file tunggal
            content = temp_srt_file.read_text(encoding="utf-8").strip()
            final_srt.write_text(content, encoding="utf-8")
            temp_srt_file.unlink()
            log_success(f"SRT berhasil disimpan ke {final_srt}.")
        else:
            log_warn(f"File SRT output tidak ditemukan: {temp_srt_file}")
    except Exception as e:
        log_error(f"Gagal memproses file SRT {temp_srt_file}: {e}")

    log_success("Transkripsi file tunggal selesai.")

## 🚀 Fungsi Main Baru (Menggunakan Argumen Posisi)
def main():
    # Cek apakah jumlah argumen cukup (nama skrip + source + model = 3)
    if len(sys.argv) < 3:
        print("Usage: python3 transcriptor_cpp.py <url_or_file> <model>")
        print(f"Model: {', '.join(VALID_MODELS)}")
        sys.exit(1)

    try:
        # 1. Cek dependensi KETAT
        whisper_cli_path = check_dependencies()

        # 2. Ambil argumen posisi (source = index 1, model = index 2)
        source = sys.argv[1]
        model_name = sys.argv[2]
        
        # 3. Validasi model dan pastikan file ada/diunduh
        model_path = ensure_model_exists(model_name)

        # 4. Tentukan & unduh audio
        if os.path.exists(source):
            audio_path = Path(source)
            log_success(f"Menggunakan file lokal: {audio_path}")
        else:
            # Karena pydub sudah tidak diimpor, kita harus pastikan nama file output aman
            # Jika user memberikan URL, kita unduh ke 'audio.mp3'
            audio_path = Path("audio.mp3") 
            log_warn(f"Input berupa URL, mengunduh ke {audio_path}...")
            download_audio(source, audio_path)

        # 5. Proses utama (Hanya transkripsi file tunggal)
        # Hapus panggilan split_audio
        transcribe_single_audio(audio_path, model_path, whisper_cli_path)

        log_success("====== PROSES SELESAI ======")
        log_info("Output akhir ada di folder ./transcripts/")

    except Exception as e:
        log_error(f"Terjadi error fatal yang tidak terduga: {e}", exit_app=False)
        print("------ STACK TRACE LENGKAP ------")
        traceback.print_exc()
        print("---------------------------------")
        sys.exit(1)

if __name__ == "__main__":
    main()