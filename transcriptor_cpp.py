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
        
    # 2. Cek 'ffmpeg' (BARU: Diperlukan untuk denoising)
    if subprocess.run(['which', 'ffmpeg'], capture_output=True).returncode != 0:
        log_error("'ffmpeg' tidak ditemukan. Harap instal 'ffmpeg' untuk pembersihan audio.")
        dependencies_ok = False
        
    # 3. Cek 'whisper-cli'
    whisper_cli_path = Path("./build/bin/whisper-cli")
    if not whisper_cli_path.exists():
        log_error(f"'{whisper_cli_path}' tidak ditemukan. Pastikan Anda telah mengompilasi whisper.cpp.")
        dependencies_ok = False
        
    if not dependencies_ok:
        log_error("Dependensi tidak lengkap. Keluar.", exit_app=True)
        
    log_success("Semua dependensi (curl, ffmpeg, whisper-cli) ditemukan.")
    return whisper_cli_path
    
def download_file(url, dest):
    # FUNGSI download_file (TIDAK BERUBAH)
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
    # FUNGSI ensure_model_exists (TIDAK BERUBAH)
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
    # FUNGSI download_audio (TIDAK BERUBAH)
    """Wrapper untuk mengunduh file audio."""
    log_info(f"Mengunduh audio dari: {url}")
    if not download_file(url, output_path):
        log_error("Gagal mengunduh audio. Membatalkan.", exit_app=True)
    log_success(f"Audio berhasil diunduh ke {output_path}")

# --- FUNGSI BARU UNTUK PENYEMPURNAAN AUDIO ---
def denoise_audio(input_path, output_path):
    """
    Membersihkan noise pada file audio menggunakan FFmpeg (filter afftdn).
    Mengubah audio ke format WAV mono 16kHz (format yang disukai Whisper).
    """
    log_info(f"Memulai penyempurnaan audio (denoising) pada: {input_path.name}")
    
    # Perintah FFmpeg:
    # -i: Input
    # -af 'afftdn': Filter Advanced Frequency-Domain Noise Reduction
    # -ac 1: Mono (Saluran tunggal)
    # -ar 16000: Sample rate 16kHz
    # -y: Timpa file output jika sudah ada
    cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-af", "afftdn=nf=4:tn=3", # Nilai filter yang umum digunakan
        "-ac", "1",
        "-ar", "16000",
        "-y", 
        str(output_path)
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log_success(f"Penyempurnaan audio selesai. Output: {output_path.name}")
        return True
    except subprocess.CalledProcessError as e:
        log_error(f"FFmpeg gagal saat denoising (return code: {e.returncode}).", exit_app=False)
        log_error("Periksa apakah FFmpeg terinstal dengan benar dan file input valid.")
        return False
    except Exception as e:
        log_error(f"Error tak terduga saat menjalankan FFmpeg: {e}")
        return False
# -----------------------------------------------

def transcribe_single_audio(audio_path, model_path, whisper_cli_path):
    # FUNGSI transcribe_single_audio (TIDAK BERUBAH)
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
            content = temp_srt_file.read_text(encoding="utf-8").strip()
            final_srt.write_text(content, encoding="utf-8")
            temp_srt_file.unlink()
            log_success(f"SRT berhasil disimpan ke {final_srt}.")
        else:
            log_warn(f"File SRT output tidak ditemukan: {temp_srt_file}")
    except Exception as e:
        log_error(f"Gagal memproses file SRT {temp_srt_file}: {e}")

    log_success("Transkripsi file tunggal selesai.")

## 🚀 Fungsi Main yang Dimodifikasi
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
        original_audio_path = Path("original_audio.mp3") # Nama default untuk audio yang diunduh/diproses
        
        if os.path.exists(source):
            # Jika input adalah file lokal, gunakan Path(source) sebagai audio_path sementara
            log_success(f"Menggunakan file lokal: {source}")
            audio_path_to_process = Path(source)
        else:
            # Jika input adalah URL, unduh ke 'original_audio.mp3'
            log_warn(f"Input berupa URL, mengunduh ke {original_audio_path}...")
            download_audio(source, original_audio_path)
            audio_path_to_process = original_audio_path

        # 5. BARU: Penyempurnaan Audio (Denoising)
        denoised_audio_path = Path(f"denoised_{audio_path_to_process.stem}.wav")
        if not denoise_audio(audio_path_to_process, denoised_audio_path):
            log_error("Gagal melakukan penyempurnaan audio. Keluar.", exit_app=True)

        # 6. Proses utama: Transkripsi file yang telah dibersihkan
        transcribe_single_audio(denoised_audio_path, model_path, whisper_cli_path)

        # 7. Pembersihan file audio sementara (Opsional)
        if original_audio_path.exists():
            original_audio_path.unlink()
        if denoised_audio_path.exists():
            denoised_audio_path.unlink()
        if os.path.exists(source) and str(Path(source)) == str(audio_path_to_process):
             log_info("File audio lokal asli TIDAK dihapus.")
        
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