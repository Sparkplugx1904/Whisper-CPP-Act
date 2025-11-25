#!/usr/bin/env python3
import sys
import os
import subprocess
import re
import traceback
from pathlib import Path
import numpy as np
from scipy.io import wavfile
import noisereduce as nr # <--- PUSTAKA BARU UNTUK DENOISING
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
        
    # 2. Cek 'ffmpeg' (MASIH DIPERLUKAN untuk konversi awal, final, dan LOUDNORM/stabilisasi volume)
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
    
# --- Fungsi lainnya (download_file, ensure_model_exists, download_audio) tetap sama ---
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
# FUNGSI BARU: PEMROSESAN AUDIO GABUNGAN (DENOISE + STABILISASI)
# -----------------------------------------------------

def convert_to_wav(input_path: Path, output_path: Path) -> Tuple[int, np.ndarray]:
    """Mengonversi file audio ke WAV Mono 16kHz menggunakan FFmpeg."""
    log_info(f"Mengonversi {input_path.name} ke WAV Mono 16kHz...")
    
    # Pastikan format WAV mono 16kHz untuk NumPy/SciPy/Whisper
    cmd = [
        "ffmpeg", "-i", str(input_path), 
        "-ac", "1", "-ar", "16000", "-y", 
        str(output_path)
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        rate, data = wavfile.read(output_path)
        log_success("Konversi awal selesai.")
        # Konversi data ke float64, diperlukan untuk noisereduce
        data = data.astype(np.float64) / 32768.0
        return rate, data
    except Exception as e:
        log_error(f"Gagal konversi ke WAV Mono 16kHz: {e}", exit_app=True)

def process_and_normalize_audio(input_data: np.ndarray, rate: int, output_base_path: Path) -> Path:
    """
    1. Melakukan Denoising (noisereduce, Python murni).
    2. Membagi menjadi segmen 1 menit.
    3. Normalisasi/Stabilisasi Volume (loudnorm, FFmpeg) pada setiap segmen.
    4. Menggabungkan segmen menjadi satu file WAV akhir.
    """
    
    # 1. Denoising dengan noisereduce (Python Murni)
    log_info("1. Melakukan Denoising pada seluruh file audio (noisereduce)...")
    # Tentukan sampel noise (Asumsi 1 detik pertama adalah noise murni).
    # Ini mungkin tidak optimal dan harus disesuaikan untuk audio nyata.
    noise_len_sec = min(1.0, len(input_data) / rate) 
    noise_sample = input_data[:int(rate * noise_len_sec)]
    
    denoised_data = nr.reduce_noise(
        audio_clip=input_data, 
        noise_clip=noise_sample, 
        verbose=False,
        sr=rate
    )
    log_success("Denoising selesai.")

    # 2. Pembagian Segmen (1 Menit) dan Normalisasi Loudnorm (FFmpeg)
    log_info("2. Membagi audio, Normalisasi Loudnorm per segmen, dan Menyimpan...")
    segment_duration_sec = 60
    segment_len = rate * segment_duration_sec
    num_segments = (len(denoised_data) + segment_len - 1) // segment_len
    
    segment_paths: List[Path] = []
    
    # Pastikan direktori sementara ada
    temp_dir = Path("./temp_segments")
    os.makedirs(temp_dir, exist_ok=True)

    for i in range(num_segments):
        start = i * segment_len
        end = min((i + 1) * segment_len, len(denoised_data))
        segment = denoised_data[start:end]
        
        temp_wav_path = temp_dir / f"seg_{i}_denoised.wav"
        temp_loudnorm_path = temp_dir / f"seg_{i}_loudnorm.wav"
        
        # Tulis segmen yang telah di-denoising ke file sementara untuk Loudnorm FFmpeg
        wavfile.write(temp_wav_path, rate, (segment * 32767).astype(np.int16))
        
        # Loudnorm (Stabilisasi Volume) menggunakan FFmpeg per segmen
        # I=-23 LUFS adalah target standar
        cmd_loudnorm = [
            "ffmpeg", "-i", str(temp_wav_path), "-y",
            "-af", "loudnorm=I=-23:LRA=7:tp=-2:print_format=json",
            str(temp_loudnorm_path)
        ]
        
        try:
            subprocess.run(cmd_loudnorm, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            segment_paths.append(temp_loudnorm_path)
            log_info(f"Segment {i+1}/{num_segments} diproses dan distabilkan.")
        except subprocess.CalledProcessError as e:
            log_error(f"FFmpeg Loudnorm gagal pada segmen {i+1}. Code: {e.returncode}", exit_app=True)

    # 3. Penggabungan Segmen
    log_info("3. Menggabungkan semua segmen yang distabilkan...")
    concat_list_path = temp_dir / "concat_list.txt"
    with open(concat_list_path, 'w') as f:
        for p in segment_paths:
            f.write(f"file '{p.name}'\n")

    final_processed_path = output_base_path.with_suffix(".wav")
    
    # Perintah FFmpeg untuk menggabungkan file WAV yang telah diproses
    cmd_concat = [
        "ffmpeg", "-f", "concat", "-safe", "0", 
        "-i", str(concat_list_path), 
        "-ac", "1", "-ar", "16000", "-y", 
        str(final_processed_path)
    ]

    try:
        subprocess.run(cmd_concat, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log_success(f"Penggabungan dan Pemrosesan Akhir Selesai: {final_processed_path.name}")
        return final_processed_path
    except subprocess.CalledProcessError as e:
        log_error(f"FFmpeg Gagal Menggabungkan Segmen. Code: {e.returncode}", exit_app=True)

def cleanup_segments(temp_dir: Path):
    """Menghapus semua file dan folder sementara."""
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
    
    # ... (Kode Transkripsi Tidak Berubah) ...
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

    # --- Pindahkan dan Bersihkan TXT/SRT (Kode Tidak Berubah) ---
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

    temp_segment_dir = Path("./temp_segments")

    try:
        whisper_cli_path = check_dependencies()
        source = sys.argv[1]
        model_name = sys.argv[2]
        model_path = ensure_model_exists(model_name)

        original_audio_path = Path("original_audio.mp3") 
        
        if os.path.exists(source):
            audio_path_to_process = Path(source)
        else:
            download_audio(source, original_audio_path)
            audio_path_to_process = original_audio_path

        # TAHAP 1: KONVERSI AWAL
        temp_wav_path = Path(f"temp_{audio_path_to_process.stem}.wav")
        rate, data = convert_to_wav(audio_path_to_process, temp_wav_path)
        
        # TAHAP 2: DENOISING & LOUDNORM PER SEGMEN (Memproduksi file akhir)
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
        cleanup_segments(temp_segment_dir)
        if 'temp_wav_path' in locals() and temp_wav_path.exists():
            temp_wav_path.unlink()
        if 'original_audio_path' in locals() and original_audio_path.exists():
            original_audio_path.unlink()
        
        log_success("====== PROSES SELESAI TOTAL ======")
        log_info("Output akhir ada di folder ./transcripts/")

if __name__ == "__main__":
    main()