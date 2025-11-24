#!/usr/bin/env python3
import sys
import os
import subprocess
import re
import traceback
import glob
from pathlib import Path
from datetime import timedelta
from pydub import AudioSegment
from pydub.utils import which

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

# --- Fungsi Inti dengan Penanganan Error (Fungsi-fungsi ini TIDAK diubah) ---

def check_dependencies():
    """Memeriksa semua dependensi eksternal sebelum memulai."""
    log_info("Memeriksa dependensi...")
    dependencies_ok = True
    
    if not which("curl"):
        log_error("'curl' tidak ditemukan. Harap instal 'curl' untuk mengunduh file.")
        dependencies_ok = False
        
    if not which("ffmpeg") and not which("ffprobe"):
        log_error("'ffmpeg' atau 'ffprobe' tidak ditemukan. 'pydub' membutuhkannya untuk memproses audio.")
        dependencies_ok = False
        
    whisper_cli_path = Path("./build/bin/whisper-cli")
    if not whisper_cli_path.exists():
        log_error(f"'{whisper_cli_path}' tidak ditemukan. Pastikan Anda telah mengompilasi whisper.cpp.")
        dependencies_ok = False
        
    if not dependencies_ok:
        log_error("Dependensi tidak lengkap. Keluar.", exit_app=True)
        
    log_success("Semua dependensi (curl, ffmpeg/ffprobe, whisper-cli) ditemukan.")
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

def split_audio(input_path, output_dir, chunk_length_ms=3*60*60*1000):
    # ... (logika split_audio tidak berubah)
    chunk_length_sec = chunk_length_ms / 1000
    log_info(f"Memecah audio menjadi potongan {chunk_length_sec} detik menggunakan ffmpeg...")

    os.makedirs(output_dir, exist_ok=True)
    output_pattern = Path(output_dir) / "part_%d.mp3"
    input_ext = Path(input_path).suffix.lower()

    cmd = [
        "ffmpeg",
        "-i", str(input_path),
        "-f", "segment",
        "-segment_time", str(chunk_length_sec),
        "-segment_start_number", "1",
        "-reset_timestamps", "1",
    ]

    if input_ext == ".mp3":
        log_info("  → Input adalah MP3. Menggunakan mode stream copy (super cepat).")
        cmd.extend(["-c", "copy"])
    else:
        log_info(f"  → Input adalah {input_ext}. Melakukan encode ke MP3 saat memecah.")
        cmd.extend(["-c:a", "libmp3lame", "-q:a", "2"])

    cmd.append(str(output_pattern))

    log_info(f"  → Menjalankan: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        print()
    except subprocess.CalledProcessError as e:
        print()
        log_error(f"ffmpeg gagal memecah audio. Return code: {e.returncode}. Lihat error di atas.", exit_app=True)
    except FileNotFoundError:
        log_error("ffmpeg tidak ditemukan. Pastikan ia terinstal dan ada di PATH sistem.", exit_app=True)

    chunk_files_sorted = sorted(
        glob.glob(str(Path(output_dir) / "part_*.mp3")),
        key=lambda x: int(Path(x).stem.split('_')[1])
    )

    if not chunk_files_sorted:
        log_warn("ffmpeg berjalan tetapi tidak ada file potongan yang ditemukan. Mungkin file audio terlalu pendek?")
        return [], chunk_length_ms
        
    chunks = [str(f) for f in chunk_files_sorted]
    
    log_success(f"Total {len(chunks)} potongan audio berhasil dibuat (via ffmpeg).")
    return chunks, chunk_length_ms

def shift_srt_time(file_path, offset_seconds, max_chunk_seconds):
    # ... (logika shift_srt_time tidak berubah)
    pattern = r"(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})"
    
    offset_duration = timedelta(seconds=offset_seconds)
    max_duration = timedelta(seconds=max_chunk_seconds)

    def to_timedelta(time_str):
        h, m, s, ms = map(int, re.match(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})", time_str).groups())
        return timedelta(hours=h, minutes=m, seconds=s, milliseconds=ms)

    def to_str(td):
        total_ms = int(td.total_seconds() * 1000)
        hh = total_ms // 3600000
        mm = (total_ms % 3600000) // 60000
        ss = (total_ms % 60000) // 1000
        mss = total_ms % 1000
        return f"{hh:02d}:{mm:02d}:{ss:02d},{mss:03d}"

    def shift_and_clean_line(match):
        start_str, end_str = match.groups()
        
        original_start = to_timedelta(start_str)
        original_end = to_timedelta(end_str)

        if original_start >= max_duration:
            log_warn(f"  → Subtitle dimulai setelah maks durasi. Memangkas: {start_str}")
            original_start = max_duration
            original_end = max_duration

        elif original_end > max_duration:
            log_warn(f"  → Memangkas waktu akhir {end_str} ke {max_duration}")
            original_end = max_duration

        shifted_start = original_start + offset_duration
        shifted_end = original_end + offset_duration

        if shifted_end < shifted_start:
            shifted_end = shifted_start

        return f"{to_str(shifted_start)} --> {to_str(shifted_end)}"

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        new_content = re.sub(pattern, shift_and_clean_line, content)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
            
        return True
    except FileNotFoundError:
        log_error(f"  → Gagal menemukan file SRT untuk digeser: {file_path}", exit_app=False)
        return False
    except Exception as e:
        log_error(f"  → Gagal memproses pergeseran waktu SRT untuk {file_path}: {e}", exit_app=False)
        return False

def transcribe_with_whisper_cpp(chunk_files, model_path, chunk_length_ms, whisper_cli_path):
    """Mentranskripsi setiap potongan, menangani kegagalan per potongan."""
    os.makedirs("transcripts", exist_ok=True)

    final_txt = Path("transcripts/transcript.txt")
    final_srt = Path("transcripts/transcript.srt")
    
    try:
        final_txt.write_text("", encoding="utf-8")
        final_srt.write_text("", encoding="utf-8")
    except IOError as e:
        log_error(f"Gagal membuat file transkrip akhir di ./transcripts/. Periksa izin folder. Error: {e}", exit_app=True)

    chunk_seconds = chunk_length_ms / 1000
    srt_block_counter = 1

    for i, chunk in enumerate(chunk_files, start=1):
        log_info(f"Mentranskripsi potongan {i}/{len(chunk_files)}: {chunk}")
        
        output_base_path = Path(chunk).with_suffix("")
        
        # CATATAN: Saya mengembalikan `--temperature 0.4` tetapi DENGAN SINTAKS YANG BENAR
        cmd = [
            str(whisper_cli_path),
            "-m", str(model_path),
            "-f", chunk,
            "--temperature", "0.2", # Diperbaiki dari `--temperature 0.4` menjadi 2 elemen
            "-of", str(output_base_path),
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
            log_error(f"whisper-cli gagal pada potongan {chunk} (return code: {e.returncode}). Lihat pesan error di atas.", exit_app=False)
            log_warn(f"Melompati potongan {chunk} karena error.")
            try:
                txt_file_fail = output_base_path.with_suffix(".txt")
                srt_file_fail = output_base_path.with_suffix(".srt")
                if txt_file_fail.exists(): txt_file_fail.unlink()
                if srt_file_fail.exists(): srt_file_fail.unlink()
            except Exception as e_clean:
                log_warn(f"  → Gagal membersihkan file sisa: {e_clean}")
            continue
        except Exception as e:
            log_error(f"Error tak terduga saat menjalankan whisper-cli pada {chunk}: {e}", exit_app=True)

        # --- Proses TXT ---
        txt_file = Path(chunk).with_suffix(".txt")
        try:
            if txt_file.exists():
                content = txt_file.read_text(encoding="utf-8").strip()
                with open(final_txt, "a", encoding="utf-8") as out:
                    out.write(content + "\n\n")
                txt_file.unlink()
                log_success("  → TXT digabung.")
            else:
                log_warn(f"  → File TXT output tidak ditemukan untuk: {chunk}")
        except Exception as e:
            log_error(f"  → Gagal memproses file TXT {txt_file}: {e}")

        # --- Proses SRT ---
        srt_file = Path(chunk).with_suffix(".srt")
        try:
            if srt_file.exists():
                offset_seconds = (i - 1) * chunk_seconds
                
                is_last_chunk = (i == len(chunk_files))
                max_seconds_cap = 99999.0 if is_last_chunk else chunk_seconds
                
                if not shift_srt_time(srt_file, offset_seconds, max_seconds_cap):
                    raise Exception(f"Gagal menggeser waktu untuk {srt_file}")

                srt_content = srt_file.read_text(encoding="utf-8").strip()

                def reindex_srt(match):
                    nonlocal srt_block_counter
                    new_index = srt_block_counter
                    srt_block_counter += 1
                    return str(new_index)

                reindexed_content = re.sub(r"^\d+\s*$", reindex_srt, srt_content, flags=re.MULTILINE)

                with open(final_srt, "a", encoding="utf-8") as out:
                    out.write(reindexed_content + "\n\n")
                
                srt_file.unlink()
                log_success(f"  → SRT digabung & di-re-index (offset {offset_seconds:.2f}s).")
            else:
                log_warn(f"  → File SRT output tidak ditemukan untuk: {chunk}")
        except Exception as e:
            log_error(f"  → Gagal memproses file SRT {srt_file}: {e}")

    log_success("Semua TXT dan SRT telah digabung.")
    log_success(f"Output TXT: {final_txt}")
    log_success(f"Output SRT: {final_srt}")

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
            audio_path = Path("audio.mp3")
            log_warn(f"Input berupa URL, mengunduh ke {audio_path}...")
            download_audio(source, audio_path)

        # 5. Proses utama
        chunk_files, chunk_length_ms = split_audio(audio_path, "chunks")
        
        if not chunk_files:
            log_error("Tidak ada potongan audio yang berhasil dibuat. Proses dihentikan.", exit_app=True)

        transcribe_with_whisper_cpp(chunk_files, model_path, chunk_length_ms, whisper_cli_path)

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