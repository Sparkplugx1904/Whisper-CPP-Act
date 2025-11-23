#!/usr/bin/env python3
import sys
import os
import subprocess
import re
import traceback
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

# --- Fungsi Inti dengan Penanganan Error ---

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
        # Menambahkan timeout 5 menit (300 detik)
        # stdout=DEVNULL dan stderr=PIPE dihapus agar progress bar curl terlihat
        subprocess.run(
            ["curl", "-L", "-o", str(dest), "-m", "300", url], 
            check=True
        )
        # Pesan sukses dipindahkan ke baris baru agar tidak bentrok dengan output curl
        print() # Tambahkan baris baru setelah progress bar curl
        log_success(f"Unduhan selesai: {dest}")
        return True
    except subprocess.CalledProcessError as e:
        # Karena stderr tidak lagi di-pipe, e.stderr akan None.
        # Pesan error curl sudah otomatis tercetak ke console.
        print() # Tambahkan baris baru jika curl error
        log_error(f"Gagal mengunduh file (curl return code: {e.returncode}). Lihat pesan error di atas.", exit_app=False)
        if dest.exists():
            dest.unlink() # Hapus file yang mungkin rusak/sebagian
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

def split_audio(input_path, output_dir, chunk_length_ms=15*1000):
    """Memecah audio dengan penanganan error untuk pydub."""
    log_info(f"Memecah audio menjadi potongan {chunk_length_ms // 60000} menit...")
    try:
        audio = AudioSegment.from_file(input_path)
    except FileNotFoundError:
        log_error(f"File audio tidak ditemukan di: {input_path}", exit_app=True)
    except Exception as e: # Menangkap error pydub/ffmpeg
        log_error(f"Gagal memuat file audio: {e}. Pastikan file tidak rusak dan format didukung.", exit_app=True)

    os.makedirs(output_dir, exist_ok=True)
    total_ms = len(audio)
    chunks = []
    
    if total_ms == 0:
         log_error("File audio kosong (durasi 0 detik). Membatalkan.", exit_app=True)

    for i in range(0, total_ms, chunk_length_ms):
        chunk_name = Path(output_dir) / f"part_{i//chunk_length_ms + 1}.mp3"
        try:
            part = audio[i:i+chunk_length_ms]
            part.export(str(chunk_name), format="mp3")
            chunks.append(str(chunk_name))
            log_info(f"  → Dibuat: {chunk_name}")
        except Exception as e:
            log_error(f"Gagal mengekspor potongan audio: {chunk_name}. Error: {e}", exit_app=False)
            log_error("Menghentikan proses pemecahan audio.", exit_app=True)

    log_success(f"Total {len(chunks)} potongan audio berhasil dibuat.")
    return chunks, chunk_length_ms


# FUNGSI BARU YANG DIPERBAIKI
def shift_srt_time(file_path, offset_seconds, max_chunk_seconds):
    """
    Menggeser waktu file SRT dengan aman DAN membersihkan/memangkas 
    timestamp yang melebihi durasi potongan maksimum.
    """
    pattern = r"(\d{2}):(\d{2}):(\d{2}),(\d{3})"
    
    # Konversi durasi ke obyek timedelta untuk perbandingan
    offset_duration = timedelta(seconds=offset_seconds)
    max_duration = timedelta(seconds=max_chunk_seconds)

    def shift_and_clean(match):
        h, m, s, ms = map(int, match.groups())
        original = timedelta(hours=h, minutes=m, seconds=s, milliseconds=ms)
        
        # --- INI ADALAH PERBAIKAN BUG ---
        # Jika timestamp (baik mulai atau akhir) melebihi durasi
        # potongan yang diizinkan, pangkas ke durasi maksimum.
        if original > max_duration:
            log_warn(f"  → Memangkas timestamp {original} ke {max_duration} di {file_path}")
            original = max_duration
        
        # Lakukan pergeseran (offset)
        shifted = original + offset_duration
        
        # Konversi total detik kembali ke format H:M:S,ms
        total_ms = int(shifted.total_seconds() * 1000)
        hh = total_ms // 3600000
        mm = (total_ms % 3600000) // 60000
        ss = (total_ms % 60000) // 1000
        mss = total_ms % 1000
        return f"{hh:02d}:{mm:02d}:{ss:02d},{mss:03d}"

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        new_content = re.sub(pattern, shift_and_clean, content)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
            
        return True
    except FileNotFoundError:
        log_error(f"  → Gagal menemukan file SRT untuk digeser: {file_path}", exit_app=False)
        return False
    except Exception as e:
        log_error(f"  → Gagal memproses pergeseran waktu SRT untuk {file_path}: {e}", exit_app=False)
        return False

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        new_content = re.sub(pattern, shift, content)
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(new_content)
        
        return True
    except FileNotFoundError:
        log_error(f"  → Gagal menemukan file SRT untuk digeser: {file_path}", exit_app=False)
        return False
    except Exception as e:
        log_error(f"  → Gagal memproses pergeseran waktu SRT untuk {file_path}: {e}", exit_app=False)
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
    srt_block_counter = 1 # Counter global untuk re-indexing SRT

    for i, chunk in enumerate(chunk_files, start=1):
        log_info(f"Mentranskripsi potongan {i}/{len(chunk_files)}: {chunk}")
        
        # --- PERUBAHAN DI SINI ---
        # Tentukan path output secara eksplisit, tanpa ekstensi
        # Cth: "chunks/part_1.mp3" -> "chunks/part_1"
        output_base_path = Path(chunk).with_suffix("")
        
        cmd = [
            str(whisper_cli_path),
            "-m", str(model_path),
            "-f", chunk,
            "-of", str(output_base_path), # <-- TAMBAHKAN FLAG INI
            "-otxt",
            "-osrt",
            "-l", "id",
            "-pp" # Prosesor progres (opsional, tapi bagus untuk debug)
        ]
        
        try:
            # --- PERUBAHAN DI SINI ---
            # Hapus stdout=DEVNULL dan stderr=PIPE agar output/progress whisper-cli terlihat
            subprocess.run(cmd, check=True)
            print() # Tambahkan baris baru setelah output whisper-cli selesai
        except subprocess.CalledProcessError as e:
            # stderr = e.stderr.decode().strip() # Baris ini tidak lagi valid, tapi tidak apa-apa
            
            # Pesan error dari whisper-cli sudah otomatis tercetak ke konsol
            print() # Tambahkan baris baru untuk spasi
            log_error(f"whisper-cli gagal pada potongan {chunk} (return code: {e.returncode}). Lihat pesan error di atas.", exit_app=False)
            log_warn(f"Melompati potongan {chunk} karena error.")

            # --- TAMBAHAN DEBUG ---
            # Jika gagal, coba hapus file output parsial agar tidak bingung
            try:
                txt_file_fail = output_base_path.with_suffix(".txt")
                srt_file_fail = output_base_path.with_suffix(".srt")
                if txt_file_fail.exists(): txt_file_fail.unlink()
                if srt_file_fail.exists(): srt_file_fail.unlink()
            except Exception as e_clean:
                log_warn(f"  → Gagal membersihkan file sisa: {e_clean}")

            continue # Lanjutkan ke potongan berikutnya
        except Exception as e:
            log_error(f"Error tak terduga saat menjalankan whisper-cli pada {chunk}: {e}", exit_app=True)

        # --- Proses TXT ---
        # Logika ini sekarang seharusnya sudah benar karena kita pakai -of
        txt_file = Path(chunk).with_suffix(".txt")
        try:
            if txt_file.exists():
                content = txt_file.read_text(encoding="utf-8").strip()
                with open(final_txt, "a", encoding="utf-8") as out:
                    out.write(content + "\n\n")
                txt_file.unlink()
                log_success("  → TXT digabung.")
            else:
                log_warn(f"  → File TXT output tidak ditemukan untuk: {chunk}")
        except Exception as e:
            log_error(f"  → Gagal memproses file TXT {txt_file}: {e}")

        # --- Proses SRT ---
        # Logika ini sekarang seharusnya sudah benar karena kita pakai -of
        srt_file = Path(chunk).with_suffix(".srt")
        # --- Proses SRT ---
        try:
            if srt_file.exists():
                offset_seconds = (i - 1) * chunk_seconds
                
                # --- INI ADALAH PERUBAHAN KEDUA ---
                # Kita oper 'chunk_seconds' (yaitu 15.0) sebagai batas pangkas
                
                # (Pengecekan ini opsional tapi bagus)
                # Jangan pangkas potongan terakhir, karena ia boleh berakhir kapan saja
                is_last_chunk = (i == len(chunk_files))
                max_seconds_cap = 99999.0 if is_last_chunk else chunk_seconds
                
                if not shift_srt_time(srt_file, offset_seconds, max_seconds_cap):
                    # Error sudah dicatat di dalam fungsi shift_srt_time
                    raise Exception(f"Gagal menggeser waktu untuk {srt_file}")

                srt_content = srt_file.read_text(encoding="utf-8").strip()

                # Fungsi untuk mengganti nomor indeks SRT
                def reindex_srt(match):
                    nonlocal srt_block_counter
                    new_index = srt_block_counter
                    srt_block_counter += 1
                    return str(new_index)

                # Regex untuk menemukan indeks blok (angka di barisnya sendiri)
                reindexed_content = re.sub(r"^\d+\s*$", reindex_srt, srt_content, flags=re.MULTILINE)

                with open(final_srt, "a", encoding="utf-8") as out:
                    out.write(reindexed_content + "\n\n")
                
                srt_file.unlink()
                log_success(f"  → SRT digabung & di-re-index (offset {offset_seconds:.2f}s).")
            else:
                log_warn(f"  → File SRT output tidak ditemukan untuk: {chunk}")
        except Exception as e:
            log_error(f"  → Gagal memproses file SRT {srt_file}: {e}")

    log_success("Semua TXT dan SRT telah digabung.")
    log_success(f"Output TXT: {final_txt}")
    log_success(f"Output SRT: {final_srt}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 transcriptor_debug.py <url_or_file> <model>")
        print(f"Model: {', '.join(VALID_MODELS)}")
        sys.exit(1)
    
    try:
        # 1. Cek dependensi KETAT
        whisper_cli_path = check_dependencies()

        # 2. Ambil argumen & validasi model
        source = sys.argv[1]
        model_name = sys.argv[2]
        model_path = ensure_model_exists(model_name)

        # 3. Tentukan & unduh audio (dengan validasi)
        if os.path.exists(source):
            audio_path = Path(source)
            log_success(f"Menggunakan file lokal: {audio_path}")
        else:
            audio_path = Path("audio.mp3")
            log_warn(f"Input berupa URL, mengunduh ke {audio_path}...")
            download_audio(source, audio_path)

        # 4. Proses utama (dengan validasi)
        chunk_files, chunk_length_ms = split_audio(audio_path, "chunks")
        
        if not chunk_files:
            log_error("Tidak ada potongan audio yang berhasil dibuat. Proses dihentikan.", exit_app=True)

        transcribe_with_whisper_cpp(chunk_files, model_path, chunk_length_ms, whisper_cli_path)

        log_success("====== PROSES SELESAI ======")
        log_info("Output akhir ada di folder ./transcripts/")

    except Exception as e:
        # Penangan error global untuk masalah yang tidak terduga
        log_error(f"Terjadi error fatal yang tidak terduga: {e}", exit_app=False)
        print("------ STACK TRACE LENGKAP ------")
        traceback.print_exc()
        print("---------------------------------")
        sys.exit(1)

if __name__ == "__main__":
    main()