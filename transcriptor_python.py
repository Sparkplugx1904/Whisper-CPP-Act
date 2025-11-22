#!/usr/bin/env python3
import sys
import os
import subprocess
import re
import traceback
import shutil
from pathlib import Path
from datetime import timedelta
from pydub import AudioSegment
from pydub.utils import which

# --- Dependensi Python Baru ---
try:
    import torch
    # Impor tambahan untuk memuat komponen secara manual
    from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoTokenizer, GenerationConfig
except ImportError:
    print("[✗] ERROR: Dependensi Python tidak ditemukan.", file=sys.stderr)
    print("Harap instal dependensi yang diperlukan dari requirements.txt", file=sys.stderr)
    print("Contoh: pip install -r requirements.txt", file=sys.stderr)
    sys.exit(1)

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

# --- Fungsi Inti dengan Penanganan Error ---

def check_ffmpeg_dependency():
    """Memeriksa dependensi ffmpeg/ffprobe untuk pydub."""
    log_info("Memeriksa dependensi 'ffmpeg'...")
    if not which("ffmpeg") and not which("ffprobe"):
        log_error("'ffmpeg' atau 'ffprobe' tidak ditemukan. 'pydub' membutuhkannya untuk memproses audio.", exit_app=True)
    log_success("'ffmpeg'/'ffprobe' ditemukan.")

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

def download_audio(url, output_path):
    """Wrapper untuk mengunduh file audio."""
    log_info(f"Mengunduh audio dari: {url}")
    if not download_file(url, output_path):
        log_error("Gagal mengunduh audio. Membatalkan.", exit_app=True)
    log_success(f"Audio berhasil diunduh ke {output_path}")

def split_audio(input_path, output_dir, chunk_length_ms=5*60*1000):
    """Memecah audio dengan penanganan error untuk pydub."""
    log_info(f"Memecah audio menjadi potongan {chunk_length_ms // 60000} menit...")
    try:
        audio = AudioSegment.from_file(input_path)
    except FileNotFoundError:
        log_error(f"File audio tidak ditemukan di: {input_path}", exit_app=True)
    except Exception as e:
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
            log_info(f"  → Dibuat: {chunk_name}")
        except Exception as e:
            log_error(f"Gagal mengekspor potongan audio: {chunk_name}. Error: {e}", exit_app=False)
            log_error("Menghentikan proses pemecahan audio.", exit_app=True)

    log_success(f"Total {len(chunks)} potongan audio berhasil dibuat.")
    return chunks, chunk_length_ms

def initialize_transcriber():
    """Menginisialisasi pipeline transkripsi dari Hugging Face."""
    model_id = "cahya/whisper-medium-id"
    log_info(f"Memuat model '{model_id}'...")
    log_warn("Ini mungkin memakan waktu lama saat pertama kali (model ~1.5GB).")
    try:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        log_info(f"Menggunakan perangkat: {device}")
        
        # --- PERUBAHAN DI SINI ---
        # 1. Muat komponen secara manual
        log_info("Memuat model dan tokenizer...")
        model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # 2. Buat GenerationConfig dan atur bahasa/tugas
        # Ini adalah langkah kunci untuk memperbaiki error timestamp
        log_info("Mengonfigurasi GenerationConfig untuk 'id' dan 'transcribe'...")
        generation_config = GenerationConfig.from_pretrained(model_id)
        generation_config.language = "id"
        generation_config.task = "transcribe"
        
        # 3. Terapkan forced_decoder_ids (sama seperti sebelumnya, tapi ke config)
        log_info("Menerapkan forced_decoder_ids...")
        forced_decoder_ids = tokenizer.get_decoder_prompt_ids(language="id", task="transcribe")
        generation_config.forced_decoder_ids = forced_decoder_ids
        
        # 4. Buat pipeline dengan SEMUA komponen yang sudah disiapkan
        log_info("Menginisialisasi pipeline...")
        transcriber = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=tokenizer,
            generation_config=generation_config, # <-- INI FIXNYA
            device=device
        )
        # --- AKHIR PERUBAHAN ---
        
        log_success("Model transkripsi berhasil dimuat.")
        return transcriber
    except Exception as e:
        log_error(f"Gagal memuat model dari Hugging Face: {e}", exit_app=True)
        traceback.print_exc() # Tampilkan trace lengkap jika gagal
        return None

def format_srt_time(seconds):
    """Mengubah detik (float) menjadi format H:M:S,ms SRT."""
    td = timedelta(seconds=seconds)
    total_ms = int(td.total_seconds() * 1000)
    hh = total_ms // 3600000
    mm = (total_ms % 3600000) // 60000
    ss = (total_ms % 60000) // 1000
    mss = total_ms % 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{mss:03d}"

def transcribe_with_transformers(transcriber, chunk_files, chunk_length_ms):
    """Mentranskripsi setiap potongan menggunakan pipeline transformers."""
    os.makedirs("transcripts", exist_ok=True)

    final_txt = Path("transcripts/transcript.txt")
    final_srt = Path("transcripts/transcript.srt")
    
    try:
        final_txt.write_text("", encoding="utf-8")
        final_srt.write_text("", encoding="utf-8")
    except IOError as e:
        log_error(f"Gagal membuat file transkrip akhir di ./transcripts/. Periksa izin folder. Error: {e}", exit_app=True)

    chunk_seconds_base = chunk_length_ms / 1000
    srt_block_counter = 1

    for i, chunk_path_str in enumerate(chunk_files, start=1):
        chunk_path = Path(chunk_path_str)
        log_info(f"Mentranskripsi potongan {i}/{len(chunk_files)}: {chunk_path.name}")
        
        time_offset_seconds = (i - 1) * chunk_seconds_base
        
        try:
            result = transcriber(
                chunk_path_str, 
                return_timestamps=True,
                chunk_length_s=30,
                stride_length_s=5
            )
            
            # 1. Proses TXT
            full_text = result.get('text', '').strip()
            if full_text:
                with open(final_txt, "a", encoding="utf-8") as out:
                    out.write(full_text + "\n\n")
                log_success("  → TXT digabung.")
            else:
                log_warn("  → Tidak ada teks yang terdeteksi di potongan ini.")

            # 2. Proses SRT
            chunks = result.get('chunks', [])
            if not chunks:
                log_warn("  → Tidak ada timestamp (chunks) yang dikembalikan untuk SRT.")
                continue 
                
            srt_content = ""
            for segment in chunks:
                timestamp = segment.get('timestamp')
                segment_text = segment.get('text', '').strip()
                
                if not timestamp or not segment_text:
                    continue

                start_time, end_time = timestamp
                
                start_time += time_offset_seconds
                end_time += time_offset_seconds
                
                srt_content += f"{srt_block_counter}\n"
                srt_content += f"{format_srt_time(start_time)} --> {format_srt_time(end_time)}\n"
                srt_content += f"{segment_text}\n\n"
                
                srt_block_counter += 1
            
            if srt_content:
                with open(final_srt, "a", encoding="utf-8") as out:
                    out.write(srt_content)
                log_success(f"  → SRT digabung & di-re-index (offset {time_offset_seconds:.2f}s).")

        except Exception as e:
            log_error(f"Gagal mentranskripsi potongan {chunk_path.name}: {e}", exit_app=False)
            print("------ STACK TRACE ------")
            traceback.print_exc()
            print("-------------------------")
            log_warn(f"Melompati potongan {chunk_path.name} karena error.")
            continue
        finally:
            try:
                chunk_path.unlink()
                log_info(f"  → File chunk {chunk_path.name} dihapus.")
            except OSError as e:
                log_warn(f"  → Gagal menghapus file chunk {chunk_path.name}: {e}")

    log_success("Semua TXT dan SRT telah digabung.")
    log_success(f"Output TXT: {final_txt}")
    log_success(f"Output SRT: {final_srt}")

def main():
    # Perbarui pesan Usage: argumen model tidak lagi diperlukan
    if len(sys.argv) < 2:
        print(f"Usage: python3 {sys.argv[0]} <url_or_file>")
        print("Script ini menggunakan model 'cahya/whisper-medium-id' (Hugging Face) secara otomatis.")
        print(f"Contoh: python3 {sys.argv[0]} \"http://example.com/audio.mp3\"")
        print(f"Contoh: python3 {sys.argv[0]} file_lokal_saya.wav")
        sys.exit(1)
    
    chunk_dir = None
    audio_path = None
    source = sys.argv[1] # Hanya ambil argumen pertama
    is_local_file = os.path.exists(source)

    try:
        check_ffmpeg_dependency()
        transcriber = initialize_transcriber()

        if is_local_file:
            audio_path = Path(source)
            log_success(f"Menggunakan file lokal: {audio_path}")
        else:
            safe_name = re.sub(r'[^a-zA-Z0-9]', '_', source.split('/')[-1])
            audio_path = Path(f"audio_{safe_name}.mp3")
            log_warn(f"Input berupa URL, mengunduh ke {audio_path}...")
            download_audio(source, audio_path)

        chunk_dir = Path(f"./chunks_{audio_path.stem}")
        chunk_files, chunk_length_ms = split_audio(audio_path, chunk_dir)
        
        if not chunk_files:
            log_error("Tidak ada potongan audio yang berhasil dibuat. Proses dihentikan.", exit_app=True)

        transcribe_with_transformers(transcriber, chunk_files, chunk_length_ms)

        log_success("====== PROSES SELESAI ======")
        log_info("Output akhir ada di folder ./transcripts/")

    except Exception as e:
        log_error(f"Terjadi error fatal yang tidak terduga: {e}", exit_app=False)
        print("------ STACK TRACE LENGKAP ------")
        traceback.print_exc()
        print("---------------------------------")
        sys.exit(1)
    finally:
        if chunk_dir and chunk_dir.exists():
            try:
                shutil.rmtree(chunk_dir)
                log_info(f"Folder chunk sementara '{chunk_dir}' telah dihapus.")
            except Exception as e:
                log_warn(f"Gagal menghapus folder chunk '{chunk_dir}': {e}")
        
        if not is_local_file and audio_path and audio_path.exists():
             try:
                audio_path.unlink()
                log_info(f"File audio unduhan '{audio_path}' telah dihapus.")
             except Exception as e:
                 log_warn(f"Gagal menghapus file audio unduhan '{audio_path}': {e}")


if __name__ == "__main__":
    main()