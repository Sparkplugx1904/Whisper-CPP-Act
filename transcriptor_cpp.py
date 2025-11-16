#!/usr/bin/env python3
import sys
import os
import subprocess
import glob
from pathlib import Path
from pydub import AudioSegment
from pydub.utils import which

def download_file(url, dest):
    print(f"[+] Mengunduh: {url}")
    try:
        subprocess.run(["curl", "-L", "-o", str(dest), url], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"[✓] Unduhan selesai: {dest}")
    except subprocess.CalledProcessError as e:
        print(f"[✗] Gagal mengunduh file: {e}")
        sys.exit(1)

def ensure_model_exists(model_name):
    os.makedirs("models", exist_ok=True)
    model_path = Path(f"./models/ggml-{model_name}.bin")
    if model_path.exists():
        print(f"[✓] Model ditemukan: {model_path}")
        return model_path
    print(f"[!] Model belum ada, mengunduh dari HuggingFace...")
    url = f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{model_name}.bin"
    download_file(url, model_path)
    return model_path

def download_audio(url, output_path):
    print(f"[+] Mengunduh audio dari: {url}")
    download_file(url, output_path)

def split_audio(input_path, output_dir, chunk_length_ms=5*60*1000):
    print(f"[+] Memecah audio menjadi potongan {chunk_length_ms // 60000} menit...")

    audio = AudioSegment.from_file(input_path)
    os.makedirs(output_dir, exist_ok=True)
    total = len(audio)
    chunks = []

    for i in range(0, total, chunk_length_ms):
        part = audio[i:i+chunk_length_ms]
        chunk_name = os.path.join(output_dir, f"part_{i//chunk_length_ms + 1}.mp3")
        part.export(chunk_name, format="mp3")
        chunks.append(chunk_name)
        print(f"    → {chunk_name}")

    print(f"[✓] Total {len(chunks)} potongan audio dibuat.")
    return chunks, chunk_length_ms

import os
import re
import subprocess
from pathlib import Path
from datetime import timedelta

def shift_srt_time(file_path, offset_seconds):
    pattern = r"(\d{2}):(\d{2}):(\d{2}),(\d{3})"

    def shift(match):
        h, m, s, ms = map(int, match.groups())
        original = timedelta(hours=h, minutes=m, seconds=s, milliseconds=ms)
        shifted = original + timedelta(seconds=offset_seconds)
        total_ms = int(shifted.total_seconds() * 1000)
        hh = total_ms // 3600000
        mm = (total_ms % 3600000) // 60000
        ss = (total_ms % 60000) // 1000
        mss = total_ms % 1000
        return f"{hh:02d}:{mm:02d}:{ss:02d},{mss:03d}"

    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    new_content = re.sub(pattern, shift, content)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)


def transcribe_with_whisper_cpp(chunk_files, model_path, chunk_length_ms):
    os.makedirs("transcripts", exist_ok=True)

    final_txt = Path("transcripts/transcript.txt")
    final_srt = Path("transcripts/transcript.srt")

    final_txt.write_text("", encoding="utf-8")
    final_srt.write_text("", encoding="utf-8")

    chunk_seconds = chunk_length_ms / 1000

    for i, chunk in enumerate(chunk_files, start=1):
        print(f"[+] Transcribing potongan {i}/{len(chunk_files)}: {chunk}")

        cmd = [
            "./build/bin/whisper-cli",
            "-m", str(model_path),
            "-f", chunk,
            "-otxt",
            "-osrt",
            "-l", "id",
            "-pp"
        ]
        subprocess.run(cmd, check=True)

        txt_file = Path(chunk).with_suffix(".txt")
        if txt_file.exists():
            with open(txt_file, "r", encoding="utf-8") as f:
                content = f.read().strip()
            with open(final_txt, "a", encoding="utf-8") as out:
                out.write(content + "\n\n")
            txt_file.unlink()
            print("    [✓] TXT appended ke transcript.txt")
        else:
            print("    [!] TXT tidak ditemukan")

        srt_file = Path(chunk).with_suffix(".srt")
        if srt_file.exists():
            offset_seconds = (i - 1) * chunk_seconds
            shift_srt_time(srt_file, offset_seconds)

            with open(srt_file, "r", encoding="utf-8") as f:
                srt_block = f.read().strip()

            with open(final_srt, "a", encoding="utf-8") as out:
                out.write(srt_block + "\n\n")

            srt_file.unlink()
            print("    [✓] SRT appended ke transcript.srt")
        else:
            print("    [!] SRT tidak ditemukan")

    print("[✓] Semua TXT dan SRT telah digabung otomatis.")
    print(f"[✓] Output: {final_txt}")
    print(f"[✓] Output: {final_srt}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 transcriptor_cpp.py <url_or_file> <model>")
        print("Model: tiny | base | small | medium | large-v1 | large-v2 | large-v3 | large-v3-turbo")
        sys.exit(1)

    source = sys.argv[1]
    model_name = sys.argv[2]
    model_path = ensure_model_exists(model_name)

    # 🔹 Tentukan nama file audio
    if os.path.exists(source):
        audio_path = Path(source)
        print(f"[✓] Menggunakan file lokal: {audio_path}")
    else:
        audio_path = Path("audio.mp3")
        print(f"[!] Input berupa URL, mengunduh ke {audio_path}")
        download_audio(source, audio_path)

    # 🔹 Proses utama
    chunk_files, chunk_length_ms = split_audio(audio_path, "chunks")
    transcribe_with_whisper_cpp(chunk_files, model_path, chunk_length_ms)

    print("[✓] Final output ada di folder ./transcripts/")
