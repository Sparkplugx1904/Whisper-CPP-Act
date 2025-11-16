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

def transcribe_with_whisper_cpp(chunk_files, model_path, chunk_length_ms):
    os.makedirs("transcripts", exist_ok=True)
    all_text = []
    chunk_minutes = chunk_length_ms / 60000

    for i, chunk in enumerate(chunk_files, start=1):
        start_min = (i - 1) * chunk_minutes
        hours = int(start_min // 60)
        minutes = int(start_min % 60)
        seconds = 0
        timestamp = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

        print(f"[+] Transcribing potongan {i}/{len(chunk_files)}: {chunk}", flush=True)
        print(f"    → Waktu mulai: {timestamp}", flush=True)

        cmd = [
            "./build/bin/whisper-cli",
            "-m", str(model_path),
            "-f", chunk,
            "-otxt",
            "-l", "id",
            "-pp"
        ]
        subprocess.run(cmd, check=True)

        txt_file = Path(chunk).with_suffix(".mp3.txt")
        if txt_file.exists():
            with open(txt_file, "r", encoding="utf-8") as f:
                all_text.append(f.read())
            os.rename(txt_file, f"./transcripts/{txt_file.name}")
            print(f"    [✓] Disimpan: ./transcripts/{txt_file.name}")
        else:
            print(f"    [!] Gagal menemukan file transkrip untuk {chunk}")
    return all_text

def combine_transcripts(all_text, output_file):
    print("[+] Menggabungkan seluruh hasil transkripsi...")
    with open(output_file, "w", encoding="utf-8") as f:
        for text in all_text:
            f.write(text.strip() + "\n\n")
    print(f"[✓] Transkrip akhir disimpan di: {output_file}")

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

    # 🔹 Buat nama file output berdasarkan nama audio
    output_name = audio_path.stem + "_transcript.txt"
    final_output = Path("./transcripts") / output_name

    # 🔹 Proses utama
    chunk_files, chunk_length_ms = split_audio(audio_path, "chunks")
    all_text = transcribe_with_whisper_cpp(chunk_files, model_path, chunk_length_ms)
    combine_transcripts(all_text, final_output)

    print(f"[✓] Final transcript: {final_output}")

if __name__ == "__main__":
    main()
