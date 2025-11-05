#!/usr/bin/env bash
# build_whisper_static.sh
# Build whisper.cpp secara bersih, statis (no shared .so), parallel make -j
#
# Usage:
#   ./build_whisper_static.sh         # pakai semua core CPU
#   ./build_whisper_static.sh 4       # pakai 4 job
#
# Catatan: Model (ggml-medium.bin) bukan bagian dari proses build — itu dipakai saat runtime.
set -euo pipefail

# jumlah job untuk make (-j). Default = jumlah core
JOBS="${1:-$(nproc)}"

# lokasi build dir (relatif ke root repo)
BUILD_DIR="build"

echo "==> Build whisper.cpp (static) — jobs: ${JOBS}"
echo "==> Membersihkan folder build lama (jika ada)..."
rm -rf "${BUILD_DIR}"

echo "==> Membuat folder build..."
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

echo "==> Menjalankan cmake (memaksa OFF shared libs)..."
# Kita set beberapa opsi agar pasti build statis: WHISPER_SHARED atau BUILD_SHARED_LIBS
cmake -DBUILD_SHARED_LIBS=OFF -DWHISPER_SHARED=OFF ..

echo "==> Menjalankan make dengan -j${JOBS}..."
make -j "${JOBS}"

echo "==> Build selesai."
echo "==> Binary ada di: $(pwd)/bin (jalankan dari folder build: LD_LIBRARY_PATH tidak diperlukan untuk binary statis)"
echo
echo "Contoh menjalankan (dari build folder):"
echo "./bin/whisper-cli -m ./models/ggml-medium.bin -f ./samples/jfk.wav"
echo
echo "Jika ingin menjalankan dari root repo, gunakan:"
echo "${BUILD_DIR}/bin/whisper-cli -m ${BUILD_DIR}/models/ggml-medium.bin -f ${BUILD_DIR}/samples/jfk.wav"
