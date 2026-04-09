#!/bin/bash
# Build DeepStream-Yolo custom parser (libnvdsinfer_custom_impl_Yolo.so)
# Run inside the container: bash /docker/scripts/build_yolo_parser.sh

set -e

WORK_DIR="/tmp/DeepStream-Yolo"
OUTPUT_DIR="/models/yolo11x"
DS_VER=$(deepstream-app -v 2>/dev/null | awk '$$1~/DeepStreamSDK/ {print substr($$2,1,3)}' || echo "9.0")

echo "=== Building DeepStream-Yolo custom parser ==="
echo "DeepStream version: ${DS_VER}"

# Clone
if [ ! -d "$WORK_DIR" ]; then
    git clone https://github.com/marcoslucianops/DeepStream-Yolo.git "$WORK_DIR"
fi

cd "$WORK_DIR"

# Detect CUDA version
CUDA_VER=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//')
if [ -z "$CUDA_VER" ]; then
    CUDA_VER="12.8"
    echo "WARNING: Could not detect CUDA version, using ${CUDA_VER}"
fi
echo "CUDA version: ${CUDA_VER}"

# Build
export CUDA_VER
make -C nvdsinfer_custom_impl_Yolo clean
make -C nvdsinfer_custom_impl_Yolo

# Copy .so to model directory
mkdir -p "$OUTPUT_DIR"
cp nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so "$OUTPUT_DIR/"

echo "=== Done ==="
echo "Parser: ${OUTPUT_DIR}/libnvdsinfer_custom_impl_Yolo.so"
