#!/bin/bash
# End-to-end setup for YOLO26-x:
#   1. Clone marcoslucianops/DeepStream-Yolo
#   2. Build libnvdsinfer_custom_impl_Yolo.so
#   3. Download yolo26x.pt weights
#   4. Export to ONNX via DeepStream-Yolo's export_yolo26.py
#   5. Place .so + ONNX + labels.txt under /models/yolo26x/
#
# Run inside the container: bash /workspace/docker/scripts/setup_yolo26x.sh

set -e

OUTPUT_DIR="/models/yolo26x"
DS_YOLO_DIR="/tmp/DeepStream-Yolo"
WORK_DIR="/tmp/yolo26x_export"
WEIGHTS_URL="https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26x.pt"

echo "=== YOLO26-x setup ==="

mkdir -p "$OUTPUT_DIR" "$WORK_DIR"

# --- 1. Clone DeepStream-Yolo ---
if [ ! -d "$DS_YOLO_DIR" ]; then
    echo "Cloning DeepStream-Yolo..."
    git clone https://github.com/marcoslucianops/DeepStream-Yolo.git "$DS_YOLO_DIR"
fi

# --- 2. Build custom parser .so ---
CUDA_VER=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //' | sed 's/,.*//')
if [ -z "$CUDA_VER" ]; then
    CUDA_VER="12.6"
    echo "WARNING: Could not detect CUDA version, using ${CUDA_VER}"
fi
echo "CUDA version: ${CUDA_VER}"

export CUDA_VER
make -C "$DS_YOLO_DIR/nvdsinfer_custom_impl_Yolo" clean
make -C "$DS_YOLO_DIR/nvdsinfer_custom_impl_Yolo"

cp "$DS_YOLO_DIR/nvdsinfer_custom_impl_Yolo/libnvdsinfer_custom_impl_Yolo.so" "$OUTPUT_DIR/"
echo "Parser: ${OUTPUT_DIR}/libnvdsinfer_custom_impl_Yolo.so"

# --- 3. Install exporter deps ---
pip install ultralytics onnx onnxslim onnxruntime --quiet

# --- 4. Download weights ---
cd "$WORK_DIR"
if [ ! -f yolo26x.pt ]; then
    echo "Downloading yolo26x.pt..."
    wget -q "$WEIGHTS_URL"
fi

# --- 5. Export to ONNX ---
cp "$DS_YOLO_DIR/utils/export_yolo26.py" "$WORK_DIR/"
python3 export_yolo26.py -w yolo26x.pt --dynamic

mv yolo26x.onnx "$OUTPUT_DIR/yolo26x.onnx"
if [ -f labels.txt ]; then
    mv labels.txt "$OUTPUT_DIR/labels.txt"
fi

# Force engine rebuild on next pipeline run
rm -f "$OUTPUT_DIR"/*.engine

rm -rf "$WORK_DIR"

echo ""
echo "=== Done ==="
echo "ONNX:   ${OUTPUT_DIR}/yolo26x.onnx"
echo "Labels: ${OUTPUT_DIR}/labels.txt"
echo "Parser: ${OUTPUT_DIR}/libnvdsinfer_custom_impl_Yolo.so"
echo "TRT engine will be generated on first pipeline run (may take >10 min)."
