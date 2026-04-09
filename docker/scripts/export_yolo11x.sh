#!/bin/bash
# Export YOLO11x to ONNX for DeepStream (two variants)
#
# Produces:
#   yolo11x_raw.onnx       [1, 84, 8400]  — for deepstream-yolo-nvinfer (callback parsing)
#   yolo11x_parsed.onnx    [1, 8400, 6]   — for deepstream-yolo-nvinfer-custom (C++ parser)
#
# Run inside the container: bash /workspace/docker/scripts/export_yolo11x.sh

set -e

OUTPUT_DIR="/models/yolo11x"
WORK_DIR="/tmp/yolo_export"
DS_YOLO_DIR="/tmp/DeepStream-Yolo"

echo "=== Exporting yolo11x to ONNX ==="

mkdir -p "$OUTPUT_DIR" "$WORK_DIR"

pip install ultralytics onnx onnxslim onnxruntime onnxscript --quiet

# Download weights
cd "$WORK_DIR"
if [ ! -f yolo11x.pt ]; then
    echo "Downloading yolo11x.pt..."
    wget -q https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11x.pt
fi

# --- 1. Raw export for callback parsing [1, 84, 8400] ---
echo ""
echo "=== Exporting raw ONNX (for deepstream-yolo-nvinfer) ==="
python3 -c "
from ultralytics import YOLO
model = YOLO('yolo11x.pt')
model.export(format='onnx', opset=16, simplify=True, dynamic=True)
"
mv yolo11x.onnx "${OUTPUT_DIR}/yolo11x_raw.onnx"
echo "Done: yolo11x_raw.onnx [1, 84, 8400]"

# --- 2. DeepStream-Yolo export for C++ parser [1, 8400, 6] ---
echo ""
echo "=== Exporting parsed ONNX (for deepstream-yolo-nvinfer-custom) ==="
if [ ! -d "$DS_YOLO_DIR" ]; then
    git clone https://github.com/marcoslucianops/DeepStream-Yolo.git "$DS_YOLO_DIR"
fi
cp "$DS_YOLO_DIR/utils/export_yolo11.py" "$WORK_DIR/"
python3 export_yolo11.py -w yolo11x.pt --simplify
mv yolo11x.onnx "${OUTPUT_DIR}/yolo11x_parsed.onnx"
echo "Done: yolo11x_parsed.onnx [1, 8400, 6]"

# Delete old engines (force rebuild)
rm -f "${OUTPUT_DIR}"/*.engine

rm -rf "$WORK_DIR"

echo ""
echo "=== Done ==="
echo "Raw:    ${OUTPUT_DIR}/yolo11x_raw.onnx     (callback parsing)"
echo "Parsed: ${OUTPUT_DIR}/yolo11x_parsed.onnx  (C++ parser)"
echo "Labels: ${OUTPUT_DIR}/labels.txt"
echo "Engines will be generated on first pipeline run"
