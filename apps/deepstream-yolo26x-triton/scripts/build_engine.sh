#!/usr/bin/env bash
# Build a TensorRT FP16 engine for yolo26x with min/opt/max batch profile.
# Run inside the DeepStream container (trtexec is on PATH).
#
# Prereq: ONNX with a DYNAMIC batch axis at:
#   /triton/model_repo/yolo26x/1/model.onnx
# Use scripts/export_yolo26x.py to produce one.

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/triton/model_repo/yolo26x/1}"
ONNX="${MODEL_DIR}/model.onnx"
PLAN="${MODEL_DIR}/model.plan"

MIN_BATCH="${MIN_BATCH:-1}"
OPT_BATCH="${OPT_BATCH:-1}"
MAX_BATCH="${MAX_BATCH:-1}"
IMGSZ="${IMGSZ:-640}"

if [[ ! -f "$ONNX" ]]; then
    echo "ERROR: $ONNX not found. Run scripts/export_yolo26x.py first." >&2
    exit 1
fi

echo "Building TensorRT FP16 engine:"
echo "  ONNX: $ONNX"
echo "  PLAN: $PLAN"
echo "  Batch profile: min=$MIN_BATCH opt=$OPT_BATCH max=$MAX_BATCH"
echo "  Image size:    ${IMGSZ}x${IMGSZ}"

trtexec \
    --onnx="$ONNX" \
    --saveEngine="$PLAN" \
    --fp16 \
    --minShapes="images:${MIN_BATCH}x3x${IMGSZ}x${IMGSZ}" \
    --optShapes="images:${OPT_BATCH}x3x${IMGSZ}x${IMGSZ}" \
    --maxShapes="images:${MAX_BATCH}x3x${IMGSZ}x${IMGSZ}"

echo "Done. Plan written to: $PLAN"
