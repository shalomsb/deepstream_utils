#!/usr/bin/env bash
# Build yolo26x_b6 TRT plan with FIXED batch=6 (one .plan per batch profile;
# the simple yolo26x app keeps its own batch=1 plan untouched).
#
# Run inside the container.

set -euo pipefail

SRC_ONNX="${SRC_ONNX:-/triton/model_repo/yolo26x/1/model.onnx}"
DST_DIR="${DST_DIR:-/triton/model_repo/yolo26x_b6/1}"
DST_PLAN="${DST_DIR}/model.plan"
IMGSZ="${IMGSZ:-640}"
BATCH=6

if [[ ! -f "$SRC_ONNX" ]]; then
    echo "ERROR: $SRC_ONNX not found." >&2
    echo "Run apps/deepstream-yolo26x-triton/scripts/export_yolo26x.py first." >&2
    exit 1
fi

mkdir -p "$DST_DIR"

echo "Building yolo26x_b6 TRT plan (fixed batch=$BATCH):"
echo "  ONNX: $SRC_ONNX"
echo "  PLAN: $DST_PLAN"

trtexec \
    --onnx="$SRC_ONNX" \
    --saveEngine="$DST_PLAN" \
    --fp16 \
    --minShapes="images:${BATCH}x3x${IMGSZ}x${IMGSZ}" \
    --optShapes="images:${BATCH}x3x${IMGSZ}x${IMGSZ}" \
    --maxShapes="images:${BATCH}x3x${IMGSZ}x${IMGSZ}"

echo "Done. Plan written to: $DST_PLAN"
