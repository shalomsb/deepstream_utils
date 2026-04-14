#!/usr/bin/env bash
# Download Grounding DINO ONNX from NGC and convert to TensorRT engine.
set -euo pipefail

MODEL_NAME="grounding_dino"
TRITON_REPO="/triton/model_repo"
MODEL_DIR="${TRITON_REPO}/${MODEL_NAME}/1"
ENGINE_FILE="${MODEL_DIR}/model.plan"
DOWNLOAD_DIR="/tmp/gdino_download"
ONNX_FILE=""

IMG_H=544
IMG_W=960
SEQ_LEN=256

# ── Skip if engine already exists ──────────────────────────────────────
if [[ -f "$ENGINE_FILE" ]]; then
    echo "TensorRT engine already exists at ${ENGINE_FILE}, skipping."
    exit 0
fi

# ── Step 1: Download from NGC ──────────────────────────────────────────
ONNX_FILE="${DOWNLOAD_DIR}/grounding_dino_swin_tiny_commercial_deployable.onnx"

if [[ ! -f "$ONNX_FILE" ]]; then
    echo "=== Downloading Grounding DINO from NGC ==="
    mkdir -p "$DOWNLOAD_DIR"
    wget --content-disposition \
        'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/grounding_dino/grounding_dino_swin_tiny_commercial_deployable_v1.0/files?redirect=true&path=grounding_dino_swin_tiny_commercial_deployable.onnx' \
        -O "$ONNX_FILE"
fi
echo "ONNX model: $ONNX_FILE"

# ── Step 2: Inspect ONNX tensor names/shapes ──────────────────────────
echo ""
echo "=== Inspecting ONNX model ==="
python3 -c "
import onnx
m = onnx.load('${ONNX_FILE}')
print('INPUTS:')
for inp in m.graph.input:
    shape = [d.dim_value if d.dim_value > 0 else d.dim_param
             for d in inp.type.tensor_type.shape.dim]
    print(f'  {inp.name}: {shape}  dtype={inp.type.tensor_type.elem_type}')
print('OUTPUTS:')
for out in m.graph.output:
    shape = [d.dim_value if d.dim_value > 0 else d.dim_param
             for d in out.type.tensor_type.shape.dim]
    print(f'  {out.name}: {shape}  dtype={out.type.tensor_type.elem_type}')
print()
print('>> Update triton config.pbtxt files if tensor names differ from defaults.')
"

# ── Step 3: Convert to TensorRT ────────────────────────────────────────
echo ""
echo "=== Converting ONNX to TensorRT engine (FP16) ==="
mkdir -p "$MODEL_DIR"

trtexec \
    --onnx="$ONNX_FILE" \
    --saveEngine="$ENGINE_FILE" \
    --fp16 \
    --minShapes=inputs:1x3x${IMG_H}x${IMG_W},input_ids:1x${SEQ_LEN},attention_mask:1x${SEQ_LEN},position_ids:1x${SEQ_LEN},token_type_ids:1x${SEQ_LEN},text_token_mask:1x${SEQ_LEN}x${SEQ_LEN} \
    --optShapes=inputs:1x3x${IMG_H}x${IMG_W},input_ids:1x${SEQ_LEN},attention_mask:1x${SEQ_LEN},position_ids:1x${SEQ_LEN},token_type_ids:1x${SEQ_LEN},text_token_mask:1x${SEQ_LEN}x${SEQ_LEN} \
    --maxShapes=inputs:1x3x${IMG_H}x${IMG_W},input_ids:1x${SEQ_LEN},attention_mask:1x${SEQ_LEN},position_ids:1x${SEQ_LEN},token_type_ids:1x${SEQ_LEN},text_token_mask:1x${SEQ_LEN}x${SEQ_LEN} \

# ── Step 4: Install transformers if missing ────────────────────────────
python3 -c "import transformers" 2>/dev/null || pip install transformers

# ── Cleanup ────────────────────────────────────────────────────────────
rm -rf "$DOWNLOAD_DIR"

echo ""
echo "=== Grounding DINO setup complete ==="
echo "Engine: ${ENGINE_FILE}"
