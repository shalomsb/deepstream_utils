# YOLO26 size sweep × SAHI 320 (true 2× zoom)

Runs all 5 YOLO26 sizes on the same image using **real SAHI tiling**:
320×320 source tiles upsampled to 640×640 by Ultralytics — giving 2× effective
zoom on every object inside each tile.

For a 1280×720 frame with overlap_ratio=0.2, this yields ~15 tiles per frame.

## The five strategies

| # | Model    | Tile size in source | Model input | Effective zoom |
|---|----------|---------------------|-------------|----------------|
| 1 | yolo26x  | 320×320             | 640×640     | **2×**         |
| 2 | yolo26l  | 320×320             | 640×640     | 2×             |
| 3 | yolo26m  | 320×320             | 640×640     | 2×             |
| 4 | yolo26s  | 320×320             | 640×640     | 2×             |
| 5 | yolo26n  | 320×320             | 640×640     | 2×             |

## How to run

```bash
cd /workspace/research/yolo26_sahi320_size
pip install -r requirements.txt

python3 compare.py \
    --image /apps/deepstream-yolo26x-sahi-triton/scripts/sahi_test.png \
    --conf 0.10
```

This is the slowest of the four research scripts (~15 forward passes × 5
models). Expect ~30 s end to end on the configured GPU.

Reuses the weights cached by `research/yolo26_size_compare/` if present.
