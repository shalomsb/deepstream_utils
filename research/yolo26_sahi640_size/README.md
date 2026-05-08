# YOLO26 size sweep × SAHI 640 (production geometry)

Runs all 5 YOLO26 sizes on the same image using the **production SAHI tile
geometry** (6 tiles of 640×640 in source → 640×640 model input, NO zoom).

Geometry exactly matches `apps/deepstream-yolo26x-sahi-triton/` — same 6 tile
offsets `(0,0)(0,512)(0,640)(80,0)(80,512)(80,640)`. Hand-rolled GREEDYNMM
with IoS metric merges across tiles.

## The five strategies

| # | Model    | Tile size | Model input | Effective zoom |
|---|----------|-----------|-------------|----------------|
| 1 | yolo26x  | 640×640   | 640×640     | 1× (no zoom)   |
| 2 | yolo26l  | 640×640   | 640×640     | 1×             |
| 3 | yolo26m  | 640×640   | 640×640     | 1×             |
| 4 | yolo26s  | 640×640   | 640×640     | 1×             |
| 5 | yolo26n  | 640×640   | 640×640     | 1×             |

## How to run

```bash
cd /workspace/research/yolo26_sahi640_size
pip install -r requirements.txt

python3 compare.py \
    --image /apps/deepstream-yolo26x-sahi-triton/scripts/sahi_test.png \
    --conf 0.10
```

Reuses the weights cached by `research/yolo26_size_compare/` if present.
