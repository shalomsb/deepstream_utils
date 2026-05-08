# YOLO26X inference-strategy comparison

Pure-Python research script — no DeepStream, no Triton, no Docker round-trips.
Runs four inference strategies on a single image and reports detection counts,
per-class breakdown, and latency.

## The four strategies

| # | Strategy                              | Effective zoom on small objects | Inference cost      |
|---|---------------------------------------|---------------------------------|---------------------|
| 1 | `yolo26x` default, `imgsz=640`        | 1× (1280×720 letterboxed → 640) | 1 forward pass      |
| 2 | `yolo26x` at `imgsz=1280`             | 2× (full frame at higher res)   | 1 forward pass (4× FLOPs) |
| 3 | 6 SAHI tiles, 640×640 in source       | 1× (tile size = model input)    | 6 forward passes    |
| 4 | SAHI tiles, 320×320 in source → 640   | **2× per tile**                 | ~15 forward passes  |

Strategy 4 is the configuration "real" SAHI papers use. Strategy 3 is what our
production DeepStream/Triton ensemble currently does — kept here for like-for-like
comparison.

The hand-rolled cross-tile merge is **GREEDYNMM with IoS metric** (the SAHI
default), ported from the Triton Python backend at
`triton/model_repo/sahi_postprocess/1/model.py`. The `sahi` PyPI package is run
as a cross-check on strategy 4 only — both implementations should produce
equivalent counts ± a few from tie-breaking.

## How to run

Inside the DeepStream container (CUDA + the yolo26x .pt weights both available):

```bash
cd /workspace/research/yolo26x_sahi_compare
pip install -r requirements.txt

python3 compare.py \
    --image /apps/deepstream-yolo26x-sahi-triton/scripts/sahi_test.png \
    --weights /mnt/desktop_data/nvidia/yolo_inference/yolo26x.pt \
    --conf 0.10
```

## Output

```
output/
├── 1_default_640.jpg        annotated image, strategy 1
├── 2_upscale_1280.jpg       strategy 2
├── 3_sahi_640.jpg           strategy 3 (production geometry)
├── 4_sahi_320.jpg           strategy 4 (real SAHI, 2× zoom)
├── 4_sahi_320_pkg.jpg       sahi-package cross-check of strategy 4
├── grid_2x2.jpg             composite of strategies 1–4 for visual diff
└── summary.csv              per-class counts + latency per strategy
```

Console prints a tabulated summary like:

```
Strategy            n   conf>=0.5   conf>=0.25   latency
1 default_640      40           5           18      0.04s
2 upscale_1280     ...
3 sahi_640         85          ...
4 sahi_320         XX
4 sahi_pkg         XX (cross-check delta)
```

## What we expect to learn

The research question is whether **smaller tiles** (#4) actually help recover
the tiny distant vehicles in the upper-left highway curve of `sahi_test.png`,
where #1 / #2 / #3 all came up empty in earlier experiments.

If #4 materially beats #3 on classes 2 (car) and 7 (truck) at low confidence,
the next step is to port the 320-pixel tile geometry back to the production
SAHI ensemble (rebuild `sahi_preprocess.plan` for ~15 tiles, rebuild
`yolo26x_b6` → `yolo26x_b15` for the larger batch). If #4 doesn't help, the
conclusion is "the model itself can't detect 5-px objects" and the next
experiment should swap to a higher-resolution backbone (yolo26x6 etc.).
