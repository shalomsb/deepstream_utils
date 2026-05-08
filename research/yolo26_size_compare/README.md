# YOLO26 model-size vs input-resolution comparison

Pure-Python research script. Runs six (model, imgsz) combinations on the same
image and reports detection counts, per-class breakdown, model size, and
latency.

## The six combinations

| # | Model    | imgsz | Effective use case                       |
|---|----------|-------|------------------------------------------|
| 1 | yolo26x  |  640  | Baseline (production default)            |
| 2 | yolo26x  | 1280  | Same model, more pixels                  |
| 3 | yolo26l  | 1280  | Smaller model, more pixels               |
| 4 | yolo26m  | 1280  | Medium model, more pixels                |
| 5 | yolo26s  | 1280  | Small model, more pixels                 |
| 6 | yolo26n  | 1280  | Nano model, more pixels                  |

The research question: **does going to higher resolution let you trade model
capacity for input pixels?** I.e. is yolo26{l,m,s,n} at 1280 better than
yolo26x at 640? Worse than yolo26x at 1280? Where's the sweet spot in the
parameter/latency vs detection-count plane?

## How to run

Inside the DeepStream container (CUDA available):

```bash
cd /workspace/research/yolo26_size_compare
pip install -r requirements.txt

python3 compare.py \
    --image /apps/deepstream-yolo26x-sahi-triton/scripts/sahi_test.png \
    --conf 0.10
```

First run downloads yolo26{n,s,m,l}.pt from the Ultralytics release (~500 MB
total). yolo26x.pt is already on the host at
`/apps/deepstream-yolo26x-triton/yolo26x.pt` and is reused if found.

## Output

```
output/
├── 1_yolo26x_640.jpg
├── 2_yolo26x_1280.jpg
├── 3_yolo26l_1280.jpg
├── 4_yolo26m_1280.jpg
├── 5_yolo26s_1280.jpg
├── 6_yolo26n_1280.jpg
├── grid_3x2.jpg
└── summary.csv
```

Console table includes parameters, weight file size, and latency for each
combination so you can read off the cost/benefit at a glance.
