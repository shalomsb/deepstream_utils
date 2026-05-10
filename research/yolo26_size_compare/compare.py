#!/usr/bin/env python3
"""YOLO26 model-size vs input-resolution comparison on a single image.

Runs six (model, imgsz) combinations:
    yolo26x @ 640
    yolo26x @ 1280
    yolo26l @ 1280
    yolo26m @ 1280
    yolo26s @ 1280
    yolo26n @ 1280

Reports detection counts (total, ≥0.5, ≥0.25), per-class breakdown,
parameter count, weight size, and latency. Saves an annotated image per
combination and a 3×2 grid composite.
"""
from __future__ import annotations

import argparse
import csv
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from tabulate import tabulate
from ultralytics import YOLO


COCO_LABELS = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]

PALETTE = [
    (0, 200, 0), (0, 200, 200), (200, 0, 0), (200, 0, 200), (0, 100, 255),
    (255, 100, 0), (100, 255, 0), (255, 200, 0), (255, 0, 200), (100, 0, 255),
]

# (model_name, imgsz). Model files are auto-downloaded by Ultralytics on first
# use unless we point at a local copy via --weights-dir.
COMBINATIONS: List[Tuple[str, int]] = [
    ("yolo26x", 640),
    ("yolo26x", 1280),
    ("yolo26l", 1280),
    ("yolo26m", 1280),
    ("yolo26s", 1280),
    ("yolo26n", 1280),
]


@dataclass
class Detection:
    x1: float; y1: float; x2: float; y2: float
    score: float
    cls: int


def _ultralytics_to_dets(results) -> List[Detection]:
    if results is None or len(results) == 0:
        return []
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return []
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(np.int32)
    return [Detection(float(x1), float(y1), float(x2), float(y2),
                      float(c), int(k))
            for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls)]


def draw(img: np.ndarray, dets: List[Detection], header: str) -> np.ndarray:
    out = img.copy()
    for d in dets:
        color = PALETTE[d.cls % len(PALETTE)]
        cv2.rectangle(out, (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2)), color, 2)
        name = COCO_LABELS[d.cls] if d.cls < len(COCO_LABELS) else f"id{d.cls}"
        text = f"{name} {d.score:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        x1 = max(0, int(d.x1)); y1 = max(0, int(d.y1))
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(out, text, (x1 + 1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.rectangle(out, (0, 0), (out.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(out, header, (8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return out


def make_grid(images: List[np.ndarray], cols: int) -> np.ndarray:
    h, w = images[0].shape[:2]
    cell_w, cell_h = w // 2, h // 2
    cells = [cv2.resize(img, (cell_w, cell_h)) for img in images]
    rows = []
    for r in range(0, len(cells), cols):
        rows.append(np.concatenate(cells[r:r + cols], axis=1))
    return np.concatenate(rows, axis=0)


def model_param_count(model: YOLO) -> int:
    try:
        return int(sum(p.numel() for p in model.model.parameters()))
    except Exception:
        return -1


def file_size_mb(p: Path) -> float:
    try:
        return round(p.stat().st_size / (1024 * 1024), 1)
    except Exception:
        return -1.0


def resolve_weights(name: str, search_dirs: List[Path]) -> str:
    """Return a local path if found (preferred), else the bare name so that
    Ultralytics downloads from its release."""
    fname = f"{name}.pt"
    for d in search_dirs:
        cand = d / fname
        if cand.is_file():
            return str(cand)
    return fname


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--weights-dir", action="append", default=None,
                    help="Directory(s) to look for *.pt before downloading. "
                         "Can repeat. Default: ./weights and "
                         "/apps/deepstream-yolo26x-triton/")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    out_dir = Path(args.output_dir) if args.output_dir else script_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_dirs = [Path(d) for d in (args.weights_dir or [])]
    weights_dirs += [
        script_dir / "weights",
        Path("/apps/deepstream-yolo26x-triton"),
    ]
    (script_dir / "weights").mkdir(exist_ok=True)

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Failed to read image: {args.image}")
    h, w = img.shape[:2]
    print(f"Image: {args.image}  ({w}×{h})")
    print(f"Conf:  {args.conf}")
    print(f"Weights search: {[str(p) for p in weights_dirs]}")
    print(f"Output: {out_dir}\n")

    summaries: list[dict] = []
    annotated: list[np.ndarray] = []

    for idx, (name, imgsz) in enumerate(COMBINATIONS, start=1):
        weight_path = resolve_weights(name, weights_dirs)
        label = f"{idx} {name}_{imgsz}"
        print(f"[{label}] loading {weight_path} …")
        try:
            model = YOLO(weight_path)
        except Exception as e:
            print(f"  ↳ skipped: {type(e).__name__}: {e}")
            continue

        params = model_param_count(model)
        # Resolve actual weight path Ultralytics ended up using
        actual_pt = Path(getattr(model, "ckpt_path", "") or weight_path)
        size_mb = file_size_mb(actual_pt)

        # Warmup so latency excludes one-time CUDA/cudnn init.
        _ = model.predict(img, imgsz=imgsz, conf=args.conf, verbose=False)

        t0 = time.perf_counter()
        res = model.predict(img, imgsz=imgsz, conf=args.conf, verbose=False)
        dt = time.perf_counter() - t0

        dets = _ultralytics_to_dets(res)
        n = len(dets)
        n50 = sum(1 for d in dets if d.score >= 0.5)
        n25 = sum(1 for d in dets if d.score >= 0.25)
        cls_counts = Counter(d.cls for d in dets)
        print(f"  ↳ {n} dets   ≥0.5={n50}  ≥0.25={n25}   "
              f"params={params/1e6:.1f}M  size={size_mb}MB   {dt:.3f}s")

        summaries.append({
            "strategy": f"{name}@{imgsz}",
            "n": n, "≥0.5": n50, "≥0.25": n25,
            "params_M": round(params / 1e6, 1),
            "weight_MB": size_mb,
            "latency_s": round(dt, 3),
            "per_class": dict(cls_counts),
        })

        annot = draw(img, dets, f"{label}  n={n}")
        annotated.append(annot)
        cv2.imwrite(str(out_dir / f"{idx}_{name}_{imgsz}.jpg"), annot)

        # Free GPU memory between models
        del model

    if annotated:
        # 3 cols × 2 rows for 6 cells. If fewer (some skipped), 2 cols.
        cols = 3 if len(annotated) >= 5 else 2
        # Pad with blank cells so the last row is complete
        while len(annotated) % cols != 0:
            annotated.append(np.zeros_like(annotated[0]))
        cv2.imwrite(str(out_dir / f"grid_{cols}x{len(annotated)//cols}.jpg"),
                    make_grid(annotated, cols=cols))

    print("\n" + tabulate(
        [[s["strategy"], s["n"], s["≥0.5"], s["≥0.25"],
          s["params_M"], s["weight_MB"], s["latency_s"]]
         for s in summaries],
        headers=["strategy", "n", "≥0.5", "≥0.25", "params_M", "weight_MB", "latency_s"],
    ))

    # CSV
    all_classes = sorted({c for s in summaries for c in s["per_class"]})
    fieldnames = ["strategy", "n", "≥0.5", "≥0.25", "params_M", "weight_MB", "latency_s"] + \
                 [f"cls_{c}_{COCO_LABELS[c] if c < len(COCO_LABELS) else c}" for c in all_classes]
    with open(out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for s in summaries:
            row = {k: s[k] for k in ("strategy", "n", "≥0.5", "≥0.25",
                                     "params_M", "weight_MB", "latency_s")}
            for c in all_classes:
                key = f"cls_{c}_{COCO_LABELS[c] if c < len(COCO_LABELS) else c}"
                row[key] = s["per_class"].get(c, 0)
            w.writerow(row)

    print(f"\nWrote {len(summaries)} annotated images, grid, and summary.csv to {out_dir}")


if __name__ == "__main__":
    main()
