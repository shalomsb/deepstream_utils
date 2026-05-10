#!/usr/bin/env python3
"""YOLO26 size sweep × SAHI 320 (real 2× zoom).

Runs each of yolo26{x,l,m,s,n} on the same image using REAL SAHI tiling:
320×320 source tiles upsampled to 640×640 by Ultralytics. ~15 tiles per
1280×720 frame at overlap_ratio=0.2. Hand-rolled GREEDYNMM with IoS metric
merges across tiles.
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

MODELS = ["yolo26x", "yolo26l", "yolo26m", "yolo26s", "yolo26n"]
TILE_SIZE = 320
OVERLAP_RATIO = 0.2
MODEL_INPUT = 640


@dataclass
class Detection:
    x1: float; y1: float; x2: float; y2: float
    score: float
    cls: int


# ────────────────────────────── tile grid ──────────────────────────────

def slice_grid(img_w: int, img_h: int, tile_w: int, tile_h: int,
               overlap_ratio: float = 0.2) -> List[Tuple[int, int, int, int]]:
    step_x = max(1, int(round(tile_w * (1 - overlap_ratio))))
    step_y = max(1, int(round(tile_h * (1 - overlap_ratio))))

    def edges(total: int, tile: int, step: int) -> list[int]:
        if total <= tile:
            return [0]
        starts = list(range(0, total - tile + 1, step))
        last = total - tile
        if starts[-1] != last:
            starts.append(last)
        return starts

    xs = edges(img_w, tile_w, step_x)
    ys = edges(img_h, tile_h, step_y)
    return [(x, y, x + tile_w, y + tile_h) for y in ys for x in xs]


# ────────────────────────────── GREEDYNMM ──────────────────────────────

def greedy_nmm_ios(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray,
                   threshold: float = 0.5,
                   class_agnostic: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SAHI's greedy non-maximum merging with IoS = inter / min(area)."""
    n = boxes.shape[0]
    if n == 0:
        return (np.zeros((0, 4), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.int32))
    boxes = boxes.astype(np.float32, copy=True)
    scores = scores.astype(np.float32, copy=False)
    classes = classes.astype(np.int32, copy=False)
    areas = (boxes[:, 2] - boxes[:, 0]).clip(min=0) * (boxes[:, 3] - boxes[:, 1]).clip(min=0)
    order = np.argsort(-scores, kind="stable")
    suppressed = np.zeros(n, dtype=bool)
    keep: list[int] = []
    for rank in range(n):
        i = int(order[rank])
        if suppressed[i]:
            continue
        keep.append(i)
        rest = order[rank + 1:]
        rest = rest[~suppressed[rest]]
        if rest.size == 0:
            continue
        if not class_agnostic:
            rest = rest[classes[rest] == classes[i]]
            if rest.size == 0:
                continue
        bi = boxes[i]; bj = boxes[rest]
        inter = (np.maximum(0, np.minimum(bi[2], bj[:, 2]) - np.maximum(bi[0], bj[:, 0])) *
                 np.maximum(0, np.minimum(bi[3], bj[:, 3]) - np.maximum(bi[1], bj[:, 1])))
        ios = inter / np.maximum(np.minimum(areas[i], areas[rest]), 1e-6)
        match = ios > threshold
        if not match.any():
            continue
        matched = rest[match]
        all_idx = np.concatenate([[i], matched])
        boxes[i, 0] = boxes[all_idx, 0].min()
        boxes[i, 1] = boxes[all_idx, 1].min()
        boxes[i, 2] = boxes[all_idx, 2].max()
        boxes[i, 3] = boxes[all_idx, 3].max()
        suppressed[matched] = True
    keep_arr = np.array(keep, dtype=np.int64)
    return boxes[keep_arr], scores[keep_arr], classes[keep_arr]


# ────────────────────────────── SAHI runner ──────────────────────────────

def run_sahi_tiled(model: YOLO, img: np.ndarray,
                   tiles_xyxy: List[Tuple[int, int, int, int]],
                   imgsz: int, conf: float, ios_thr: float = 0.5,
                   class_agnostic: bool = False) -> List[Detection]:
    h, w = img.shape[:2]
    all_b: list[np.ndarray] = []
    all_s: list[np.ndarray] = []
    all_c: list[np.ndarray] = []
    for (x1, y1, x2, y2) in tiles_xyxy:
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w, x2), min(h, y2)
        if x2c <= x1c or y2c <= y1c:
            continue
        crop = img[y1c:y2c, x1c:x2c]
        res = model.predict(crop, imgsz=imgsz, conf=conf, verbose=False)
        if not res or res[0].boxes is None or len(res[0].boxes) == 0:
            continue
        b = res[0].boxes.xyxy.cpu().numpy().astype(np.float32)
        s = res[0].boxes.conf.cpu().numpy().astype(np.float32)
        c = res[0].boxes.cls.cpu().numpy().astype(np.int32)
        b[:, [0, 2]] += x1c
        b[:, [1, 3]] += y1c
        all_b.append(b); all_s.append(s); all_c.append(c)
    if not all_b:
        return []
    boxes, scores, classes = greedy_nmm_ios(
        np.concatenate(all_b), np.concatenate(all_s), np.concatenate(all_c),
        threshold=ios_thr, class_agnostic=class_agnostic,
    )
    return [Detection(*b.tolist(), score=float(s), cls=int(c))
            for b, s, c in zip(boxes, scores, classes)]


# ────────────────────────────── drawing/utils ──────────────────────────────

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
    cell = (w // 2, h // 2)
    cells = [cv2.resize(i, cell) for i in images]
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
    fname = f"{name}.pt"
    for d in search_dirs:
        cand = d / fname
        if cand.is_file():
            return str(cand)
    return fname


# ────────────────────────────── main ──────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--ios", type=float, default=0.5)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--weights-dir", action="append", default=None)
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    out_dir = Path(args.output_dir) if args.output_dir else script_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    weights_dirs = [Path(d) for d in (args.weights_dir or [])]
    weights_dirs += [
        script_dir / "weights",
        script_dir.parent / "yolo26_size_compare" / "weights",
        Path("/apps/deepstream-yolo26x-triton"),
    ]
    (script_dir / "weights").mkdir(exist_ok=True)

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Failed to read image: {args.image}")
    h, w = img.shape[:2]
    tiles = slice_grid(w, h, TILE_SIZE, TILE_SIZE, OVERLAP_RATIO)
    print(f"Image: {args.image}  ({w}×{h})")
    print(f"SAHI geometry: {len(tiles)} tiles of {TILE_SIZE}×{TILE_SIZE} "
          f"→ {MODEL_INPUT}×{MODEL_INPUT} model input "
          f"({MODEL_INPUT // TILE_SIZE}× zoom)")
    print(f"Conf: {args.conf}   IoS: {args.ios}\n")

    summaries: list[dict] = []
    annotated: list[np.ndarray] = []

    for idx, name in enumerate(MODELS, start=1):
        wp = resolve_weights(name, weights_dirs)
        label = f"{idx} {name}_sahi320"
        print(f"[{label}] loading {wp} …")
        try:
            model = YOLO(wp)
        except Exception as e:
            print(f"  ↳ skipped: {type(e).__name__}: {e}")
            continue

        params = model_param_count(model)
        actual_pt = Path(getattr(model, "ckpt_path", "") or wp)
        size_mb = file_size_mb(actual_pt)

        # Warmup
        _ = model.predict(img[:TILE_SIZE, :TILE_SIZE],
                          imgsz=MODEL_INPUT, conf=args.conf, verbose=False)

        t0 = time.perf_counter()
        dets = run_sahi_tiled(model, img, tiles, imgsz=MODEL_INPUT,
                              conf=args.conf, ios_thr=args.ios)
        dt = time.perf_counter() - t0

        n = len(dets)
        n50 = sum(1 for d in dets if d.score >= 0.5)
        n25 = sum(1 for d in dets if d.score >= 0.25)
        cls_counts = Counter(d.cls for d in dets)
        print(f"  ↳ {n} dets   ≥0.5={n50}  ≥0.25={n25}   "
              f"params={params/1e6:.1f}M  size={size_mb}MB   {dt:.3f}s")

        summaries.append({
            "strategy": f"{name} sahi320",
            "n": n, "≥0.5": n50, "≥0.25": n25,
            "params_M": round(params / 1e6, 1),
            "weight_MB": size_mb,
            "latency_s": round(dt, 3),
            "per_class": dict(cls_counts),
        })
        annot = draw(img, dets, f"{label}  n={n}")
        annotated.append(annot)
        cv2.imwrite(str(out_dir / f"{idx}_{name}_sahi320.jpg"), annot)
        del model

    if annotated:
        cols = 3
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
    print(f"\nWrote {len(summaries)} annotated images, grid, summary.csv → {out_dir}")


if __name__ == "__main__":
    main()
