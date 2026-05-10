#!/usr/bin/env python3
"""YOLO26X inference-strategy comparison on a single image.

Four strategies, side-by-side:
  1. default_640   - model.predict(imgsz=640) on the full image
  2. upscale_1280  - model.predict(imgsz=1280) on the full image
  3. sahi_640      - 6 SAHI tiles 640x640 in source (= production geometry, no zoom)
  4. sahi_320      - SAHI tiles 320x320 in source -> upsampled to 640 (2x zoom)

Plus a cross-check of #4 using the `sahi` PyPI package.
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


# COCO labels (yolo26x default head)
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

# Production SAHI tile offsets — must match
# apps/deepstream-yolo26x-sahi-triton/scripts/build_sahi_preprocess_plan.py.
# Tiles are (y_start, x_start) into a 1280x720 frame; tile size 640x640.
SAHI_640_TILES_YX = [
    (0,  0), (0,  512), (0,  640),
    (80, 0), (80, 512), (80, 640),
]

PALETTE = [
    (0, 200, 0), (0, 200, 200), (200, 0, 0), (200, 0, 200), (0, 100, 255),
    (255, 100, 0), (100, 255, 0), (255, 200, 0), (255, 0, 200), (100, 0, 255),
]


@dataclass
class Detection:
    x1: float; y1: float; x2: float; y2: float
    score: float
    cls: int


# ────────────────────────────── GREEDYNMM ──────────────────────────────

def greedy_nmm_ios(boxes: np.ndarray, scores: np.ndarray, classes: np.ndarray,
                   threshold: float = 0.5,
                   class_agnostic: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SAHI's greedy non-maximum merging with IoS metric (numpy port).

    Numpy port of the torch implementation in
    triton/model_repo/sahi_postprocess/1/model.py:_greedy_nmm_ios.

    Args:
        boxes:   [N, 4] xyxy
        scores:  [N]
        classes: [N] int
    Returns:
        merged_boxes [M, 4], merged_scores [M], merged_classes [M] (M <= N).
    """
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

        bi = boxes[i]
        bj = boxes[rest]
        xx1 = np.maximum(bi[0], bj[:, 0])
        yy1 = np.maximum(bi[1], bj[:, 1])
        xx2 = np.minimum(bi[2], bj[:, 2])
        yy2 = np.minimum(bi[3], bj[:, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        min_area = np.minimum(areas[i], areas[rest])
        ios = inter / np.maximum(min_area, 1e-6)
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


# ────────────────────────────── tile geometry ──────────────────────────────

def slice_grid(img_w: int, img_h: int, tile_w: int, tile_h: int,
               overlap_ratio: float = 0.2) -> List[Tuple[int, int, int, int]]:
    """Return a list of (x1,y1,x2,y2) tile rects covering the full frame.

    Step = round(tile * (1 - overlap_ratio)). The last column/row is shifted
    so the right/bottom edge is fully covered (matches SAHI behavior).
    """
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
    rects = []
    for y in ys:
        for x in xs:
            rects.append((x, y, x + tile_w, y + tile_h))
    return rects


# ────────────────────────────── strategies ──────────────────────────────

def _ultralytics_to_dets(results, offset_x: float = 0.0, offset_y: float = 0.0,
                         scale_x: float = 1.0, scale_y: float = 1.0) -> List[Detection]:
    """Pull boxes/scores/classes off an Ultralytics Results object and shift
    into image-global coords."""
    dets: list[Detection] = []
    if results is None or len(results) == 0:
        return dets
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return dets
    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy().astype(np.int32)
    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i]
        dets.append(Detection(
            x1=float(x1) * scale_x + offset_x,
            y1=float(y1) * scale_y + offset_y,
            x2=float(x2) * scale_x + offset_x,
            y2=float(y2) * scale_y + offset_y,
            score=float(conf[i]),
            cls=int(cls[i]),
        ))
    return dets


def run_default_640(model: YOLO, img: np.ndarray, conf: float) -> List[Detection]:
    res = model.predict(img, imgsz=640, conf=conf, verbose=False)
    return _ultralytics_to_dets(res)


def run_upscale_1280(model: YOLO, img: np.ndarray, conf: float) -> List[Detection]:
    res = model.predict(img, imgsz=1280, conf=conf, verbose=False)
    return _ultralytics_to_dets(res)


def _run_tiled(model: YOLO, img: np.ndarray,
               tiles_xyxy: List[Tuple[int, int, int, int]],
               imgsz: int, conf: float, ios_threshold: float,
               class_agnostic: bool) -> List[Detection]:
    """Generic SAHI-style tiled runner with GREEDYNMM merge.

    For each tile rect (in source-image coords), crop, run model.predict,
    translate boxes back to image coords, then merge across tiles.
    """
    h, w = img.shape[:2]
    all_boxes: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_classes: list[np.ndarray] = []

    for (x1, y1, x2, y2) in tiles_xyxy:
        x1c, y1c = max(0, x1), max(0, y1)
        x2c, y2c = min(w, x2), min(h, y2)
        if x2c <= x1c or y2c <= y1c:
            continue
        crop = img[y1c:y2c, x1c:x2c]

        # Ultralytics resizes the crop internally to imgsz×imgsz with letterbox,
        # then returns boxes in CROP-pixel coords. We just need to translate
        # those back to the source image by adding (x1c, y1c).
        res = model.predict(crop, imgsz=imgsz, conf=conf, verbose=False)
        if not res or res[0].boxes is None or len(res[0].boxes) == 0:
            continue
        b = res[0].boxes.xyxy.cpu().numpy().astype(np.float32)
        s = res[0].boxes.conf.cpu().numpy().astype(np.float32)
        c = res[0].boxes.cls.cpu().numpy().astype(np.int32)
        b[:, [0, 2]] += x1c
        b[:, [1, 3]] += y1c
        all_boxes.append(b)
        all_scores.append(s)
        all_classes.append(c)

    if not all_boxes:
        return []

    boxes = np.concatenate(all_boxes, axis=0)
    scores = np.concatenate(all_scores, axis=0)
    classes = np.concatenate(all_classes, axis=0)
    boxes, scores, classes = greedy_nmm_ios(
        boxes, scores, classes, threshold=ios_threshold, class_agnostic=class_agnostic
    )
    return [Detection(*b.tolist(), score=float(s), cls=int(c))
            for b, s, c in zip(boxes, scores, classes)]


def run_sahi_640(model: YOLO, img: np.ndarray, conf: float,
                 ios_threshold: float = 0.5,
                 class_agnostic: bool = False) -> List[Detection]:
    """Strategy 3 — production geometry (6 tiles, 640×640 source, 640 input)."""
    # SAHI_640_TILES_YX is (y_start, x_start) — convert to (x1,y1,x2,y2).
    tiles = [(x, y, x + 640, y + 640) for (y, x) in SAHI_640_TILES_YX]
    return _run_tiled(model, img, tiles, imgsz=640, conf=conf,
                      ios_threshold=ios_threshold, class_agnostic=class_agnostic)


def run_sahi_320(model: YOLO, img: np.ndarray, conf: float,
                 ios_threshold: float = 0.5,
                 class_agnostic: bool = False) -> List[Detection]:
    """Strategy 4 — 320×320 source tiles upsampled to 640×640 (2× zoom)."""
    h, w = img.shape[:2]
    tiles = slice_grid(w, h, tile_w=320, tile_h=320, overlap_ratio=0.2)
    return _run_tiled(model, img, tiles, imgsz=640, conf=conf,
                      ios_threshold=ios_threshold, class_agnostic=class_agnostic)


def run_sahi_pkg(weights_path: str, img_path: str, conf: float
                 ) -> Tuple[List[Detection], str]:
    """Strategy-4 cross-check via the `sahi` PyPI package. Returns dets + version."""
    import sahi
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction

    detection_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=weights_path,
        confidence_threshold=conf,
        device="cuda:0",
    )
    result = get_sliced_prediction(
        image=img_path,
        detection_model=detection_model,
        slice_height=320, slice_width=320,
        overlap_height_ratio=0.2, overlap_width_ratio=0.2,
        postprocess_type="GREEDYNMM",
        postprocess_match_metric="IOS",
        postprocess_match_threshold=0.5,
        postprocess_class_agnostic=False,
        verbose=0,
    )
    dets: list[Detection] = []
    for p in result.object_prediction_list:
        bb = p.bbox
        dets.append(Detection(
            x1=bb.minx, y1=bb.miny, x2=bb.maxx, y2=bb.maxy,
            score=p.score.value,
            cls=p.category.id,
        ))
    return dets, getattr(sahi, "__version__", "?")


# ────────────────────────────── drawing ──────────────────────────────

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


def make_grid(images: List[np.ndarray], titles: List[str], cols: int = 2) -> np.ndarray:
    h, w = images[0].shape[:2]
    target = (w // 2, h // 2)
    cells = [cv2.resize(img, target) for img in images]
    rows = []
    for r in range(0, len(cells), cols):
        rows.append(np.concatenate(cells[r:r + cols], axis=1))
    return np.concatenate(rows, axis=0)


# ────────────────────────────── summary ──────────────────────────────

def summarize(name: str, dets: List[Detection], elapsed_s: float) -> dict:
    n = len(dets)
    n50 = sum(1 for d in dets if d.score >= 0.5)
    n25 = sum(1 for d in dets if d.score >= 0.25)
    cls_counts = Counter(d.cls for d in dets)
    return {
        "strategy": name,
        "n": n,
        "conf>=0.5": n50,
        "conf>=0.25": n25,
        "latency_s": round(elapsed_s, 3),
        "per_class": dict(cls_counts),
    }


def write_csv(rows: List[dict], path: Path) -> None:
    all_classes = sorted({c for r in rows for c in r["per_class"]})
    fieldnames = ["strategy", "n", "conf>=0.5", "conf>=0.25", "latency_s"] + \
                 [f"cls_{c}_{COCO_LABELS[c] if c < len(COCO_LABELS) else c}" for c in all_classes]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            row = {k: r[k] for k in ("strategy", "n", "conf>=0.5", "conf>=0.25", "latency_s")}
            for c in all_classes:
                key = f"cls_{c}_{COCO_LABELS[c] if c < len(COCO_LABELS) else c}"
                row[key] = r["per_class"].get(c, 0)
            w.writerow(row)


# ────────────────────────────── main ──────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--weights", required=True,
                    help="Path to yolo26x.pt")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--ios", type=float, default=0.5,
                    help="GREEDYNMM IoS threshold")
    ap.add_argument("--output-dir", default=None,
                    help="Defaults to ./output relative to this script")
    ap.add_argument("--skip-pkg-crosscheck", action="store_true",
                    help="Skip the sahi-package cross-check on strategy 4")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    out_dir = Path(args.output_dir) if args.output_dir else script_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Failed to read image: {args.image}")
    h, w = img.shape[:2]
    print(f"Image: {args.image}  ({w}×{h})")
    print(f"Weights: {args.weights}")
    print(f"Conf: {args.conf}   IoS: {args.ios}")
    print(f"Output: {out_dir}\n")

    print("Loading model…")
    model = YOLO(args.weights)
    # Warm up CUDA — first call includes engine compile etc.
    _ = model.predict(img, imgsz=640, conf=args.conf, verbose=False)

    summaries = []
    annotated = []
    titles = []

    def run_and_record(name: str, fn, label: str, save_name: str):
        t0 = time.perf_counter()
        dets = fn()
        dt = time.perf_counter() - t0
        print(f"[{name}] {len(dets)} dets   {dt:.3f}s")
        summaries.append(summarize(name, dets, dt))
        annot = draw(img, dets, f"{label}  n={len(dets)}")
        cv2.imwrite(str(out_dir / save_name), annot)
        annotated.append(annot)
        titles.append(label)
        return dets

    run_and_record("default_640", lambda: run_default_640(model, img, args.conf),
                   "1 default_640", "1_default_640.jpg")
    run_and_record("upscale_1280", lambda: run_upscale_1280(model, img, args.conf),
                   "2 upscale_1280", "2_upscale_1280.jpg")
    run_and_record("sahi_640", lambda: run_sahi_640(model, img, args.conf, args.ios),
                   "3 sahi_640", "3_sahi_640.jpg")
    run_and_record("sahi_320", lambda: run_sahi_320(model, img, args.conf, args.ios),
                   "4 sahi_320", "4_sahi_320.jpg")

    if not args.skip_pkg_crosscheck:
        try:
            t0 = time.perf_counter()
            pkg_dets, pkg_ver = run_sahi_pkg(args.weights, args.image, args.conf)
            dt = time.perf_counter() - t0
            print(f"[sahi_pkg v{pkg_ver}] {len(pkg_dets)} dets   {dt:.3f}s")
            summaries.append(summarize(f"sahi_pkg v{pkg_ver}", pkg_dets, dt))
            annot = draw(img, pkg_dets, f"4 sahi_pkg  n={len(pkg_dets)}")
            cv2.imwrite(str(out_dir / "4_sahi_320_pkg.jpg"), annot)
        except Exception as e:
            print(f"[sahi_pkg] skipped — {type(e).__name__}: {e}")

    grid = make_grid(annotated[:4], titles[:4], cols=2)
    cv2.imwrite(str(out_dir / "grid_2x2.jpg"), grid)

    print("\n" + tabulate(
        [[s["strategy"], s["n"], s["conf>=0.5"], s["conf>=0.25"], s["latency_s"]]
         for s in summaries],
        headers=["strategy", "n", "≥0.5", "≥0.25", "latency_s"],
    ))

    write_csv(summaries, out_dir / "summary.csv")
    print(f"\nWrote {len(annotated)} annotated images, grid, and summary.csv to {out_dir}")


if __name__ == "__main__":
    main()
