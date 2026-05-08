#!/usr/bin/env python3
"""Standalone Triton client to test detections on a single image.

Supports two pipelines:
  --pipeline sahi    -> sahi_pipeline ensemble  (resize 1280x720 + 6 tiles + GREEDYNMM)
  --pipeline simple  -> yolo26x model           (single 640x640 letterboxed inference)

Usage:
    # Terminal 1 (inside the DS container):
    tritonserver --model-repository=/triton/model_repo

    # Terminal 2 (inside the same container):
    pip install tritonclient[http]   # if not already installed
    python3 scripts/test_image.py --pipeline sahi   --image scripts/sahi_test.png
    python3 scripts/test_image.py --pipeline simple --image scripts/sahi_test.png
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import tritonclient.http as httpclient


SAHI_W, SAHI_H = 1280, 720
NET = 640


# ────────────────────────── preprocessing ──────────────────────────

def preprocess_sahi(img_bgr: np.ndarray) -> np.ndarray:
    """BGR uint8 → fp32 [1,3,720,1280] in [0,1] RGB NCHW (no aspect maintain)."""
    resized = cv2.resize(img_bgr, (SAHI_W, SAHI_H), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return chw[np.newaxis, ...]


def preprocess_simple(img_bgr: np.ndarray):
    """BGR uint8 → fp32 [1,3,640,640] letterboxed RGB NCHW.

    Returns (tensor, scale, pad_x, pad_y) for un-letterboxing the output.
    Mirrors nvinferserver's maintain_aspect_ratio=1 + symmetric_padding=1.
    """
    h, w = img_bgr.shape[:2]
    scale = min(NET / w, NET / h)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.zeros((NET, NET, 3), dtype=np.uint8)
    pad_x = (NET - new_w) // 2
    pad_y = (NET - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    return chw[np.newaxis, ...], scale, pad_x, pad_y


# ────────────────────────── inference ──────────────────────────

def infer_sahi(client, img_bgr, conf_threshold=0.10):
    inp = preprocess_sahi(img_bgr)
    triton_in = httpclient.InferInput("raw_frame", inp.shape, "FP32")
    triton_in.set_data_from_numpy(inp)
    outs = [httpclient.InferRequestedOutput(n)
            for n in ("boxes", "scores", "classes", "num_dets")]
    r = client.infer(model_name="sahi_pipeline", inputs=[triton_in], outputs=outs)
    n = int(r.as_numpy("num_dets").flatten()[0])
    boxes = r.as_numpy("boxes").reshape(-1, 4)[:n]
    scores = r.as_numpy("scores").flatten()[:n]
    classes = r.as_numpy("classes").flatten()[:n].astype(np.int32)

    # Client-side conf filter on top of the postprocess's broad floor.
    keep = scores > conf_threshold
    boxes, scores, classes = boxes[keep], scores[keep], classes[keep]

    # Scale 1280x720 inference space → original image dims.
    h, w = img_bgr.shape[:2]
    sx, sy = w / SAHI_W, h / SAHI_H
    if len(boxes):
        boxes = boxes.copy()
        boxes[:, [0, 2]] *= sx
        boxes[:, [1, 3]] *= sy
    return boxes, scores, classes


def infer_simple(client, img_bgr, conf_threshold=0.10):
    inp, scale, pad_x, pad_y = preprocess_simple(img_bgr)
    triton_in = httpclient.InferInput("images", inp.shape, "FP32")
    triton_in.set_data_from_numpy(inp)
    outs = [httpclient.InferRequestedOutput("output0")]
    r = client.infer(model_name="yolo26x", inputs=[triton_in], outputs=outs)
    raw = r.as_numpy("output0")          # [1, 300, 6] or [300, 6]
    if raw.ndim == 3:
        raw = raw[0]
    conf = raw[:, 4]
    mask = conf > conf_threshold
    raw = raw[mask]
    if raw.size == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int32)

    # Un-letterbox network coords → original image coords.
    boxes = raw[:, :4].copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    return boxes, raw[:, 4], raw[:, 5].astype(np.int32)


# ────────────────────────── drawing ──────────────────────────

_PALETTE = [(0, 200, 0), (0, 200, 200), (200, 0, 0), (200, 0, 200), (0, 100, 255),
            (255, 100, 0), (100, 255, 0), (255, 200, 0), (255, 0, 200), (100, 0, 255)]


def load_labels(path: Path) -> list:
    if not path.exists():
        return []
    with open(path) as f:
        return [ln.strip() for ln in f if ln.strip()]


def draw(img, boxes, scores, classes, labels, header_text=None):
    for (x1, y1, x2, y2), s, c in zip(boxes, scores, classes):
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        color = _PALETTE[int(c) % len(_PALETTE)]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        name = labels[int(c)] if int(c) < len(labels) else f"id{int(c)}"
        text = f"{name} {s:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 4), (x1 + tw + 2, y1), color, -1)
        cv2.putText(img, text, (x1 + 1, y1 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if header_text:
        cv2.rectangle(img, (0, 0), (img.shape[1], 28), (0, 0, 0), -1)
        cv2.putText(img, header_text, (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


# ────────────────────────── entry ──────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--pipeline", choices=("sahi", "simple"), default="sahi")
    ap.add_argument("--output", default=None)
    ap.add_argument("--url", default="localhost:8000")
    ap.add_argument("--conf", type=float, default=0.10,
                    help="Client-side confidence threshold (default 0.10)")
    ap.add_argument("--labels", default="/models/yolo26/labels.txt")
    args = ap.parse_args()

    img_path = Path(args.image)
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise SystemExit(f"Failed to read image: {img_path}")

    client = httpclient.InferenceServerClient(url=args.url, verbose=False)
    target_model = "sahi_pipeline" if args.pipeline == "sahi" else "yolo26x"
    if not client.is_model_ready(target_model):
        raise SystemExit(f"Model '{target_model}' not ready on {args.url}.")

    if args.pipeline == "sahi":
        boxes, scores, classes = infer_sahi(client, img_bgr, conf_threshold=args.conf)
    else:
        boxes, scores, classes = infer_simple(client, img_bgr, conf_threshold=args.conf)

    n = len(boxes)
    print(f"[{args.pipeline}] Detections: {n}")
    for i in range(n):
        x1, y1, x2, y2 = boxes[i]
        print(f"  [{i:3d}] cls={int(classes[i]):2d} conf={scores[i]:.3f} "
              f"xyxy=({x1:7.1f},{y1:7.1f},{x2:7.1f},{y2:7.1f})")

    labels = load_labels(Path(args.labels))
    out_img = img_bgr.copy()
    draw(out_img, boxes, scores, classes, labels,
         header_text=f"{args.pipeline.upper()}  n={n}")

    out_path = Path(args.output) if args.output \
        else img_path.with_name(f"{img_path.stem}_{args.pipeline}.jpg")
    cv2.imwrite(str(out_path), out_img)
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
