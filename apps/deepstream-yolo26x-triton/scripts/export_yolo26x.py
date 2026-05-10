#!/usr/bin/env python3
"""Export Ultralytics YOLO26X to ONNX with a dynamic batch axis.

Run inside the container after `pip install ultralytics`:

    python3 scripts/export_yolo26x.py --weights yolo26x.pt \
        --out /triton/model_repo/yolo26x/1/model.onnx
"""
import argparse
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="yolo26x.pt",
                    help="Path to .pt weights (Ultralytics will download if missing)")
    ap.add_argument("--out", default="/triton/model_repo/yolo26x/1/model.onnx",
                    help="Destination path for the exported ONNX")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.weights)
    # dynamic=True -> dynamic batch axis on input/output. simplify=True runs onnx-simplifier.
    exported = model.export(format="onnx", imgsz=args.imgsz, dynamic=True,
                            opset=args.opset, simplify=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(exported), str(out_path))
    print(f"Exported -> {out_path}")


if __name__ == "__main__":
    main()
