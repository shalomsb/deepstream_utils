#!/usr/bin/env python3
"""Build the sahi_preprocess TensorRT plan.

Emits a tiny ONNX with 6 Slice nodes + Concat (NCHW slicing along H,W),
then runs trtexec to produce model.plan.

Input :  raw_frame   fp32 [1, 3, 720, 1280]   (RGB, [0,1])
Output:  tiles       fp32 [6, 3, 640, 640]
"""
import argparse
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


# Tile (y_start, x_start) — must match SAHI_TILE_OFFSETS in sahi_postprocess/model.py.
TILES_YX = [
    (0,  0),
    (0,  512),
    (0,  640),
    (80, 0),
    (80, 512),
    (80, 640),
]
TILE_H = TILE_W = 640
FRAME_H = 720
FRAME_W = 1280


def build_onnx(out_path: Path) -> None:
    initializers = []
    slice_outputs = []
    nodes = []

    for i, (y, x) in enumerate(TILES_YX):
        starts = numpy_helper.from_array(
            np.array([y, x], dtype=np.int64), name=f"starts_{i}"
        )
        ends = numpy_helper.from_array(
            np.array([y + TILE_H, x + TILE_W], dtype=np.int64), name=f"ends_{i}"
        )
        axes = numpy_helper.from_array(
            np.array([2, 3], dtype=np.int64), name=f"axes_{i}"
        )
        initializers += [starts, ends, axes]

        out_name = f"tile_{i}"
        nodes.append(helper.make_node(
            "Slice",
            inputs=["raw_frame", f"starts_{i}", f"ends_{i}", f"axes_{i}"],
            outputs=[out_name],
            name=f"slice_{i}",
        ))
        slice_outputs.append(out_name)

    nodes.append(helper.make_node(
        "Concat",
        inputs=slice_outputs,
        outputs=["tiles"],
        name="concat_tiles",
        axis=0,
    ))

    graph = helper.make_graph(
        nodes=nodes,
        name="sahi_preprocess",
        inputs=[helper.make_tensor_value_info(
            "raw_frame", TensorProto.FLOAT, [1, 3, FRAME_H, FRAME_W])],
        outputs=[helper.make_tensor_value_info(
            "tiles", TensorProto.FLOAT, [6, 3, TILE_H, TILE_W])],
        initializer=initializers,
    )
    model = helper.make_model(graph, producer_name="sahi_preprocess_builder",
                              opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, str(out_path))


def build_plan(onnx_path: Path, plan_path: Path) -> None:
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={plan_path}",
        # Slicing/concat — fp32 is fine, no perf gain from fp16 here.
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-plan", default="/triton/model_repo/sahi_preprocess/1/model.plan")
    ap.add_argument("--keep-onnx", action="store_true",
                    help="Keep the intermediate ONNX next to the plan for debugging.")
    args = ap.parse_args()

    plan_path = Path(args.out_plan)
    onnx_path = plan_path.with_suffix(".onnx") if args.keep_onnx else \
                Path(tempfile.mkstemp(suffix=".onnx")[1])

    print(f"Building ONNX -> {onnx_path}")
    build_onnx(onnx_path)
    print(f"Building TRT plan -> {plan_path}")
    build_plan(onnx_path, plan_path)
    if not args.keep_onnx:
        os.unlink(onnx_path)
    print("Done.")


if __name__ == "__main__":
    main()
