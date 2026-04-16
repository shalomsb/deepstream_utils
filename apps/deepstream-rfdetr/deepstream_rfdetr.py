#!/usr/bin/env python3

import math
import os
import sys
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
from common.platform_info import PlatformInfo
from common.FPS import PERF_DATA
from ds_pipeline import (
    Logger,
    create_pipeline, attach_sources, create_streammux,
    create_pgie, create_tracker, create_nvvidconv, create_nvosd, create_sink,
    create_tiler, create_queue, create_rtsp_output_bin, start_rtsp_server,
    run_pipeline, link_chain,
    get_batch_meta, iter_frames,
)
from config import Config
from callbacks import pgie_src_probe, osd_probe

CONFIGS = {
    "nano":   "config_nano.yaml",
    "small":  "config_small.yaml",
    "base":   "config_base.yaml",
    "medium": "config_medium.yaml",
    "l":      "config_l.yaml",
    "xl":     "config_xl.yaml",
    "2xl":    "config_2xl.yaml",
}

perf_data = None


def fps_probe(pad, info, u_data):
    batch_meta, _ = get_batch_meta(info)
    if batch_meta:
        for frame in iter_frames(batch_meta):
            perf_data.update_fps(f"stream{frame.pad_index}")
    return Gst.PadProbeReturn.OK


def main():
    parser = argparse.ArgumentParser(description="RF-DETR DeepStream pipeline")
    parser.add_argument("--model", choices=CONFIGS.keys(), default="l",
                        help="Model size: nano (384), small (512), base (560), medium (576), l (704), xl (700), 2xl (880)")
    args = parser.parse_args()

    config = Config(yaml_filename=CONFIGS[args.model])
    name = f"deepstream-rfdetr-{args.model}"
    logger = Logger(name)
    platform_info = PlatformInfo()
    Gst.init(None)

    global perf_data
    perf_data = PERF_DATA(num_streams=config.num_sources)

    pipeline = create_pipeline(name, logger)

    streammux = create_streammux("rfdetr", batch_size=config.streammux_batch_size,
                                 width=config.streammux_width,
                                 height=config.streammux_height, logger=logger)
    pipeline.add(streammux)

    is_live = attach_sources(pipeline, streammux, config.sources, logger)
    if not is_live and config.num_sources > 1:
        streammux.set_property("sync-inputs", 1)

    pgie = create_pgie("rfdetr", config.pgie_config, logger)
    pgie.set_property("batch-size", config.num_sources)
    with open(config.pgie_config) as f:
        for line in f:
            if line.startswith("onnx-file="):
                onnx_path = line.split("=", 1)[1].strip()
                engine_path = f"{onnx_path}_b{config.num_sources}_gpu0_fp16.engine"
                pgie.set_property("model-engine-file", engine_path)
                break
    tracker = create_tracker("rfdetr", config.tracker_config, logger)

    rows = max(1, int(math.sqrt(config.num_sources)))
    cols = math.ceil(config.num_sources / rows)
    tiler = create_tiler("rfdetr", rows, cols,
                         config.tiler_width, config.tiler_height,
                         platform_info, logger)

    nvvidconv = create_nvvidconv("rfdetr", logger)
    nvosd = create_nvosd("rfdetr", logger)

    headless = not os.environ.get("DISPLAY")

    if headless:
        logger.info("No DISPLAY detected — using RTSP output")
        sink = create_rtsp_output_bin(
            "rfdetr", config.rtsp_codec, config.rtsp_bitrate,
            config.rtsp_enc_type, platform_info, logger,
        )
    else:
        sink = create_sink("rfdetr", platform_info, logger)
        sink.set_property("sync", 0)
        sink.set_property("qos", 0)

    queues = [create_queue(f"rfdetr-q{i}", logger) for i in range(5)]

    for el in [pgie, tracker, tiler, nvvidconv, nvosd, sink, *queues]:
        pipeline.add(el)

    link_chain(streammux, queues[0], pgie, queues[1], tracker, queues[2],
               tiler, queues[3], nvvidconv, queues[4], nvosd, sink)

    pgie.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, pgie_src_probe, config)
    tracker.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, fps_probe, 0)
    # nvosd.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, osd_probe, config)

    GLib.timeout_add_seconds(2, perf_data.perf_print_callback)

    if headless:
        start_rtsp_server(
            config.rtsp_port, config.rtsp_udp_port,
            config.rtsp_mount, config.rtsp_codec, logger,
        )

    run_pipeline(pipeline, logger)


if __name__ == '__main__':
    sys.exit(main())
