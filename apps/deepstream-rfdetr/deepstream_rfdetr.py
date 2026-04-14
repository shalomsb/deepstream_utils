#!/usr/bin/env python3

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
    create_pipeline, create_source_bin, create_streammux,
    create_pgie, create_tracker, create_nvvidconv, create_nvosd, create_sink,
    run_pipeline,
)
from config import Config
from callbacks import pgie_src_probe, osd_probe

CONFIGS = {
    "l":   "config_l.yaml",
    "2xl": "config_2xl.yaml",
}

perf_data = PERF_DATA(num_streams=1)


def fps_probe(pad, info, u_data):
    perf_data.update_fps("stream0")
    return Gst.PadProbeReturn.OK


def main():
    parser = argparse.ArgumentParser(description="RF-DETR DeepStream pipeline")
    parser.add_argument("--model", choices=CONFIGS.keys(), default="l",
                        help="Model size: l (Large, 704x704) or 2xl (2XLarge, 880x880)")
    args = parser.parse_args()

    config = Config(yaml_filename=CONFIGS[args.model])
    name = f"deepstream-rfdetr-{args.model}"
    logger = Logger(name)
    platform_info = PlatformInfo()
    Gst.init(None)

    pipeline = create_pipeline(name, logger)

    source_bin = create_source_bin(0, config.source, logger)
    streammux = create_streammux("rfdetr", batch_size=config.streammux_batch_size,
                                 width=config.streammux_width,
                                 height=config.streammux_height, logger=logger)
    pgie = create_pgie("rfdetr", config.pgie_config, logger)
    tracker = create_tracker("rfdetr", config.tracker_config, logger)
    nvvidconv = create_nvvidconv("rfdetr", logger)
    nvosd = create_nvosd("rfdetr", logger)
    sink = create_sink("rfdetr", platform_info, logger)
    sink.set_property("sync", 0)
    sink.set_property("qos", 0)

    for el in [source_bin, streammux, pgie, tracker, nvvidconv, nvosd, sink]:
        pipeline.add(el)

    srcpad = source_bin.get_static_pad("src")
    sinkpad = streammux.request_pad_simple("sink_0")
    srcpad.link(sinkpad)

    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(sink)

    pgie.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, pgie_src_probe, config)
    nvosd.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, osd_probe, config)
    nvosd.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, fps_probe, 0)

    GLib.timeout_add_seconds(2, perf_data.perf_print_callback)

    run_pipeline(pipeline, logger)


if __name__ == '__main__':
    sys.exit(main())
