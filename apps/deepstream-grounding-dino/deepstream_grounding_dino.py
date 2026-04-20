#!/usr/bin/env python3

import os
import sys
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
    create_pgie_inferserver, create_tracker,
    create_nvvidconv, create_nvosd, create_sink,
    create_rtsp_output_bin, start_rtsp_server,
    run_pipeline,
)
from config import Config
from callbacks import pgie_src_probe, osd_probe

perf_data = PERF_DATA(num_streams=1)


def fps_probe(pad, info, u_data):
    perf_data.update_fps("stream0")
    return Gst.PadProbeReturn.OK


def main():
    config = Config()
    logger = Logger("deepstream-grounding-dino")
    platform_info = PlatformInfo()
    Gst.init(None)

    pipeline = create_pipeline("deepstream-grounding-dino", logger)

    source_bin = create_source_bin(0, config.source, logger)
    streammux = create_streammux(
        "gdino", batch_size=config.streammux_batch_size,
        width=config.streammux_width,
        height=config.streammux_height, logger=logger,
    )
    pgie = create_pgie_inferserver("gdino", config.pgie_config, logger)
    pgie.set_property("interval", config.interval)
    tracker = create_tracker("gdino", config.tracker_config, logger)
    nvvidconv = create_nvvidconv("gdino", logger)
    nvosd = create_nvosd("gdino", logger)

    headless = not os.environ.get("DISPLAY")

    if headless:
        logger.info("No DISPLAY detected — using RTSP output")
        sink = create_rtsp_output_bin(
            "gdino", config.rtsp_codec, config.rtsp_bitrate,
            platform_info, logger,
        )
    else:
        sink = create_sink("gdino", platform_info, logger)
        sink.set_property("sync", 0)

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

    # Parse ensemble tensor outputs -> NvDsObjectMeta (before tracker)
    pgie.get_static_pad("src").add_probe(
        Gst.PadProbeType.BUFFER, pgie_src_probe, config,
    )

    # Display info on OSD
    nvosd.get_static_pad("sink").add_probe(
        Gst.PadProbeType.BUFFER, osd_probe, config,
    )

    # FPS counter
    nvosd.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, fps_probe, 0)
    GLib.timeout_add_seconds(2, perf_data.perf_print_callback)

    if headless:
        start_rtsp_server(
            config.rtsp_port, config.rtsp_udp_port,
            config.rtsp_mount, config.rtsp_codec, logger,
        )

    run_pipeline(pipeline, logger)


if __name__ == "__main__":
    sys.exit(main())
