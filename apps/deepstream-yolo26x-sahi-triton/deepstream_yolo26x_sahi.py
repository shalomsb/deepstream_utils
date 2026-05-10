#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
from common.platform_info import PlatformInfo
from ds_pipeline import (
    Logger,
    create_pipeline, create_source_bin, create_streammux,
    create_pgie_inferserver, create_tracker, create_nvvidconv, create_nvosd, create_sink,
    run_pipeline,
)
from config import Config
from callbacks import pgie_src_probe, osd_probe


def main():
    config = Config()
    logger = Logger("deepstream-yolo26x-sahi-triton")
    platform_info = PlatformInfo()
    Gst.init(None)

    pipeline = create_pipeline("deepstream-yolo26x-sahi-triton", logger)

    source_bin = create_source_bin(0, config.source, logger)
    streammux = create_streammux("sahi", batch_size=config.streammux_batch_size,
                                 width=config.streammux_width,
                                 height=config.streammux_height, logger=logger)
    pgie = create_pgie_inferserver("sahi", config.pgie_config, logger)
    tracker = create_tracker("sahi", config.tracker_config, logger)
    nvvidconv = create_nvvidconv("sahi", logger)
    nvosd = create_nvosd("sahi", logger)
    sink = create_sink("sahi", platform_info, logger)

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

    # Unpack ensemble's already-merged detections into NvDsObjectMeta.
    pgie.get_static_pad("src").add_probe(Gst.PadProbeType.BUFFER, pgie_src_probe, config)

    # Display info on OSD.
    nvosd.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, osd_probe, config)

    run_pipeline(pipeline, logger)


if __name__ == '__main__':
    sys.exit(main())
