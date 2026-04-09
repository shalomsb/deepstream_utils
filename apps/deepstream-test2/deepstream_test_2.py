#!/usr/bin/env python3

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
from common.platform_info import PlatformInfo
from logger import Logger
from ds_bins import create_source_bin
from ds_factory import (
    create_pipeline, create_streammux, create_pgie, create_sgie,
    create_tracker, create_nvvidconv, create_nvosd, create_sink,
)
from ds_callbacks import osd_probe_test2
from ds_utils import run_pipeline
from constants import Test2Config


def main():
    logger = Logger("deepstream-test2")
    platform_info = PlatformInfo()
    Gst.init(None)

    pipeline = create_pipeline("deepstream-test2", logger)

    # Create elements
    source_bin = create_source_bin(0, Test2Config.VIDEO, logger)
    streammux = create_streammux("test2", batch_size=1, logger=logger)
    pgie = create_pgie("test2", Test2Config.PGIE, logger)
    tracker = create_tracker("test2", Test2Config.TRACKER, logger)
    sgie1 = create_sgie("test2-sgie1", Test2Config.SGIE1, logger)
    sgie2 = create_sgie("test2-sgie2", Test2Config.SGIE2, logger)
    nvvidconv = create_nvvidconv("test2", logger)
    nvosd = create_nvosd("test2", logger)
    sink = create_sink("test2", platform_info, logger)

    # Add to pipeline
    for element in [source_bin, streammux, pgie, tracker, sgie1, sgie2, nvvidconv, nvosd, sink]:
        pipeline.add(element)

    # Link source bin to streammux
    sinkpad = streammux.request_pad_simple("sink_0")
    srcpad = source_bin.get_static_pad("src")
    srcpad.link(sinkpad)

    # Link rest of the pipeline
    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(sgie1)
    sgie1.link(sgie2)
    sgie2.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(sink)

    # Add probe on OSD sink pad
    osdsinkpad = nvosd.get_static_pad("sink")
    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_probe_test2, 0)

    run_pipeline(pipeline, logger)


if __name__ == '__main__':
    sys.exit(main())
