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
from ds_factory import create_pipeline, create_streammux, create_pgie, create_nvvidconv, create_nvosd, create_sink
from ds_callbacks import osd_probe_test1
from ds_utils import run_pipeline
from constants import Test1Config


def main():
    logger = Logger("deepstream-test1-usbcam")
    platform_info = PlatformInfo()
    # Initialize GStreamer — must be called before any GStreamer API usage
    Gst.init(None)

    pipeline = create_pipeline("deepstream-test1-usbcam", logger)

    # Create elements
    source_bin = create_source_bin(0, Test1Config.USBCAM, logger)
    streammux = create_streammux("test1", batch_size=1, logger=logger)
    pgie = create_pgie("test1", Test1Config.PGIE, logger)
    nvvidconv = create_nvvidconv("test1", logger)
    nvosd = create_nvosd("test1", logger)
    sink = create_sink("test1", platform_info, logger)
    # Avoid late frame drops on live camera input
    sink.set_property("sync", False)

    # Add to pipeline
    pipeline.add(source_bin)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    pipeline.add(sink)

    # Link source bin to streammux
    sinkpad = streammux.request_pad_simple("sink_0")
    srcpad = source_bin.get_static_pad("src")
    srcpad.link(sinkpad)

    # Link rest of the pipeline
    streammux.link(pgie)
    pgie.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(sink)

    # Add probe on OSD sink pad
    osdsinkpad = nvosd.get_static_pad("sink")
    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_probe_test1, 0)

    run_pipeline(pipeline, logger)


if __name__ == '__main__':
    sys.exit(main())
