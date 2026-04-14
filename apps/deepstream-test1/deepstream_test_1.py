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
    create_pgie, create_nvvidconv, create_nvosd, create_sink,
    run_pipeline,
)
from config import Config
from callbacks import osd_probe


def main():
    config = Config()
    logger = Logger("deepstream-test1")
    platform_info = PlatformInfo()
    # Initialize GStreamer — must be called before any GStreamer API usage
    Gst.init(None)

    pipeline = create_pipeline("deepstream-test1", logger)

    source_bin = create_source_bin(0, config.source, logger)
    streammux = create_streammux("test1", batch_size=1, logger=logger)
    pgie = create_pgie("test1", config.pgie_config, logger)
    nvvidconv = create_nvvidconv("test1", logger)
    nvosd = create_nvosd("test1", logger)
    sink = create_sink("test1", platform_info, logger)

    for el in [source_bin, streammux, pgie, nvvidconv, nvosd, sink]:
        pipeline.add(el)

    srcpad = source_bin.get_static_pad("src")
    sinkpad = streammux.request_pad_simple("sink_0")
    srcpad.link(sinkpad)

    streammux.link(pgie)
    pgie.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(sink)

    nvosd.get_static_pad("sink").add_probe(Gst.PadProbeType.BUFFER, osd_probe, 0)

    run_pipeline(pipeline, logger)


if __name__ == '__main__':
    sys.exit(main())
