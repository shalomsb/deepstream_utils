#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
from common.platform_info import PlatformInfo
from logger import Logger
from ds_bins import create_source_bin, create_rtsp_output_bin
from ds_factory import create_pipeline, create_streammux, create_pgie, create_nvvidconv, create_nvosd
from ds_callbacks import osd_probe_test1
from ds_utils import run_pipeline, start_rtsp_server
from constants import Test1RtspConfig


def parse_args():
    parser = argparse.ArgumentParser(description="RTSP Output Sample Application")
    parser.add_argument("-i", "--input", default=Test1RtspConfig.VIDEO,
                        help="Path to input H264 elementary stream")
    parser.add_argument("-c", "--codec", default="H264", choices=["H264", "H265"],
                        help="RTSP Streaming Codec, default=H264")
    parser.add_argument("-b", "--bitrate", default=4000000, type=int,
                        help="Set the encoding bitrate")
    return parser.parse_args()


def main():
    args = parse_args()
    logger = Logger("deepstream-test1-rtsp-out")
    platform_info = PlatformInfo()
    Gst.init(None)

    pipeline = create_pipeline("deepstream-test1-rtsp-out", logger)

    # Create elements
    source_bin = create_source_bin(0, args.input, logger)
    streammux = create_streammux("rtsp", batch_size=1, logger=logger)
    pgie = create_pgie("rtsp", Test1RtspConfig.PGIE, logger)
    nvvidconv = create_nvvidconv("rtsp", logger)
    nvosd = create_nvosd("rtsp", logger)
    rtsp_out = create_rtsp_output_bin("rtsp", args.codec, args.bitrate, platform_info, logger)

    # Add to pipeline
    for element in [source_bin, streammux, pgie, nvvidconv, nvosd, rtsp_out]:
        pipeline.add(element)

    # Link source bin to streammux
    sinkpad = streammux.request_pad_simple("sink_0")
    srcpad = source_bin.get_static_pad("src")
    srcpad.link(sinkpad)

    # Link: streammux -> pgie -> nvvidconv -> nvosd -> rtsp_output_bin
    streammux.link(pgie)
    pgie.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(rtsp_out)

    # Add probe on OSD sink pad
    osdsinkpad = nvosd.get_static_pad("sink")
    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_probe_test1, 0)

    # Start RTSP server
    start_rtsp_server(Test1RtspConfig.RTSP_PORT, Test1RtspConfig.UDP_PORT,
                      Test1RtspConfig.RTSP_MOUNT, args.codec, logger)

    run_pipeline(pipeline, logger)


if __name__ == '__main__':
    sys.exit(main())
