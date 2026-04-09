#!/usr/bin/env python3

import sys
import math
import argparse
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from os import environ
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
from common.platform_info import PlatformInfo
from common.FPS import PERF_DATA
from logger import Logger
from ds_bins import create_source_bin
from ds_factory import (
    create_pipeline, create_streammux, create_pgie, create_pgie_inferserver,
    create_tiler, create_nvvidconv, create_nvosd, create_sink, create_fakesink,
    create_queue, make_element,
)
from ds_callbacks import pgie_src_probe_test3
from ds_utils import run_pipeline
from constants import Test3Config


def parse_args():
    parser = argparse.ArgumentParser(
        prog="deepstream_test_3",
        description="deepstream-test3 multi stream, multi model inference reference app",
    )
    parser.add_argument("-i", "--input", help="Path to input streams", nargs="+",
                        metavar="URIs", required=True)
    parser.add_argument("-c", "--configfile", default=None,
                        help="Choose the config-file to be used with specified pgie")
    parser.add_argument("-g", "--pgie", default=None,
                        choices=["nvinfer", "nvinferserver", "nvinferserver-grpc"],
                        help="Choose Primary GPU Inference Engine")
    parser.add_argument("--no-display", action="store_true", default=False, dest="no_display")
    parser.add_argument("--file-loop", action="store_true", default=False, dest="file_loop")
    parser.add_argument("--disable-probe", action="store_true", default=False, dest="disable_probe")
    parser.add_argument("-s", "--silent", action="store_true", default=False, dest="silent")

    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    if args.configfile and not args.pgie or args.pgie and not args.configfile:
        sys.stderr.write("\nEither pgie or configfile is missing. Please specify both!\n\n")
        parser.print_help()
        sys.exit(1)
    if args.configfile and not Path(args.configfile).is_file():
        sys.stderr.write(f"Specified config-file: {args.configfile} doesn't exist.\n")
        sys.exit(1)

    return args


def main():
    args = parse_args()
    logger = Logger("deepstream-test3")
    platform_info = PlatformInfo()
    Gst.init(None)

    number_sources = len(args.input)
    perf_data = PERF_DATA(number_sources)

    pipeline = create_pipeline("deepstream-test3", logger)

    # Streammux
    streammux = create_streammux("test3", batch_size=number_sources, logger=logger)
    if args.file_loop:
        mem_type = 4 if platform_info.is_integrated_gpu() else 2
        streammux.set_property("nvbuf-memory-type", mem_type)
    pipeline.add(streammux)

    # Add sources
    is_live = False
    for i, uri in enumerate(args.input):
        if uri.startswith("rtsp://"):
            is_live = True
        source_bin = create_source_bin(i, uri, logger, file_loop=args.file_loop)
        pipeline.add(source_bin)
        sinkpad = streammux.request_pad_simple(f"sink_{i}")
        srcpad = source_bin.get_static_pad("src")
        srcpad.link(sinkpad)

    if is_live:
        logger.info("At least one source is live")
        streammux.set_property("live-source", 1)

    # Primary inference
    if args.pgie in ("nvinferserver", "nvinferserver-grpc"):
        pgie = create_pgie_inferserver("test3", args.configfile, logger)
    else:
        config = args.configfile if args.configfile else Test3Config.PGIE
        pgie = create_pgie("test3", config, logger)

    # Adjust batch size to match number of sources
    pgie_batch_size = pgie.get_property("batch-size")
    if pgie_batch_size != number_sources:
        logger.warning(f"Overriding infer-config batch-size {pgie_batch_size} with number of sources {number_sources}")
        pgie.set_property("batch-size", number_sources)

    # Tiler
    tiler_rows = int(math.sqrt(number_sources))
    tiler_columns = int(math.ceil(number_sources / tiler_rows))
    tiler = create_tiler("test3", tiler_rows, tiler_columns,
                         Test3Config.TILED_OUTPUT_WIDTH, Test3Config.TILED_OUTPUT_HEIGHT,
                         platform_info, logger)

    # Rest of pipeline
    nvvidconv = create_nvvidconv("test3", logger)
    nvosd = create_nvosd("test3", logger)
    nvosd.set_property("process-mode", Test3Config.OSD_PROCESS_MODE)
    nvosd.set_property("display-text", Test3Config.OSD_DISPLAY_TEXT)

    if args.no_display:
        sink = create_fakesink("test3", logger)
    else:
        sink = create_sink("test3", platform_info, logger)
    # When qos is enabled, the sink drop late frames when the pipeline is not able to keep up with the source. 
    # This can help reduce latency when the system is under heavy load, but may result in lower accuracy due to dropped frames.
    sink.set_property("qos", 0)

    # Queues
    queues = [create_queue(f"test3-{i}", logger) for i in range(5)]

    # Optional nvdslogger
    nvdslogger = None
    if args.disable_probe:
        nvdslogger = make_element("nvdslogger", "nvdslogger", logger)

    # Add all to pipeline
    for element in [pgie, tiler, nvvidconv, nvosd, sink] + queues:
        pipeline.add(element)
    if nvdslogger:
        pipeline.add(nvdslogger)

    # Link: streammux -> q1 -> pgie -> q2 -> [nvdslogger ->] tiler -> q3 -> nvvidconv -> q4 -> nvosd -> q5 -> sink
    streammux.link(queues[0])
    queues[0].link(pgie)
    pgie.link(queues[1])
    if nvdslogger:
        queues[1].link(nvdslogger)
        nvdslogger.link(tiler)
    else:
        queues[1].link(tiler)
    tiler.link(queues[2])
    queues[2].link(nvvidconv)
    nvvidconv.link(queues[3])
    queues[3].link(nvosd)
    nvosd.link(queues[4])
    queues[4].link(sink)

    # Probe on pgie src pad
    if not args.disable_probe:
        pgie_src_pad = pgie.get_static_pad("src")
        probe_data = {
            "perf_data": perf_data,
            "silent": args.silent,
            "measure_latency": environ.get("NVDS_ENABLE_LATENCY_MEASUREMENT") == "1",
        }
        pgie_src_pad.add_probe(Gst.PadProbeType.BUFFER, pgie_src_probe_test3, probe_data)
        GLib.timeout_add(5000, perf_data.perf_print_callback)

    if probe_data.get("measure_latency"):
        logger.info("Pipeline Latency Measurement enabled! Set NVDS_ENABLE_COMPONENT_LATENCY_MEASUREMENT=1 for component level.")

    # Print sources
    for i, source in enumerate(args.input):
        logger.info(f"  Source {i}: {source}")

    run_pipeline(pipeline, logger)


if __name__ == '__main__':
    sys.exit(main())
