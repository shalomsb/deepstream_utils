"""GStreamer element factory functions.

Internal module — import from ds_pipeline directly, not from ds_pipeline._elements.
"""

import os
import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
from .constants import MUXER_BATCH_TIMEOUT_USEC


def make_element(factory_name, element_name, logger):
    """Create a GStreamer element, exit on failure."""
    element = Gst.ElementFactory.make(factory_name, element_name)
    if not element:
        logger.error(f"Unable to create {element_name} ({factory_name})")
        sys.exit(1)
    return element


def has_nvenc():
    """True iff the platform has NVIDIA HW video encoder (nvv4l2h264enc).

    Orin Nano has no NVENC; Orin NX / AGX / x86 dGPU do. Used to auto-select
    HW vs SW encoder path without a YAML flag, and to gate other Nano-only
    workarounds (nvosd CPU draw, output-bin nvvideoconvert compute-hw).
    """
    return Gst.ElementFactory.find("nvv4l2h264enc") is not None


def create_pipeline(name, logger):
    """Create a GStreamer pipeline."""
    logger.info(f"Creating Pipeline: {name}")
    return Gst.Pipeline.new(name)


def create_streammux(name, batch_size, logger, width=1920, height=1080,
                     timeout_usec=MUXER_BATCH_TIMEOUT_USEC,
                     file_loop=False, platform_info=None):
    """Create nvstreammux with properties.

    `file_loop` + `platform_info` match the reference pattern in
    `deepstream-test3`: when looping a file via nvurisrcbin, streammux needs an
    explicit `nvbuf-memory-type` — 4 (NVBUF_MEM_SYSTEM) on Jetson/aarch64,
    2 (NVBUF_MEM_CUDA_DEVICE) on x86 dGPU.
    """
    streammux = make_element("nvstreammux", f"streammux-{name}", logger)
    if os.environ.get("USE_NEW_NVSTREAMMUX") != "yes":
        streammux.set_property("width", width)
        streammux.set_property("height", height)
        streammux.set_property("batched-push-timeout", timeout_usec)
    streammux.set_property("batch-size", batch_size)
    if file_loop and platform_info is not None:
        is_jetson_or_aarch64 = (
            platform_info.is_integrated_gpu() or platform_info.is_platform_aarch64()
        )
        streammux.set_property("nvbuf-memory-type", 4 if is_jetson_or_aarch64 else 2)
    return streammux


def create_pgie(name, config_file, logger):
    """Create nvinfer element with config."""
    pgie = make_element("nvinfer", f"pgie-{name}", logger)
    pgie.set_property("config-file-path", config_file)
    return pgie


def create_sgie(name, config_file, logger):
    """Create secondary nvinfer element with config."""
    sgie = make_element("nvinfer", f"sgie-{name}", logger)
    sgie.set_property("config-file-path", config_file)
    return sgie


def create_tracker(name, config_file, logger):
    """Create nvtracker element and load properties from config file."""
    import configparser
    tracker = make_element("nvtracker", f"tracker-{name}", logger)
    config = configparser.ConfigParser()
    config.read(config_file)
    for key in config['tracker']:
        if key == 'tracker-width':
            tracker.set_property('tracker-width', config.getint('tracker', key))
        elif key == 'tracker-height':
            tracker.set_property('tracker-height', config.getint('tracker', key))
        elif key == 'gpu-id':
            tracker.set_property('gpu_id', config.getint('tracker', key))
        elif key == 'll-lib-file':
            tracker.set_property('ll-lib-file', config.get('tracker', key))
        elif key == 'll-config-file':
            tracker.set_property('ll-config-file', config.get('tracker', key))
        elif key == 'user-meta-pool-size':
            tracker.set_property('user-meta-pool-size', config.getint('tracker', key))
    return tracker


def create_pgie_inferserver(name, config_file, logger):
    """Create nvinferserver element with config."""
    pgie = make_element("nvinferserver", f"pgie-{name}", logger)
    pgie.set_property("config-file-path", config_file)
    return pgie


def create_tiler(name, rows, columns, width, height, platform_info, logger):
    """Create nvmultistreamtiler element.

    Per NVIDIA reference apps:
      - iGPU     → VIC compute (`compute-hw=2`), memory type unset
      - x86 dGPU → GPU compute (`compute-hw=1`), `nvbuf-memory-type=3` (UNIFIED)

    Memory-type gating uses `is_platform_aarch64()` not `is_integrated_gpu()`
    because the latter misreports False on Orin Nano (CUDA probe fails). Setting
    `nvbuf-memory-type=3` on Jetson produces CUDA_UNIFIED surfaces that later
    nvvideoconvert transforms cannot handle. See
    `project_is_integrated_gpu_broken_on_orin_nano.md`.
    """
    tiler = make_element("nvmultistreamtiler", f"tiler-{name}", logger)
    tiler.set_property("rows", rows)
    tiler.set_property("columns", columns)
    tiler.set_property("width", width)
    tiler.set_property("height", height)
    tiler.set_property("compute-hw", 2 if platform_info.is_integrated_gpu() else 1)
    if not platform_info.is_platform_aarch64():
        tiler.set_property("nvbuf-memory-type", 3)
    return tiler


def create_queue(name, logger):
    """Create a queue element."""
    return make_element("queue", f"queue-{name}", logger)


def create_fakesink(name, logger):
    """Create a fakesink element for headless mode."""
    sink = make_element("fakesink", f"fakesink-{name}", logger)
    sink.set_property("enable-last-sample", 0)
    sink.set_property("sync", 0)
    return sink


def create_nvvidconv(name, logger, platform_info=None):
    """Create nvvideoconvert element.

    Per NVIDIA reference apps:
      - x86 dGPU → `nvbuf-memory-type=3` (CUDA_UNIFIED)
      - iGPU     → memory type left at default (reference does not set it)

    Note: `platform_info.is_integrated_gpu()` can return False on Orin Nano
    because `cudaGetDeviceProperties` fails at init (`cudaErrorInsufficientDriver`).
    Use `is_platform_aarch64()` as a Jetson-safe fallback: on aarch64 we're
    either Jetson or SBSA-ARM, and neither wants CUDA_UNIFIED here.
    """
    elem = make_element("nvvideoconvert", f"nvvidconv-{name}", logger)
    if platform_info is not None and not platform_info.is_platform_aarch64():
        elem.set_property("nvbuf-memory-type", 3)
    return elem


def create_nvosd(name, logger):
    """Create nvdsosd element."""
    return make_element("nvdsosd", f"nvosd-{name}", logger)


def create_sink(name, platform_info, logger):
    """Create display sink based on platform.

    Per NVIDIA reference apps:
      - iGPU or aarch64 → `nv3dsink` (no `gpu_id`)
      - x86 dGPU        → `nveglglessink` with `gpu_id=0`
    """
    if platform_info.is_integrated_gpu() or platform_info.is_platform_aarch64():
        logger.info("Creating nv3dsink")
        return make_element("nv3dsink", f"nv3dsink-{name}", logger)
    else:
        logger.info("Creating EGLSink")
        sink = make_element("nveglglessink", f"eglsink-{name}", logger)
        sink.set_property("gpu_id", 0)
        return sink


# --- Internal: used by bins.py, not re-exported through __init__.py ---

def create_encoder(name, codec, bitrate, platform_info, logger):
    """Create video encoder element, auto-selecting HW (NVENC) vs SW (x264/x265).

    HW is used whenever `nvv4l2h26{4,5}enc` is available (Orin NX/AGX, x86
    dGPU); otherwise falls back to `x26{4,5}enc` (Orin Nano). Bitrate is in
    bits/s for HW and kbit/s for SW — converted transparently here.
    """
    hw_map = {"H264": "nvv4l2h264enc", "H265": "nvv4l2h265enc"}
    sw_map = {"H264": "x264enc", "H265": "x265enc"}
    use_hw = has_nvenc()
    factory = hw_map[codec] if use_hw else sw_map[codec]
    logger.info(f"Creating {codec} encoder ({factory})")
    encoder = make_element(factory, f"encoder-{name}", logger)
    encoder.set_property("bitrate", bitrate if use_hw else bitrate // 1000)
    if use_hw:
        encoder.set_property("insert-sps-pps", 1)
        if platform_info.is_integrated_gpu():
            encoder.set_property("preset-level", 1)
    elif factory == "x264enc":
        # Low-latency CPU encode tuning for platforms without NVENC (Orin Nano).
        encoder.set_property("tune", "zerolatency")
        encoder.set_property("speed-preset", "ultrafast")
        encoder.set_property("key-int-max", 30)
        encoder.set_property("bframes", 0)
        encoder.set_property("threads", 4)
    return encoder


def create_rtppay(name, codec, logger):
    """Create RTP payloader for H264 or H265."""
    factory = "rtph264pay" if codec == "H264" else "rtph265pay"
    return make_element(factory, f"rtppay-{name}", logger)


def create_udpsink(name, host, port, logger):
    """Create UDP sink for RTSP streaming."""
    sink = make_element("udpsink", f"udpsink-{name}", logger)
    sink.set_property("host", host)
    sink.set_property("port", port)
    sink.set_property("async", False)
    sink.set_property("sync", 1)
    return sink


def create_capsfilter(name, caps_str, logger):
    """Create a capsfilter element with given caps string."""
    caps = make_element("capsfilter", f"capsfilter-{name}", logger)
    caps.set_property("caps", Gst.Caps.from_string(caps_str))
    return caps
