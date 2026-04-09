import os
import sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
from constants import MUXER_BATCH_TIMEOUT_USEC


def create_pipeline(name, logger):
    """Create a GStreamer pipeline."""
    logger.info(f"Creating Pipeline: {name}")
    return Gst.Pipeline.new(name)


def make_element(factory_name, element_name, logger):
    """Create a GStreamer element, exit on failure."""
    element = Gst.ElementFactory.make(factory_name, element_name)
    if not element:
        logger.error(f"Unable to create {element_name} ({factory_name})")
        sys.exit(1)
    return element


def create_streammux(name, batch_size, logger, width=1920, height=1080, timeout_usec=MUXER_BATCH_TIMEOUT_USEC):
    """Create nvstreammux with properties."""
    streammux = make_element("nvstreammux", f"streammux-{name}", logger)
    if os.environ.get("USE_NEW_NVSTREAMMUX") != "yes":
        streammux.set_property("width", width)
        streammux.set_property("height", height)
        streammux.set_property("batched-push-timeout", timeout_usec)
    streammux.set_property("batch-size", batch_size)
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
    return tracker


def create_pgie_inferserver(name, config_file, logger):
    """Create nvinferserver element with config."""
    pgie = make_element("nvinferserver", f"pgie-{name}", logger)
    pgie.set_property("config-file-path", config_file)
    return pgie


def create_tiler(name, rows, columns, width, height, platform_info, logger):
    """Create nvmultistreamtiler element."""
    tiler = make_element("nvmultistreamtiler", f"tiler-{name}", logger)
    tiler.set_property("rows", rows)
    tiler.set_property("columns", columns)
    tiler.set_property("width", width)
    tiler.set_property("height", height)
    tiler.set_property("compute-hw", 2 if platform_info.is_integrated_gpu() else 1)
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


def create_nvvidconv(name, logger):
    """Create nvvideoconvert element."""
    return make_element("nvvideoconvert", f"nvvidconv-{name}", logger)


def create_nvosd(name, logger):
    """Create nvdsosd element."""
    return make_element("nvdsosd", f"nvosd-{name}", logger)


def create_sink(name, platform_info, logger):
    """Create display sink based on platform."""
    if platform_info.is_integrated_gpu() or platform_info.is_platform_aarch64():
        logger.info("Creating nv3dsink")
        return make_element("nv3dsink", f"nv3dsink-{name}", logger)
    else:
        logger.info("Creating EGLSink")
        return make_element("nveglglessink", f"eglsink-{name}", logger)


def create_encoder(name, codec, bitrate, enc_type, platform_info, logger):
    """Create video encoder element.

    Args:
        codec: "H264" or "H265"
        enc_type: 0 = hardware, 1 = software
    """
    hw_map = {"H264": "nvv4l2h264enc", "H265": "nvv4l2h265enc"}
    sw_map = {"H264": "x264enc", "H265": "x265enc"}
    factory = hw_map[codec] if enc_type == 0 else sw_map[codec]
    logger.info(f"Creating {codec} encoder ({factory})")
    encoder = make_element(factory, f"encoder-{name}", logger)
    encoder.set_property("bitrate", bitrate)
    if enc_type == 0:
        encoder.set_property("insert-sps-pps", 1)
        if platform_info.is_integrated_gpu():
            encoder.set_property("preset-level", 1)
    return encoder


def create_rtppay(name, codec, logger):
    """Create RTP payloader for H264 or H265."""
    factory = "rtph264pay" if codec == "H264" else "rtph265pay"
    logger.info(f"Creating {codec} rtppay")
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
