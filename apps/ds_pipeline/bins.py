"""GStreamer Bin compositions for DeepStream pipelines.

Source bins, output bins — composite elements with ghost pads.
"""

import os
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
from ._elements import make_element


# --- Source bins ---

def create_usbcam_source_bin(name, device, logger):
    """Create a source bin for USB camera input."""
    bin_name = f"{name}-usbcam-bin"
    nbin = Gst.Bin.new(bin_name)
    source = make_element("v4l2src", f"usbcam-source-{name}", logger)
    caps_v4l2src = make_element("capsfilter", f"v4l2src_caps_{name}", logger)
    vidconvsrc = make_element("videoconvert", f"convertor_src_{name}", logger)
    nvvidconvsrc = make_element("nvvideoconvert", f"nvconvertor_src_{name}", logger)
    caps_vidconvsrc = make_element("capsfilter", f"nvmm_caps_{name}", logger)

    nbin.add(source)
    nbin.add(caps_v4l2src)
    nbin.add(vidconvsrc)
    nbin.add(nvvidconvsrc)
    nbin.add(caps_vidconvsrc)

    source.link(caps_v4l2src)
    caps_v4l2src.link(vidconvsrc)
    vidconvsrc.link(nvvidconvsrc)
    nvvidconvsrc.link(caps_vidconvsrc)

    source.set_property('device', device)
    caps_v4l2src.set_property('caps', Gst.Caps.from_string("video/x-raw, framerate=30/1"))
    caps_vidconvsrc.set_property('caps', Gst.Caps.from_string("video/x-raw(memory:NVMM)"))

    srcpad = caps_vidconvsrc.get_static_pad("src")
    ghost_src = Gst.GhostPad.new("src", srcpad)
    nbin.add_pad(ghost_src)

    return nbin


def _cb_newpad_nvurisrcbin(decodebin, decoder_src_pad, data):
    """Callback for when nvurisrcbin creates a new pad."""
    source_bin, logger = data
    logger.info("nvurisrcbin pad-added callback")

    bin_ghost_pad = source_bin.get_static_pad("src")
    if not bin_ghost_pad.set_target(decoder_src_pad):
        logger.error("Failed to link nvurisrcbin src pad to source bin ghost pad")


def create_nvurisrcbin_bin(name, uri, logger, file_loop=False):
    """Create a source bin using nvurisrcbin."""
    logger.info("Creating source bin with nvurisrcbin")

    bin_name = f"source-bin-{name}"
    nbin = Gst.Bin.new(bin_name)
    if not nbin:
        logger.error("Unable to create source bin")
        return None

    uri_src_bin = make_element("nvurisrcbin", f"uri-src-bin-{name}", logger)
    uri_src_bin.set_property("uri", uri)
    if file_loop:
        uri_src_bin.set_property("file-loop", 1)
        uri_src_bin.set_property("cudadec-memtype", 0)
    uri_src_bin.connect("pad-added", _cb_newpad_nvurisrcbin, (nbin, logger))

    Gst.Bin.add(nbin, uri_src_bin)

    bin_pad = nbin.add_pad(Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC))
    if not bin_pad:
        logger.error("Failed to add ghost pad in source bin")
        return None

    return nbin


def create_filesrc_bin(name, location, logger):
    """Create a source bin for h264 file input: filesrc -> h264parse -> nvv4l2decoder."""
    bin_name = f"{name}-filesrc-bin"
    nbin = Gst.Bin.new(bin_name)

    source = make_element("filesrc", f"filesrc-{name}", logger)
    h264parser = make_element("h264parse", f"h264parser-{name}", logger)
    decoder = make_element("nvv4l2decoder", f"decoder-{name}", logger)

    source.set_property("location", location)

    nbin.add(source)
    nbin.add(h264parser)
    nbin.add(decoder)

    source.link(h264parser)
    h264parser.link(decoder)

    srcpad = decoder.get_static_pad("src")
    ghost_src = Gst.GhostPad.new("src", srcpad)
    nbin.add_pad(ghost_src)

    return nbin


def create_source_bin(index, uri, logger, file_loop=False):
    """Create a source bin for reading from a URI, file path, or USB camera."""
    name = f"input_{index}"

    if uri.startswith("/dev/video"):
        return create_usbcam_source_bin(name, uri, logger)
    else:
        if not uri.startswith(("rtsp://", "http://", "https://", "file://")):
            uri = "file://" + os.path.abspath(uri)
        return create_nvurisrcbin_bin(name, uri, logger, file_loop=file_loop)


# --- Output bins ---

def create_rtsp_output_bin(name, codec, bitrate, enc_type, platform_info, logger):
    """Create an RTSP output bin: nvvideoconvert -> capsfilter -> encoder -> rtppay -> udpsink.

    Args:
        codec: "H264" or "H265"
        bitrate: encoding bitrate (e.g. 4000000)
        enc_type: 0 = hardware encoder, 1 = software encoder
        platform_info: PlatformInfo instance
    """
    bin_name = f"{name}-rtsp-output-bin"
    nbin = Gst.Bin.new(bin_name)

    nvvidconv = make_element("nvvideoconvert", f"nvvidconv-postosd-{name}", logger)

    caps_str = "video/x-raw(memory:NVMM), format=I420" if enc_type == 0 else "video/x-raw, format=I420"
    capsfilter = make_element("capsfilter", f"capsfilter-{name}", logger)
    capsfilter.set_property("caps", Gst.Caps.from_string(caps_str))

    hw_map = {"H264": "nvv4l2h264enc", "H265": "nvv4l2h265enc"}
    sw_map = {"H264": "x264enc", "H265": "x265enc"}
    factory = hw_map[codec] if enc_type == 0 else sw_map[codec]
    encoder = make_element(factory, f"encoder-{name}", logger)
    encoder.set_property("bitrate", bitrate)
    if enc_type == 0:
        encoder.set_property("insert-sps-pps", 1)
        if platform_info.is_integrated_gpu():
            encoder.set_property("preset-level", 1)

    rtppay_factory = "rtph264pay" if codec == "H264" else "rtph265pay"
    rtppay = make_element(rtppay_factory, f"rtppay-{name}", logger)

    udpsink = make_element("udpsink", f"udpsink-{name}", logger)
    udpsink.set_property("host", "224.224.255.255")
    udpsink.set_property("port", 5400)
    udpsink.set_property("async", False)
    udpsink.set_property("sync", 0)

    nbin.add(nvvidconv)
    nbin.add(capsfilter)
    nbin.add(encoder)
    nbin.add(rtppay)
    nbin.add(udpsink)

    nvvidconv.link(capsfilter)
    capsfilter.link(encoder)
    encoder.link(rtppay)
    rtppay.link(udpsink)

    sinkpad = nvvidconv.get_static_pad("sink")
    ghost_sink = Gst.GhostPad.new("sink", sinkpad)
    nbin.add_pad(ghost_sink)

    return nbin
