"""GStreamer Bin compositions for DeepStream pipelines.

Source bins, output bins — composite elements with ghost pads.
"""

import os
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
from ._elements import make_element, has_nvenc, create_encoder


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


def create_nvurisrcbin_bin(name, uri, logger, file_loop=False, platform_info=None):
    """Create a source bin using nvurisrcbin.

    Per NVIDIA reference apps (deepstream-test3:L190), when looping a file the
    underlying decoder needs `cudadec-memtype=0` on x86 dGPU only; on Jetson
    iGPU the decoder uses its default Tegra memory type.
    """
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
        if platform_info is not None and not platform_info.is_integrated_gpu():
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


def create_source_bin(index, uri, logger, file_loop=False, platform_info=None):
    """Create a source bin for reading from a URI, file path, or USB camera."""
    name = f"input_{index}"

    if uri.startswith("/dev/video"):
        return create_usbcam_source_bin(name, uri, logger)
    else:
        if not uri.startswith(("rtsp://", "http://", "https://", "file://")):
            uri = "file://" + os.path.abspath(uri)
        return create_nvurisrcbin_bin(name, uri, logger, file_loop=file_loop,
                                      platform_info=platform_info)


def attach_sources(pipeline, streammux, uris, logger, file_loop=False, platform_info=None):
    """Create one source bin per URI and link each to streammux.sink_{i}.

    Also flips streammux into live-source mode when any URI is RTSP.
    Returns True if any URI is a live (RTSP) source, False otherwise.
    """
    is_live = False
    for i, uri in enumerate(uris):
        if uri.startswith("rtsp://"):
            is_live = True
        src = create_source_bin(i, uri, logger, file_loop=file_loop,
                                platform_info=platform_info)
        pipeline.add(src)
        sinkpad = streammux.request_pad_simple(f"sink_{i}")
        srcpad = src.get_static_pad("src")
        srcpad.link(sinkpad)
    if is_live:
        streammux.set_property("live-source", 1)
    return is_live


# --- Output bins ---

def _build_postosd_nvvidconv(name, platform_info, logger):
    """Post-OSD nvvideoconvert shared by RTSP + record output bins."""
    nvvidconv = make_element("nvvideoconvert", f"nvvidconv-postosd-{name}", logger)
    if not has_nvenc() and platform_info.is_platform_aarch64():
        # Orin Nano (aarch64 + no NVENC): force GPU compute. Without this,
        # nvvideoconvert can fall to VIC which cannot transform CUDA_UNIFIED
        # surfaces that arrive when `is_integrated_gpu()` misreports False due
        # to a failing `cudaGetDeviceProperties` probe.
        nvvidconv.set_property("compute-hw", 1)
    elif not platform_info.is_platform_aarch64():
        # x86 dGPU: CUDA_UNIFIED memory per NVIDIA reference apps.
        nvvidconv.set_property("nvbuf-memory-type", 3)
    return nvvidconv


def _build_enc_caps_and_queue(name, logger):
    """Encoder-input caps + leaky queue shared by RTSP + record output bins.

    SW x264/x265enc needs system-memory caps; HW NVENC wants NVMM. Leaky queue
    drops oldest frames instead of stalling when the encoder can't keep up.
    """
    caps_str = "video/x-raw(memory:NVMM), format=I420" if has_nvenc() else "video/x-raw, format=I420"
    capsfilter = make_element("capsfilter", f"capsfilter-{name}", logger)
    capsfilter.set_property("caps", Gst.Caps.from_string(caps_str))

    enc_queue = make_element("queue", f"enc-queue-{name}", logger)
    enc_queue.set_property("leaky", 2)  # 2 = downstream (drop oldest)
    enc_queue.set_property("max-size-buffers", 4)
    enc_queue.set_property("max-size-bytes", 0)
    enc_queue.set_property("max-size-time", 0)
    return capsfilter, enc_queue


def create_rtsp_output_bin(name, codec, bitrate, platform_info, logger):
    """Create an RTSP output bin: nvvideoconvert -> capsfilter -> encoder -> rtppay -> udpsink.

    Args:
        codec: "H264" or "H265"
        bitrate: encoding bitrate (e.g. 4000000, bits/s — converted for SW enc)
        platform_info: PlatformInfo instance

    HW vs SW encoder is auto-selected via `has_nvenc()`. No `enc_type` flag
    is needed in app YAMLs.
    """
    bin_name = f"{name}-rtsp-output-bin"
    nbin = Gst.Bin.new(bin_name)

    nvvidconv = _build_postosd_nvvidconv(name, platform_info, logger)
    capsfilter, enc_queue = _build_enc_caps_and_queue(name, logger)
    encoder = create_encoder(name, codec, bitrate, platform_info, logger)

    rtppay_factory = "rtph264pay" if codec == "H264" else "rtph265pay"
    rtppay = make_element(rtppay_factory, f"rtppay-{name}", logger)

    udpsink = make_element("udpsink", f"udpsink-{name}", logger)
    udpsink.set_property("host", "224.224.255.255")
    udpsink.set_property("port", 5400)
    udpsink.set_property("async", False)
    udpsink.set_property("sync", 0)

    for el in (nvvidconv, capsfilter, enc_queue, encoder, rtppay, udpsink):
        nbin.add(el)

    nvvidconv.link(capsfilter)
    capsfilter.link(enc_queue)
    enc_queue.link(encoder)
    encoder.link(rtppay)
    rtppay.link(udpsink)

    ghost_sink = Gst.GhostPad.new("sink", nvvidconv.get_static_pad("sink"))
    nbin.add_pad(ghost_sink)

    return nbin


def create_record_output_bin(name, location, codec, bitrate, platform_info, logger):
    """Create a file-record output bin: nvvidconv -> capsfilter -> encoder -> parser -> matroskamux -> filesink.

    Same encoder config as create_rtsp_output_bin, but terminates in filesink
    instead of udpsink. Matroska container tolerates truncation on abrupt
    shutdown better than mp4.
    """
    bin_name = f"{name}-record-output-bin"
    nbin = Gst.Bin.new(bin_name)

    nvvidconv = _build_postosd_nvvidconv(name, platform_info, logger)
    capsfilter, enc_queue = _build_enc_caps_and_queue(name, logger)
    encoder = create_encoder(name, codec, bitrate, platform_info, logger)

    parser_factory = "h264parse" if codec == "H264" else "h265parse"
    parser = make_element(parser_factory, f"parser-{name}", logger)

    muxer = make_element("matroskamux", f"mkvmux-{name}", logger)

    filesink = make_element("filesink", f"filesink-{name}", logger)
    filesink.set_property("location", location)
    filesink.set_property("sync", False)
    filesink.set_property("async", False)

    for el in (nvvidconv, capsfilter, enc_queue, encoder, parser, muxer, filesink):
        nbin.add(el)

    nvvidconv.link(capsfilter)
    capsfilter.link(enc_queue)
    enc_queue.link(encoder)
    encoder.link(parser)
    parser.link(muxer)
    muxer.link(filesink)

    ghost_sink = Gst.GhostPad.new("sink", nvvidconv.get_static_pad("sink"))
    nbin.add_pad(ghost_sink)
    return nbin
