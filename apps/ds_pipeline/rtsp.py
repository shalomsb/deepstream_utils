"""RTSP server utilities."""

import gi
gi.require_version('GstRtspServer', '1.0')
from gi.repository import GstRtspServer


def start_rtsp_server(rtsp_port, udp_port, mount, codec, logger):
    """Start a GStreamer RTSP server that reads from a UDP source.

    Must be called before run_pipeline() so the server is ready
    when the pipeline starts pushing data to the UDP sink.

    Returns the server object (must be kept alive).
    """
    server = GstRtspServer.RTSPServer.new()
    server.props.service = str(rtsp_port)
    server.attach(None)

    factory = GstRtspServer.RTSPMediaFactory.new()
    factory.set_launch(
        '( udpsrc name=pay0 port=%d buffer-size=524288 '
        'caps="application/x-rtp, media=video, clock-rate=90000, '
        'encoding-name=(string)%s, payload=96" )' % (udp_port, codec)
    )
    factory.set_shared(True)
    server.get_mount_points().add_factory(mount, factory)

    logger.info(f"RTSP Streaming at rtsp://localhost:{rtsp_port}{mount}")
    return server
