import sys
import gi
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import GLib, Gst, GstRtspServer
from common.bus_call import bus_call


def make_element(factory_name, element_name, logger):
    """Create a GStreamer element, exit on failure."""
    element = Gst.ElementFactory.make(factory_name, element_name)
    if not element:
        logger.error(f"Unable to create {element_name} ({factory_name})")
        sys.exit(1)
    return element


def run_pipeline(pipeline, logger):
    """Start pipeline, run GLib main loop, clean up on exit.

    GStreamer pipelines run asynchronously — elements process data in their own
    threads.  A GLib MainLoop keeps the calling thread alive and dispatches bus
    messages (errors, EOS, state-changes) so we can react to them.

    Flow:
        1. Create a MainLoop for event processing.
        2. Wire the pipeline's message bus into the loop via bus_call(),
           which quits the loop on EOS or error.
        3. Set the pipeline to PLAYING — data starts flowing.
        4. Block on loop.run() until bus_call() calls loop.quit().
        5. Set the pipeline to NULL — releases all resources (GPU memory,
           file handles, decoder sessions, etc.).
    """
    # Event loop that keeps the program alive while the pipeline runs
    loop = GLib.MainLoop()
    # Pipeline message bus — every element posts messages here
    bus = pipeline.get_bus()
    # Emit GLib signals for bus messages so the MainLoop can handle them
    bus.add_signal_watch()
    # Route all bus messages to bus_call(); it calls loop.quit() on EOS/error
    bus.connect("message", bus_call, loop)

    logger.info("Starting pipeline")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass
    pipeline.set_state(Gst.State.NULL)


def start_rtsp_server(rtsp_port, udp_port, mount, codec, logger):
    """Start a GStreamer RTSP server that reads from a UDP source.

    Must be called before run_pipeline() so the server is ready
    when the pipeline starts pushing data to the UDP sink.
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
