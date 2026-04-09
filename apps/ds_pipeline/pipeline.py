"""Pipeline lifecycle and linking utilities."""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import GLib, Gst
from common.bus_call import bus_call


def link_chain(*elements):
    """Link a chain of GStreamer elements in order.

    Raises RuntimeError if any link fails.

    Usage:
        link_chain(streammux, q1, pgie, q2, tracker, nvvidconv, nvosd, sink)
    """
    for a, b in zip(elements, elements[1:]):
        if not a.link(b):
            raise RuntimeError(
                f"Failed to link {a.get_name()} -> {b.get_name()}")


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
    loop = GLib.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect("message", bus_call, loop)

    logger.info("Starting pipeline")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted")
    pipeline.set_state(Gst.State.NULL)
