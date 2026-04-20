from gi.repository import Gst
from ds_pipeline import get_batch_meta, iter_frames


def osd_probe(pad, info, u_data):
    """Optional probe on nvosd sink — iterates frames.

    The YOLO26 C++ parser already writes NvDsObjectMeta with class labels
    and confidences, so no per-object rewrite is needed here. Hook point
    kept for future overlays (custom text, per-stream counters, etc.).
    """
    batch_meta, _ = get_batch_meta(info)
    if batch_meta is None:
        return Gst.PadProbeReturn.OK
    for _frame in iter_frames(batch_meta):
        pass
    return Gst.PadProbeReturn.OK
