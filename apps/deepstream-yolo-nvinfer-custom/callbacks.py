import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
from ds_pipeline import (
    get_batch_meta, iter_frames, iter_objects,
    add_osd_text,
)


def osd_probe(pad, info, u_data):
    """Probe on OSD sink pad: display detection count.

    nvinfer + custom parser already creates NvDsObjectMeta with labels
    from the labelfile. No tensor parsing needed here.
    """
    batch_meta, _ = get_batch_meta(info)
    if not batch_meta:
        return Gst.PadProbeReturn.OK
    for frame in iter_frames(batch_meta):
        text = f"Frame={frame.frame_num} Objects={frame.num_obj_meta}"
        add_osd_text(batch_meta, frame, text)
        print(text)
    return Gst.PadProbeReturn.OK
