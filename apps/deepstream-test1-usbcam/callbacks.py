import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
from ds_pipeline import (
    get_batch_meta, iter_frames, iter_objects,
    count_objects, add_osd_text, set_border_color,
    PGIE_CLASS_ID_VEHICLE, PGIE_CLASS_ID_PERSON,
)


def osd_probe(pad, info, u_data):
    batch_meta, _ = get_batch_meta(info)
    if not batch_meta:
        return Gst.PadProbeReturn.OK
    for frame in iter_frames(batch_meta):
        counts = count_objects(frame)
        for obj in iter_objects(frame):
            set_border_color(obj, 0.0, 0.0, 1.0)
        text = (f"Frame={frame.frame_num} Objects={frame.num_obj_meta} "
                f"Vehicles={counts.get(PGIE_CLASS_ID_VEHICLE, 0)} "
                f"Persons={counts.get(PGIE_CLASS_ID_PERSON, 0)}")
        add_osd_text(batch_meta, frame, text)
        print(text)
    return Gst.PadProbeReturn.OK
