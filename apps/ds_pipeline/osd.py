"""OSD (On-Screen Display) helpers for DeepStream pipelines.

Convenience functions for simple detection apps. For production apps with
custom rendering, use pyds directly.
"""

import pyds
from .meta import iter_objects


def add_osd_text(batch_meta, frame_meta, text,
                 x=10, y=12,
                 font_name="Serif", font_size=10,
                 font_color=(1.0, 1.0, 1.0, 1.0),
                 bg_color=(0.0, 0.0, 0.0, 1.0)):
    """Acquire display_meta from pool and add an OSD text overlay to a frame."""
    display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
    display_meta.num_labels = 1
    params = display_meta.text_params[0]
    params.display_text = text
    params.x_offset = x
    params.y_offset = y
    params.font_params.font_name = font_name
    params.font_params.font_size = font_size
    params.font_params.font_color.set(*font_color)
    params.set_bg_clr = 1
    params.text_bg_clr.set(*bg_color)
    pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)


def set_border_color(obj_meta, r, g, b, a=0.8):
    """Set bounding box border color on an object meta."""
    obj_meta.rect_params.border_color.set(r, g, b, a)


def set_obj_label(obj_meta, text,
                  font_name="Serif", font_size=10,
                  font_color=(1.0, 1.0, 1.0, 1.0),
                  bg_color=(0.0, 0.0, 0.0, 0.6)):
    """Set display label on an object's bounding box.

    Positions the text above the top-left corner of the bounding box.
    """
    obj_meta.text_params.display_text = text
    obj_meta.text_params.font_params.font_name = font_name
    obj_meta.text_params.font_params.font_size = font_size
    obj_meta.text_params.font_params.font_color.set(*font_color)
    obj_meta.text_params.set_bg_clr = 1
    obj_meta.text_params.text_bg_clr.set(*bg_color)
    obj_meta.text_params.x_offset = int(obj_meta.rect_params.left)
    obj_meta.text_params.y_offset = max(0, int(obj_meta.rect_params.top) - 12)


def count_objects(frame_meta, class_ids=None):
    """Count objects per class_id in a frame.

    Args:
        frame_meta: NvDsFrameMeta
        class_ids: Optional list/set of class_ids to count.
                   If None, counts all class_ids found.

    Returns:
        dict mapping class_id -> count
    """
    counts = {}
    if class_ids is not None:
        counts = {cid: 0 for cid in class_ids}

    for obj in iter_objects(frame_meta):
        cid = obj.class_id
        if class_ids is None or cid in counts:
            counts[cid] = counts.get(cid, 0) + 1

    return counts
