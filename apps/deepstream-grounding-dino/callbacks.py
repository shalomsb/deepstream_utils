import ctypes
import numpy as np
import pyds
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
from ds_pipeline import (
    get_batch_meta, iter_frames, iter_output_tensors,
    get_layer_data, add_obj_meta, iter_objects,
    add_osd_text, set_obj_label,
)
from config import Config


def _parse_gdino_output(tensor_meta, frame_w, frame_h):
    """Read postprocessed ensemble outputs: boxes, scores, class_ids, num_detections.

    Layer ordering may vary — look up by name.
    """
    outputs = {}
    for i in range(tensor_meta.num_output_layers):
        layer = pyds.get_nvds_LayerInfo(tensor_meta, i)
        name = layer.layerName
        if name in ("num_detections", "class_ids"):
            outputs[name] = get_layer_data(tensor_meta, i, dtype=ctypes.c_int32)
        else:
            outputs[name] = get_layer_data(tensor_meta, i)

    num_det_arr = outputs.get("num_detections")
    if num_det_arr is None or num_det_arr.size == 0:
        return []
    num_det = int(num_det_arr.flat[0])
    if num_det == 0:
        return []

    boxes = outputs.get("boxes")
    scores = outputs.get("scores")
    class_ids = outputs.get("class_ids")
    if boxes is None or scores is None:
        return []

    # Ensure 1D/2D even when Triton returns scalars for single detections
    scores = np.atleast_1d(scores)
    boxes = np.atleast_2d(boxes)
    if class_ids is not None:
        class_ids = np.atleast_1d(class_ids)

    detections = []
    for i in range(min(num_det, len(scores))):
        x1, y1, x2, y2 = boxes[i]
        # Clamp to frame and convert xyxy -> left, top, width, height
        left = float(max(0, x1))
        top = float(max(0, y1))
        width = float(min(x2, frame_w) - left)
        height = float(min(y2, frame_h) - top)
        if width <= 0 or height <= 0:
            continue
        detections.append({
            "class_id": int(class_ids[i]) if class_ids is not None else 0,
            "confidence": float(scores[i]),
            "left": left,
            "top": top,
            "width": width,
            "height": height,
        })
    return detections


def pgie_src_probe(pad, info, u_data: Config):
    """Parse ensemble tensor outputs -> NvDsObjectMeta (before tracker)."""
    config = u_data
    batch_meta, _ = get_batch_meta(info)
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    for frame in iter_frames(batch_meta):
        for tensor_meta in iter_output_tensors(frame):
            detections = _parse_gdino_output(
                tensor_meta,
                config.streammux_width,
                config.streammux_height,
            )
            for det in detections:
                add_obj_meta(
                    batch_meta, frame,
                    det["left"], det["top"], det["width"], det["height"],
                    class_id=det["class_id"],
                    confidence=det["confidence"],
                    unique_id=tensor_meta.unique_id,
                )

    return Gst.PadProbeReturn.OK


def osd_probe(pad, info, u_data: Config):
    """Set labels on OSD + print frame stats."""
    config = u_data
    labels = config.labels
    batch_meta, _ = get_batch_meta(info)
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    for frame in iter_frames(batch_meta):
        for obj in iter_objects(frame):
            label = labels[obj.class_id] if obj.class_id < len(labels) else "unknown"
            set_obj_label(obj, f"{label} {obj.confidence:.2f} ID:{obj.object_id}")
        text = f"Frame={frame.frame_num} Objects={frame.num_obj_meta}"
        add_osd_text(batch_meta, frame, text)

    return Gst.PadProbeReturn.OK
