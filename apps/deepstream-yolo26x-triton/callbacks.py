import numpy as np
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
from ds_pipeline import (
    get_batch_meta, iter_frames, iter_objects, iter_output_tensors,
    get_layer_data, add_obj_meta, add_osd_text, set_obj_label,
)
from config import Config


_labels = None

def _load_labels(path):
    global _labels
    if _labels is None:
        with open(path, 'r') as f:
            _labels = [line.strip() for line in f if line.strip()]
    return _labels


def parse_yolo26_output(tensor_meta, batch_id, net_w, net_h, mux_w, mux_h,
                       conf_threshold):
    """Parse YOLO26 (and YOLOv10+) post-NMS output into detections.

    Output tensor: [batch, 300, 6] -- already post-NMS.
    Each row: (x1, y1, x2, y2, conf, cls) in NETWORK pixel coords (0..640).
    Inactive slots have conf=0 -- filter on conf_threshold.

    Returns: list of dicts in MUXER coordinates.
    """
    out = get_layer_data(tensor_meta, 0)

    # nvinferserver may emit either [N, 6] (already per-frame) or
    # [batch, N, 6] (full batch). Handle both.
    if out.ndim == 3:
        out = out[batch_id]

    if out.shape[-1] != 6:
        return []

    conf = out[:, 4]
    mask = conf > conf_threshold
    if not mask.any():
        return []

    out = out[mask]
    x1 = out[:, 0]; y1 = out[:, 1]; x2 = out[:, 2]; y2 = out[:, 3]
    conf = out[:, 4]; cls = out[:, 5].astype(np.int32)

    # Network -> muxer coords (maintain-aspect-ratio + symmetric-padding)
    scale = min(net_w / mux_w, net_h / mux_h)
    pad_x = (net_w - mux_w * scale) / 2
    pad_y = (net_h - mux_h * scale) / 2
    left = (x1 - pad_x) / scale
    top = (y1 - pad_y) / scale
    width = (x2 - x1) / scale
    height = (y2 - y1) / scale

    detections = []
    for i in range(len(left)):
        detections.append({
            'class_id': int(cls[i]),
            'confidence': float(conf[i]),
            'left': float(left[i]),
            'top': float(top[i]),
            'width': float(width[i]),
            'height': float(height[i]),
        })
    return detections


def pgie_src_probe(pad, info, u_data: Config):
    """Probe on PGIE src pad: parse YOLO26 tensors -> NvDsObjectMeta."""
    config = u_data
    batch_meta, _ = get_batch_meta(info)
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    for frame in iter_frames(batch_meta):
        for tensor_meta in iter_output_tensors(frame):
            detections = parse_yolo26_output(
                tensor_meta, frame.batch_id,
                config.network_width, config.network_height,
                config.streammux_width, config.streammux_height,
                config.conf_threshold,
            )
            for det in detections:
                add_obj_meta(
                    batch_meta, frame,
                    det['left'], det['top'], det['width'], det['height'],
                    class_id=det['class_id'],
                    confidence=det['confidence'],
                    unique_id=tensor_meta.unique_id,
                )
            frame.bInferDone = True

    return Gst.PadProbeReturn.OK


def osd_probe(pad, info, u_data: Config):
    """Probe on OSD sink pad: set labels + display count."""
    config = u_data
    labels = _load_labels(config.labels_file)
    batch_meta, _ = get_batch_meta(info)
    if not batch_meta:
        return Gst.PadProbeReturn.OK
    for frame in iter_frames(batch_meta):
        for obj in iter_objects(frame):
            if obj.class_id < len(labels):
                set_obj_label(obj, f"{labels[obj.class_id]} {obj.confidence:.2f} ID:{obj.object_id}")
        text = f"Frame={frame.frame_num} Objects={frame.num_obj_meta}"
        add_osd_text(batch_meta, frame, text)
        print(text)
    return Gst.PadProbeReturn.OK
