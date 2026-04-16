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


def parse_rfdetr_output(tensor_meta, net_w, net_h, mux_w, mux_h,
                         conf_threshold):
    """Parse RF-DETR output tensors into detections.

    RF-DETR outputs two tensors:
      - pred_boxes:  [num_queries, 4] -- cxcywh normalized [0,1]
      - pred_logits: [num_queries, num_classes] -- raw logits (pre-softmax)

    No NMS needed -- transformer decoder handles deduplication.
    """
    boxes = get_layer_data(tensor_meta, 0)
    logits = get_layer_data(tensor_meta, 1)

    # Handle optional batch dimension
    if boxes.ndim == 3:
        boxes = boxes[0]
    if logits.ndim == 3:
        logits = logits[0]

    # Auto-detect tensor order by shape
    if boxes.shape[-1] != 4:
        boxes, logits = logits, boxes

    # Softmax on logits
    logits_shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_logits = np.exp(logits_shifted)
    scores = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    class_ids = np.argmax(scores, axis=-1)
    max_scores = np.max(scores, axis=-1)

    # Confidence filter
    mask = max_scores > conf_threshold
    if not mask.any():
        return []

    boxes = boxes[mask]
    class_ids = class_ids[mask]
    max_scores = max_scores[mask]

    # Convert normalized cxcywh to pixel-space left,top,w,h
    cx = boxes[:, 0] * net_w
    cy = boxes[:, 1] * net_h
    w  = boxes[:, 2] * net_w
    h  = boxes[:, 3] * net_h
    left = cx - w / 2
    top  = cy - h / 2

    # Scale from network input to muxer output coordinates
    # (maintain-aspect-ratio=1 + symmetric-padding=1)
    scale = min(net_w / mux_w, net_h / mux_h)
    pad_x = (net_w - mux_w * scale) / 2
    pad_y = (net_h - mux_h * scale) / 2

    detections = []
    for i in range(len(left)):
        detections.append({
            'class_id': int(class_ids[i]),
            'confidence': float(max_scores[i]),
            'left': float((left[i] - pad_x) / scale),
            'top': float((top[i] - pad_y) / scale),
            'width': float(w[i] / scale),
            'height': float(h[i] / scale),
        })
    return detections


def pgie_src_probe(pad, info, u_data: Config):
    """Probe on PGIE src pad: parse raw RF-DETR tensors -> NvDsObjectMeta."""
    config = u_data
    batch_meta, _ = get_batch_meta(info)
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    for frame in iter_frames(batch_meta):
        for tensor_meta in iter_output_tensors(frame):
            detections = parse_rfdetr_output(
                tensor_meta,
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
    return Gst.PadProbeReturn.OK
