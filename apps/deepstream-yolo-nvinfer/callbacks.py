import numpy as np
import cv2
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


def parse_yolo_output(tensor_meta, net_w, net_h, mux_w, mux_h,
                       conf_threshold, nms_threshold):
    """Parse raw YOLO11x output [84, 8400] into detections.

    Coordinates are converted from network input space (640x640) to
    muxer output space (mux_w x mux_h), which is what nvinfer sees.
    """
    output = get_layer_data(tensor_meta, 0)  # [84, 8400]
    output = output.T                         # [8400, 84]

    boxes_raw = output[:, :4]                 # cx, cy, w, h (network input space)
    scores = output[:, 4:]                    # [8400, 80] class scores

    class_ids = np.argmax(scores, axis=1)
    max_scores = np.max(scores, axis=1)

    mask = max_scores > conf_threshold
    if not mask.any():
        return []

    boxes_raw = boxes_raw[mask]
    class_ids = class_ids[mask]
    max_scores = max_scores[mask]

    # cx,cy,w,h → x,y,w,h for NMS
    x = boxes_raw[:, 0] - boxes_raw[:, 2] / 2
    y = boxes_raw[:, 1] - boxes_raw[:, 3] / 2
    w = boxes_raw[:, 2]
    h = boxes_raw[:, 3]

    nms_boxes = np.stack([x, y, w, h], axis=1)
    indices = cv2.dnn.NMSBoxes(
        nms_boxes.tolist(), max_scores.tolist(), conf_threshold, nms_threshold,
    )
    if len(indices) == 0:
        return []
    keep = indices.flatten()

    # Scale from network input to muxer output coordinates
    # nvinfer with maintain-aspect-ratio=1 + symmetric-padding=1:
    scale = min(net_w / mux_w, net_h / mux_h)
    pad_x = (net_w - mux_w * scale) / 2
    pad_y = (net_h - mux_h * scale) / 2

    detections = []
    for i in keep:
        detections.append({
            'class_id': int(class_ids[i]),
            'confidence': float(max_scores[i]),
            'left': float((x[i] - pad_x) / scale),
            'top': float((y[i] - pad_y) / scale),
            'width': float(w[i] / scale),
            'height': float(h[i] / scale),
        })
    return detections


def pgie_src_probe(pad, info, u_data: Config):
    """Probe on PGIE src pad: parse raw YOLO tensors → NvDsObjectMeta."""
    config = u_data
    batch_meta, _ = get_batch_meta(info)
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    for frame in iter_frames(batch_meta):
        for tensor_meta in iter_output_tensors(frame):
            detections = parse_yolo_output(
                tensor_meta,
                config.network_width, config.network_height,
                config.streammux_width, config.streammux_height,
                config.conf_threshold, config.nms_threshold,
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
