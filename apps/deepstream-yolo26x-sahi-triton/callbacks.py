"""Probes for the SAHI ensemble app.

The Triton ensemble (sahi_preprocess -> yolo26x_b6 -> sahi_postprocess) returns
already-merged detections in 1280x720 inference coords. This probe only needs
to scale them to streammux coords and create NvDsObjectMeta -- no NMS, no
coord translation, no merge.
"""
import ctypes
import numpy as np
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import pyds
from ds_pipeline import (
    get_batch_meta, iter_frames, iter_objects, iter_output_tensors,
    add_obj_meta, add_osd_text, set_obj_label,
)
from config import Config


_labels = None

def _load_labels(path):
    global _labels
    if _labels is None:
        with open(path, 'r') as f:
            _labels = [line.strip() for line in f if line.strip()]
    return _labels


def _layer_array(tensor_meta, layer_idx, dtype):
    """Read an output layer as a numpy array (handles non-fp32 outputs).

    nvinferserver may strip leading-1 dims, so a [1]-shaped tensor can arrive
    with numDims=0. Treat that as a 1-element scalar.
    """
    layer = pyds.get_nvds_LayerInfo(tensor_meta, layer_idx)
    ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(dtype))
    shape = tuple(layer.inferDims.d[i] for i in range(layer.inferDims.numDims))
    if not shape:
        shape = (1,)
    return np.ctypeslib.as_array(ptr, shape=shape)


def _layers_by_name(tensor_meta):
    """Map layer name -> (index, layer) for the named ensemble outputs."""
    out = {}
    for i in range(tensor_meta.num_output_layers):
        layer = pyds.get_nvds_LayerInfo(tensor_meta, i)
        out[layer.layerName] = (i, layer)
    return out


def pgie_src_probe(pad, info, u_data: Config):
    """Read merged detections from sahi_pipeline output and emit NvDsObjectMeta."""
    config = u_data
    batch_meta, _ = get_batch_meta(info)
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    scale_x = config.streammux_width / config.inference_width
    scale_y = config.streammux_height / config.inference_height

    for frame in iter_frames(batch_meta):
        for tensor_meta in iter_output_tensors(frame):
            layers = _layers_by_name(tensor_meta)
            if "num_dets" not in layers or "boxes" not in layers:
                continue

            num_dets_idx, _ = layers["num_dets"]
            boxes_idx, _ = layers["boxes"]
            scores_idx, _ = layers["scores"]
            classes_idx, _ = layers["classes"]

            num = int(_layer_array(tensor_meta, num_dets_idx, ctypes.c_int32).flat[0])
            if num <= 0:
                frame.bInferDone = True
                continue

            # boxes/scores/classes have dynamic dim -1; nvinferserver reports
            # the runtime size. Use np.atleast_2d / np.atleast_1d to defend
            # against the same dim-stripping quirk as num_dets.
            boxes_flat = _layer_array(tensor_meta, boxes_idx, ctypes.c_float)
            boxes = np.atleast_2d(boxes_flat).reshape(-1, 4)[:num]
            scores = np.atleast_1d(_layer_array(tensor_meta, scores_idx, ctypes.c_float))[:num]
            classes = np.atleast_1d(_layer_array(tensor_meta, classes_idx, ctypes.c_int32))[:num]

            for i in range(num):
                x1, y1, x2, y2 = boxes[i]
                add_obj_meta(
                    batch_meta, frame,
                    left=float(x1 * scale_x),
                    top=float(y1 * scale_y),
                    width=float((x2 - x1) * scale_x),
                    height=float((y2 - y1) * scale_y),
                    class_id=int(classes[i]),
                    confidence=float(scores[i]),
                    unique_id=tensor_meta.unique_id,
                )
            frame.bInferDone = True

    return Gst.PadProbeReturn.OK


def osd_probe(pad, info, u_data: Config):
    """Set labels + display count on OSD."""
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
