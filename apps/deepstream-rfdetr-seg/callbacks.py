import numpy as np
import cv2
import pyds
import torch
import torch.nn.functional as F
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
from ds_pipeline import (
    get_batch_meta, iter_frames, iter_objects, iter_output_tensors,
    get_layer_data, add_osd_text, set_obj_label,
)
from config import Config


_labels = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_labels(path):
    global _labels
    if _labels is None:
        with open(path, 'r') as f:
            _labels = [line.strip() for line in f if line.strip()]
    return _labels


def _add_obj_meta_with_mask(batch_meta, frame_meta, left, top, width, height,
                             class_id, confidence, unique_id,
                             mask=None, mask_threshold=0.5):
    """Create NvDsObjectMeta with optional instance mask for nvosd GPU rendering."""
    obj_meta = pyds.nvds_acquire_obj_meta_from_pool(batch_meta)
    obj_meta.unique_component_id = unique_id
    obj_meta.confidence = confidence
    obj_meta.class_id = class_id
    obj_meta.object_id = 0xffffffffffffffff

    rect = obj_meta.rect_params
    rect.left = left
    rect.top = top
    rect.width = width
    rect.height = height
    rect.border_width = 2
    rect.border_color.set(0.0, 1.0, 0.0, 0.8)

    if mask is not None and mask.size > 0:
        mask_params = pyds.NvOSD_MaskParams.cast(obj_meta.mask_params)
        mask_params.height, mask_params.width = mask.shape[:2]
        mask_params.size = mask_params.height * mask_params.width * 4
        mask_params.threshold = mask_threshold
        allocated_mask_array = mask_params.alloc_mask_array()
        mask_resized = cv2.resize(mask, (int(mask_params.width), int(mask_params.height)),
                                   interpolation=cv2.INTER_LINEAR)
        np.copyto(allocated_mask_array, mask_resized.flatten())
        del allocated_mask_array

    pyds.nvds_add_obj_meta_to_frame(frame_meta, obj_meta, None)
    return obj_meta


def parse_rfdetr_seg_output(tensor_meta, net_w, net_h, mux_w, mux_h,
                             conf_threshold):
    """Parse RF-DETR Seg tensors using torch on GPU.

    RF-DETR Seg outputs:
      - pred_boxes:  [num_queries, 4] -- cxcywh normalized [0,1]
      - pred_logits: [num_queries, num_classes] -- raw logits
      - pred_masks:  [num_queries, mask_h, mask_w] -- mask logits
    """
    # Read tensors as numpy
    layer0 = get_layer_data(tensor_meta, 0)
    layer1 = get_layer_data(tensor_meta, 1)

    if layer0.ndim == 3:
        layer0 = layer0[0]
    if layer1.ndim == 3:
        layer1 = layer1[0]

    try:
        masks_np = get_layer_data(tensor_meta, 2)
        if masks_np.ndim == 4:
            masks_np = masks_np[0]
    except Exception:
        masks_np = None

    # Identify boxes vs logits
    if layer0.shape[-1] == 4:
        boxes_np, logits_np = layer0, layer1
    else:
        logits_np, boxes_np = layer0, layer1

    # Move to GPU
    boxes_t = torch.from_numpy(boxes_np.copy()).to(_device)
    logits_t = torch.from_numpy(logits_np.copy()).to(_device)

    # Softmax + confidence filter on GPU
    scores_t = torch.softmax(logits_t, dim=-1)
    max_scores_t, class_ids_t = scores_t.max(dim=-1)
    keep = max_scores_t > conf_threshold

    if not keep.any():
        return []

    boxes_t = boxes_t[keep]
    class_ids_t = class_ids_t[keep]
    max_scores_t = max_scores_t[keep]

    # Convert cxcywh normalized to pixel coords in muxer space
    scale = min(net_w / mux_w, net_h / mux_h)
    pad_x = (net_w - mux_w * scale) / 2
    pad_y = (net_h - mux_h * scale) / 2

    cx = boxes_t[:, 0] * net_w
    cy = boxes_t[:, 1] * net_h
    w = boxes_t[:, 2] * net_w
    h = boxes_t[:, 3] * net_h
    left_t = (cx - w / 2 - pad_x) / scale
    top_t = (cy - h / 2 - pad_y) / scale
    w_t = w / scale
    h_t = h / scale

    # Move detection results to CPU
    left_cpu = left_t.cpu().numpy()
    top_cpu = top_t.cpu().numpy()
    w_cpu = w_t.cpu().numpy()
    h_cpu = h_t.cpu().numpy()
    class_ids_cpu = class_ids_t.cpu().numpy()
    max_scores_cpu = max_scores_t.cpu().numpy()

    # Process masks on GPU: sigmoid + upsample logits to net size, remove padding, resize to muxer
    mask_crops = [None] * len(left_cpu)
    if masks_np is not None:
        masks_t = torch.from_numpy(masks_np.copy()).to(_device)
        masks_t = masks_t[keep]  # [K, mask_h, mask_w]

        # Upsample logits to network input size on GPU (before sigmoid for sharper edges)
        masks_t = masks_t.unsqueeze(1)  # [K, 1, mask_h, mask_w]
        masks_t = F.interpolate(masks_t, size=(net_h, net_w), mode='bilinear', align_corners=False)

        # Remove padding
        y_start, y_end = int(pad_y), int(net_h - pad_y)
        x_start, x_end = int(pad_x), int(net_w - pad_x)
        masks_t = masks_t[:, :, y_start:y_end, x_start:x_end]

        # Resize to muxer resolution
        masks_t = F.interpolate(masks_t, size=(mux_h, mux_w), mode='bilinear', align_corners=False)

        # Sigmoid on GPU (after upsample for sharp edges)
        masks_t = torch.sigmoid(masks_t).squeeze(1)  # [K, mux_h, mux_w]

        # Crop each mask to its bbox and transfer to CPU
        for i in range(len(left_cpu)):
            ix1 = max(0, int(left_cpu[i]))
            iy1 = max(0, int(top_cpu[i]))
            ix2 = min(int(left_cpu[i] + w_cpu[i]), mux_w)
            iy2 = min(int(top_cpu[i] + h_cpu[i]), mux_h)

            if (ix2 - ix1) >= 4 and (iy2 - iy1) >= 4:
                crop = masks_t[i, iy1:iy2, ix1:ix2].cpu().numpy()
                mask_crops[i] = (np.clip(crop, 0, 1) * 255).astype(np.uint8)

    results = []
    for i in range(len(left_cpu)):
        results.append({
            'class_id': int(class_ids_cpu[i]),
            'confidence': float(max_scores_cpu[i]),
            'left': float(left_cpu[i]),
            'top': float(top_cpu[i]),
            'width': float(w_cpu[i]),
            'height': float(h_cpu[i]),
            'mask': mask_crops[i],
        })
    return results


def pgie_src_probe(pad, info, u_data: Config):
    """Probe on PGIE src pad: parse RF-DETR Seg tensors -> NvDsObjectMeta + masks."""
    config = u_data
    batch_meta, _ = get_batch_meta(info)
    if not batch_meta:
        return Gst.PadProbeReturn.OK

    for frame in iter_frames(batch_meta):
        for tensor_meta in iter_output_tensors(frame):
            results = parse_rfdetr_seg_output(
                tensor_meta,
                config.network_width, config.network_height,
                config.streammux_width, config.streammux_height,
                config.conf_threshold,
            )
            for det in results:
                _add_obj_meta_with_mask(
                    batch_meta, frame,
                    det['left'], det['top'], det['width'], det['height'],
                    class_id=det['class_id'],
                    confidence=det['confidence'],
                    unique_id=tensor_meta.unique_id,
                    mask=det['mask'],
                    mask_threshold=config.mask_threshold,
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
