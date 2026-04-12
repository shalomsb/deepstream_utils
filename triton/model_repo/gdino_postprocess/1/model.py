"""Grounding DINO postprocessor: positive-map scoring, topK, NMS, box decode.

Post-processing based on public open-source implementations:
  - IDEA-Research/GroundingDINO (Apache 2.0): groundingdino/models/GroundingDINO/groundingdino.py
  - HuggingFace Transformers (Apache 2.0): transformers/models/grounding_dino/modeling_grounding_dino.py
  - Paper: https://arxiv.org/abs/2303.05499

Input:  pred_logits [1, 900, 256] float32 — per-token logits
        pred_boxes  [1, 900, 4]   float32 — cxcywh normalized [0,1]
        raw_frame   [1, 3, H, W]  float32 — for frame dimensions
        pos_map     [1, num_cats, 256] float32 — token-to-category mapping
Output: boxes          [-1, 4]  float32 — xyxy frame-space pixels
        scores         [-1]     float32
        class_ids      [-1]     int32
        num_detections [1]      int32
"""

import json
import sys
import numpy as np
import triton_python_backend_utils as pb_utils

MAX_DETECTIONS = 300


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        params = model_config.get("parameters", {})
        config_path = params.get("config_file", {}).get("string_value", "")

        import yaml
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.categories = config.get("labels", [])
        self.conf_threshold = float(config.get("conf_threshold", 0.3))
        self.nms_threshold = float(config.get("nms_threshold", 0.5))
        self.num_select = int(config.get("num_select", 300))
        self._debug_count = 0

    def execute(self, requests):
        responses = []
        for request in requests:
            pred_logits = pb_utils.get_input_tensor_by_name(request, "pred_logits").as_numpy()
            pred_boxes = pb_utils.get_input_tensor_by_name(request, "pred_boxes").as_numpy()
            raw_frame = pb_utils.get_input_tensor_by_name(request, "raw_frame").as_numpy()
            pos_map = pb_utils.get_input_tensor_by_name(request, "pos_map").as_numpy()

            _, _, frame_h, frame_w = raw_frame.shape

            # Handle NaN from model (e.g. first black frame)
            if np.isnan(pred_logits).any() or np.isnan(pred_boxes).any():
                pred_logits = np.nan_to_num(pred_logits, nan=-100.0)
                pred_boxes = np.nan_to_num(pred_boxes, nan=0.0)

            # Based on: IDEA-Research/GroundingDINO (Apache 2.0)
            # Source: groundingdino/models/GroundingDINO/groundingdino.py :: forward()
            # Step 1: sigmoid on per-token logits -> per-token probabilities
            prob_to_token = sigmoid(pred_logits[0])  # [900, 256]

            # Step 2: normalize pos_map rows so each category sums to 1
            # Based on: groundingdino/models/GroundingDINO/groundingdino.py
            pm = pos_map[0].astype(np.float64)  # [num_cats, 256]
            row_sums = pm.sum(axis=1, keepdims=True)
            row_sums = np.where(row_sums != 0, row_sums, 1.0)
            pm = (pm / row_sums).astype(np.float32)

            # Step 3: map token probs to category probs via pos_map
            # prob_to_label[q, c] = sum of token probs for category c at query q
            prob_to_label = prob_to_token @ pm.T  # [900, num_cats]

            if self._debug_count < 2:
                self._debug_count += 1
                print(f"[POSTPROCESS] prob_to_label shape={prob_to_label.shape} "
                      f"max={prob_to_label.max():.4f}", file=sys.stderr, flush=True)

            # Based on: HuggingFace Transformers (Apache 2.0)
            # Source: transformers/models/grounding_dino/modeling_grounding_dino.py
            # Step 4: topK selection across all (query, category) pairs
            flat_scores = prob_to_label.flatten()  # [900 * num_cats]
            num_select = min(self.num_select, len(flat_scores))
            topk_indices = np.argpartition(flat_scores, -num_select)[-num_select:]
            topk_indices = topk_indices[np.argsort(flat_scores[topk_indices])[::-1]]

            scores = flat_scores[topk_indices]
            query_indices = topk_indices // prob_to_label.shape[1]
            class_ids = (topk_indices % prob_to_label.shape[1]).astype(np.int32)

            # Step 5: confidence filter
            above_threshold = scores > self.conf_threshold
            if not above_threshold.any():
                responses.append(self._empty_response())
                continue

            scores = scores[above_threshold]
            class_ids = class_ids[above_threshold]
            query_indices = query_indices[above_threshold]

            # Step 6: decode boxes cxcywh -> xyxy, scale to frame pixels
            # Based on: groundingdino/util/box_ops.py :: box_cxcywh_to_xyxy()
            selected_boxes = pred_boxes[0][query_indices]  # [N, 4]
            boxes_xyxy = box_cxcywh_to_xyxy(selected_boxes)

            # Scale normalized [0,1] to frame pixels and clamp
            target_sizes = np.array([frame_w, frame_h, frame_w, frame_h], dtype=np.float32)
            boxes_scaled = boxes_xyxy * target_sizes
            boxes_scaled[:, 0::2] = np.clip(boxes_scaled[:, 0::2], 0, frame_w)
            boxes_scaled[:, 1::2] = np.clip(boxes_scaled[:, 1::2], 0, frame_h)

            # Step 7: per-class NMS
            keep = np.array(
                nms_per_class(boxes_scaled, scores, class_ids, self.nms_threshold),
                dtype=np.int64,
            )
            if len(keep) == 0:
                responses.append(self._empty_response())
                continue

            kept = keep[:MAX_DETECTIONS]
            boxes_out = np.ascontiguousarray(boxes_scaled[kept], dtype=np.float32)
            scores_out = np.ascontiguousarray(scores[kept], dtype=np.float32)
            class_ids_out = np.ascontiguousarray(class_ids[kept], dtype=np.int32)
            num_det = np.array([len(kept)], dtype=np.int32)

            responses.append(pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("boxes", boxes_out),
                pb_utils.Tensor("scores", scores_out),
                pb_utils.Tensor("class_ids", class_ids_out),
                pb_utils.Tensor("num_detections", num_det),
            ]))
        return responses

    @staticmethod
    def _empty_response():
        return pb_utils.InferenceResponse(output_tensors=[
            pb_utils.Tensor("boxes", np.zeros((1, 4), dtype=np.float32)),
            pb_utils.Tensor("scores", np.zeros((1,), dtype=np.float32)),
            pb_utils.Tensor("class_ids", np.zeros((1,), dtype=np.int32)),
            pb_utils.Tensor("num_detections", np.array([0], dtype=np.int32)),
        ])

    def finalize(self):
        pass


# ---------------------------------------------------------------------------
# Post-processing utilities
# ---------------------------------------------------------------------------


def sigmoid(x):
    """Numerically stable sigmoid."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def box_cxcywh_to_xyxy(boxes):
    """Convert boxes from center format to corner format.

    Based on: IDEA-Research/GroundingDINO (Apache 2.0)
    Source: groundingdino/util/box_ops.py :: box_cxcywh_to_xyxy()
    """
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    return np.stack([x1, y1, x2, y2], axis=-1)


def iou_batch(box_a, boxes_b):
    """Compute IoU between one box [1,4] and N boxes [N,4]. All xyxy format."""
    x1 = np.maximum(box_a[:, 0], boxes_b[:, 0])
    y1 = np.maximum(box_a[:, 1], boxes_b[:, 1])
    x2 = np.minimum(box_a[:, 2], boxes_b[:, 2])
    y2 = np.minimum(box_a[:, 3], boxes_b[:, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (box_a[:, 2] - box_a[:, 0]) * (box_a[:, 3] - box_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    return inter / (area_a + area_b - inter + 1e-6)


def nms_per_class(boxes, scores, class_ids, iou_threshold):
    """Per-class greedy NMS using pure numpy."""
    keep = []
    for cls in np.unique(class_ids):
        cls_mask = class_ids == cls
        cls_idx = np.where(cls_mask)[0]
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]

        order = cls_scores.argsort()[::-1]
        cls_idx = cls_idx[order]
        cls_boxes = cls_boxes[order]

        while len(cls_idx) > 0:
            keep.append(cls_idx[0])
            if len(cls_idx) == 1:
                break
            ious = iou_batch(cls_boxes[0:1], cls_boxes[1:])
            remaining = ious <= iou_threshold
            cls_idx = cls_idx[1:][remaining]
            cls_boxes = cls_boxes[1:][remaining]
    return keep
