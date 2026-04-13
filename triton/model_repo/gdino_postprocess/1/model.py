"""Grounding DINO postprocessor: positive-map scoring, topK, NMS, box decode.

Post-processing based on NVIDIA TAO Deploy (Apache 2.0 License)
and IDEA-Research/GroundingDINO (Apache 2.0 License).
  - TAO Deploy: nvidia_tao_deploy/cv/grounding_dino/utils.py :: post_process()
  - TAO Deploy: nvidia_tao_deploy/cv/deformable_detr/utils.py :: sigmoid(), box_cxcywh_to_xyxy()
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

    def execute(self, requests):
        responses = []
        for request in requests:
            pred_logits = pb_utils.get_input_tensor_by_name(request, "pred_logits").as_numpy()
            pred_boxes = pb_utils.get_input_tensor_by_name(request, "pred_boxes").as_numpy()
            raw_frame = pb_utils.get_input_tensor_by_name(request, "raw_frame").as_numpy()
            pos_map = pb_utils.get_input_tensor_by_name(request, "pos_map").as_numpy()

            _, _, frame_h, frame_w = raw_frame.shape
            bs = pred_logits.shape[0]

            # Handle NaN from model (e.g. first black frame in streaming)
            if np.isnan(pred_logits).any() or np.isnan(pred_boxes).any():
                pred_logits = np.nan_to_num(pred_logits, nan=-100.0)
                pred_boxes = np.nan_to_num(pred_boxes, nan=0.0)

            # ── Post-processing (TAO Deploy style) ────────────────────────
            # Based on: nvidia_tao_deploy/cv/grounding_dino/utils.py :: post_process()

            # Step 1: sigmoid on per-token logits
            prob_to_token = sigmoid(pred_logits)

            # Step 2: normalize pos_map rows so each category sums to 1
            pos_maps = pos_map[0].copy()
            for label_ind in range(len(pos_maps)):
                if pos_maps[label_ind].sum() != 0:
                    pos_maps[label_ind] = pos_maps[label_ind] / pos_maps[label_ind].sum()

            # Step 3: map token probs to category probs via pos_map
            prob_to_label = prob_to_token @ pos_maps.T
            prob = prob_to_label

            # Step 4: topK selection across all (query, category) pairs
            num_select = min(self.num_select, prob.reshape((bs, -1)).shape[1])
            topk_indices = np.argsort(prob.reshape((bs, -1)), axis=1)[:, ::-1][:, :num_select]

            # Step 5: extract scores
            scores = np.array([
                per_batch_prob[ind]
                for per_batch_prob, ind in zip(prob.reshape((bs, -1)), topk_indices)
            ])

            # Step 6: get corresponding boxes and labels
            topk_boxes = topk_indices // prob.shape[2]
            labels = topk_indices % prob.shape[2]

            # Step 7: convert to x1, y1, x2, y2 format
            boxes = box_cxcywh_to_xyxy(pred_boxes)

            # Step 8: take corresponding topk boxes
            boxes = np.take_along_axis(
                boxes,
                np.repeat(np.expand_dims(topk_boxes, -1), 4, axis=-1),
                axis=1,
            )

            # Step 9: scale back to frame pixels and clamp
            target_sizes = np.array([[frame_w, frame_h, frame_w, frame_h]], dtype=np.float32)
            boxes = boxes * target_sizes[:, None, :]

            for i, target_size in enumerate(target_sizes):
                w, h = target_size[0], target_size[1]
                boxes[i, :, 0::2] = np.clip(boxes[i, :, 0::2], 0.0, w)
                boxes[i, :, 1::2] = np.clip(boxes[i, :, 1::2], 0.0, h)

            # ── Our additions for DeepStream streaming ────────────────────

            # Flatten batch dim (bs=1 in DeepStream)
            scores_flat = scores[0]
            labels_flat = labels[0].astype(np.int32)
            boxes_flat = boxes[0]

            # Step 10: confidence threshold filter
            above_threshold = scores_flat > self.conf_threshold
            if not above_threshold.any():
                responses.append(self._empty_response())
                continue

            scores_flat = scores_flat[above_threshold]
            labels_flat = labels_flat[above_threshold]
            boxes_flat = boxes_flat[above_threshold]

            # Step 11: per-class NMS
            keep = np.array(
                nms_per_class(boxes_flat, scores_flat, labels_flat, self.nms_threshold),
                dtype=np.int64,
            )
            if len(keep) == 0:
                responses.append(self._empty_response())
                continue

            kept = keep[:MAX_DETECTIONS]
            boxes_out = np.ascontiguousarray(boxes_flat[kept], dtype=np.float32)
            scores_out = np.ascontiguousarray(scores_flat[kept], dtype=np.float32)
            class_ids_out = np.ascontiguousarray(labels_flat[kept], dtype=np.int32)
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
# Based on: NVIDIA TAO Deploy (Apache 2.0)
# Source: nvidia_tao_deploy/cv/deformable_detr/utils.py
# ---------------------------------------------------------------------------


def sigmoid(x):
    """Numpy-based sigmoid function."""
    return 1 / (1 + np.exp(-x))


def box_cxcywh_to_xyxy(x):
    """Convert box from cxcywh to xyxy.

    Based on: nvidia_tao_deploy/cv/deformable_detr/utils.py :: box_cxcywh_to_xyxy()
    """
    x_c, y_c, w, h = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return np.stack(b, axis=-1)


# ---------------------------------------------------------------------------
# NMS utility (not in TAO Deploy — needed for DeepStream streaming)
# ---------------------------------------------------------------------------


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
