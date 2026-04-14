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

import yaml
import torch
from torch.utils.dlpack import from_dlpack, to_dlpack
from torchvision.ops import batched_nms
import triton_python_backend_utils as pb_utils

MAX_DETECTIONS = 300


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        params = model_config.get("parameters", {})
        config_path = params.get("config_file", {}).get("string_value", "")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.categories = config.get("labels", [])
        self.conf_threshold = float(config.get("conf_threshold", 0.3))
        self.nms_threshold = float(config.get("nms_threshold", 0.5))
        self.num_select = int(config.get("num_select", 300))

    def execute(self, requests):
        responses = []
        for request in requests:
            pred_logits = from_dlpack(
                pb_utils.get_input_tensor_by_name(request, "pred_logits").to_dlpack()
            )
            pred_boxes = from_dlpack(
                pb_utils.get_input_tensor_by_name(request, "pred_boxes").to_dlpack()
            )
            raw_frame = from_dlpack(
                pb_utils.get_input_tensor_by_name(request, "raw_frame").to_dlpack()
            )
            pos_map = from_dlpack(
                pb_utils.get_input_tensor_by_name(request, "pos_map").to_dlpack()
            )

            _, _, frame_h, frame_w = raw_frame.shape
            bs = pred_logits.shape[0]

            # Sigmoid + pos_map normalize + matmul (on GPU to avoid CPU scheduling jitter)
            prob_to_token = torch.sigmoid(pred_logits.cuda())
            pos_maps = pos_map[0].cuda()
            row_sums = pos_maps.sum(dim=1, keepdim=True).clamp(min=1.0)
            pos_maps = pos_maps / row_sums
            prob = (prob_to_token @ pos_maps.T).cpu()

            # TopK + index decode
            num_select = min(self.num_select, prob.reshape(bs, -1).shape[1])
            topk_scores, topk_indices = torch.topk(prob.reshape(bs, -1), num_select, dim=1)
            topk_box_indices = topk_indices // prob.shape[2]
            labels = topk_indices % prob.shape[2]

            # Box decode + scale + clamp
            boxes = box_cxcywh_to_xyxy(pred_boxes)
            boxes = torch.gather(boxes, 1, topk_box_indices.unsqueeze(-1).expand(-1, -1, 4))
            scale = torch.tensor([[frame_w, frame_h, frame_w, frame_h]],
                                 dtype=boxes.dtype, device=boxes.device)
            boxes = boxes * scale.unsqueeze(1)
            boxes[..., 0::2] = boxes[..., 0::2].clamp(0.0, float(frame_w))
            boxes[..., 1::2] = boxes[..., 1::2].clamp(0.0, float(frame_h))

            # Confidence filter
            scores_flat = topk_scores[0]
            labels_flat = labels[0]
            boxes_flat = boxes[0]
            above = scores_flat > self.conf_threshold
            if not above.any():
                responses.append(self._empty_response())
                continue
            scores_flat = scores_flat[above]
            labels_flat = labels_flat[above]
            boxes_flat = boxes_flat[above]

            # NMS
            keep = batched_nms(boxes_flat, scores_flat, labels_flat, self.nms_threshold)
            if len(keep) == 0:
                responses.append(self._empty_response())
                continue

            keep = keep[:MAX_DETECTIONS]
            boxes_out = boxes_flat[keep].contiguous()
            scores_out = scores_flat[keep].contiguous()
            class_ids_out = labels_flat[keep].to(torch.int32).contiguous()
            num_det = torch.tensor([len(keep)], dtype=torch.int32, device=boxes.device)

            responses.append(pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor.from_dlpack("boxes", to_dlpack(boxes_out)),
                pb_utils.Tensor.from_dlpack("scores", to_dlpack(scores_out)),
                pb_utils.Tensor.from_dlpack("class_ids", to_dlpack(class_ids_out)),
                pb_utils.Tensor.from_dlpack("num_detections", to_dlpack(num_det)),
            ]))
        return responses

    @staticmethod
    def _empty_response():
        return pb_utils.InferenceResponse(output_tensors=[
            pb_utils.Tensor.from_dlpack("boxes", to_dlpack(torch.zeros(1, 4, dtype=torch.float32))),
            pb_utils.Tensor.from_dlpack("scores", to_dlpack(torch.zeros(1, dtype=torch.float32))),
            pb_utils.Tensor.from_dlpack("class_ids", to_dlpack(torch.zeros(1, dtype=torch.int32))),
            pb_utils.Tensor.from_dlpack("num_detections", to_dlpack(torch.tensor([0], dtype=torch.int32))),
        ])


# ---------------------------------------------------------------------------
# Post-processing utilities
# Based on: NVIDIA TAO Deploy (Apache 2.0)
# Source: nvidia_tao_deploy/cv/deformable_detr/utils.py
# ---------------------------------------------------------------------------


def box_cxcywh_to_xyxy(x):
    """Convert box from cxcywh to xyxy.

    Based on: nvidia_tao_deploy/cv/deformable_detr/utils.py :: box_cxcywh_to_xyxy()
    """
    x_c, y_c, w, h = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)
