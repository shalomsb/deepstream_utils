"""SAHI postprocess: tile-local -> global coords + GREEDYNMM with IoS.

Receives  : tile_dets [6, 300, 6]  -- (x1, y1, x2, y2, conf, cls), tile-local
Emits     : boxes [M, 4], scores [M], classes [M], num_dets [1]
            All coords in 1280x720 frame space, post-merge.

GREEDYNMM (SAHI default): sort by score, for each survivor merge any
remaining same-class boxes whose IoS = intersection / min(area_a, area_b)
exceeds match_threshold; merged box = bounding union of all matches.
"""

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from torch.utils.dlpack import from_dlpack, to_dlpack


# Tile (x_offset, y_offset) in the 1280x720 frame.
# Matches sahi_preprocess.onnx: T1..T6 = (0,0),(512,0),(640,0),(0,80),(512,80),(640,80).
TILE_OFFSETS = np.array([
    [0,   0],
    [512, 0],
    [640, 0],
    [0,   80],
    [512, 80],
    [640, 80],
], dtype=np.float32)

CONF_THRESHOLD = 0.05  # broad floor; client filters further
IOS_THRESHOLD = 0.5
CLASS_AGNOSTIC = False


def _greedy_nmm_ios(boxes, scores, classes, ios_threshold, class_agnostic):
    """SAHI's greedy non-maximum merging with IoS metric (GPU torch).

    Args:
        boxes:   [N, 4]  xyxy
        scores:  [N]
        classes: [N] int
    Returns:
        merged_boxes [M, 4], merged_scores [M], merged_classes [M]
    """
    n = boxes.shape[0]
    if n == 0:
        device = boxes.device
        return (torch.zeros((0, 4), dtype=torch.float32, device=device),
                torch.zeros((0,), dtype=torch.float32, device=device),
                torch.zeros((0,), dtype=torch.int32, device=device))

    # Mutable copy — we update boxes when merging.
    boxes = boxes.clone()
    areas = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    order = scores.argsort(descending=True)
    suppressed = torch.zeros(n, dtype=torch.bool, device=boxes.device)
    keep = []

    for rank in range(n):
        i = order[rank].item()
        if suppressed[i]:
            continue
        keep.append(i)

        # Candidates: lower-ranked + still alive.
        rest_idx = order[rank + 1:]
        rest_idx = rest_idx[~suppressed[rest_idx]]
        if rest_idx.numel() == 0:
            continue

        if not class_agnostic:
            same_cls = classes[rest_idx] == classes[i]
            rest_idx = rest_idx[same_cls]
            if rest_idx.numel() == 0:
                continue

        bi = boxes[i]
        bj = boxes[rest_idx]
        xx1 = torch.maximum(bi[0], bj[:, 0])
        yy1 = torch.maximum(bi[1], bj[:, 1])
        xx2 = torch.minimum(bi[2], bj[:, 2])
        yy2 = torch.minimum(bi[3], bj[:, 3])
        inter = (xx2 - xx1).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        min_area = torch.minimum(areas[i], areas[rest_idx])
        ios = inter / min_area.clamp(min=1e-6)
        match = ios > ios_threshold
        if not match.any():
            continue

        matched = rest_idx[match]
        # Bounding union of i + matched.
        all_idx = torch.cat([torch.tensor([i], device=boxes.device), matched])
        boxes[i, 0] = boxes[all_idx, 0].min()
        boxes[i, 1] = boxes[all_idx, 1].min()
        boxes[i, 2] = boxes[all_idx, 2].max()
        boxes[i, 3] = boxes[all_idx, 3].max()
        suppressed[matched] = True

    keep_idx = torch.tensor(keep, dtype=torch.long, device=boxes.device)
    return boxes[keep_idx], scores[keep_idx], classes[keep_idx]


class TritonPythonModel:
    def initialize(self, args):
        self.device = torch.device("cuda")
        self.tile_offsets = torch.from_numpy(TILE_OFFSETS).to(self.device)  # [6, 2]

    def execute(self, requests):
        responses = []
        for request in requests:
            tile_dets = from_dlpack(
                pb_utils.get_input_tensor_by_name(request, "tile_dets").to_dlpack()
            ).to(self.device)  # [6, 300, 6]

            # Per-tile filter + global coord shift.
            all_boxes = []
            all_scores = []
            all_classes = []
            for t in range(6):
                d = tile_dets[t]              # [300, 6]
                conf = d[:, 4]
                mask = conf > CONF_THRESHOLD
                if not mask.any():
                    continue
                v = d[mask]
                ox, oy = self.tile_offsets[t, 0], self.tile_offsets[t, 1]
                boxes = torch.stack([
                    v[:, 0] + ox,
                    v[:, 1] + oy,
                    v[:, 2] + ox,
                    v[:, 3] + oy,
                ], dim=-1)
                all_boxes.append(boxes)
                all_scores.append(v[:, 4])
                all_classes.append(v[:, 5].to(torch.int32))

            if all_boxes:
                boxes_cat = torch.cat(all_boxes, dim=0)
                scores_cat = torch.cat(all_scores, dim=0)
                classes_cat = torch.cat(all_classes, dim=0)
                merged_boxes, merged_scores, merged_classes = _greedy_nmm_ios(
                    boxes_cat, scores_cat, classes_cat,
                    IOS_THRESHOLD, CLASS_AGNOSTIC,
                )
                m = merged_boxes.shape[0]
            else:
                merged_boxes = torch.zeros((0, 4), dtype=torch.float32, device=self.device)
                merged_scores = torch.zeros((0,), dtype=torch.float32, device=self.device)
                merged_classes = torch.zeros((0,), dtype=torch.int32, device=self.device)
                m = 0

            # Triton requires output tensors to have at least one element.
            # Pad with a sentinel row when empty; consumer trusts num_dets.
            if m == 0:
                merged_boxes = torch.zeros((1, 4), dtype=torch.float32, device=self.device)
                merged_scores = torch.zeros((1,), dtype=torch.float32, device=self.device)
                merged_classes = torch.zeros((1,), dtype=torch.int32, device=self.device)

            response = pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor.from_dlpack("boxes", to_dlpack(merged_boxes.contiguous())),
                pb_utils.Tensor.from_dlpack("scores", to_dlpack(merged_scores.contiguous())),
                pb_utils.Tensor.from_dlpack("classes", to_dlpack(merged_classes.contiguous())),
                pb_utils.Tensor("num_dets", np.array([m], dtype=np.int32)),
            ])
            responses.append(response)
        return responses

    def finalize(self):
        pass
