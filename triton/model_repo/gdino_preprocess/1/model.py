"""Grounding DINO preprocessor: GPU image resize + BERT tokenization.

Text preprocessing based on NVIDIA TAO Deploy (Apache 2.0 License)
and IDEA-Research/GroundingDINO (Apache 2.0 License).
  - TAO Deploy: nvidia_tao_deploy/cv/grounding_dino/utils.py
  - Paper: https://arxiv.org/abs/2303.05499 (Section 3.4)
  - Repo:  https://github.com/IDEA-Research/GroundingDINO

Input:  raw_frame [1, 3, H, W] float32 RGB [0,255]
Output: inputs [1, 3, 544, 960] float32
        input_ids [1, 256] int64
        attention_mask [1, 256] bool
        position_ids [1, 256] int64
        token_type_ids [1, 256] int64
        text_token_mask [1, 256, 256] bool
        pos_map [1, num_cats, 256] float32
"""

import json

import yaml
import torch
import torch.nn.functional as F
from torch.utils.dlpack import from_dlpack, to_dlpack
from transformers import AutoTokenizer
import triton_python_backend_utils as pb_utils

TARGET_H = 544
TARGET_W = 960
MAX_TEXT_LEN = 256
DEVICE = "cpu"


class TritonPythonModel:
    def initialize(self, args):
        model_config = json.loads(args["model_config"])
        params = model_config.get("parameters", {})
        config_path = params.get("config_file", {}).get("string_value", "")

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        self.categories = config.get("labels", [])
        self._prepare_text_inputs()

    def _prepare_text_inputs(self):
        """Tokenize categories and build all text tensors as CUDA tensors (cached at init)."""
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        cat_list = [c.lower().strip() for c in self.categories]
        caption = [" . ".join(cat_list) + " ."]

        (
            self.input_ids,
            self.attention_mask,
            self.position_ids,
            self.token_type_ids,
            self.text_token_mask,
            pos_map,
        ) = tokenize_captions(tokenizer, cat_list, caption, MAX_TEXT_LEN)

        self.pos_map = pos_map.unsqueeze(0)  # [num_cats, 256] -> [1, num_cats, 256]

    def execute(self, requests):
        responses = []
        for request in requests:
            raw = from_dlpack(
                pb_utils.get_input_tensor_by_name(request, "raw_frame").to_dlpack()
            )

            images = F.interpolate(
                raw.float(), size=(TARGET_H, TARGET_W), mode="bilinear", align_corners=False
            )

            response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor.from_dlpack("inputs", to_dlpack(images.contiguous())),
                    pb_utils.Tensor.from_dlpack("input_ids", to_dlpack(self.input_ids.contiguous())),
                    pb_utils.Tensor.from_dlpack("attention_mask", to_dlpack(self.attention_mask.contiguous())),
                    pb_utils.Tensor.from_dlpack("position_ids", to_dlpack(self.position_ids.contiguous())),
                    pb_utils.Tensor.from_dlpack("token_type_ids", to_dlpack(self.token_type_ids.contiguous())),
                    pb_utils.Tensor.from_dlpack("text_token_mask", to_dlpack(self.text_token_mask.contiguous())),
                    pb_utils.Tensor.from_dlpack("pos_map", to_dlpack(self.pos_map.contiguous())),
                ]
            )
            responses.append(response)
        return responses


# ---------------------------------------------------------------------------
# Text preprocessing utilities
# Based on: NVIDIA TAO Deploy (Apache 2.0)
# Source: nvidia_tao_deploy/cv/grounding_dino/utils.py
# ---------------------------------------------------------------------------


def tokenize_captions(tokenizer, cat_list, caption, max_text_len=256):
    """Tokenize captions and build all text tensors on CUDA.

    Based on: nvidia_tao_deploy/cv/grounding_dino/utils.py :: tokenize_captions()
    """
    special_tokens = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])
    tokenized = tokenizer(caption, padding="max_length", return_tensors="pt", max_length=max_text_len)

    input_ids = tokenized["input_ids"]        # [1, num_token]
    attention_mask = tokenized["attention_mask"]
    token_type_ids = tokenized["token_type_ids"]

    label_list = list(range(len(cat_list)))
    pos_map = create_positive_map(tokenized, label_list, cat_list, caption[0], max_text_len=max_text_len)

    text_self_attention_masks, position_ids = generate_masks_with_special_tokens_and_transfer_map(
        input_ids, special_tokens
    )

    if text_self_attention_masks.shape[1] > max_text_len:
        text_self_attention_masks = text_self_attention_masks[:, :max_text_len, :max_text_len]
        position_ids = position_ids[:, :max_text_len]
        input_ids = input_ids[:, :max_text_len]
        attention_mask = attention_mask[:, :max_text_len]
        token_type_ids = token_type_ids[:, :max_text_len]

    return (
        input_ids.to(dtype=torch.int64, device=DEVICE),
        attention_mask.to(dtype=torch.bool, device=DEVICE),
        position_ids.to(dtype=torch.int64, device=DEVICE),
        token_type_ids.to(dtype=torch.int64, device=DEVICE),
        text_self_attention_masks.to(dtype=torch.bool, device=DEVICE),
        pos_map.to(dtype=torch.float32, device=DEVICE),
    )


def generate_masks_with_special_tokens_and_transfer_map(input_ids, special_tokens_list):
    """Generate attention mask between each pair of special tokens.

    Based on: nvidia_tao_deploy/cv/grounding_dino/utils.py
              :: generate_masks_with_special_tokens_and_transfer_map()

    Grounding DINO uses sub-sentence level text representation (Paper Section 3.4).
    Each category phrase forms an independent attention block. Tokens within a
    block attend to each other but NOT across blocks.

    The mask starts as identity (diagonal), so every position self-attends.
    This prevents softmax(all -inf) = NaN for padding rows.
    """
    bs, num_token = input_ids.shape

    special_tokens_mask = torch.zeros(bs, num_token, dtype=torch.bool)
    for special_token in special_tokens_list:
        special_tokens_mask |= (input_ids == special_token)

    idxs = torch.nonzero(special_tokens_mask)  # [N, 2] — (row, col)

    attention_mask = torch.eye(num_token, dtype=torch.bool).unsqueeze(0).repeat(bs, 1, 1)
    position_ids = torch.zeros(bs, num_token, dtype=torch.int64)

    previous_col = 0
    for i in range(idxs.shape[0]):
        row, col = idxs[i][0].item(), idxs[i][1].item()
        if col in (0, num_token - 1):
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[row, previous_col + 1: col + 1, previous_col + 1: col + 1] = True
            position_ids[row, previous_col + 1: col + 1] = torch.arange(0, col - previous_col)
        previous_col = col

    return attention_mask, position_ids


def create_positive_map(tokenized, tokens_positive, cat_list, caption, max_text_len=256):
    """Construct a map such that positive_map[i,j] = True iff class i is associated to token j.

    Based on: nvidia_tao_deploy/cv/grounding_dino/utils.py :: create_positive_map()
    """
    positive_map = torch.zeros(len(tokens_positive), max_text_len)

    for j, label in enumerate(tokens_positive):
        start_ind = caption.find(cat_list[label])
        end_ind = start_ind + len(cat_list[label]) - 1
        beg_pos = tokenized.char_to_token(start_ind)
        try:
            end_pos = tokenized.char_to_token(end_ind)
        except Exception:
            end_pos = None
        if end_pos is None:
            try:
                end_pos = tokenized.char_to_token(end_ind - 1)
                if end_pos is None:
                    end_pos = tokenized.char_to_token(end_ind - 2)
            except Exception:
                end_pos = None

        if beg_pos is None or end_pos is None:
            continue
        if beg_pos < 0 or end_pos < 0:
            continue
        if beg_pos > end_pos:
            continue
        positive_map[j, beg_pos: end_pos + 1] = 1.0

    return positive_map
