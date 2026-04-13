"""Grounding DINO preprocessor: image resize + BERT tokenization.

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
import numpy as np
from transformers import AutoTokenizer
import triton_python_backend_utils as pb_utils

TARGET_H = 544
TARGET_W = 960
MAX_TEXT_LEN = 256


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
        """Tokenize categories and build all text tensors (cached at init)."""
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        cat_list = [c.lower().strip() for c in self.categories]
        caption = [" . ".join(cat_list) + " ."]

        (
            input_ids, attention_mask, position_ids,
            token_type_ids, text_self_attention_masks, pos_map,
        ) = tokenize_captions(tokenizer, cat_list, caption, MAX_TEXT_LEN)

        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.position_ids = position_ids
        self.token_type_ids = token_type_ids
        self.text_token_mask = text_self_attention_masks
        self.pos_map = pos_map.reshape(1, -1, MAX_TEXT_LEN).astype(np.float32)

    def execute(self, requests):
        import cv2
        responses = []
        for request in requests:
            raw = pb_utils.get_input_tensor_by_name(request, "raw_frame").as_numpy()

            img = raw[0].transpose(1, 2, 0)  # CHW -> HWC
            resized = cv2.resize(img, (TARGET_W, TARGET_H), interpolation=cv2.INTER_LINEAR)
            images = resized.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)

            response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor("inputs", images),
                    pb_utils.Tensor("input_ids", self.input_ids),
                    pb_utils.Tensor("attention_mask", self.attention_mask),
                    pb_utils.Tensor("position_ids", self.position_ids),
                    pb_utils.Tensor("token_type_ids", self.token_type_ids),
                    pb_utils.Tensor("text_token_mask", self.text_token_mask),
                    pb_utils.Tensor("pos_map", self.pos_map),
                ]
            )
            responses.append(response)
        return responses

    def finalize(self):
        pass


# ---------------------------------------------------------------------------
# Text preprocessing utilities
# Based on: NVIDIA TAO Deploy (Apache 2.0)
# Source: nvidia_tao_deploy/cv/grounding_dino/utils.py
# ---------------------------------------------------------------------------


def tokenize_captions(tokenizer, cat_list, caption, max_text_len=256):
    """Tokenize captions and build all text tensors.

    Based on: nvidia_tao_deploy/cv/grounding_dino/utils.py :: tokenize_captions()
    """
    special_tokens = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])
    tokenized = tokenizer(caption, padding="max_length", return_tensors="np", max_length=max_text_len)

    label_list = np.arange(len(cat_list))
    pos_map = create_positive_map(tokenized, label_list, cat_list, caption[0], max_text_len=max_text_len)

    (
        text_self_attention_masks,
        position_ids,
    ) = generate_masks_with_special_tokens_and_transfer_map(
        tokenized, special_tokens
    )

    if text_self_attention_masks.shape[1] > max_text_len:
        text_self_attention_masks = text_self_attention_masks[
            :, :max_text_len, :max_text_len
        ]
        position_ids = position_ids[:, :max_text_len]
        tokenized["input_ids"] = tokenized["input_ids"][:, :max_text_len]
        tokenized["attention_mask"] = tokenized["attention_mask"][:, :max_text_len]
        tokenized["token_type_ids"] = tokenized["token_type_ids"][:, :max_text_len]

    input_ids = tokenized["input_ids"].astype(np.int64)
    attention_mask = tokenized["attention_mask"].astype(np.bool_)
    position_ids = position_ids.astype(np.int64)
    token_type_ids = tokenized["token_type_ids"].astype(np.int64)
    text_self_attention_masks = text_self_attention_masks.astype(np.bool_)

    return input_ids, attention_mask, position_ids, token_type_ids, text_self_attention_masks, pos_map


def generate_masks_with_special_tokens_and_transfer_map(tokenized, special_tokens_list):
    """Generate attention mask between each pair of special tokens.

    Based on: nvidia_tao_deploy/cv/grounding_dino/utils.py
              :: generate_masks_with_special_tokens_and_transfer_map()

    Grounding DINO uses sub-sentence level text representation (Paper Section 3.4).
    Each category phrase forms an independent attention block. Tokens within a
    block attend to each other but NOT across blocks.

    The mask starts as identity (diagonal), so every position self-attends.
    This prevents softmax(all -inf) = NaN for padding rows.
    """
    input_ids = tokenized["input_ids"]
    bs, num_token = input_ids.shape

    special_tokens_mask = np.zeros((bs, num_token), dtype=bool)
    for special_token in special_tokens_list:
        special_tokens_mask |= input_ids == special_token

    idxs = np.stack(np.nonzero(special_tokens_mask), axis=1)

    attention_mask = (
        np.tile(np.expand_dims(np.eye(num_token, dtype=bool), axis=0), (bs, 1, 1))
    )
    position_ids = np.zeros((bs, num_token))

    previous_col = 0
    for i in range(idxs.shape[0]):
        row, col = idxs[i]
        if col in (0, num_token - 1):
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            attention_mask[row, previous_col + 1: col + 1, previous_col + 1: col + 1] = True
            position_ids[row, previous_col + 1: col + 1] = np.arange(
                0, col - previous_col
            )
        previous_col = col

    return attention_mask, position_ids


def create_positive_map(tokenized, tokens_positive, cat_list, caption, max_text_len=256):
    """Construct a map such that positive_map[i,j] = True iff class i is associated to token j.

    Based on: nvidia_tao_deploy/cv/grounding_dino/utils.py :: create_positive_map()
    """
    positive_map = np.zeros((len(tokens_positive), max_text_len), dtype=float)

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
        positive_map[j, beg_pos: end_pos + 1].fill(1)

    return positive_map
