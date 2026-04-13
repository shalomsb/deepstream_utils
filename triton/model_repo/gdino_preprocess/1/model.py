"""Grounding DINO preprocessor: image resize + BERT tokenization.

Text preprocessing based on the official IDEA-Research/GroundingDINO
implementation (Apache 2.0 License).
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

        # Based on: IDEA-Research/GroundingDINO (Apache 2.0)
        # Source: groundingdino/util/inference.py :: preprocess_caption()
        # Caption format: lowercase, categories separated by " . ", trailing " ."
        caption = preprocess_caption(self.categories)

        encoding = tokenizer(
            caption,
            padding="max_length",
            max_length=MAX_TEXT_LEN,
            truncation=True,
            return_tensors="np",
            return_offsets_mapping=True,
        )

        # Based on: IDEA-Research/GroundingDINO (Apache 2.0)
        # Source: groundingdino/models/GroundingDINO/bertwarper.py
        special_token_ids = tokenizer.convert_tokens_to_ids(["[CLS]", "[SEP]", ".", "?"])

        text_self_attn_masks, position_ids = generate_masks_with_special_tokens_and_transfer_map(
            encoding, special_token_ids
        )

        # Based on: IDEA-Research/GroundingDINO (Apache 2.0)
        # Source: groundingdino/util/vl_utils.py :: create_positive_map_from_span()
        pos_map = create_positive_map_from_span(
            encoding, self.categories, caption
        )

        # Truncate to MAX_TEXT_LEN
        self.input_ids = encoding["input_ids"][:, :MAX_TEXT_LEN].astype(np.int64)
        self.attention_mask = encoding["attention_mask"][:, :MAX_TEXT_LEN].astype(np.bool_)
        self.token_type_ids = encoding["token_type_ids"][:, :MAX_TEXT_LEN].astype(np.int64)
        self.position_ids = position_ids[:, :MAX_TEXT_LEN].astype(np.int64)
        self.text_token_mask = text_self_attn_masks[:, :MAX_TEXT_LEN, :MAX_TEXT_LEN].astype(np.bool_)
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
# Based on: IDEA-Research/GroundingDINO (Apache 2.0 License)
# https://github.com/IDEA-Research/GroundingDINO
# ---------------------------------------------------------------------------


def preprocess_caption(categories):
    """Build caption string from category list.

    Based on: groundingdino/util/inference.py :: preprocess_caption()
    Categories are joined with " . " and a trailing " ." is appended.
    """
    return " . ".join(cat.lower().strip() for cat in categories) + " ."


def generate_masks_with_special_tokens_and_transfer_map(tokenized, special_token_ids):
    """Generate block-diagonal text self-attention masks and per-block position IDs.

    Based on: groundingdino/models/GroundingDINO/bertwarper.py
              :: generate_masks_with_special_tokens_and_transfer_map()

    Grounding DINO uses sub-sentence level text representation (Paper Section 3.4).
    Each category phrase forms an independent attention block. Tokens within a
    block attend to each other but NOT across blocks. This prevents cross-phrase
    attention contamination.

    The mask starts as identity (diagonal), so every position — including padding —
    self-attends. This prevents softmax(all -inf) = NaN for padding rows.
    """
    input_ids = tokenized["input_ids"]  # [bs, num_token]
    bs, num_token = input_ids.shape

    # Mark special token positions
    special_tokens_mask = np.zeros((bs, num_token), dtype=bool)
    for tok_id in special_token_ids:
        special_tokens_mask |= (input_ids == tok_id)

    # Collect (row, col) indices of all special tokens
    idxs = np.stack(np.nonzero(special_tokens_mask), axis=1)  # [N, 2]

    # Start from identity: every token can attend to itself
    attention_mask = np.tile(
        np.eye(num_token, dtype=bool)[np.newaxis], (bs, 1, 1)
    )
    position_ids = np.zeros((bs, num_token), dtype=np.int64)

    previous_col = 0
    for i in range(idxs.shape[0]):
        row, col = idxs[i]
        if col == 0 or col == num_token - 1:
            # [CLS] at start or [SEP]/last token: self-attend only, position 0
            attention_mask[row, col, col] = True
            position_ids[row, col] = 0
        else:
            # Tokens between previous_col+1 and col (inclusive) form a block.
            # They can all attend to each other within this block.
            start = previous_col + 1
            end = col + 1
            attention_mask[row, start:end, start:end] = True
            position_ids[row, start:end] = np.arange(end - start)
        previous_col = col

    return attention_mask, position_ids


def create_positive_map_from_span(tokenized, categories, caption):
    """Map each category label to the BERT tokens that cover its character span.

    Based on: groundingdino/util/vl_utils.py :: create_positive_map_from_span()

    Uses offset_mapping from the tokenizer to find which tokens overlap with
    each category's character range in the caption string. Returns a
    [num_cats, max_text_len] float matrix used by postprocessing to convert
    per-token logits (dim 256) to per-category scores.
    """
    offset_mapping = tokenized.get("offset_mapping")
    pos_map = np.zeros((len(categories), MAX_TEXT_LEN), dtype=np.float32)

    if offset_mapping is None:
        return pos_map

    offsets = offset_mapping[0]  # [num_tokens, 2]

    for cat_idx, label in enumerate(categories):
        # Find character span of this category in the caption
        start_char = caption.find(label.lower().strip())
        if start_char == -1:
            continue
        end_char = start_char + len(label.lower().strip())

        # Find all tokens whose character range overlaps with the label span
        for tok_idx in range(min(len(offsets), MAX_TEXT_LEN)):
            tok_start, tok_end = offsets[tok_idx]
            if tok_start == 0 and tok_end == 0:
                continue  # skip special tokens ([CLS], [SEP], [PAD])
            if tok_end > start_char and tok_start < end_char:
                pos_map[cat_idx, tok_idx] = 1.0

    return pos_map
