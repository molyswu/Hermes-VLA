"""
GR00T N1.7 Processor (Preprocessing / Postprocessing).

Handles input preparation and output decoding for GR00T N1.7 inference.

Adapted from huggingface/lerobot: src/lerobot/policies/groot/processor_groot.py

Copyright 2025 NVIDIA Corporation and The HuggingFace Inc. team.
Licensed under Apache License 2.0.
"""

import logging
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .configuration_groot_n1_7 import GR00TN17Config

logger = logging.getLogger(__name__)


class GR00TN17Processor:
    """
    Input preprocessor and output postprocessor for GR00T N1.7.

    Handles:
    1. Image resizing and normalization for Qwen3-VL backbone
    2. State padding to max_state_dim
    3. Language instruction formatting
    4. Action denormalization and dimensionality trimming

    GR00T N1.7 expects:
    - Images: [B, V, 3, H, W], normalized to model-specific range
    - State: [B, max_state_dim], zero-padded
    - Language: raw text string (processed by tokenizer inside model)
    """

    def __init__(self, config: GR00TN17Config):
        self.config = config
        self.image_size = config.image_size

        # Default normalization: ImageNet stats
        self.image_mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
        self.image_std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

    def preprocess(
        self,
        images: np.ndarray,
        state: np.ndarray,
        task: str,
    ) -> Dict[str, torch.Tensor]:
        """
        Preprocess raw inputs into model-ready tensors.

        Args:
            images: Camera images.
                    Shape: [V, H, W, C] (single) or [B, V, H, W, C] (batch)
                    dtype: uint8 [0, 255] or float32 [0, 1]
            state: Proprioceptive state.
                   Shape: [state_dim] or [B, state_dim]
            task: Language instruction.

        Returns:
            Dict with keys depending on model interface:
            - pixel_values: [B, V, 3, H, W] processed images
            - state: [B, max_state_dim] padded state
            - text: list of language strings
            - input_ids, attention_mask (if tokenizer available)
        """
        batch = {}

        # ---- Images ----
        images_tensor = self._preprocess_images(images)
        batch["pixel_values"] = images_tensor

        # ---- State ----
        state_tensor = self._preprocess_state(state)
        batch["state"] = state_tensor

        # ---- Language ----
        batch["task"] = task  # Some models expect raw text
        batch["text"] = [task] if isinstance(task, str) else task

        # Try to tokenize if we have a tokenizer available
        try:
            from transformers import AutoTokenizer
            # GR00T N1.7 uses Qwen3-VL's tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model_path,
                trust_remote_code=True,
            )
            encoded = tokenizer(
                [task] if isinstance(task, str) else task,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            batch["input_ids"] = encoded["input_ids"]
            batch["attention_mask"] = encoded["attention_mask"]
        except Exception:
            logger.debug("Could not load tokenizer; model will handle text internally.")

        return batch

    def _preprocess_images(self, images: np.ndarray) -> torch.Tensor:
        """
        Preprocess images for the GR00T N1.7 visual backbone.

        Steps:
        1. Handle uint8 [0,255] → float32 [0,1]
        2. Add batch dim if needed
        3. Resize to config.image_size
        4. Normalize with ImageNet stats
        5. Convert to [B, V, C, H, W] layout
        """
        # Convert to float32 [0, 1] if uint8
        if images.dtype == np.uint8:
            images = images.astype(np.float32) / 255.0

        # Ensure at least 4D: [V, H, W, C]
        if images.ndim == 3:
            images = images[np.newaxis, ...]  # [1, H, W, C]

        # Add batch dim if 4D: [B, V, H, W, C]
        if images.ndim == 4:
            images = images[np.newaxis, ...]  # [1, V, H, W, C]

        B, V, H, W, C = images.shape

        # Resize each view
        resized = np.zeros((B, V, self.image_size, self.image_size, C), dtype=np.float32)
        for b in range(B):
            for v in range(V):
                img = images[b, v]
                # Convert to PIL for reliable resize
                pil_img = Image.fromarray((img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8))
                pil_img = pil_img.resize((self.image_size, self.image_size), Image.BICUBIC)
                resized[b, v] = np.array(pil_img, dtype=np.float32) / 255.0

        # Normalize: [0,1] → ImageNet normalized
        resized = (resized - self.image_mean.reshape(1, 1, 1, 1, 3)) / self.image_std.reshape(1, 1, 1, 1, 3)

        # Convert to [B, V, C, H, W] (PyTorch format)
        resized = np.transpose(resized, (0, 1, 4, 2, 3))

        return torch.from_numpy(resized.copy())

    def _preprocess_state(self, state: np.ndarray) -> torch.Tensor:
        """
        Preprocess state: ensure float, add batch dim, pad to max_state_dim.
        """
        state = np.asarray(state, dtype=np.float32)

        # Add batch dim if 1D
        if state.ndim == 1:
            state = state[np.newaxis, :]  # [1, state_dim]

        B, S = state.shape
        max_dim = self.config.max_state_dim

        if S < max_dim:
            # Pad to max_state_dim
            padded = np.zeros((B, max_dim), dtype=np.float32)
            padded[:, :S] = state
            state = padded

        return torch.from_numpy(state)

    def postprocess(self, action_chunk: torch.Tensor) -> np.ndarray:
        """
        Postprocess model output to numpy action array.

        Args:
            action_chunk: Raw model output.
                          Shape: [B, chunk_size, action_dim] or [chunk_size, action_dim]

        Returns:
            action_chunk: [chunk_size, action_dim] numpy array
        """
        # Convert to numpy
        if isinstance(action_chunk, torch.Tensor):
            action_chunk = action_chunk.detach().cpu().numpy()

        # Remove batch dim if present
        if action_chunk.ndim == 3:
            action_chunk = action_chunk[0]

        return action_chunk.astype(np.float32)

    def get_model_input_names(self) -> list:
        """Return the expected input keys for the model."""
        return [
            "pixel_values",      # Images
            "input_ids",         # Tokenized text
            "attention_mask",    # Text attention mask
            "state",             # Proprioceptive state
        ]


# ---------------------------------------------------------------------------
# Utility: image preprocessing without Processor class
# ---------------------------------------------------------------------------

def preprocess_images_simple(
    images: np.ndarray,
    image_size: int = 224,
    image_mean: Optional[np.ndarray] = None,
    image_std: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """
    Simple image preprocessing for GR00T N1.7.

    Args:
        images: [V, H, W, C] or [H, W, C], uint8 or float32
        image_size: Target resize dimension
        image_mean: Per-channel mean (default: ImageNet)
        image_std: Per-channel std (default: ImageNet)

    Returns:
        Tensor [1, V, C, image_size, image_size]
    """
    if image_mean is None:
        image_mean = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    if image_std is None:
        image_std = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

    if images.dtype == np.uint8:
        images = images.astype(np.float32) / 255.0

    if images.ndim == 3:
        images = images[np.newaxis, ...]

    V, H, W, C = images.shape
    resized = np.zeros((V, image_size, image_size, C), dtype=np.float32)
    for v in range(V):
        img = Image.fromarray((images[v] * 255).astype(np.uint8))
        img = img.resize((image_size, image_size), Image.BICUBIC)
        resized[v] = np.array(img, dtype=np.float32) / 255.0

    resized = (resized - image_mean.reshape(1, 1, 1, 3)) / image_std.reshape(1, 1, 1, 3)
    resized = np.transpose(resized, (0, 3, 1, 2))  # [V, C, H, W]
    resized = resized[np.newaxis, ...]  # [1, V, C, H, W]

    return torch.from_numpy(resized.copy())
