"""
Hermes-VLA Pre/Post Processor Pipeline.

Handles data preprocessing and postprocessing for the Hermes-VLA policy,
compatible with LeRobot-style data formats.

Preprocessing:
  1. Normalize images to [-1, 1] range (ViT standard)
  2. Normalize state/action using dataset statistics
  3. Tokenize language instructions
  4. Pad state/action to max dimensions
  5. Add batch dimension
  6. Move to device

Postprocessing:
  1. Unnormalize actions to original scale
  2. Move to CPU
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch

from hermes_vla.configuration_hermes_vla import HermesVLAConfig


# ---------------------------------------------------------------------------
# Normalization Utilities
# ---------------------------------------------------------------------------

class RunningNormalizer:
    """Online mean/std normalizer for robotics data."""

    def __init__(self, shape: tuple, eps: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.std = np.ones(shape, dtype=np.float32)
        self.eps = eps
        self._count = 0
        self._M2 = np.zeros(shape, dtype=np.float32)

    def update(self, x: np.ndarray):
        """Welford's online algorithm for mean/variance."""
        self._count += 1
        delta = x - self.mean
        self.mean += delta / self._count
        delta2 = x - self.mean
        self._M2 += delta * delta2
        if self._count > 1:
            self.std = np.sqrt(self._M2 / (self._count - 1)) + self.eps

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def unnormalize(self, x: np.ndarray) -> np.ndarray:
        return x * self.std + self.mean

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std, "count": self._count}

    def load_state_dict(self, state: dict):
        self.mean = state["mean"]
        self.std = state["std"]
        self._count = state["count"]


# ---------------------------------------------------------------------------
# Image Preprocessing
# ---------------------------------------------------------------------------

def preprocess_images(
    images: np.ndarray,
    image_size: int = 224,
    mean: tuple = (0.5, 0.5, 0.5),
    std: tuple = (0.5, 0.5, 0.5),
) -> torch.Tensor:
    """
    Preprocess images for ViT input.

    Handles multiple formats:
    - Single image: [H, W, C] or [C, H, W]
    - Multiple cameras: [V, H, W, C] or [V, C, H, W]
    - Batched: [B, V, H, W, C] or [B, V, C, H, W]

    Steps:
    1. Convert to [B, V, C, H, W] format
    2. Resize to image_size x image_size
    3. Normalize to [-1, 1] range
    """
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy()

    # Handle single image
    if images.ndim == 3:
        if images.shape[-1] == 3:  # [H, W, C]
            images = images.transpose(2, 0, 1)  # -> [C, H, W]
        images = np.expand_dims(np.expand_dims(images, 0), 0)  # -> [1, 1, C, H, W]

    # Handle multiple cameras without batch
    elif images.ndim == 4:
        if images.shape[-1] == 3:  # [V, H, W, C]
            images = images.transpose(0, 3, 1, 2)  # -> [V, C, H, W]
        images = np.expand_dims(images, 0)  # -> [1, V, C, H, W]

    # Already [B, V, C, H, W]
    elif images.ndim == 5:
        pass
    else:
        raise ValueError(f"Unexpected image shape: {images.shape}")

    # Convert to tensor and resize
    images_tensor = torch.from_numpy(images).float()

    if images_tensor.shape[-1] != image_size or images_tensor.shape[-2] != image_size:
        B, V = images_tensor.shape[:2]
        images_tensor = images_tensor.reshape(B * V, *images_tensor.shape[2:])
        images_tensor = torch.nn.functional.interpolate(
            images_tensor, size=(image_size, image_size), mode="bilinear", align_corners=False
        )
        images_tensor = images_tensor.reshape(B, V, images_tensor.shape[1], image_size, image_size)

    # Normalize to [-1, 1]
    mean_t = torch.tensor(mean, dtype=torch.float32).view(1, 1, 3, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(1, 1, 3, 1, 1)
    images_tensor = (images_tensor / 255.0 - mean_t) / std_t

    return images_tensor


# ---------------------------------------------------------------------------
# State / Action Padding
# ---------------------------------------------------------------------------

def pad_to_dim(data: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Pad the last dimension of a tensor to target_dim with zeros."""
    if data.shape[-1] >= target_dim:
        return data
    return torch.nn.functional.pad(data, (0, target_dim - data.shape[-1]))


# ---------------------------------------------------------------------------
# Processor Pipeline
# ---------------------------------------------------------------------------

@dataclass
class HermesVLAPreprocessor:
    """
    Preprocessor for Hermes-VLA.

    Transforms raw observation data into model-ready inputs.
    """

    config: HermesVLAConfig
    state_normalizer: Optional[RunningNormalizer] = None
    action_normalizer: Optional[RunningNormalizer] = None

    # Image normalization parameters (ViT standard: [-1, 1])
    image_mean: tuple = (0.5, 0.5, 0.5)
    image_std: tuple = (0.5, 0.5, 0.5)

    def __call__(self, observation: dict) -> dict:
        """
        Preprocess a single observation step.

        Args:
            observation: Dict with keys:
                - "images" or "observation.images": np.ndarray, camera images
                - "observation.state" or "state": np.ndarray, proprioceptive state
                - "task" or "language_instruction": str, task description
                - "action" (optional): np.ndarray, ground-truth actions (training only)

        Returns:
            Dict with keys: "images", "input_ids", "attention_mask",
                             "state", "action" (if provided)
        """
        batch = {}

        # ---- Images ----
        images = observation.get("images", observation.get("observation.images"))
        if images is not None:
            batch["images"] = preprocess_images(
                images, self.config.image_size, self.image_mean, self.image_std
            )

        # ---- State ----
        state = observation.get("observation.state", observation.get("state"))
        if state is not None:
            state_tensor = torch.from_numpy(np.asarray(state)).float()
            if self.state_normalizer is not None:
                state_tensor = torch.from_numpy(
                    self.state_normalizer.normalize(state_tensor.numpy())
                ).float()
            state_tensor = pad_to_dim(state_tensor, self.config.max_state_dim)
            batch["observation.state"] = state_tensor
            batch["state"] = state_tensor

        # ---- Language ----
        task = observation.get("task", observation.get("language_instruction", ""))
        # Tokenization is delegated to the tokenizer in the pipeline
        batch["task"] = task

        # ---- Action (training only) ----
        action = observation.get("action")
        if action is not None:
            action_tensor = torch.from_numpy(np.asarray(action)).float()
            if self.action_normalizer is not None:
                action_tensor = torch.from_numpy(
                    self.action_normalizer.normalize(action_tensor.numpy())
                ).float()
            action_tensor = pad_to_dim(action_tensor, self.config.max_action_dim)
            batch["action"] = action_tensor

        return batch


@dataclass
class HermesVLAPostprocessor:
    """
    Postprocessor for Hermes-VLA.

    Unnormalizes predicted actions back to the original scale.
    """

    action_normalizer: Optional[RunningNormalizer] = None

    def __call__(self, action: torch.Tensor) -> np.ndarray:
        """
        Postprocess predicted actions.

        Args:
            action: [B, action_dim] or [action_dim] raw model output

        Returns:
            np.ndarray of unnormalized actions
        """
        action_np = action.cpu().numpy() if isinstance(action, torch.Tensor) else action
        if self.action_normalizer is not None:
            action_np = self.action_normalizer.unnormalize(action_np)
        return action_np


# ---------------------------------------------------------------------------
# Tokenizer Wrapper
# ---------------------------------------------------------------------------

@dataclass
class SimpleTokenizer:
    """
    Simple BPE-like tokenizer wrapper for language processing.

    In production, use HuggingFace tokenizers (e.g., google/paligemma-3b-pt-224).
    This provides a fallback implementation.
    """

    vocab_size: int = 256000
    max_length: int = 128
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    def tokenize(self, texts: list[str]) -> dict:
        """
        Tokenize a list of text strings.

        Args:
            texts: List of language instruction strings

        Returns:
            Dict with keys: "input_ids", "attention_mask"
        """
        batch_size = len(texts)

        # Simple hash-based tokenization (placeholder for real tokenizer)
        input_ids = torch.zeros(batch_size, self.max_length, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, self.max_length, dtype=torch.long)

        for i, text in enumerate(texts):
            # Simple word-to-id mapping via hashing
            words = text.lower().replace(",", " ").replace(".", " ").split()
            tokens = []
            for w in words:
                tid = hash(w) % (self.vocab_size - 10) + 10  # avoid special tokens
                tokens.append(tid)

            # Truncate or pad
            tokens = tokens[:self.max_length - 2]
            token_ids = [self.bos_token_id] + tokens + [self.eos_token_id]

            input_ids[i, :len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
            attention_mask[i, :len(token_ids)] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


# ---------------------------------------------------------------------------
# Pipeline Factory
# ---------------------------------------------------------------------------

def make_hermes_vla_pre_post_processors(
    config: HermesVLAConfig,
    dataset_stats: Optional[dict] = None,
) -> tuple[HermesVLAPreprocessor, HermesVLAPostprocessor, SimpleTokenizer]:
    """
    Create preprocessor, postprocessor, and tokenizer for Hermes-VLA.

    Args:
        config: HermesVLAConfig instance
        dataset_stats: Optional dict with keys "state" and "action",
                       each containing {"mean": np.ndarray, "std": np.ndarray}

    Returns:
        (preprocessor, postprocessor, tokenizer) tuple
    """
    # Create normalizers from dataset stats
    state_normalizer = None
    action_normalizer = None

    if dataset_stats is not None:
        state_stats = dataset_stats.get("observation.state") or dataset_stats.get("state")
        if state_stats is not None:
            state_normalizer = RunningNormalizer(shape=state_stats["mean"].shape)
            state_normalizer.load_state_dict({
                "mean": state_stats["mean"],
                "std": state_stats["std"],
                "count": state_stats.get("count", 1),
            })

        action_stats = dataset_stats.get("action")
        if action_stats is not None:
            action_normalizer = RunningNormalizer(shape=action_stats["mean"].shape)
            action_normalizer.load_state_dict({
                "mean": action_stats["mean"],
                "std": action_stats["std"],
                "count": action_stats.get("count", 1),
            })

    preprocessor = HermesVLAPreprocessor(
        config=config,
        state_normalizer=state_normalizer,
        action_normalizer=action_normalizer,
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
    )

    postprocessor = HermesVLAPostprocessor(action_normalizer=action_normalizer)

    tokenizer = SimpleTokenizer(max_length=config.tokenizer_max_length)

    return preprocessor, postprocessor, tokenizer
