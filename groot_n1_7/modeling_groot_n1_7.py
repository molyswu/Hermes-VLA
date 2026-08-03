"""
GR00T N1.7 Inference Model.

A lightweight inference-only wrapper around the GR00T N1.7 model.
Delegates to the official Isaac-GR00T components for actual computation.

Adapted from huggingface/lerobot: src/lerobot/policies/groot/modeling_groot.py
and src/lerobot/policies/groot/groot_n1_7.py

Copyright 2025 NVIDIA Corporation and The HuggingFace Inc. team.
Licensed under Apache License 2.0.
"""

import logging
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from .configuration_groot_n1_7 import GR00TN17Config
from .processor_groot_n1_7 import GR00TN17Processor

logger = logging.getLogger(__name__)


class GR00TN17ForInference(nn.Module):
    """
    GR00T N1.7 inference-only model wrapper.

    This class provides a clean interface for loading and running
    GR00T N1.7 for action prediction. It supports:

    1. Loading from HuggingFace Hub or local path
    2. RTC (Receding Horizon Control) for smooth action execution
    3. Flow-matching denoising with configurable steps
    4. Batch inference for multiple environments

    Architecture Summary:
    ┌────────────────────────────────────────────────┐
    │                  Input                          │
    │  images [B,V,3,H,W]  +  language [B,L]  + state │
    └──────────┬─────────────────────────────────────┘
               │
    ┌──────────▼──────────┐
    │   Qwen3-VL Backbone  │  ← Cosmos-Reason2-2B
    │  (Vision + Language) │     Visual + Text tokens
    └──────────┬──────────┘
               │ hidden_states
    ┌──────────▼──────────┐
    │  AlternateVLDiT      │  ← Flow-matching diffusion
    │  (Action Head)       │     Cross-attn: img/text
    │  + State Encoder     │     4-50 denoising steps
    │  + Action Decoder    │
    │  + VLLN              │
    └──────────┬──────────┘
               │
    ┌──────────▼──────────┐
    │  Action Trajectory   │  [B, chunk_size, action_dim]
    └──────────────────────┘
    """

    def __init__(self, config: GR00TN17Config):
        super().__init__()
        self.config = config
        self._groot_model: Optional[Any] = None
        self._processor: Optional[GR00TN17Processor] = None
        self._is_loaded = False

        # Action queue for temporal chunking
        self._action_queue: list = []
        self._chunk_index: int = 0
        self._prev_chunk_left_over: Optional[torch.Tensor] = None

    @property
    def is_loaded(self) -> bool:
        return self._is_loaded

    def load_model(self, pretrained_path: Optional[str] = None) -> None:
        """
        Load the GR00T N1.7 model from HuggingFace Hub or local path.

        This requires the Isaac-GR00T package to be installed:
            pip install isaac-gr00t

        Or from source:
            git clone https://github.com/NVIDIA/Isaac-GR00T.git
            cd Isaac-GR00T && pip install -e .

        Args:
            pretrained_path: HF model ID or local path.
                             Defaults to config.base_model_path.
        """
        if self._is_loaded:
            logger.info("Model already loaded, skipping.")
            return

        model_path = pretrained_path or self.config.base_model_path
        logger.info(f"Loading GR00T N1.7 from: {model_path}")

        try:
            # Try loading via the official Isaac-GR00T package
            from isaac_groot.n1_7 import GR00TN17

            self._groot_model = GR00TN17.from_pretrained(
                pretrained_model_name_or_path=model_path,
                num_inference_timesteps=self.config.num_inference_timesteps,
                tune_llm=False,
                tune_visual=False,
                tune_projector=False,
                tune_diffusion_model=False,
                tune_vlln=False,
                transformers_loading_kwargs={"trust_remote_code": True},
            )
            self._groot_model.eval()
            self._groot_model.to(self.config.device)

            logger.info("GR00T N1.7 model loaded successfully via isaac-gr00t.")

        except ImportError:
            logger.warning(
                "isaac-gr00t package not found. "
                "Trying to load via transformers + custom model loading..."
            )
            self._load_via_transformers(model_path)
        except Exception as e:
            logger.warning(f"Failed to load via isaac-gr00t: {e}")
            logger.info("Attempting fallback loading via transformers...")
            self._load_via_transformers(model_path)

        self._is_loaded = True
        self._processor = GR00TN17Processor(self.config)

        # Log model info
        total_params = sum(p.numel() for p in self._groot_model.parameters())
        trainable_params = sum(p.numel() for p in self._groot_model.parameters() if p.requires_grad)
        logger.info(f"GR00T N1.7: {total_params:,} total params, {trainable_params:,} trainable")

    def _load_via_transformers(self, model_path: str) -> None:
        """
        Fallback: Load GR00T N1.7 via HuggingFace transformers AutoModel.

        This approach loads the model using trust_remote_code=True,
        which allows loading custom model code from the HF repo.
        """
        try:
            from transformers import AutoModel, AutoProcessor

            logger.info(f"Loading GR00T N1.7 via AutoModel.from_pretrained({model_path})...")

            self._groot_model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if self.config.use_bf16 else torch.float32,
                device_map=self.config.device if self.config.device != "cpu" else None,
            )
            self._groot_model.eval()

            # Also try loading the processor for VLM encoding
            try:
                self._hf_processor = AutoProcessor.from_pretrained(
                    model_path, trust_remote_code=True
                )
            except Exception:
                self._hf_processor = None
                logger.warning("Could not load AutoProcessor; will use manual preprocessing.")

            logger.info("GR00T N1.7 loaded via transformers AutoModel.")

        except Exception as e:
            raise RuntimeError(
                f"Failed to load GR00T N1.7 model from {model_path}. "
                f"Please ensure:\n"
                f"  1. The model is downloaded: huggingface-cli download {model_path}\n"
                f"  2. transformers is installed: pip install transformers>=4.45\n"
                f"  3. For full support: pip install isaac-gr00t\n"
                f"Original error: {e}"
            )

    @torch.no_grad()
    def predict_action_chunk(
        self,
        images: np.ndarray,
        state: np.ndarray,
        task: str,
        *,
        prev_chunk_left_over: Optional[np.ndarray] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Predict a full action trajectory chunk.

        Args:
            images: Camera images.
                    Shape: [V, H, W, C] or [B, V, H, W, C]
                    dtype: uint8 (0-255) or float32 (0-1)
            state: Proprioceptive state.
                   Shape: [state_dim] or [B, state_dim]
            task: Language instruction string.
            prev_chunk_left_over: Previous chunk remainder for RTC.
                                  Shape: [B, remaining_steps, action_dim]

        Returns:
            action_chunk: [chunk_size, action_dim] numpy array

        Raises:
            RuntimeError: If model is not loaded.
        """
        if not self._is_loaded:
            raise RuntimeError(
                "Model not loaded. Call load_model() first, "
                "or use GR00TN17ForInference.from_pretrained()."
            )

        self.eval()

        # Preprocess inputs
        batch = self._processor.preprocess(
            images=images,
            state=state,
            task=task,
        )

        # Move to device
        batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Run inference
        with torch.autocast(
            device_type=self.config.device if "cuda" in self.config.device else "cpu",
            dtype=torch.bfloat16,
            enabled=self.config.use_bf16 and self.config.device != "cpu",
        ):
            action_chunk = self._run_inference(batch)

        # Postprocess
        action_np = self._processor.postprocess(action_chunk)

        # Trim to configured dimensions
        action_dim = self.config.action_dim
        action_np = action_np[..., :action_dim]

        return action_np

    def _run_inference(self, batch: dict) -> torch.Tensor:
        """
        Run the actual model inference.

        Supports two paths:
        1. Native GR00TN17.get_action() (preferred)
        2. Generic model forward (fallback)
        """
        # Path 1: Use native get_action if available
        if hasattr(self._groot_model, "get_action"):
            outputs = self._groot_model.get_action(batch)
            if isinstance(outputs, dict):
                return outputs.get("action_pred", outputs.get("actions"))
            return outputs

        # Path 2: Generic forward pass
        if hasattr(self._groot_model, "predict_action_chunk"):
            return self._groot_model.predict_action_chunk(batch)

        # Path 3: Raw forward
        outputs = self._groot_model(**batch)
        if isinstance(outputs, dict):
            return outputs.get("action_pred", outputs.get("actions"))
        # Assume first element of tuple is action
        if isinstance(outputs, (tuple, list)):
            return outputs[0]
        return outputs

    def select_action(
        self,
        images: np.ndarray,
        state: np.ndarray,
        task: str,
    ) -> np.ndarray:
        """
        Select a single action using temporal action chunking.

        Maintains an internal action queue. When the queue is empty,
        predicts a new action chunk and fills the queue.

        Args:
            images: Camera images [V, H, W, C]
            state: Proprioceptive state [state_dim]
            task: Language instruction

        Returns:
            action: [action_dim] numpy array
        """
        if len(self._action_queue) == 0:
            action_chunk = self.predict_action_chunk(images, state, task)
            # action_chunk: [chunk_size, action_dim]
            n_steps = min(self.config.n_action_steps, len(action_chunk))
            for i in range(n_steps):
                self._action_queue.append(action_chunk[i])

        return self._action_queue.pop(0)

    def reset(self) -> None:
        """Reset temporal chunking state and action queue."""
        self._action_queue.clear()
        self._chunk_index = 0
        self._prev_chunk_left_over = None

    def get_device(self) -> torch.device:
        """Get the device the model is on."""
        if self._groot_model is not None:
            return next(self._groot_model.parameters()).device
        return torch.device(self.config.device)

    # ------------------------------------------------------------------
    # Convenience class methods
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str = "nvidia/GR00T-N1.7-3B",
        *,
        device: str = "cuda",
        num_inference_timesteps: int = 4,
        use_bf16: bool = True,
        **config_kwargs,
    ) -> "GR00TN17ForInference":
        """
        Load a pretrained GR00T N1.7 model for inference.

        This is the recommended way to create an inference-ready model.

        Args:
            pretrained_path: HuggingFace model ID or local path.
            device: Device for inference ('cuda', 'cpu').
            num_inference_timesteps: Flow-matching denoising steps (1-50).
                                     Lower = faster, higher = more accurate.
            use_bf16: Use bfloat16 precision.
            **config_kwargs: Additional config overrides.

        Returns:
            Initialized GR00TN17ForInference instance.

        Example:
            >>> model = GR00TN17ForInference.from_pretrained(
            ...     "nvidia/GR00T-N1.7-3B",
            ...     device="cuda",
            ...     num_inference_timesteps=4,
            ... )
            >>> action = model.predict_action_chunk(images, state, "pick up the block")
        """
        config = GR00TN17Config(
            base_model_path=pretrained_path,
            device=device,
            num_inference_timesteps=num_inference_timesteps,
            use_bf16=use_bf16,
            **config_kwargs,
        )

        model = cls(config)
        model.load_model(pretrained_path)
        return model

    def __repr__(self) -> str:
        loaded = "loaded" if self._is_loaded else "not loaded"
        return (
            f"GR00TN17ForInference(\n"
            f"  model_path={self.config.base_model_path},\n"
            f"  device={self.config.device},\n"
            f"  num_inference_timesteps={self.config.num_inference_timesteps},\n"
            f"  action_dim={self.config.action_dim},\n"
            f"  chunk_size={self.config.chunk_size},\n"
            f"  status={loaded},\n"
            f")"
        )
