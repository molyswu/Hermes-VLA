"""
GR00T N1.7 Configuration.

Adapted from huggingface/lerobot: src/lerobot/policies/groot/configuration_groot.py

Copyright 2025 NVIDIA Corporation and The HuggingFace Inc. team.
Licensed under Apache License 2.0.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class GR00TN17Config:
    """Configuration for GR00T N1.7 inference.

    GR00T N1.7 is a 3B Vision-Language-Action model built on Qwen3-VL
    (Cosmos-Reason2-2B) with a flow-matching diffusion action head.

    Key architecture:
    - Backbone: Qwen3-VL (visual + language encoder)
    - Action Head: AlternateVLDiT (alternating visual-language cross-attention DiT)
    - State Encoder: CategorySpecificMLP (multi-embodiment support)
    - Action Decoder: CategorySpecificMLP
    - VLLN: Vision-Language LayerNorm for feature fusion

    Reference:
        NVIDIA Isaac GR00T N1.7 Technical Report
        https://github.com/NVIDIA/Isaac-GR00T
        https://huggingface.co/nvidia/GR00T-N1.7-3B
    """

    # ---- Model Path ----
    base_model_path: str = "nvidia/GR00T-N1.7-3B"
    """HuggingFace model ID or local path to the GR00T N1.7 checkpoint."""

    # ---- Input Dimensions ----
    n_obs_steps: int = 1
    """Number of observation steps to condition on."""

    chunk_size: int = 40
    """Number of action steps to predict (action horizon)."""

    n_action_steps: int = 40
    """Number of action steps to execute from each chunk."""

    max_state_dim: int = 132
    """Maximum state dimension (shorter states are zero-padded)."""

    max_action_dim: int = 132
    """Maximum action dimension (shorter actions are zero-padded)."""

    # ---- Image / Video Settings ----
    num_cameras: int = 1
    """Number of camera views. Default 1 for single-arm; 2+ for bimanual."""

    image_size: int = 224
    """Image resize dimension for the visual encoder."""

    # ---- Embodiment ----
    embodiment_tag: str = "new_embodiment"
    """Embodiment identifier for multi-embodiment model routing."""

    # ---- Action Decode ----
    action_decode_transform: str = "auto"
    """Action decode transform. "auto" auto-detects; LIBERO uses special handling."""

    # ---- Tuning Flags (for fine-tuning, not inference) ----
    tune_llm: bool = False
    tune_visual: bool = False
    tune_projector: bool = False
    tune_diffusion_model: bool = False
    tune_vlln: bool = False
    tune_top_llm_layers: int = 0

    # ---- Inference ----
    num_inference_timesteps: Optional[int] = 4
    """Number of flow-matching denoising steps. 4 = fast, 10 = balanced, 50 = best quality."""

    rtc_ramp_rate: Optional[float] = 6.0
    """RTC (Receding Horizon Control) blending rate for action chunk overlap."""

    use_flash_attention: bool = False
    """Enable Flash Attention 2 for faster inference (requires flash-attn package)."""

    use_bf16: bool = True
    """Use bfloat16 precision for inference. Recommended for GR00T N1.7."""

    # ---- Device ----
    device: str = "cuda"
    """Device for inference: 'cuda', 'cpu', 'mps'."""

    # ---- Output ----
    output_features: Optional[dict] = None
    """Optional output feature specification dict. If None, uses max_action_dim."""

    @property
    def action_dim(self) -> int:
        if self.output_features is not None and "action" in self.output_features:
            return self.output_features["action"].shape[0]
        return self.max_action_dim

    @property
    def state_dim(self) -> int:
        if self.output_features is not None and "observation.state" in self.output_features:
            return self.output_features["observation.state"].shape[0]
        return self.max_state_dim


# Default N1.7 configs for common use cases
GR00T_N1_7_DEFAULTS = {
    "model_name": "nvidia/Cosmos-Reason2-2B",
    "hidden_size": 1024,
    "action_horizon": 40,
    "max_action_dim": 132,
    "max_state_dim": 132,
    "num_inference_timesteps": 4,
}
