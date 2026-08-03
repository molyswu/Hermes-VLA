"""
Hermes-VLA Configuration Module.

Configuration dataclass for the Hermes-VLA dual-stream hierarchical VLA framework.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class VisualEncoderType(str, Enum):
    """Supported visual encoder backbone types."""
    SIGLIP = "siglip"
    DINOV2 = "dinov2"
    CLIP = "clip"
    EAGLE2 = "eagle2"


class LanguageEncoderType(str, Enum):
    """Supported language encoder backbone types."""
    GEMMA_300M = "gemma_300m"
    GEMMA_2B = "gemma_2b"
    SMOLLM = "smollm"
    QWEN2_05B = "qwen2_0.5b"


class FusionType(str, Enum):
    """Cross-stream fusion mechanisms."""
    CROSS_ATTENTION = "cross_attention"
    GATED_FUSION = "gated_fusion"
    CONCAT_LINEAR = "concat_linear"


@dataclass
class HermesVLAConfig:
    """
    Configuration for Hermes-VLA: A Dual-Stream Hierarchical VLA Framework.

    The architecture consists of:
    1. Dual-Stream Encoder: Visual stream + Language stream
    2. Cross-Stream Fusion: Bidirectional cross-attention fusion module
    3. High-Level Planner (System 2): Produces latent sub-goal embeddings
    4. Low-Level Controller (System 1): Flow matching action generation

    Attributes:
        visual_encoder_type: Type of visual backbone (siglip, dinov2, clip, eagle2)
        language_encoder_type: Type of language backbone (gemma_300m, gemma_2b, smollm)
        fusion_type: Cross-stream fusion mechanism
        num_fusion_layers: Number of cross-attention fusion layers
        planner_layers: Number of transformer layers in the high-level planner
        controller_layers: Number of transformer layers in the low-level controller
        hidden_dim: Hidden dimension for all transformer components
        num_attention_heads: Number of attention heads
        intermediate_dim: MLP intermediate dimension
        planner_temporal_window: Number of observation frames the planner processes
        plan_dim: Dimensionality of the latent plan embedding
        plan_horizon: Number of future steps the planner predicts sub-goals for
        chunk_size: Number of action steps to predict per inference
        n_action_steps: Number of action steps to execute per chunk
        max_state_dim: Maximum state dimension (padded if smaller)
        max_action_dim: Maximum action dimension (padded if smaller)
        image_size: Input image resolution
        num_cameras: Number of camera views
        tokenizer_max_length: Maximum token length for language input
        num_inference_steps: Flow matching ODE solver steps during inference
        dropout_rate: Dropout rate for regularization
        use_flash_attention: Whether to use flash attention
        gradient_checkpointing: Enable gradient checkpointing
        compile_model: Whether to use torch.compile
        optimizer_lr: Learning rate
        optimizer_weight_decay: Weight decay
        scheduler_warmup_steps: Warmup steps
        scheduler_decay_steps: Total decay steps
        freeze_visual_encoder: Freeze visual encoder during finetuning
        freeze_language_encoder: Freeze language encoder during finetuning
        train_planner_only: Only train the planner component
        train_controller_only: Only train the controller component
    """

    # ---- Architecture ----
    visual_encoder_type: VisualEncoderType = VisualEncoderType.SIGLIP
    language_encoder_type: LanguageEncoderType = LanguageEncoderType.GEMMA_2B
    fusion_type: FusionType = FusionType.CROSS_ATTENTION
    num_fusion_layers: int = 4

    # ---- Hierarchical Components ----
    planner_layers: int = 4
    controller_layers: int = 8
    hidden_dim: int = 1024
    num_attention_heads: int = 16
    intermediate_dim: int = 4096

    # ---- Planner Settings ----
    planner_temporal_window: int = 4
    plan_dim: int = 256
    plan_horizon: int = 8  # number of sub-goals to plan

    # ---- Controller / Action Settings ----
    chunk_size: int = 50
    n_action_steps: int = 50
    max_state_dim: int = 32
    max_action_dim: int = 32

    # ---- Input Settings ----
    image_size: int = 224
    num_cameras: int = 2
    tokenizer_max_length: int = 128

    # ---- Flow Matching ----
    num_inference_steps: int = 10
    flow_sigma_min: float = 0.001

    # ---- Regularization ----
    dropout_rate: float = 0.0
    use_flash_attention: bool = False

    # ---- Training ----
    gradient_checkpointing: bool = False
    compile_model: bool = False
    device: Optional[str] = None

    # ---- Optimizer ----
    optimizer_lr: float = 1e-4
    optimizer_weight_decay: float = 0.01
    scheduler_warmup_steps: int = 500
    scheduler_decay_steps: int = 30000
    scheduler_decay_lr: float = 1e-5

    # ---- Finetuning Control ----
    freeze_visual_encoder: bool = False
    freeze_language_encoder: bool = False
    train_planner_only: bool = False
    train_controller_only: bool = False

    # ---- Normalization ----
    # Maps feature type string to normalization mode
    normalization_mapping: dict = field(
        default_factory=lambda: {
            "VISUAL": "IDENTITY",
            "STATE": "MEAN_STD",
            "ACTION": "MEAN_STD",
        }
    )

    def __post_init__(self):
        """Validate configuration."""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) exceeds chunk_size ({self.chunk_size})"
            )
        if self.hidden_dim % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_dim ({self.hidden_dim}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})"
            )
        if self.plan_horizon < 1:
            raise ValueError("plan_horizon must be >= 1")
        if self.num_inference_steps < 1:
            raise ValueError("num_inference_steps must be >= 1")


# Register as "hermes_vla" for integration with lerobot-style factory
HermesVLAConfig.__lerobot_policy_name__ = "hermes_vla"
