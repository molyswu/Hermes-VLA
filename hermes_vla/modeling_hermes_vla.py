"""
Hermes-VLA: Dual-Stream Hierarchical Vision-Language-Action Model.

Core architecture implementing:
  1. Dual-Stream Encoder (Visual + Language with Cross-Attention Fusion)
  2. High-Level Planner (System 2) - produces latent plan / sub-goal embeddings
  3. Low-Level Controller (System 1) - flow-matching action generation

Reference inspirations:
  - PI0.5: VLM + Flow Matching action generation (Physical Intelligence)
  - GR00T N1.5: Eagle backbone + Flow Matching action head (NVIDIA)
  - Helix / Figure: Dual-system (slow reasoning + fast control) architecture
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from hermes_vla.configuration_hermes_vla import (
    FusionType,
    HermesVLAConfig,
    LanguageEncoderType,
    VisualEncoderType,
)


# ---------------------------------------------------------------------------
# Utility Modules
# ---------------------------------------------------------------------------

def sinusoidal_pos_embedding(timesteps: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """Creates sinusoidal timestep embeddings (as in diffusion/flow models)."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half)
    freqs = freqs.to(timesteps.device)
    args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (used in Gemma / LLaMA-style models)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)


class MLP(nn.Module):
    """Standard MLP with GELU activation."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with optional flash attention."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0, use_flash: bool = False):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.use_flash = use_flash
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if self.use_flash and hasattr(F, "scaled_dot_product_attention"):
            x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout.p if self.training else 0.0)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if attention_mask is not None:
                attn = attn + attention_mask
            attn = F.softmax(attn, dim=-1)
            attn = self.dropout(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        return self.out_proj(x)


class TransformerBlock(nn.Module):
    """Transformer block with pre-norm, self-attention, and MLP."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0,
                 use_flash: bool = False):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = MultiHeadSelfAttention(dim, num_heads, dropout, use_flash)
        self.norm2 = RMSNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.mlp(self.norm2(x))
        return x


class CrossAttentionBlock(nn.Module):
    """Cross-attention block: query stream attends to key/value stream."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0, use_flash: bool = False):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_flash = use_flash

        self.norm_q = RMSNorm(dim)
        self.norm_kv = RMSNorm(dim)
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        B, Nq, C = query.shape
        _, Nkv, _ = key_value.shape

        q = self.q_proj(self.norm_q(query)).reshape(B, Nq, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(self.norm_kv(key_value)).reshape(B, Nkv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(self.norm_kv(key_value)).reshape(B, Nkv, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, Nq, C)
        return self.out_proj(out)


class GatedFusionBlock(nn.Module):
    """Gated fusion block: learnable gating between visual and language features."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm_vis = RMSNorm(dim)
        self.norm_lang = RMSNorm(dim)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, vis: torch.Tensor, lang: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Bidirectional gated fusion."""
        vis_pooled = vis.mean(dim=1, keepdim=True)  # [B, 1, C]
        lang_pooled = lang.mean(dim=1, keepdim=True)  # [B, 1, C]

        combined = torch.cat([vis_pooled, lang_pooled], dim=-1)
        gate_vis = self.gate(combined)
        gate_lang = 1.0 - gate_vis

        vis_out = self.out_proj(self.norm_vis(vis)) * gate_vis + vis
        lang_out = self.out_proj(self.norm_lang(lang)) * gate_lang + lang
        return vis_out, lang_out


# ---------------------------------------------------------------------------
# Dual-Stream Encoder
# ---------------------------------------------------------------------------

class VisualEncoder(nn.Module):
    """
    Visual encoder (Vision Stream).

    Supports multiple backbones:
    - SigLIP: Lightweight ViT from google/siglip-so400m-patch14-224
    - DINOv2: Self-supervised ViT from Meta
    - CLIP: OpenAI CLIP ViT
    - Eagle2: NVIDIA Eagle2 (used in GR00T N1.5)

    For simplicity, we implement a configurable ViT-style encoder.
    In production, you would wrap pretrained models from huggingface/transformers.
    """

    def __init__(self, config: HermesVLAConfig):
        super().__init__()
        self.encoder_type = config.visual_encoder_type
        self.hidden_dim = config.hidden_dim
        self.image_size = config.image_size
        self.num_cameras = config.num_cameras

        # ViT-style patch embedding
        self.patch_size = 14
        self.num_patches = (self.image_size // self.patch_size) ** 2

        self.patch_embed = nn.Conv2d(
            3, self.hidden_dim,
            kernel_size=self.patch_size, stride=self.patch_size, bias=False
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, self.hidden_dim))

        self.num_vit_layers = {
            VisualEncoderType.SIGLIP: 12,
            VisualEncoderType.DINOV2: 12,
            VisualEncoderType.CLIP: 12,
            VisualEncoderType.EAGLE2: 16,
        }.get(self.encoder_type, 12)

        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=self.hidden_dim,
                num_heads=config.num_attention_heads,
                mlp_ratio=4.0,
                dropout=config.dropout_rate,
                use_flash=config.use_flash_attention,
            )
            for _ in range(self.num_vit_layers)
        ])
        self.norm = RMSNorm(self.hidden_dim)

        # Camera embedding (distinguishes different camera views)
        if self.num_cameras > 1:
            self.camera_embed = nn.Embedding(self.num_cameras, self.hidden_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [B, V, C, H, W] or [B, C, H, W]
                B=batch, V=num_cameras, C=3, H=W=image_size

        Returns:
            visual_features: [B, V * (1 + num_patches), hidden_dim]
        """
        if images.dim() == 4:
            images = images.unsqueeze(1)  # [B, 1, C, H, W]

        B, V, C, H, W = images.shape
        images = images.reshape(B * V, C, H, W)

        # Patch embedding
        x = self.patch_embed(images)  # [B*V, hidden_dim, H', W']
        x = x.flatten(2).transpose(1, 2)  # [B*V, num_patches, hidden_dim]

        # Add cls token
        cls_tokens = self.cls_token.expand(B * V, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed

        # Add camera embedding
        if self.num_cameras > 1 and hasattr(self, "camera_embed"):
            cam_ids = torch.arange(V, device=x.device).repeat_interleave(
                self.num_patches + 1
            ).unsqueeze(0).expand(B, -1)
            x = x + self.camera_embed(cam_ids)

        # ViT blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Reshape back to batch-wise
        x = x.reshape(B, V, self.num_patches + 1, self.hidden_dim)
        x = x.reshape(B, V * (self.num_patches + 1), self.hidden_dim)
        return x


class LanguageEncoder(nn.Module):
    """
    Language encoder (Language Stream).

    Processes natural language task instructions into token embeddings.
    Supports Gemma, SmolLM, and Qwen-style backbones.

    For simplicity, we implement a small transformer-based language model.
    In production, this would wrap a pretrained LLM.
    """

    def __init__(self, config: HermesVLAConfig):
        super().__init__()
        self.encoder_type = config.language_encoder_type
        self.hidden_dim = config.hidden_dim
        self.max_length = config.tokenizer_max_length

        # Token embedding
        vocab_size = 256000  # Standard LLM vocab size
        self.token_embed = nn.Embedding(vocab_size, self.hidden_dim)

        # Transformer layers
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=self.hidden_dim,
                num_heads=config.num_attention_heads,
                mlp_ratio=4.0,
                dropout=config.dropout_rate,
                use_flash=config.use_flash_attention,
            )
            for _ in range(12)  # 12-layer language encoder
        ])
        self.norm = RMSNorm(self.hidden_dim)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_ids: [B, L] tokenized language input
            attention_mask: [B, L] padding mask

        Returns:
            language_features: [B, L, hidden_dim]
        """
        x = self.token_embed(input_ids)  # [B, L, hidden_dim]

        # Create causal + padding mask
        if attention_mask is not None:
            causal_mask = torch.tril(torch.ones(x.shape[1], x.shape[1], device=x.device)).bool()
            padding_mask = attention_mask.bool().unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
            attn_mask = causal_mask & padding_mask.squeeze(1).squeeze(1).unsqueeze(1)
            # Convert to additive mask: 0 -> 0, False -> -inf
            attn_mask = torch.where(attn_mask, 0.0, float('-inf'))
        else:
            attn_mask = None

        for block in self.blocks:
            x = block(x, attn_mask)

        return self.norm(x)


# ---------------------------------------------------------------------------
# Cross-Stream Fusion Module
# ---------------------------------------------------------------------------

class CrossStreamFusion(nn.Module):
    """
    Dual-stream cross-attention fusion module.

    Fuses visual and language representations via:
    - Cross-attention: Vision queries Language, Language queries Vision
    - Or gated fusion: Learned gates controlling information flow

    Produces unified multimodal representations for downstream planning & control.
    """

    def __init__(self, config: HermesVLAConfig):
        super().__init__()
        self.fusion_type = config.fusion_type
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_fusion_layers
        self.num_heads = config.num_attention_heads

        if self.fusion_type == FusionType.CROSS_ATTENTION:
            # Bidirectional cross-attention layers
            self.vis_to_lang_layers = nn.ModuleList([
                CrossAttentionBlock(
                    self.hidden_dim, self.num_heads,
                    config.dropout_rate, config.use_flash_attention
                )
                for _ in range(self.num_layers)
            ])
            self.lang_to_vis_layers = nn.ModuleList([
                CrossAttentionBlock(
                    self.hidden_dim, self.num_heads,
                    config.dropout_rate, config.use_flash_attention
                )
                for _ in range(self.num_layers)
            ])

            # Feedforward after cross-attention
            self.vis_ffn = nn.ModuleList([
                MLP(self.hidden_dim, self.hidden_dim * 4, config.dropout_rate)
                for _ in range(self.num_layers)
            ])
            self.lang_ffn = nn.ModuleList([
                MLP(self.hidden_dim, self.hidden_dim * 4, config.dropout_rate)
                for _ in range(self.num_layers)
            ])
        elif self.fusion_type == FusionType.GATED_FUSION:
            self.gated_fusion = GatedFusionBlock(self.hidden_dim)
        else:
            # Simple concat + linear projection
            self.concat_proj = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

    def forward(
        self, vis_features: torch.Tensor, lang_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            vis_features: [B, VN, hidden_dim] visual stream output
            lang_features: [B, L, hidden_dim] language stream output

        Returns:
            fused_vis: [B, VN, hidden_dim]
            fused_lang: [B, L, hidden_dim]
        """
        if self.fusion_type == FusionType.CROSS_ATTENTION:
            v, l = vis_features, lang_features
            for i in range(self.num_layers):
                # Language-informed visual features
                v_res = self.vis_to_lang_layers[i](v, l)  # vision queries language
                v = v + self.vis_ffn[i](v_res)

                # Vision-informed language features
                l_res = self.lang_to_vis_layers[i](l, v)  # language queries vision
                l = l + self.lang_ffn[i](l_res)
            return v, l

        elif self.fusion_type == FusionType.GATED_FUSION:
            return self.gated_fusion(vis_features, lang_features)

        else:
            # Concat + project
            vis_pooled = vis_features.mean(dim=1, keepdim=True).expand_as(vis_features)
            lang_pooled = lang_features.mean(dim=1, keepdim=True).expand_as(lang_features)
            fused = self.concat_proj(torch.cat([vis_pooled, lang_pooled], dim=-1))
            return fused, fused


# ---------------------------------------------------------------------------
# High-Level Planner (System 2) - Slow Reasoning
# ---------------------------------------------------------------------------

class HighLevelPlanner(nn.Module):
    """
    High-Level Planner (System 2) — Slow, deliberate reasoning.

    Processes the fused multimodal representation to produce:
    1. A global task understanding embedding
    2. A sequence of latent sub-goal / plan embeddings

    This module operates at a coarser temporal granularity — it reasons about
    "what to do" at the task level rather than frame-by-frame control.
    """

    def __init__(self, config: HermesVLAConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.plan_dim = config.plan_dim
        self.plan_horizon = config.plan_horizon
        self.temporal_window = config.planner_temporal_window

        # Learnable sub-goal queries
        self.subgoal_queries = nn.Parameter(torch.zeros(1, self.plan_horizon, self.hidden_dim))
        nn.init.trunc_normal_(self.subgoal_queries, std=0.02)

        # Learnable task understanding token
        self.task_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))
        nn.init.trunc_normal_(self.task_token, std=0.02)

        # Transformer layers for reasoning over fused features
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=self.hidden_dim,
                num_heads=config.num_attention_heads,
                mlp_ratio=4.0,
                dropout=config.dropout_rate,
                use_flash=config.use_flash_attention,
            )
            for _ in range(config.planner_layers)
        ])

        # Project to plan dimension
        self.plan_proj = nn.Linear(self.hidden_dim, self.plan_dim)

        # Cross-attention: subgoal queries attend to fused multimodal features
        self.query_cross_attn = CrossAttentionBlock(
            self.hidden_dim, config.num_attention_heads,
            config.dropout_rate, config.use_flash_attention,
        )

        self.norm = RMSNorm(self.hidden_dim)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        fused_features: torch.Tensor,  # [B, N_fused, hidden_dim]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Produce task understanding and sub-goal plans.

        Args:
            fused_features: Fused multimodal features

        Returns:
            task_embedding: [B, 1, plan_dim] — global task representation
            plan_embeddings: [B, plan_horizon, plan_dim] — sequence of sub-goals
        """
        B = fused_features.shape[0]

        # Add task token
        task_tokens = self.task_token.expand(B, -1, -1)
        x = torch.cat([task_tokens, fused_features], dim=1)

        # Process through transformer
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Extract task embedding
        task_embedding = self.plan_proj(x[:, :1, :])  # [B, 1, plan_dim]

        # Generate sub-goal plans via cross-attention
        subgoal_queries = self.subgoal_queries.expand(B, -1, -1)
        plan_embeddings = self.query_cross_attn(subgoal_queries, x)
        plan_embeddings = self.plan_proj(plan_embeddings)  # [B, plan_horizon, plan_dim]

        return task_embedding, plan_embeddings


# ---------------------------------------------------------------------------
# Low-Level Controller (System 1) - Fast Action Generation
# ---------------------------------------------------------------------------

class ActionHead(nn.Module):
    """Action decoder head: maps hidden states to action vectors."""

    def __init__(self, hidden_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LowLevelController(nn.Module):
    """
    Low-Level Controller (System 1) — Fast, reactive action generation.

    Combines:
    - Fused multimodal features (from dual-stream encoder + fusion)
    - Plan embeddings (from high-level planner)
    - Current state (proprioception)
    - Noisy actions + timestep (flow matching)

    Uses Flow Matching to generate smooth action trajectories.
    Reference: PI0.5 flow matching (modeling_pi05.py), GR00T flow matching action head.
    """

    def __init__(self, config: HermesVLAConfig):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.plan_dim = config.plan_dim
        self.action_dim = config.max_action_dim
        self.state_dim = config.max_state_dim
        self.chunk_size = config.chunk_size
        self.plan_horizon = config.plan_horizon
        self.num_inference_steps = config.num_inference_steps

        # Plan conditioner: broadcast plan over action chunk
        self.plan_conditioner = nn.Sequential(
            nn.Linear(self.plan_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # State encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        # Time embedding for flow matching
        self.time_mlp = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(self.hidden_dim * 4, self.hidden_dim),
        )

        # Action + state projection (for noisy actions input)
        self.action_state_proj = nn.Linear(self.action_dim + self.state_dim, self.hidden_dim)

        # Position embeddings for action chunk
        self.action_pos_embed = nn.Parameter(torch.zeros(1, config.chunk_size, self.hidden_dim))
        nn.init.trunc_normal_(self.action_pos_embed, std=0.02)

        # Transformer blocks for denoising
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=self.hidden_dim,
                num_heads=config.num_attention_heads,
                mlp_ratio=4.0,
                dropout=config.dropout_rate,
                use_flash=config.use_flash_attention,
            )
            for _ in range(config.controller_layers)
        ])

        self.norm = RMSNorm(self.hidden_dim)

        # Action head
        self.action_head = ActionHead(self.hidden_dim, self.action_dim)

    def _get_flow_velocity(
        self,
        noisy_actions: torch.Tensor,  # [B, chunk_size, action_dim]
        timestep: torch.Tensor,       # [B] or [B, 1]
        state: torch.Tensor,          # [B, state_dim]
        plan_conditioning: torch.Tensor,  # [B, hidden_dim]
        fused_context: torch.Tensor,  # [B, N, hidden_dim]
    ) -> torch.Tensor:
        """
        Predict flow velocity v = dx/dt given noisy actions and timestep.

        This is the neural network that parameterizes the flow vector field.
        """
        B, T, _ = noisy_actions.shape

        # Time embedding
        time_emb = sinusoidal_pos_embedding(timestep, self.hidden_dim)
        time_emb = self.time_mlp(time_emb)  # [B, hidden_dim]

        # Encode state
        state_emb = self.state_encoder(state)  # [B, hidden_dim]

        # Encode noisy actions with state
        state_expanded = state.unsqueeze(1).expand(-1, T, -1)
        action_state = torch.cat([noisy_actions, state_expanded], dim=-1)
        action_emb = self.action_state_proj(action_state)  # [B, T, hidden_dim]

        # Add position embeddings
        action_emb = action_emb + self.action_pos_embed

        # Combine conditioning signals
        # time_emb, state_emb, plan_conditioning -> per-timestep conditioning
        cond = (time_emb.unsqueeze(1) + state_emb.unsqueeze(1) + plan_conditioning.unsqueeze(1))
        cond = cond.expand(-1, T, -1)

        # Fuse with action embeddings
        x = action_emb + cond

        # Cross-attention to fused context for visual-language grounding
        # (mean-pool context for simplicity; could use full cross-attn)
        context_summary = fused_context.mean(dim=1, keepdim=True)  # [B, 1, hidden_dim]
        x = x + context_summary

        # Denoising transformer
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Predict flow velocity
        velocity = self.action_head(x)  # [B, T, action_dim]
        return velocity

    def forward(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        state: torch.Tensor,
        plan_conditioning: torch.Tensor,
        fused_context: torch.Tensor,
    ) -> torch.Tensor:
        """Training forward pass: predict flow velocity for loss computation."""
        return self._get_flow_velocity(
            noisy_actions, timestep, state, plan_conditioning, fused_context
        )

    @torch.no_grad()
    def sample_actions(
        self,
        state: torch.Tensor,
        plan_conditioning: torch.Tensor,
        fused_context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample actions via Flow Matching ODE integration (Euler method).

        Starts from Gaussian noise and integrates the learned velocity field
        to produce clean actions at t=0.

        Args:
            state: [B, state_dim]
            plan_conditioning: [B, hidden_dim]
            fused_context: [B, N, hidden_dim]

        Returns:
            actions: [B, chunk_size, action_dim]
        """
        B = state.shape[0]
        device = state.device

        # Start from noise at t=1
        x_t = torch.randn(B, self.chunk_size, self.action_dim, device=device)

        # Euler integration from t=1 to t=0
        dt = -1.0 / self.num_inference_steps
        for step in range(self.num_inference_steps):
            t = 1.0 + step * dt  # Current time (1.0, 1.0-dt, ..., dt)
            t_tensor = torch.full((B,), t, device=device)

            # Predict velocity
            v = self._get_flow_velocity(x_t, t_tensor, state, plan_conditioning, fused_context)

            # Euler step: x_{t+dt} = x_t + v * dt
            x_t = x_t + v * dt

        return x_t


# ---------------------------------------------------------------------------
# Hermes-VLA Full Model
# ---------------------------------------------------------------------------

class HermesVLA(nn.Module):
    """
    Hermes-VLA: A Dual-Stream Hierarchical VLA Framework.

    Full architecture:
    ┌─────────────────────────────────────────────────┐
    │                 Input                            │
    │  images [B,V,3,H,W]    language [B,L]  state[B,S]│
    └───────┬─────────────────────┬───────────────────┘
            │                     │
    ┌───────▼──────┐   ┌──────────▼──────────┐
    │ Visual Stream│   │  Language Stream     │
    │  (ViT-based) │   │  (Transformer LLM)   │
    └───────┬──────┘   └──────────┬──────────┘
            │                     │
            └─────────┬───────────┘
                      │
          ┌───────────▼───────────┐
          │  Cross-Stream Fusion  │  ← Dual-Stream Key Innovation
          │ (Bidirectional X-Attn)│
          └───────────┬───────────┘
                      │
          ┌───────────▼───────────┐
          │  High-Level Planner    │  ← System 2: Slow Reasoning
          │  (Sub-goal Planning)   │     What to do?
          │  task emb + plan emb   │
          └───────────┬───────────┘
                      │
          ┌───────────▼───────────┐
          │  Low-Level Controller  │  ← System 1: Fast Control
          │  (Flow Matching)       │     How to do it?
          │  action trajectory     │
          └────────────────────────┘
    """

    def __init__(self, config: HermesVLAConfig):
        super().__init__()
        self.config = config
        self.plan_dim = config.plan_dim
        self.hidden_dim = config.hidden_dim
        self.chunk_size = config.chunk_size
        self.action_dim = config.max_action_dim
        self.state_dim = config.max_state_dim
        self.num_inference_steps = config.num_inference_steps

        # Dual-Stream Encoders
        self.visual_encoder = VisualEncoder(config)
        self.language_encoder = LanguageEncoder(config)

        # Cross-Stream Fusion
        self.fusion = CrossStreamFusion(config)

        # Hierarchical Planning & Control
        self.planner = HighLevelPlanner(config)
        self.controller = LowLevelController(config)

        # ---- Finetuning Control ----
        if config.freeze_visual_encoder:
            for p in self.visual_encoder.parameters():
                p.requires_grad = False
        if config.freeze_language_encoder:
            for p in self.language_encoder.parameters():
                p.requires_grad = False
        if config.train_planner_only:
            for p in self.controller.parameters():
                p.requires_grad = False
        if config.train_controller_only:
            for p in self.planner.parameters():
                p.requires_grad = False
            for p in self.fusion.parameters():
                p.requires_grad = False

    def encode_multimodal(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Dual-stream encoding + cross-stream fusion.

        Returns:
            fused_features: [B, VN+L, hidden_dim] fused multimodal representation
            task_embedding: [B, 1, plan_dim]
            plan_embeddings: [B, plan_horizon, plan_dim]
        """
        # Visual stream
        vis_features = self.visual_encoder(images)  # [B, VN, hidden_dim]

        # Language stream
        lang_features = self.language_encoder(input_ids, attention_mask)  # [B, L, hidden_dim]

        # Cross-stream fusion
        fused_vis, fused_lang = self.fusion(vis_features, lang_features)

        # Concatenate fused features
        fused_features = torch.cat([fused_vis, fused_lang], dim=1)  # [B, VN+L, hidden_dim]

        # High-level planning
        task_embedding, plan_embeddings = self.planner(fused_features)

        return fused_features, task_embedding, plan_embeddings

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        state: torch.Tensor,
        actions: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Training forward pass with Flow Matching loss.

        Args:
            images: [B, V, C, H, W]
            input_ids: [B, L]
            attention_mask: [B, L]
            state: [B, S] current proprioceptive state
            actions: [B, chunk_size, A] ground-truth action trajectory

        Returns:
            dict with keys: 'loss', 'predicted_actions', 'task_embedding', 'plan_embeddings'
        """
        B = images.shape[0]
        device = images.device

        # Multimodal encoding + planning
        fused_features, task_embedding, plan_embeddings = self.encode_multimodal(
            images, input_ids, attention_mask
        )

        # Collapse plan into single conditioning vector
        plan_conditioning = plan_embeddings.mean(dim=1)  # [B, plan_dim]
        plan_conditioning = self.controller.plan_conditioner(plan_conditioning)  # [B, hidden_dim]

        # Flow matching: sample noise and timestep
        if noise is None:
            noise = torch.randn_like(actions)

        # Sample timestep from Beta(alpha, beta) distribution (like PI0.5)
        alpha, beta = 1.5, 1.0
        time_dist = torch.distributions.Beta(alpha, beta)
        timestep = time_dist.sample((B,)).to(device)  # [B]
        timestep = timestep.clamp(min=1e-3)

        # Interpolate: x_t = t * noise + (1 - t) * actions
        timestep_expanded = timestep.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
        noisy_actions = timestep_expanded * noise + (1 - timestep_expanded) * actions

        # Predict flow velocity
        pred_velocity = self.controller(
            noisy_actions, timestep, state, plan_conditioning, fused_features
        )

        # Flow matching loss: ||v_pred - (noise - actions)||^2
        target_velocity = noise - actions
        loss = F.mse_loss(pred_velocity, target_velocity)

        # Additionally add a small planning auxiliary loss (encourages plan diversity)
        plan_var = plan_embeddings.var(dim=1).mean()
        planning_aux_loss = -0.01 * plan_var  # neg encourages diversity

        total_loss = loss + planning_aux_loss

        return {
            "loss": total_loss,
            "flow_loss": loss,
            "planning_aux_loss": planning_aux_loss,
            "predicted_actions": None,  # computed during inference
            "task_embedding": task_embedding,
            "plan_embeddings": plan_embeddings,
        }

    @torch.no_grad()
    def predict_action_chunk(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        Inference: predict action chunk via Flow Matching ODE integration.

        Args:
            images: [B, V, C, H, W]
            input_ids: [B, L]
            attention_mask: [B, L]
            state: [B, S]

        Returns:
            actions: [B, chunk_size, action_dim]
        """
        # Multimodal encoding + planning
        fused_features, task_embedding, plan_embeddings = self.encode_multimodal(
            images, input_ids, attention_mask
        )

        # Plan conditioning
        plan_conditioning = plan_embeddings.mean(dim=1)
        plan_conditioning = self.controller.plan_conditioner(plan_conditioning)

        # Sample actions via flow ODE
        actions = self.controller.sample_actions(
            state, plan_conditioning, fused_features
        )

        return actions

    @torch.no_grad()
    def select_action(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        state: torch.Tensor,
        action_index: int = 0,
    ) -> torch.Tensor:
        """
        Select a single action from the predicted chunk at the given index.

        Useful for step-by-step inference with temporal action chunking.
        """
        action_chunk = self.predict_action_chunk(images, input_ids, attention_mask, state)
        return action_chunk[:, action_index, :]


# ---------------------------------------------------------------------------
# Hermes-VLA Policy Wrapper (lerobot-compatible interface)
# ---------------------------------------------------------------------------

class HermesVLAPolicy(nn.Module):
    """
    Hermes-VLA Policy wrapper compatible with LeRobot-style interface.

    Provides standard methods:
    - forward(): Training step
    - predict_action_chunk(): Generate action trajectory
    - select_action(): Pick single action from chunk
    - reset(): Reset internal state
    """

    config_class = HermesVLAConfig
    name = "hermes_vla"

    def __init__(self, config: HermesVLAConfig):
        super().__init__()
        self.config = config
        self.model = HermesVLA(config)

        # Internal state for temporal chunking
        self._action_chunk: Optional[torch.Tensor] = None
        self._chunk_index: int = 0

    def forward(self, batch: dict) -> dict:
        """Training forward pass. Expects dict with keys matching model inputs."""
        return self.model(
            images=batch.get("images", batch.get("observation.images")),
            input_ids=batch.get("input_ids"),
            attention_mask=batch.get("attention_mask"),
            state=batch.get("observation.state", batch.get("state")),
            actions=batch.get("action"),
        )

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict) -> torch.Tensor:
        """Generate full action trajectory."""
        return self.model.predict_action_chunk(
            images=batch.get("images", batch.get("observation.images")),
            input_ids=batch.get("input_ids"),
            attention_mask=batch.get("attention_mask"),
            state=batch.get("observation.state", batch.get("state")),
        )

    @torch.no_grad()
    def select_action(self, batch: dict) -> torch.Tensor:
        """Select action with temporal chunking support."""
        n_action_steps = getattr(self.config, 'n_action_steps', self.config.chunk_size)

        if self._action_chunk is None or self._chunk_index >= n_action_steps:
            self._action_chunk = self.predict_action_chunk(batch)
            self._chunk_index = 0

        action = self._action_chunk[:, self._chunk_index, :]
        self._chunk_index += 1
        return action

    def reset(self):
        """Reset temporal chunking state."""
        self._action_chunk = None
        self._chunk_index = 0

    def to(self, device):
        self.model = self.model.to(device)
        return self

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def eval(self):
        self.model.eval()
        return self
