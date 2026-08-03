"""
Hermes-VLA: A Dual-Stream Hierarchical VLA Framework for Embodied Intelligence.

A Vision-Language-Action framework featuring:
- Dual-Stream: Parallel visual and language processing streams with cross-attention fusion
- Hierarchical: High-level planner (System 2) + Low-level controller (System 1)
- Flow Matching: Continuous normalizing flow-based action generation
"""

from .configuration_hermes_vla import HermesVLAConfig
from .modeling_hermes_vla import HermesVLAPolicy
from .processor_hermes_vla import make_hermes_vla_pre_post_processors

__all__ = ["HermesVLAConfig", "HermesVLAPolicy", "make_hermes_vla_pre_post_processors"]
