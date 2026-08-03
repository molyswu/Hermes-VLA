"""
GR00T N1.7 Inference Module.

A standalone inference module for NVIDIA's GR00T N1.7 VLA model,
adapted from LeRobot's implementation (huggingface/lerobot).

GR00T N1.7 is a 3B-parameter Vision-Language-Action model featuring:
- Qwen3-VL backbone (Cosmos-Reason2)
- Flow-matching diffusion action head (AlternateVLDiT)
- Dual-system architecture: System 2 (reasoning) + System 1 (action)
- RTC (Receding Horizon Control) support
"""

from .configuration_groot_n1_7 import GR00TN17Config
from .modeling_groot_n1_7 import GR00TN17ForInference
from .processor_groot_n1_7 import GR00TN17Processor

__all__ = ["GR00TN17Config", "GR00TN17ForInference", "GR00TN17Processor"]
