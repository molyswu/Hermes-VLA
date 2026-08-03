# Hermes-VLA

**A Dual-Stream Hierarchical VLA Framework for Embodied Intelligence**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

Hermes-VLA is a novel **Vision-Language-Action (VLA)** framework that introduces two key innovations for robust robot manipulation:

### 1. Dual-Stream Architecture
Two parallel processing streams — **Visual** and **Language** — interact through **bidirectional cross-attention fusion**, enabling deep multimodal understanding before any planning or control decisions are made.

### 2. Hierarchical Planning & Control
| Component | System Type | Role | Frequency |
|-----------|------------|------|-----------|
| **High-Level Planner** | System 2 (Slow) | Task understanding, sub-goal generation | Coarse (task-level) |
| **Low-Level Controller** | System 1 (Fast) | Flow Matching action generation | Fine (per-step) |

### 3. Flow Matching Action Generation
Uses continuous normalizing flows (inspired by PI0.5 and GR00T N1.5) to generate smooth, high-quality action trajectories.

## Architecture

```
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
      │  Cross-Stream Fusion  │  ← Dual-Stream: Bidirectional
      │ (Bidirectional X-Attn)│     Cross-Attention
      └───────────┬───────────┘
                  │
      ┌───────────▼───────────┐
      │  High-Level Planner    │  ← System 2: Sub-goal Planning
      │  task emb + plan emb   │     "What to do?"
      └───────────┬───────────┘
                  │
      ┌───────────▼───────────┐
      │  Low-Level Controller  │  ← System 1: Flow Matching
      │  action trajectory     │     "How to do it?"
      └────────────────────────┘
```

## Installation

```bash
git clone https://github.com/molyswu/Hermes-VLA.git
cd Hermes-VLA
pip install -e .

# With LeRobot integration support:
pip install -e ".[lerobot]"
```

## Quick Start

### Training

```bash
python train_hermes_vla.py \
    --dataset_path /path/to/your/dataset \
    --output_dir ./outputs/hermes_vla \
    --batch_size 32 \
    --num_epochs 100 \
    --lr 1e-4 \
    --visual_encoder siglip \
    --language_encoder gemma_2b \
    --fusion_type cross_attention \
    --num_cameras 2 \
    --device cuda
```

**Key training flags:**
- `--freeze_visual`: Freeze visual encoder (for finetuning)
- `--freeze_language`: Freeze language encoder
- `--train_planner_only`: Train only the high-level planner
- `--train_controller_only`: Train only the low-level controller
- `--gradient_checkpointing`: Enable for memory-constrained GPUs

### Inference

```python
from hermes_vla import HermesVLAConfig, HermesVLAPolicy
from inference_hermes_vla import HermesVLAInference

# Load trained model
engine = HermesVLAInference(
    checkpoint_path="./outputs/hermes_vla/best_model.pt",
    device="cuda",
)

# Process observation and predict actions
action_chunk = engine.predict_action_chunk(
    images=camera_images,    # [V, H, W, 3] numpy array
    state=robot_state,       # [state_dim] numpy array
    task="pick up the red block",
)

# Or get single actions with temporal chunking
engine.reset()
for step in range(100):
    action = engine.get_action(camera_images, robot_state, task)
    robot.execute(action)
```

### Test with Dummy Inputs

```bash
python inference_hermes_vla.py \
    --checkpoint ./outputs/hermes_vla/best_model.pt \
    --task "pick up the red block and place it on the table" \
    --test

# Benchmark inference speed
python inference_hermes_vla.py \
    --checkpoint ./outputs/hermes_vla/best_model.pt \
    --benchmark \
    --num_runs 100
```

## Model Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `visual_encoder_type` | Visual backbone (siglip/dinov2/clip/eagle2) | `siglip` |
| `language_encoder_type` | Language backbone (gemma_2b/gemma_300m/smollm) | `gemma_2b` |
| `fusion_type` | Cross-stream fusion (cross_attention/gated_fusion) | `cross_attention` |
| `hidden_dim` | Hidden dimension | `1024` |
| `planner_layers` | Planner transformer layers | `4` |
| `controller_layers` | Controller transformer layers | `8` |
| `plan_horizon` | Number of sub-goals to plan | `8` |
| `chunk_size` | Action trajectory length | `50` |
| `num_inference_steps` | Flow matching ODE steps | `10` |
| `num_cameras` | Number of camera views | `2` |

## Integration with LeRobot

Hermes-VLA follows the LeRobot policy interface conventions. To use within the LeRobot ecosystem:

```python
from hermes_vla import HermesVLAConfig, HermesVLAPolicy

config = HermesVLAConfig(...)
policy = HermesVLAPolicy(config)

# LeRobot-compatible interface:
loss = policy.forward(batch)          # Training
actions = policy.predict_action_chunk(batch)  # Inference
action = policy.select_action(batch)  # Step-by-step
policy.reset()                        # Reset state
```

## References

- **PI0.5**: [Physical Intelligence](https://www.physicalintelligence.company/blog/pi05) — VLM + Flow Matching for robot control
- **GR00T N1.5**: [NVIDIA GR00T](https://developer.nvidia.com/gr00t) — Generalist humanoid foundation model
- **LeRobot**: [huggingface/lerobot](https://github.com/huggingface/lerobot) — State-of-the-art AI for real-world robotics

## License

MIT License. See [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{hermes_vla,
  title = {Hermes-VLA: A Dual-Stream Hierarchical VLA Framework for Embodied Intelligence},
  author = {molyswu},
  year = {2026},
  url = {https://github.com/molyswu/Hermes-VLA},
}
```
