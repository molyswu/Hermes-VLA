# GR00T N1.7 Inference

基于 [LeRobot](https://github.com/huggingface/lerobot) 的 GR00T N1.7 推理代码，独立可运行的推理模块。

## 概述

**GR00T N1.7** 是 NVIDIA 开源的 3B 参数 Vision-Language-Action (VLA) 基础模型，专为通用机器人操控设计。

### 核心架构

```
┌──────────────────────────────────────────────────┐
│                    Input                          │
│  images [B,V,3,H,W]  +  language [B,L]  + state   │
└──────────┬───────────────────────────────────────┘
           │
┌──────────▼──────────┐
│  Qwen3-VL Backbone   │  ← Cosmos-Reason2-2B
│  Vision + Language   │     Cross-modal encoding
└──────────┬──────────┘
           │ hidden_states
┌──────────▼──────────┐
│  AlternateVLDiT       │  ← Flow-matching diffusion
│  + State Encoder      │     Cross-attention:
│  + Action Decoder     │     alternating img/text
│  + VLLN               │     4-50 denoising steps
└──────────┬──────────┘
           │
┌──────────▼──────────┐
│  Action Trajectory    │  [B, 40, action_dim]
└──────────────────────┘
```

**关键特性:**
- **双系统架构**: System 2 (Qwen3-VL推理) + System 1 (DiT扩散动作生成)
- **Flow Matching**: 连续归一化流进行动作去噪
- **RTC (Receding Horizon Control)**: 动作块重叠执行，平滑过渡
- **多具身支持**: CategorySpecificMLP 支持不同机器人形态
- **开源商用**: Apache 2.0 许可证

## 环境配置

### 基础依赖

```bash
pip install torch>=2.0 transformers>=4.45 accelerate numpy pillow
```

### 完整 GR00T 支持 (推荐)

```bash
# 从源码安装 Isaac-GR00T
git clone https://github.com/NVIDIA/Isaac-GR00T.git
cd Isaac-GR00T
pip install -e .

# 或直接 pip 安装
pip install isaac-gr00t
```

### 下载模型

```bash
# 从 HuggingFace 下载
huggingface-cli download nvidia/GR00T-N1.7-3B

# 或使用 Python
python -c "from huggingface_hub import snapshot_download; snapshot_download('nvidia/GR00T-N1.7-3B')"
```

## 快速开始

### 1. 测试模式 (无需下载模型)

```bash
python inference_groot_n1_7.py --test
```

### 2. 加载模型并推理

```python
from groot_n1_7 import GR00TN17ForInference
import numpy as np

# 加载模型
model = GR00TN17ForInference.from_pretrained(
    "nvidia/GR00T-N1.7-3B",
    device="cuda",
    num_inference_timesteps=4,  # 4=fast, 10=balanced, 50=best
)

# 准备输入
images = np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8)  # [V, H, W, C]
state = np.random.randn(32).astype(np.float32)  # [state_dim]
task = "pick up the red block"

# 预测完整动作轨迹
action_chunk = model.predict_action_chunk(images, state, task)
print(f"Action chunk shape: {action_chunk.shape}")  # [40, action_dim]

# 单步动作 (带 temporal chunking)
model.reset()
for step in range(100):
    action = model.select_action(images, state, task)
    # robot.execute(action)
```

### 3. 使用推理引擎

```python
from inference_groot_n1_7 import GR00TN17InferenceEngine

engine = GR00TN17InferenceEngine(
    model_path="nvidia/GR00T-N1.7-3B",
    device="cuda",
    num_inference_timesteps=4,
)

# 推理
action_chunk = engine.predict_action_chunk(images, state, "open the drawer")

# 单步控制
engine.reset()
action = engine.select_action(images, state, "open the drawer")

# 性能测试
stats = engine.benchmark(num_runs=100)
```

### 4. 命令行推理

```bash
# 加载模型并测试
python inference_groot_n1_7.py \
    --model nvidia/GR00T-N1.7-3B \
    --task "pick up the object" \
    --device cuda \
    --num-timesteps 4

# 性能基准测试
python inference_groot_n1_7.py \
    --model nvidia/GR00T-N1.7-3B \
    --benchmark \
    --num-runs 100 \
    --num-cameras 1 \
    --state-dim 32

# 使用本地模型
python inference_groot_n1_7.py \
    --model /path/to/local/GR00T-N1.7-3B \
    --task "grasp the cup"
```

## 推理参数调优

| 参数 | 值 | 延迟 | 质量 | 推荐场景 |
|------|-----|------|------|----------|
| `num_inference_timesteps` | 4 | ~50ms | 良好 | 实时控制 (20Hz+) |
| | 10 | ~120ms | 很好 | 平衡速度和质量 |
| | 50 | ~600ms | 最佳 | 离线/最高精度 |

**RTC (Receding Horizon Control)**:
- 默认 `rtc_ramp_rate=6.0`
- 控制动作块之间的重叠混合
- 值越大过渡越平滑，但计算量略增

## 文件结构

```
groot_n1_7/
├── __init__.py                      # 模块入口
├── configuration_groot_n1_7.py      # 配置类
├── modeling_groot_n1_7.py           # 推理模型封装
├── processor_groot_n1_7.py          # 输入预处理/输出后处理
└── README.md                        # 本文档

inference_groot_n1_7.py              # 独立推理脚本 (CLI + Python API)
```

## LeRobot 原始实现参考

本代码基于 LeRobot 中 `src/lerobot/policies/groot/` 的实现:

| LeRobot 文件 | 功能 |
|-------------|------|
| `configuration_groot.py` | GrootConfig 配置类 |
| `modeling_groot.py` | GrootPolicy 策略包装器 |
| `groot_n1_7.py` | GR00TN17 核心模型 |
| `processor_groot.py` | 预/后处理器 |
| `action_head/cross_attention_dit.py` | AlternateVLDiT 实现 |
| `utils.py` | 工具函数 (归一化, 位姿变换等) |

完整 LeRobot 集成方式:
```python
from lerobot.policies.groot import GrootPolicy, GrootConfig

config = GrootConfig(base_model_path="nvidia/GR00T-N1.7-3B")
policy = GrootPolicy(config)
policy.load_model()

# 或使用 LeRobot 的标准接口
from lerobot.policies import make_policy
policy = make_policy("groot", pretrained_path="nvidia/GR00T-N1.7-3B")
```

## 参考文献

```bibtex
@article{gr00t2025,
  title={GR00T N1: An Open Foundation Model for Generalist Humanoid Robots},
  author={NVIDIA},
  journal={arXiv preprint arXiv:2503.14734},
  year={2025}
}

@software{lerobot2024,
  title={LeRobot: State-of-the-art AI for Real-World Robotics},
  author={Cadene, Remi and Alibert, Simon and Soare, Alexander and others},
  url={https://github.com/huggingface/lerobot},
  year={2024}
}
```

## License

Apache License 2.0 (与 NVIDIA GR00T N1.7 保持一致)
