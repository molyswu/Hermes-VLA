#!/usr/bin/env python3
"""
GR00T N1.7 Inference Script.

Run inference with NVIDIA's GR00T N1.7 Vision-Language-Action model.

GR00T N1.7 is a 3B-parameter open-source VLA model by NVIDIA,
built on Qwen3-VL (Cosmos-Reason2-2B) with a flow-matching diffusion
action head. It supports multi-embodiment robot control.

Usage:
    # Quick test with dummy inputs (no model download needed)
    python inference_groot_n1_7.py --test

    # Load and run inference
    python inference_groot_n1_7.py \\
        --model nvidia/GR00T-N1.7-3B \\
        --task "pick up the red block" \\
        --device cuda

    # With local model
    python inference_groot_n1_7.py \\
        --model /path/to/GR00T-N1.7-3B \\
        --task "open the drawer"

    # Benchmark mode
    python inference_groot_n1_7.py \\
        --model nvidia/GR00T-N1.7-3B \\
        --benchmark \\
        --num-runs 100

    # With real camera input
    python inference_groot_n1_7.py \\
        --model nvidia/GR00T-N1.7-3B \\
        --task "grasp the cup" \\
        --camera-topic /camera/color \\
        --state-topic /robot/joint_states

References:
    - Paper: https://arxiv.org/abs/2503.14734
    - Model: https://huggingface.co/nvidia/GR00T-N1.7-3B
    - Code:  https://github.com/NVIDIA/Isaac-GR00T
    - LeRobot integration: https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies/groot

Copyright 2025 NVIDIA Corporation and The HuggingFace Inc. team.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

# Add parent directory to path for local imports
sys.path.insert(0, str(Path(__file__).parent))

from groot_n1_7 import GR00TN17Config, GR00TN17ForInference, GR00TN17Processor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("groot_n1_7")


# ---------------------------------------------------------------------------
# Inference Engine
# ---------------------------------------------------------------------------

class GR00TN17InferenceEngine:
    """
    High-level inference engine for GR00T N1.7.

    Provides a simple API for robot control with GR00T N1.7:
    - predict_action_chunk(): Generate full action trajectory
    - select_action(): Get single action with temporal chunking
    - benchmark(): Measure inference speed
    - reset(): Clear action queue
    """

    def __init__(
        self,
        model_path: str = "nvidia/GR00T-N1.7-3B",
        *,
        device: str = "cuda",
        num_inference_timesteps: int = 4,
        use_bf16: bool = True,
        config: Optional[GR00TN17Config] = None,
    ):
        self.device = device
        self.num_inference_timesteps = num_inference_timesteps

        # Create or use provided config
        if config is not None:
            self.config = config
        else:
            self.config = GR00TN17Config(
                base_model_path=model_path,
                device=device,
                num_inference_timesteps=num_inference_timesteps,
                use_bf16=use_bf16,
            )

        # Load model
        logger.info(f"Initializing GR00T N1.7 from {model_path}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Inference timesteps: {num_inference_timesteps}")
        logger.info(f"  BF16: {use_bf16}")

        self.model = GR00TN17ForInference.from_pretrained(
            pretrained_path=model_path,
            device=device,
            num_inference_timesteps=num_inference_timesteps,
            use_bf16=use_bf16,
        )
        self.processor = GR00TN17Processor(self.config)

        # Latency tracking
        self._latency_history: list[float] = []

        logger.info(f"GR00T N1.7 inference engine ready.")
        logger.info(f"  Action dim: {self.config.action_dim}")
        logger.info(f"  Chunk size: {self.config.chunk_size}")
        logger.info(f"  State dim: {self.config.max_state_dim}")

    def predict_action_chunk(
        self,
        images: np.ndarray,
        state: np.ndarray,
        task: str,
    ) -> np.ndarray:
        """
        Generate a full action trajectory.

        Args:
            images: [V, H, W, C] numpy array (uint8 0-255 or float32 0-1)
            state: [state_dim] numpy array
            task: Language instruction

        Returns:
            action_chunk: [chunk_size, action_dim] numpy array
        """
        t_start = time.perf_counter()
        action_chunk = self.model.predict_action_chunk(images, state, task)
        latency_ms = (time.perf_counter() - t_start) * 1000
        self._latency_history.append(latency_ms)
        return action_chunk

    def select_action(
        self,
        images: np.ndarray,
        state: np.ndarray,
        task: str,
    ) -> np.ndarray:
        """
        Get single action step using temporal action chunking.

        Uses an internal action queue. When exhausted, predicts a new chunk.
        """
        return self.model.select_action(images, state, task)

    def reset(self) -> None:
        """Reset action queue for a new episode."""
        self.model.reset()

    def get_latency_stats(self) -> dict:
        """Get inference latency statistics."""
        if not self._latency_history:
            return {}
        latencies = np.array(self._latency_history)
        return {
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "p50_ms": float(np.percentile(latencies, 50)),
            "p95_ms": float(np.percentile(latencies, 95)),
            "p99_ms": float(np.percentile(latencies, 99)),
            "count": len(latencies),
        }

    def benchmark(
        self,
        num_runs: int = 100,
        num_cameras: int = 1,
        image_size: int = 224,
        state_dim: int = 32,
        task: str = "pick up the object",
    ) -> dict:
        """
        Run inference benchmark with dummy inputs.

        Args:
            num_runs: Number of inference iterations
            num_cameras: Number of simulated camera views
            image_size: Simulated image resolution
            state_dim: Simulated state dimension
            task: Task instruction for benchmark

        Returns:
            Latency statistics dict
        """
        logger.info(f"Running benchmark: {num_runs} iterations...")
        logger.info(f"  Cameras: {num_cameras}, Image size: {image_size}x{image_size}")
        logger.info(f"  State dim: {state_dim}, Task: '{task}'")

        # Create dummy inputs
        dummy_images = np.random.randint(
            0, 255,
            (num_cameras, image_size, image_size, 3),
            dtype=np.uint8,
        )
        dummy_state = np.random.randn(state_dim).astype(np.float32)

        # Warmup
        logger.info("Warming up (10 iterations)...")
        for _ in range(10):
            _ = self.predict_action_chunk(dummy_images, dummy_state, task)

        # Reset stats
        self._latency_history = []

        # Benchmark
        logger.info(f"Benchmarking ({num_runs} iterations)...")
        for i in range(num_runs):
            _ = self.predict_action_chunk(dummy_images, dummy_state, task)
            if (i + 1) % 20 == 0:
                logger.info(f"  Progress: {i + 1}/{num_runs}")

        stats = self.get_latency_stats()
        logger.info("\n" + "=" * 50)
        logger.info("Benchmark Results")
        logger.info("=" * 50)
        for k, v in stats.items():
            if k != "count":
                logger.info(f"  {k}: {v:.2f} ms" if isinstance(v, float) else f"  {k}: {v}")
        logger.info(f"  Iterations: {stats['count']}")
        logger.info(f"  Approximate FPS: {1000 / stats['mean_ms']:.1f} Hz")
        logger.info("=" * 50)

        return stats


# ---------------------------------------------------------------------------
# Test mode: dry-run without loading the actual model
# ---------------------------------------------------------------------------

class GR00TN17MockInference:
    """
    Mock inference engine for testing without downloading the model.

    Simulates GR00T N1.7 inference with random outputs.
    Useful for:
    - Testing the inference pipeline
    - Integration testing
    - Estimating I/O overhead
    """

    def __init__(
        self,
        action_dim: int = 26,
        chunk_size: int = 40,
        device: str = "cpu",
    ):
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.device = device
        self._action_queue: list = []
        self._latency_history: list[float] = []

        logger.info(f"GR00T N1.7 Mock Inference Engine (no model download)")
        logger.info(f"  Action dim: {action_dim}")
        logger.info(f"  Chunk size: {chunk_size}")
        logger.info(f"  Device: {device}")

    def predict_action_chunk(
        self,
        images: np.ndarray,
        state: np.ndarray,
        task: str,
    ) -> np.ndarray:
        """Generate random action chunk (mock)."""
        t_start = time.perf_counter()
        # Simulate inference latency (~50ms)
        time.sleep(0.05)
        action_chunk = np.random.randn(self.chunk_size, self.action_dim).astype(np.float32)
        latency_ms = (time.perf_counter() - t_start) * 1000
        self._latency_history.append(latency_ms)
        return action_chunk

    def select_action(
        self,
        images: np.ndarray,
        state: np.ndarray,
        task: str,
    ) -> np.ndarray:
        """Get single action from mock queue."""
        if len(self._action_queue) == 0:
            chunk = self.predict_action_chunk(images, state, task)
            for i in range(len(chunk)):
                self._action_queue.append(chunk[i])
        return self._action_queue.pop(0)

    def reset(self) -> None:
        self._action_queue.clear()

    def get_latency_stats(self) -> dict:
        if not self._latency_history:
            return {}
        latencies = np.array(self._latency_history)
        return {
            "mean_ms": float(np.mean(latencies)),
            "std_ms": float(np.std(latencies)),
            "min_ms": float(np.min(latencies)),
            "max_ms": float(np.max(latencies)),
            "count": len(latencies),
        }

    def benchmark(self, num_runs: int = 100, **kwargs) -> dict:
        logger.info(f"Mock benchmark: {num_runs} iterations...")
        dummy_images = np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8)
        dummy_state = np.random.randn(32).astype(np.float32)
        for _ in range(10):
            _ = self.predict_action_chunk(dummy_images, dummy_state, "test")
        self._latency_history = []
        for _ in range(num_runs):
            _ = self.predict_action_chunk(dummy_images, dummy_state, "test")
        return self.get_latency_stats()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="GR00T N1.7 Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test without model
  python inference_groot_n1_7.py --test

  # Load and run inference
  python inference_groot_n1_7.py --model nvidia/GR00T-N1.7-3B --task "pick up the block"

  # Benchmark
  python inference_groot_n1_7.py --model nvidia/GR00T-N1.7-3B --benchmark --num-runs 50
        """,
    )

    # ---- Model ----
    parser.add_argument(
        "--model", type=str, default="nvidia/GR00T-N1.7-3B",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to config JSON (optional)",
    )

    # ---- Device ----
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cuda", "cpu", "mps"],
        help="Device for inference",
    )
    parser.add_argument(
        "--no-bf16", action="store_true",
        help="Disable bfloat16 precision",
    )

    # ---- Inference ----
    parser.add_argument(
        "--task", type=str, default="pick up the red block",
        help="Task instruction for inference",
    )
    parser.add_argument(
        "--num-timesteps", type=int, default=4,
        help="Flow-matching denoising steps (1-50). Lower = faster.",
    )

    # ---- Modes ----
    parser.add_argument(
        "--test", action="store_true",
        help="Run with mock engine (no model download)",
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run inference benchmark",
    )
    parser.add_argument(
        "--num-runs", type=int, default=100,
        help="Number of benchmark runs",
    )
    parser.add_argument(
        "--num-cameras", type=int, default=1,
        help="Number of cameras for benchmark",
    )
    parser.add_argument(
        "--state-dim", type=int, default=32,
        help="State dimension for benchmark",
    )

    # ---- ROS / Real Robot (optional) ----
    parser.add_argument(
        "--camera-topic", type=str, default=None,
        help="ROS camera topic for real robot inference",
    )
    parser.add_argument(
        "--state-topic", type=str, default=None,
        help="ROS joint state topic for real robot inference",
    )

    args = parser.parse_args()

    # ---- Test Mode (no model download) ----
    if args.test:
        logger.info("Running in TEST mode (mock engine, no model download)")
        engine = GR00TN17MockInference(
            action_dim=26,
            chunk_size=40,
            device=args.device,
        )

        dummy_images = np.random.randint(
            0, 255,
            (args.num_cameras, 224, 224, 3),
            dtype=np.uint8,
        )
        dummy_state = np.random.randn(args.state_dim).astype(np.float32)

        # Test action chunk prediction
        logger.info(f"Testing with task: '{args.task}'")
        action_chunk = engine.predict_action_chunk(dummy_images, dummy_state, args.task)
        logger.info(f"Predicted action chunk shape: {action_chunk.shape}")
        logger.info(f"Action stats: mean={action_chunk.mean():.4f}, std={action_chunk.std():.4f}")
        logger.info(f"First action: {action_chunk[0]}")
        logger.info(f"Last action: {action_chunk[-1]}")

        # Test temporal chunking
        logger.info("\nTesting temporal action chunking...")
        engine.reset()
        for i in range(5):
            action = engine.select_action(dummy_images, dummy_state, args.task)
            logger.info(f"  Step {i}: action shape={action.shape}")

        # Test benchmark
        if args.benchmark:
            stats = engine.benchmark(num_runs=args.num_runs)
        return

    # ---- Load real model ----
    logger.info(f"Loading GR00T N1.7 from: {args.model}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Inference timesteps: {args.num_timesteps}")

    try:
        engine = GR00TN17InferenceEngine(
            model_path=args.model,
            device=args.device,
            num_inference_timesteps=args.num_timesteps,
            use_bf16=not args.no_bf16,
        )
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error(
            "\nTo use GR00T N1.7, you need:\n"
            "  1. Install dependencies:\n"
            "     pip install transformers>=4.45 torch accelerate\n"
            "     pip install isaac-gr00t  # for full support\n"
            "  2. Download the model:\n"
            "     huggingface-cli download nvidia/GR00T-N1.7-3B\n"
            "  3. Or run with --test for mock inference."
        )
        sys.exit(1)

    # ---- Benchmark Mode ----
    if args.benchmark:
        engine.benchmark(
            num_runs=args.num_runs,
            num_cameras=args.num_cameras,
            image_size=224,
            state_dim=args.state_dim,
            task=args.task,
        )
        return

    # ---- Interactive / Demo Mode ----
    logger.info(f"\nGR00T N1.7 ready for inference.")
    logger.info(f"Task: '{args.task}'")
    logger.info(f"\nUsage from Python:")
    logger.info(f"  action_chunk = engine.predict_action_chunk(images, state, task)")
    logger.info(f"  action = engine.select_action(images, state, task)")
    logger.info(f"  engine.reset()  # Start new episode")

    # If ROS topics specified, try ROS integration
    if args.camera_topic or args.state_topic:
        _run_ros_inference(engine, args)

    # Otherwise, run a quick test inference
    logger.info("\nRunning quick test inference with dummy inputs...")
    dummy_images = np.random.randint(
        0, 255,
        (args.num_cameras, 224, 224, 3),
        dtype=np.uint8,
    )
    dummy_state = np.random.randn(args.state_dim).astype(np.float32)

    try:
        action_chunk = engine.predict_action_chunk(dummy_images, dummy_state, args.task)
        logger.info(f"Success! Action chunk shape: {action_chunk.shape}")
        logger.info(f"Action range: [{action_chunk.min():.4f}, {action_chunk.max():.4f}]")
        logger.info(f"First action: {action_chunk[0]}")
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        logger.info("This may be due to missing dependencies. Try --test for mock mode.")


def _run_ros_inference(engine, args):
    """Optional ROS integration for real robot inference."""
    try:
        import rospy
        from sensor_msgs.msg import Image as RosImage
        from sensor_msgs.msg import JointState
        from cv_bridge import CvBridge

        logger.info("ROS integration detected. Starting ROS node...")
        rospy.init_node("groot_n1_7_inference", anonymous=True)
        bridge = CvBridge()

        # This is a skeleton; users should customize for their robot
        rate = rospy.Rate(20)  # 20 Hz control loop

        while not rospy.is_shutdown():
            # Placeholder: users implement their own observation collection
            rate.sleep()

    except ImportError:
        logger.warning("ROS not available. Install with: pip install rospy sensor-msgs cv-bridge")


if __name__ == "__main__":
    main()
