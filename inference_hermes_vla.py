"""
Hermes-VLA Inference Script.

Run inference with a trained Hermes-VLA model on real robot observations.

Usage:
    python inference_hermes_vla.py \
        --checkpoint ./outputs/hermes_vla/best_model.pt \
        --task "pick up the red block" \
        --device cuda

    # For testing with random inputs:
    python inference_hermes_vla.py \
        --checkpoint ./outputs/hermes_vla/best_model.pt \
        --test
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from hermes_vla.configuration_hermes_vla import HermesVLAConfig
from hermes_vla.modeling_hermes_vla import HermesVLAPolicy
from hermes_vla.processor_hermes_vla import (
    HermesVLAPreprocessor,
    HermesVLAPostprocessor,
    SimpleTokenizer,
    preprocess_images,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class HermesVLAInference:
    """
    Inference wrapper for Hermes-VLA.

    Provides a simple API for real-time robot control:
    - process_observation(): Convert raw obs to model inputs
    - predict(): Generate action chunk
    - get_action(): Get single action with temporal chunking
    """

    def __init__(
        self,
        checkpoint_path: str,
        config: Optional[HermesVLAConfig] = None,
        device: str = "cuda",
    ):
        self.device = device

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

        # Load config
        if config is None:
            if "config" in checkpoint:
                config_data = checkpoint["config"]
                if isinstance(config_data, dict):
                    config = HermesVLAConfig(**config_data)
                else:
                    config = config_data
            else:
                raise ValueError("No config found in checkpoint. Please provide config explicitly.")

        self.config = config
        self.config.device = device

        # Create model
        logger.info("Loading Hermes-VLA model...")
        self.model = HermesVLAPolicy(config)
        self.model.to(device)

        # Load weights
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.eval()

        # Create processors
        self.preprocessor = HermesVLAPreprocessor(config=config)
        self.postprocessor = HermesVLAPostprocessor()
        self.tokenizer = SimpleTokenizer(max_length=config.tokenizer_max_length)

        # Track inference latency
        self._latency_history = []

        logger.info(f"Model loaded successfully. Params: {sum(p.numel() for p in self.model.parameters()):,}")
        logger.info(f"Visual encoder: {config.visual_encoder_type}")
        logger.info(f"Language encoder: {config.language_encoder_type}")
        logger.info(f"Fusion type: {config.fusion_type}")
        logger.info(f"Plan horizon: {config.plan_horizon}, Chunk size: {config.chunk_size}")

    def process_observation(
        self,
        images: np.ndarray,
        state: np.ndarray,
        task: str,
    ) -> dict:
        """
        Process raw observation into model-ready inputs.

        Args:
            images: Camera images, shape [V, H, W, C] or [H, W, C]
            state: Proprioceptive state, shape [state_dim]
            task: Language task description

        Returns:
            Dict with keys: "images", "input_ids", "attention_mask", "state"
        """
        # Preprocess images
        images_tensor = preprocess_images(images, self.config.image_size)

        # Tokenize task
        tokenized = self.tokenizer.tokenize([task])

        # Process state
        state_tensor = torch.from_numpy(np.asarray(state)).float()
        if state_tensor.shape[-1] < self.config.max_state_dim:
            state_tensor = torch.nn.functional.pad(
                state_tensor, (0, self.config.max_state_dim - state_tensor.shape[-1])
            )

        batch = {
            "images": images_tensor.to(self.device),
            "input_ids": tokenized["input_ids"].to(self.device),
            "attention_mask": tokenized["attention_mask"].to(self.device),
            "state": state_tensor.unsqueeze(0).to(self.device),  # Add batch dim
        }
        return batch

    @torch.no_grad()
    def predict_action_chunk(
        self,
        images: np.ndarray,
        state: np.ndarray,
        task: str,
    ) -> np.ndarray:
        """
        Generate a full action trajectory.

        Args:
            images: Camera images
            state: Current proprioceptive state
            task: Language instruction

        Returns:
            action_chunk: [chunk_size, action_dim] numpy array
        """
        batch = self.process_observation(images, state, task)

        t0 = torch.cuda.Event(enable_timing=True) if self.device.startswith("cuda") else None
        t1 = torch.cuda.Event(enable_timing=True) if self.device.startswith("cuda") else None

        if t0:
            t0.record()
            action_chunk = self.model.predict_action_chunk(batch)
            t1.record()
            torch.cuda.synchronize()
            latency = t0.elapsed_time(t1)
            self._latency_history.append(latency)
        else:
            import time
            start = time.time()
            action_chunk = self.model.predict_action_chunk(batch)
            latency = (time.time() - start) * 1000
            self._latency_history.append(latency)

        # Postprocess
        action_np = self.postprocessor(action_chunk.squeeze(0))
        return action_np

    def get_action(
        self,
        images: np.ndarray,
        state: np.ndarray,
        task: str,
    ) -> np.ndarray:
        """
        Get a single action step using temporal action chunking.

        Maintains internal state to re-use action chunks efficiently.
        """
        if self.model._action_chunk is None or self.model._chunk_index >= self.config.n_action_steps:
            self.model._action_chunk = self.predict_action_chunk(images, state, task)
            self.model._chunk_index = 0

        action = self.model._action_chunk[self.model._chunk_index]
        self.model._chunk_index += 1
        return action

    def reset(self):
        """Reset temporal chunking state."""
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
            "count": len(latencies),
        }

    def benchmark(self, num_runs: int = 100) -> dict:
        """Run inference benchmark with random inputs."""
        logger.info(f"Running benchmark: {num_runs} iterations...")

        # Create random inputs matching config
        dummy_images = np.random.randint(0, 255, (self.config.num_cameras, self.config.image_size, self.config.image_size, 3), dtype=np.uint8)
        dummy_state = np.random.randn(self.config.max_state_dim).astype(np.float32)
        task = "pick up the object and place it on the table"

        # Warmup
        for _ in range(10):
            _ = self.predict_action_chunk(dummy_images, dummy_state, task)

        # Benchmark
        import time
        self._latency_history = []
        for _ in range(num_runs):
            start = time.time()
            _ = self.predict_action_chunk(dummy_images, dummy_state, task)
            self._latency_history.append((time.time() - start) * 1000)

        stats = self.get_latency_stats()
        logger.info(f"Benchmark results:")
        for k, v in stats.items():
            logger.info(f"  {k}: {v}")

        return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Hermes-VLA Inference")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config JSON (optional, loaded from checkpoint if not provided)")
    parser.add_argument("--task", type=str, default="pick up the red block",
                        help="Task instruction for inference")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run inference on")

    # Mode
    parser.add_argument("--test", action="store_true",
                        help="Run with random dummy inputs for testing")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run inference benchmark")
    parser.add_argument("--num_runs", type=int, default=100,
                        help="Number of benchmark runs")

    args = parser.parse_args()

    # Load config if provided
    config = None
    if args.config:
        with open(args.config) as f:
            config_data = json.load(f)
        config = HermesVLAConfig(**config_data)

    # Create inference engine
    engine = HermesVLAInference(
        checkpoint_path=args.checkpoint,
        config=config,
        device=args.device,
    )

    if args.benchmark:
        engine.benchmark(num_runs=args.num_runs)
        return

    if args.test:
        # Test with dummy inputs
        logger.info(f"Testing inference with task: '{args.task}'")
        dummy_images = np.random.randint(
            0, 255,
            (engine.config.num_cameras, engine.config.image_size, engine.config.image_size, 3),
            dtype=np.uint8
        )
        dummy_state = np.random.randn(engine.config.max_state_dim).astype(np.float32)

        # Predict action chunk
        action_chunk = engine.predict_action_chunk(dummy_images, dummy_state, args.task)
        logger.info(f"Predicted action chunk shape: {action_chunk.shape}")
        logger.info(f"Action chunk stats: mean={action_chunk.mean():.4f}, std={action_chunk.std():.4f}")
        logger.info(f"First 3 actions:\n{action_chunk[:3]}")
        logger.info(f"Last 3 actions:\n{action_chunk[-3:]}")

        # Test temporal chunking
        logger.info("\nTesting temporal action chunking...")
        engine.reset()
        for i in range(5):
            action = engine.get_action(dummy_images, dummy_state, args.task)
            logger.info(f"  Step {i}: action shape={action.shape}, chunk_idx={engine.model._chunk_index}")
    else:
        logger.info(f"Hermes-VLA model loaded. Ready for inference.")
        logger.info(f"Use engine.predict_action_chunk(images, state, task) to generate actions.")


if __name__ == "__main__":
    main()
