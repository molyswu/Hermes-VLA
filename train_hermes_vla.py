"""
Hermes-VLA Training Script.

Trains the Hermes-VLA dual-stream hierarchical VLA model on robot manipulation datasets.

Usage:
    python train_hermes_vla.py \
        --dataset_path /path/to/dataset \
        --output_dir ./outputs/hermes_vla \
        --batch_size 32 \
        --num_epochs 100 \
        --lr 1e-4 \
        --visual_encoder siglip \
        --language_encoder gemma_2b

Integration with LeRobot datasets:
    Supports LeRobot-format datasets automatically.
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from hermes_vla.configuration_hermes_vla import (
    FusionType,
    HermesVLAConfig,
    LanguageEncoderType,
    VisualEncoderType,
)
from hermes_vla.modeling_hermes_vla import HermesVLAPolicy
from hermes_vla.processor_hermes_vla import (
    HermesVLAPreprocessor,
    HermesVLAPostprocessor,
    SimpleTokenizer,
    RunningNormalizer,
    make_hermes_vla_pre_post_processors,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class HermesVLADataset(Dataset):
    """
    Dataset wrapper for Hermes-VLA training.

    Supports both raw numpy files and LeRobot-format datasets.

    Expected directory structure (raw format):
        dataset_path/
            episode_0/
                step_0.npz   # contains: images, state, action
                step_1.npz
                ...
            episode_1/
                ...
            task_descriptions.json  # mapping episode_id -> task text
            stats.json              # normalization stats (optional)
    """

    def __init__(
        self,
        dataset_path: str,
        chunk_size: int = 50,
        num_cameras: int = 1,
        preprocessor: Optional[HermesVLAPreprocessor] = None,
    ):
        self.dataset_path = Path(dataset_path)
        self.chunk_size = chunk_size
        self.num_cameras = num_cameras
        self.preprocessor = preprocessor

        # Load task descriptions
        task_file = self.dataset_path / "task_descriptions.json"
        if task_file.exists():
            with open(task_file) as f:
                self.task_descriptions = json.load(f)
        else:
            self.task_descriptions = {}

        # Discover episodes and steps
        self.samples = []
        self._load_episodes()

        logger.info(f"Loaded {len(self.samples)} samples from {dataset_path}")

    def _load_episodes(self):
        """Discover all episodes and index valid (obs, action) pairs."""
        # Try LeRobot format first
        meta_path = self.dataset_path / "meta" / "info.json"
        if meta_path.exists():
            self._load_lerobot_format()
        else:
            self._load_raw_format()

    def _load_lerobot_format(self):
        """Load from LeRobot dataset format."""
        episodes_dir = self.dataset_path / "data"
        if episodes_dir.exists():
            for ep_file in sorted(episodes_dir.glob("episode_*.parquet")):
                import pandas as pd
                df = pd.read_parquet(ep_file)
                ep_id = ep_file.stem
                n_steps = len(df)
                for i in range(max(0, n_steps - self.chunk_size)):
                    self.samples.append({
                        "episode": ep_id,
                        "start_idx": i,
                    })

    def _load_raw_format(self):
        """Load from raw NPZ format."""
        for ep_dir in sorted(self.dataset_path.iterdir()):
            if not ep_dir.is_dir() or not ep_dir.name.startswith("episode"):
                continue
            ep_id = ep_dir.name
            step_files = sorted(ep_dir.glob("step_*.npz"))
            for i in range(max(0, len(step_files) - self.chunk_size)):
                self.samples.append({
                    "episode": ep_id,
                    "start_idx": i,
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        ep_id = sample["episode"]
        start = sample["start_idx"]

        # Load steps
        ep_dir = self.dataset_path / ep_id if (self.dataset_path / ep_id).exists() else self.dataset_path / "data"
        step_files = sorted((ep_dir).glob("step_*.npz"))

        images_batch = []
        states_batch = []
        actions_batch = []

        for j in range(start, min(start + self.chunk_size, len(step_files))):
            data = np.load(str(step_files[j]))
            images_batch.append(data.get("images", data.get("image")))
            states_batch.append(data["state"])
            actions_batch.append(data["action"])

        # Pad if needed
        while len(actions_batch) < self.chunk_size:
            actions_batch.append(actions_batch[-1])
            states_batch.append(states_batch[-1])
            images_batch.append(images_batch[-1])

        # Stack
        images = np.stack(images_batch, axis=0)  # [chunk, ...]
        state = np.stack(states_batch, axis=0)    # [chunk, state_dim]
        actions = np.stack(actions_batch, axis=0)  # [chunk, action_dim]

        # Get task description
        task = self.task_descriptions.get(ep_id, "perform the task")

        return {
            "images": images,  # Expected: [chunk, ...]
            "observation.state": state[0],  # Current state only
            "state": state[0],
            "action": actions,
            "task": task,
        }


# ---------------------------------------------------------------------------
# Training Loop
# ---------------------------------------------------------------------------

def train_hermes_vla(
    config: HermesVLAConfig,
    dataset_path: str,
    output_dir: str,
    batch_size: int = 32,
    num_epochs: int = 100,
    lr: float = 1e-4,
    log_interval: int = 100,
    save_interval: int = 1000,
    resume_from: Optional[str] = None,
    device: str = "cuda",
):
    """Main training function for Hermes-VLA."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Override config with CLI args
    config.optimizer_lr = lr
    config.device = device

    # Create model
    logger.info("Initializing Hermes-VLA model...")
    model = HermesVLAPolicy(config)
    model.to(device)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Create processors
    preprocessor = HermesVLAPreprocessor(config=config)
    postprocessor = HermesVLAPostprocessor()
    tokenizer = SimpleTokenizer(max_length=config.tokenizer_max_length)

    # Create dataset & dataloader
    logger.info(f"Loading dataset from {dataset_path}...")
    dataset = HermesVLADataset(
        dataset_path=dataset_path,
        chunk_size=config.chunk_size,
        num_cameras=config.num_cameras,
        preprocessor=preprocessor,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.optimizer_lr,
        weight_decay=config.optimizer_weight_decay,
    )

    # Learning rate scheduler (cosine decay with warmup)
    warmup_steps = config.scheduler_warmup_steps
    total_steps = num_epochs * len(dataloader)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Automatic mixed precision
    scaler = torch.cuda.amp.GradScaler() if device.startswith("cuda") else None

    # Training loop
    global_step = 0
    best_loss = float("inf")

    logger.info(f"Starting training: {num_epochs} epochs, {len(dataloader)} steps/epoch")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_flow_loss = 0.0
        epoch_plan_loss = 0.0
        t0 = time.time()

        for batch_idx, raw_batch in enumerate(dataloader):
            # Preprocess
            batch = {}
            tasks = raw_batch.get("task", raw_batch.get("language_instruction", [""]))
            if isinstance(tasks, (list, tuple, np.ndarray)):
                tasks = list(tasks)
            else:
                tasks = [tasks]

            # Tokenize
            tokenized = tokenizer.tokenize(tasks)
            batch["input_ids"] = tokenized["input_ids"].to(device)
            batch["attention_mask"] = tokenized["attention_mask"].to(device)

            # Images
            images = raw_batch["images"]
            if isinstance(images, np.ndarray):
                images = torch.from_numpy(images).float().to(device)
            elif isinstance(images, torch.Tensor):
                images = images.float().to(device)
            # Normalize to [-1, 1]
            images = images / 255.0 * 2.0 - 1.0

            # Handle image shape: [B, T, C, H, W] -> use first frame; or [B, V, C, H, W]
            if images.dim() == 5 and images.shape[1] > 1:
                # Take first frame for simplicity (could do temporal encoding)
                images = images[:, 0:1, ...]  # [B, 1, C, H, W]
            batch["images"] = images

            # State
            state = raw_batch.get("observation.state", raw_batch.get("state"))
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state).float().to(device)
            elif isinstance(state, torch.Tensor):
                state = state.float().to(device)
            # State shape: [B, state_dim] or [B, 1, state_dim]
            if state.dim() == 3:
                state = state.squeeze(1)
            batch["state"] = state

            # Actions
            actions = raw_batch["action"]
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions).float().to(device)
            elif isinstance(actions, torch.Tensor):
                actions = actions.float().to(device)
            batch["actions"] = actions

            # Forward pass
            optimizer.zero_grad()

            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model.model(
                        images=batch["images"],
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        state=batch["state"],
                        actions=batch["actions"],
                    )
                    loss = outputs["loss"]
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model.model(
                    images=batch["images"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    state=batch["state"],
                    actions=batch["actions"],
                )
                loss = outputs["loss"]
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

            epoch_loss += loss.item()
            epoch_flow_loss += outputs.get("flow_loss", 0)
            epoch_plan_loss += outputs.get("planning_aux_loss", 0)
            global_step += 1

            if global_step % log_interval == 0:
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs} "
                    f"Step {global_step} "
                    f"Loss: {loss.item():.6f} "
                    f"(Flow: {outputs.get('flow_loss', 0):.6f}, "
                    f"Plan: {outputs.get('planning_aux_loss', 0):.6f}) "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )

            if global_step % save_interval == 0:
                checkpoint_path = output_dir / f"checkpoint_step_{global_step}.pt"
                torch.save({
                    "step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "loss": loss.item(),
                }, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")

        avg_loss = epoch_loss / len(dataloader)
        elapsed = time.time() - t0
        logger.info(
            f"Epoch {epoch+1} complete | "
            f"Avg Loss: {avg_loss:.6f} | "
            f"Avg Flow: {epoch_flow_loss/len(dataloader):.6f} | "
            f"Avg Plan: {epoch_plan_loss/len(dataloader):.6f} | "
            f"Time: {elapsed:.1f}s"
        )

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), output_dir / "best_model.pt")
            logger.info(f"New best model! Loss: {best_loss:.6f}")

    # Save final model
    torch.save(model.state_dict(), output_dir / "final_model.pt")
    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.__dict__, f, indent=2, default=str)
    logger.info(f"Training complete. Model saved to {output_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train Hermes-VLA")

    # Data
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Path to dataset directory")
    parser.add_argument("--output_dir", type=str, default="./outputs/hermes_vla",
                        help="Output directory for checkpoints")

    # Model architecture
    parser.add_argument("--visual_encoder", type=str, default="siglip",
                        choices=["siglip", "dinov2", "clip", "eagle2"])
    parser.add_argument("--language_encoder", type=str, default="gemma_2b",
                        choices=["gemma_300m", "gemma_2b", "smollm", "qwen2_0.5b"])
    parser.add_argument("--fusion_type", type=str, default="cross_attention",
                        choices=["cross_attention", "gated_fusion", "concat_linear"])
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--num_fusion_layers", type=int, default=4)
    parser.add_argument("--planner_layers", type=int, default=4)
    parser.add_argument("--controller_layers", type=int, default=8)
    parser.add_argument("--plan_dim", type=int, default=256)
    parser.add_argument("--plan_horizon", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=50)
    parser.add_argument("--max_state_dim", type=int, default=32)
    parser.add_argument("--max_action_dim", type=int, default=32)
    parser.add_argument("--num_cameras", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=224)

    # Training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=4)

    # Device
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume_from", type=str, default=None)

    # Finetuning
    parser.add_argument("--freeze_visual", action="store_true")
    parser.add_argument("--freeze_language", action="store_true")
    parser.add_argument("--train_planner_only", action="store_true")
    parser.add_argument("--train_controller_only", action="store_true")

    args = parser.parse_args()

    # Create config
    config = HermesVLAConfig(
        visual_encoder_type=VisualEncoderType(args.visual_encoder),
        language_encoder_type=LanguageEncoderType(args.language_encoder),
        fusion_type=FusionType(args.fusion_type),
        hidden_dim=args.hidden_dim,
        num_fusion_layers=args.num_fusion_layers,
        planner_layers=args.planner_layers,
        controller_layers=args.controller_layers,
        plan_dim=args.plan_dim,
        plan_horizon=args.plan_horizon,
        chunk_size=args.chunk_size,
        max_state_dim=args.max_state_dim,
        max_action_dim=args.max_action_dim,
        num_cameras=args.num_cameras,
        image_size=args.image_size,
        gradient_checkpointing=args.gradient_checkpointing,
        optimizer_lr=args.lr,
        optimizer_weight_decay=args.weight_decay,
        scheduler_warmup_steps=args.warmup_steps,
        freeze_visual_encoder=args.freeze_visual,
        freeze_language_encoder=args.freeze_language,
        train_planner_only=args.train_planner_only,
        train_controller_only=args.train_controller_only,
        device=args.device,
    )

    train_hermes_vla(
        config=config,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        lr=args.lr,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_from=args.resume_from,
        device=args.device,
    )


if __name__ == "__main__":
    main()
