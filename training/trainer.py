"""Trimmed DDP trainer for Surflo flow-matching training.

Kept faithful to the canonical training runs while dropping non-canonical
machinery. Retained:

* DDP + AMP (bf16) + gradient accumulation
* Gradient clipping on ``surface_net`` (fvcore-style ``where``-driven LR/WD schedulers)
* EMA of the model weights (``ema_pytorch``)
* Preemption-safe checkpoint save / resume
* Scalar loss logging (Weights & Biases, scalar-only)
* Periodic validation loss + validation Chamfer distance, incl. OOD validation

Dropped from the original: qualitative visuals (orbit videos / point-cloud /
mesh renders), TensorBoard image/histogram logging, profiling, mean-flow /
elastic / self-distillation branches, batch-repetition augmentation, and the
raw-image (non-cached) preprocessing path. Training consumes preprocessed
(cached) DL3DV data only.
"""
import contextlib
import gc
import json
import logging
import os
import time
from datetime import timedelta
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
from hydra.utils import instantiate

from surflo.metrics.chamfer import compute_chamfer_distance
from utils.checkpoint import DDPCheckpointSaver, load_checkpoint_with_fallback
from utils.distributed import get_machine_local_and_dist_rank
from utils.general import (
    AverageMeter,
    DurationMeter,
    ProgressMeter,
    copy_data_to_device,
    get_resume_checkpoint,
    is_dist_avail_and_initialized,
    model_summary,
    safe_makedirs,
    set_seeds,
)
from utils.logging import setup_logging
from utils.optimizer import construct_optimizers

# Module-level alias: the ``__init__`` argument ``logging`` (the Hydra logging
# config) shadows the stdlib ``logging`` module inside that method, so use this
# alias there.
_log = logging

# --- Environment variables for performance / debugging ---
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"


class Trainer:
    """A generic DDP trainer (also supports multi-node).

    Orchestrates the full training / validation process: distributed setup,
    model / loss / optimizer / dataloader construction, checkpointing, the
    training loop, and validation (loss + Chamfer, in-distribution and OOD).
    """

    EPSILON = 1e-8

    def __init__(
        self,
        *,
        data: Dict[str, Any],
        model: Dict[str, Any],
        logging: Dict[str, Any],
        checkpoint: Dict[str, Any],
        max_epochs: int,
        mode: str = "train",
        device: str = "cuda",
        seed_value: int = 123,
        val_epoch_freq: int = 1,
        slow_val_epoch_freq: Optional[int] = None,
        ood_val_epoch_freq: Optional[int] = None,
        distributed: Dict[str, bool] = None,
        cuda: Dict[str, bool] = None,
        limit_train_batches: Optional[int] = None,
        limit_val_batches: Optional[int] = None,
        optim: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        accum_steps: int = 1,
        **kwargs,
    ):
        """Initialize the Trainer.

        Args:
            data: Hydra config for datasets and dataloaders.
            model: Hydra config for the model.
            logging: Hydra config for logging (wandb writer, log frequencies).
            checkpoint: Hydra config for checkpointing.
            max_epochs: Total number of epochs to train.
            mode: ``"train"`` for training + validation, ``"val"`` for validation only.
            device: ``"cuda"`` or ``"cpu"``.
            seed_value: Random seed for reproducibility.
            val_epoch_freq: Frequency (in epochs) to run fast validation.
            slow_val_epoch_freq: Frequency (in epochs) to run full validation.
            ood_val_epoch_freq: Frequency (in epochs) to run OOD validation. ``None`` disables it.
            distributed: Hydra config for DDP settings.
            cuda: Hydra config for CUDA-specific settings (cuDNN / TF32).
            limit_train_batches: Limit training batches per epoch.
            limit_val_batches: Limit validation batches per epoch.
            optim: Hydra config for optimizers and schedulers.
            loss: Hydra config for the loss function.
            env_variables: Extra environment variables to set.
            accum_steps: Gradient-accumulation steps before an optimizer step.
        """
        self._setup_env_variables(env_variables)
        self._setup_timers()

        # Store Hydra configurations
        self.data_conf = data
        self.model_conf = model
        self.loss_conf = loss
        self.logging_conf = logging
        self.checkpoint_conf = checkpoint
        self.optim_conf = optim

        # Store hyperparameters
        self.accum_steps = accum_steps
        self.max_epochs = max_epochs
        self.mode = mode
        self.val_epoch_freq = val_epoch_freq
        self.slow_val_epoch_freq = slow_val_epoch_freq
        self.ood_val_epoch_freq = ood_val_epoch_freq
        self.limit_train_batches = limit_train_batches
        self.limit_val_batches = limit_val_batches
        self.seed_value = seed_value

        # 'where' tracks training progress from 0.0 to 1.0 for schedulers
        self.where = 0.0

        self.optims = None
        self.ema_model = None

        self._setup_device(device)
        self._setup_torch_dist_and_backend(cuda, distributed)

        # Setup logging directory and configure logger
        exp_dir = os.path.join(self.logging_conf.log_dir, self.logging_conf.writer.exp_name)
        safe_makedirs(exp_dir)
        setup_logging(
            __name__,
            output_dir=exp_dir,
            rank=self.rank,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
            all_ranks=self.logging_conf.all_ranks,
        )
        set_seeds(seed_value, self.max_epochs, self.distributed_rank)

        assert is_dist_avail_and_initialized(), "Torch distributed needs to be initialized before calling the trainer."

        # Instantiate components (model, loss, etc.)
        self._setup_components()
        self._setup_dataloaders()

        # Move model to the correct device
        self.model.to(self.device)
        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.4f")

        # Construct optimizers (after moving model to device)
        if self.mode != "val":
            self.optims = construct_optimizers(self.model, self.optim_conf)

        # Load checkpoint if available or specified
        if self.checkpoint_conf.resume_checkpoint_path is not None:
            self._load_resuming_checkpoint(self.checkpoint_conf.resume_checkpoint_path)
        else:
            ckpt_path = get_resume_checkpoint(self.checkpoint_conf.save_dir)
            if ckpt_path is not None:
                self._load_resuming_checkpoint(ckpt_path)

        # Wrap the model with DDP
        self._setup_ddp_distributed_training(distributed, device)

        # Setup EMA model
        if self.optim_conf.ema.enabled:
            self._setup_ema_model()
            # Restore EMA state from checkpoint if available
            if getattr(self, "_pending_ema_state", None) is not None:
                self.ema_model.load_state_dict(self._pending_ema_state)
                _log.info("EMA state restored from checkpoint.")
                self._pending_ema_state = None

        # Barrier to ensure all processes are synchronized before starting
        dist.barrier()

    def _setup_timers(self):
        """Initializes timers for tracking total elapsed time."""
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0

    def _setup_env_variables(self, env_variables_conf: Optional[Dict[str, Any]]) -> None:
        """Sets environment variables from the configuration."""
        if env_variables_conf:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value
        logging.info(f"Environment:\n{json.dumps(dict(os.environ), sort_keys=True, indent=2)}")

    def _setup_torch_dist_and_backend(self, cuda_conf: Dict, distributed_conf: Dict) -> None:
        """Initializes the distributed process group and configures PyTorch backends."""
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
            torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = cuda_conf.allow_tf32
            torch.backends.cudnn.allow_tf32 = cuda_conf.allow_tf32

        dist.init_process_group(backend=distributed_conf.backend, timeout=timedelta(minutes=distributed_conf.timeout_mins))
        self.rank = dist.get_rank()

    def _load_resuming_checkpoint(self, ckpt_path: str):
        """Loads a checkpoint from the given path to resume training.

        Uses ``load_checkpoint_with_fallback`` so that a corrupted/truncated
        primary checkpoint (e.g. the job was killed mid-save) automatically
        falls back to ``<ckpt_path>.bak``.
        """
        logging.info(f"Resuming training from {ckpt_path} (rank {self.rank})")

        checkpoint, loaded_from = load_checkpoint_with_fallback(ckpt_path, map_location="cpu")
        if loaded_from != ckpt_path:
            logging.warning(
                f"Primary checkpoint {ckpt_path} was unusable; resumed from "
                f"backup {loaded_from}. Training will re-save a fresh primary "
                f"at the next checkpoint."
            )

        # Load model state
        model_state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        missing, unexpected = self.model.load_state_dict(model_state_dict, strict=self.checkpoint_conf.strict)
        if self.rank == 0:
            logging.info(f"Model state loaded. Missing keys: {missing or 'None'}. Unexpected keys: {unexpected or 'None'}.")

        # Load optimizer state if available and in training mode
        if "optimizer" in checkpoint and self.optims is not None:
            self.optims[0].optimizer.load_state_dict(checkpoint["optimizer"])

        # Load training progress. The saved epoch is the *completed* epoch;
        # resume from the next one.
        completed_epoch = checkpoint.get("epoch", None)
        if completed_epoch is not None:
            self.epoch = completed_epoch + 1
        self.steps = checkpoint.get("steps", {"train": 0, "val": 0, "ood_val": 0})
        if "ood_val" not in self.steps:
            self.steps["ood_val"] = 0
        self.ckpt_time_elapsed = checkpoint.get("time_elapsed", 0)

        # Load AMP scaler state if available
        if self.optim_conf.amp.enabled and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        # Stash EMA state for loading after _setup_ema_model()
        self._pending_ema_state = checkpoint.get("ema_state", None)
        if self._pending_ema_state is not None:
            logging.info("EMA state found in checkpoint — will be loaded after EMA setup.")

    def _setup_device(self, device: str):
        """Sets up the device for training (CPU or CUDA)."""
        self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()
        if device == "cuda":
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        elif device == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported device: {device}")

    def _setup_components(self):
        """Initializes all core training components using Hydra configs."""
        logging.info("Setting up components: Model, Loss, Logger, etc.")
        self.epoch = 0
        self.steps = {"train": 0, "val": 0, "ood_val": 0}

        # Instantiate components from configs
        self.writer = instantiate(self.logging_conf.writer, _recursive_=False)
        self.model = instantiate(self.model_conf, _recursive_=False)
        self.loss = instantiate(self.loss_conf, _recursive_=False)
        self.gradient_clipper = instantiate(self.optim_conf.gradient_clip)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.optim_conf.amp.enabled)

        # Log model summary on rank 0
        if self.rank == 0:
            model_summary_path = os.path.join(self.logging_conf.log_dir, "model.txt")
            model_summary(self.model, log_file=model_summary_path)
            logging.info(f"Model summary saved to {model_summary_path}")

        logging.info("Successfully initialized training components.")

    def _setup_dataloaders(self):
        """Initializes train, validation, and OOD validation datasets."""
        self.train_dataset = None
        self.val_dataset = None
        self.ood_val_dataset = None

        if self.mode in ["train", "val"]:
            self.val_dataset = instantiate(self.data_conf.get("val", None), _recursive_=False)
            if self.val_dataset is not None:
                self.val_dataset.seed = self.seed_value

            if self.ood_val_epoch_freq is not None:
                self.ood_val_dataset = instantiate(self.data_conf.get("ood_val", None), _recursive_=False)
                if self.ood_val_dataset is not None:
                    self.ood_val_dataset.seed = self.seed_value
                    logging.info(f"OOD validation dataset initialized (freq={self.ood_val_epoch_freq})")

        if self.mode in ["train"]:
            self.train_dataset = instantiate(self.data_conf.train, _recursive_=False)
            self.train_dataset.seed = self.seed_value

    def _setup_ddp_distributed_training(self, distributed_conf: Dict, device: str):
        """Wraps the model with DistributedDataParallel (DDP)."""
        assert isinstance(self.model, torch.nn.Module)

        ddp_options = dict(
            find_unused_parameters=distributed_conf.find_unused_parameters,
            gradient_as_bucket_view=distributed_conf.gradient_as_bucket_view,
            bucket_cap_mb=distributed_conf.bucket_cap_mb,
            broadcast_buffers=distributed_conf.broadcast_buffers,
        )

        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if device == "cuda" else [],
            **ddp_options,
        )

    def _setup_ema_model(self):
        """Sets up the EMA model.

        When ``use_learned_tokens`` is enabled, only the ``surface_net`` (decoder
        + learned tokens) is wrapped in EMA, avoiding unnecessary work on frozen
        parameters. Otherwise the full model is wrapped.
        """
        model = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model

        if model.surface_net.use_learned_tokens:
            logging.info("Setting up EMA model for surface_net only (learned tokens mode)")
            self.ema_model = instantiate(self.optim_conf.ema.model, model=model.surface_net)
            self._ema_surface_net_only = True
        else:
            logging.info("Setting up EMA model for full model")
            self.ema_model = instantiate(self.optim_conf.ema.model, model=model)
            self._ema_surface_net_only = False

    def _get_ema_model(self):
        """Returns the full model with EMA weights applied.

        When EMA wraps only ``surface_net``, swaps the EMA'd ``surface_net`` into
        the full model and returns the original so it can be restored afterwards.
        """
        model = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model

        if self._ema_surface_net_only:
            original_surface_net = model.surface_net
            model.surface_net = self.ema_model.ema_model
            return model, original_surface_net
        else:
            return self.ema_model.ema_model, None

    def _restore_surface_net(self, model, original_surface_net):
        """Restore the original surface_net after using EMA model for inference."""
        if original_surface_net is not None:
            model.surface_net = original_surface_net

    def save_checkpoint(self, epoch: int, checkpoint_names: Optional[List[str]] = None):
        """Saves a training checkpoint.

        Args:
            epoch: The current epoch number.
            checkpoint_names: Names for the checkpoint file(s). If ``None``, saves
                ``checkpoint`` and (on frequency) ``checkpoint_{epoch}``.
        """
        checkpoint_folder = self.checkpoint_conf.save_dir
        safe_makedirs(checkpoint_folder)
        if checkpoint_names is None:
            checkpoint_names = ["checkpoint"]
            if (
                self.checkpoint_conf.save_freq > 0
                and int(epoch) % self.checkpoint_conf.save_freq == 0
                and (int(epoch) > 0 or self.checkpoint_conf.save_freq == 1)
            ):
                checkpoint_names.append(f"checkpoint_{int(epoch)}")

        checkpoint_content = {
            "epoch": epoch,
            "steps": self.steps,
            "time_elapsed": self.time_elapsed_meter.val,
            "optimizer": [optim.optimizer.state_dict() for optim in self.optims],
        }

        if len(self.optims) == 1:
            checkpoint_content["optimizer"] = checkpoint_content["optimizer"][0]
        if self.optim_conf.amp.enabled:
            checkpoint_content["scaler"] = self.scaler.state_dict()

        # Save EMA state separately so it can be fully restored on resume
        if self.optim_conf.ema.enabled and self.ema_model is not None:
            checkpoint_content["ema_state"] = self.ema_model.state_dict()

        saver = DDPCheckpointSaver(
            checkpoint_folder,
            checkpoint_names=checkpoint_names,
            rank=self.distributed_rank,
            epoch=epoch,
        )

        # Always save the training model (unwrapped from DDP), not the EMA model
        model = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
        saver.save_checkpoint(model=model, **checkpoint_content)

    def _get_scalar_log_keys(self, phase: str) -> List[str]:
        """Retrieves keys for scalar values to be logged for a given phase."""
        if self.logging_conf.scalar_keys_to_log:
            return self.logging_conf.scalar_keys_to_log[phase].keys_to_log
        return []

    def _amp_dtype(self):
        """Resolve the configured AMP dtype."""
        amp_type = self.optim_conf.amp.amp_dtype
        assert amp_type in ["bfloat16", "float16"], f"Invalid Amp type: {amp_type}"
        return torch.bfloat16 if amp_type == "bfloat16" else torch.float16

    def run(self):
        """Main entry point to start the training or validation process."""
        assert self.mode in ["train", "val"], f"Invalid mode: {self.mode}"
        if self.mode == "train":
            self.run_train()
            # Run a full validation after all training is done
            self.run_val(limit_batches=None)
            self.run_ood_val(limit_batches=None)
        elif self.mode == "val":
            self.run_val(limit_batches=None)
            self.run_ood_val(limit_batches=None)
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

    def run_train(self):
        """Runs the main training loop over all epochs."""
        self._prime_viz_caches()
        while self.epoch < self.max_epochs:
            set_seeds(self.seed_value + self.epoch * 100, self.max_epochs, self.distributed_rank)

            dataloader = self.train_dataset.get_loader(epoch=int(self.epoch))
            self.train_epoch(dataloader)

            # Save checkpoint after each training epoch
            self.save_checkpoint(self.epoch)

            # Clean up memory
            del dataloader
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Run validation at the specified frequency. Skips validation after
            # the last training epoch, as it can be run separately.
            if self.epoch < self.max_epochs - 1:
                is_slow_val_epoch = (
                    self.slow_val_epoch_freq is not None
                    and self.epoch > 0
                    and self.epoch % self.slow_val_epoch_freq == 0
                )
                is_fast_val_epoch = self.epoch % self.val_epoch_freq == 0

                if is_slow_val_epoch:
                    self.run_val(limit_batches=None)
                elif is_fast_val_epoch:
                    self.run_val()

                # OOD validation at its own frequency
                if (
                    self.ood_val_epoch_freq is not None
                    and self.ood_val_dataset is not None
                    and self.epoch % self.ood_val_epoch_freq == 0
                ):
                    self.run_ood_val()

            self.epoch += 1

        self.epoch -= 1

    def run_val(self, limit_batches: Optional[int] = -1):
        """Runs a validation epoch if a validation dataset is available.

        Args:
            limit_batches: Max batches to evaluate. ``-1`` (default) uses
                ``self.limit_val_batches`` (fast val). ``None`` means no limit
                (slow / full val).
        """
        if not self.val_dataset:
            logging.info("No validation dataset configured. Skipping validation.")
            return

        if limit_batches == -1:
            limit_batches = self.limit_val_batches

        is_full = limit_batches is None
        tag = "val_full" if is_full else "val"
        logging.info(f"Running {tag} (limit_batches={limit_batches})")

        set_seeds(self.seed_value, self.max_epochs, self.distributed_rank)
        dataloader = self.val_dataset.get_loader(epoch=int(self.epoch))
        self.val_epoch(dataloader, limit_batches=limit_batches, phase=tag)

        del dataloader
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    def run_ood_val(self, limit_batches: Optional[int] = -1):
        """Runs an OOD validation epoch if an OOD dataset is available.

        Mirrors :meth:`run_val` but uses the ``ood_val`` dataset and logs under
        the ``ood_val`` / ``ood_val_full`` phase tags.
        """
        if not self.ood_val_dataset:
            return

        if limit_batches == -1:
            limit_batches = self.limit_val_batches

        is_full = limit_batches is None
        tag = "ood_val_full" if is_full else "ood_val"
        logging.info(f"Running {tag} (limit_batches={limit_batches})")

        set_seeds(self.seed_value, self.max_epochs, self.distributed_rank)
        dataloader = self.ood_val_dataset.get_loader(epoch=int(self.epoch))
        self.val_epoch(dataloader, limit_batches=limit_batches, phase=tag)

        del dataloader
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    @torch.no_grad()
    def val_epoch(
        self,
        val_loader,
        limit_batches: Optional[int] = None,
        phase: str = "val",
    ):
        """Run one validation epoch (loss + Chamfer distance).

        Args:
            val_loader: Validation dataloader.
            limit_batches: Max number of batches to evaluate. ``None`` means the
                entire loader.
            phase: Logging tag (``"val"``/``"val_full"``/``"ood_val"``/``"ood_val_full"``).
        """
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        chamfer_meter = AverageMeter("Chamfer", self.device, ":.6f")
        data_times = []

        # Use "val" loss keys for all validation-type phases (val, val_full,
        # ood_val, ood_val_full) since scalar_keys_to_log only defines a "val"
        # entry. The step counter and per-step scalar logging go through
        # loss_phase="val"; the aggregate Chamfer metric is logged under the
        # actual *phase* tag so OOD metrics are distinguishable.
        loss_phase = "val"
        loss_names = self._get_scalar_log_keys(loss_phase)
        loss_names = [f"Loss/{loss_phase}_{name}" for name in loss_names]
        loss_meters = {name: AverageMeter(name, self.device, ":.4f") for name in loss_names}

        progress = ProgressMeter(
            num_batches=len(val_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                chamfer_meter,
                self.time_elapsed_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix=f"{phase} Epoch: [{self.epoch}]",
        )

        self.model.eval()
        end = time.time()

        iters_per_epoch = len(val_loader)
        max_batches = iters_per_epoch if limit_batches is None else limit_batches

        # ---- Padding-aware scene counting --------------------------------
        # DistributedSampler pads the dataset so every rank gets the same number
        # of samples. When aggregating Chamfer across ranks we must exclude the
        # padded (duplicated) scenes to avoid double-counting.
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        dataset_size = len(val_loader.dataset)
        real_samples_this_rank = dataset_size // world_size + (
            1 if self.rank < dataset_size % world_size else 0
        )
        # When limit_batches is set we never reach the tail, so padding is
        # irrelevant — treat every scene as real.
        if limit_batches is not None:
            real_samples_this_rank = float("inf")

        total_scenes_seen = 0  # running count of scenes iterated on this rank

        amp_type = self._amp_dtype()

        for data_iter, batch in enumerate(val_loader):
            if data_iter >= max_batches:
                break

            data_time.update(time.time() - end)
            data_times.append(data_time.val)

            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            # Fixed-seed generator per batch index so validation noise is
            # reproducible across epochs (both val loss and Chamfer distance).
            val_generator = torch.Generator(device=self.device)
            val_generator.manual_seed(42 + data_iter)

            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=self.optim_conf.amp.enabled, dtype=amp_type):
                    self._step(batch, self.model, loss_phase, loss_meters, generator=val_generator, log_phase=phase)

            # Chamfer distance — distributed across all ranks
            if "target_3d_points" in batch:
                gt_key = "chamfer_target_3d_points" if "chamfer_target_3d_points" in batch else "target_3d_points"
                batch_size = batch[gt_key].shape[0]
                real_in_batch = int(min(batch_size, max(0, real_samples_this_rank - total_scenes_seen)))
                total_scenes_seen += batch_size

                if real_in_batch > 0:
                    chamfer_generator = torch.Generator(device=self.device)
                    chamfer_generator.manual_seed(42 + data_iter)
                    with torch.amp.autocast('cuda', enabled=self.optim_conf.amp.enabled, dtype=amp_type):
                        chamfer_sum, chamfer_n = self._compute_val_chamfer(
                            batch, generator=chamfer_generator, max_scenes=real_in_batch,
                        )
                    chamfer_meter.update(chamfer_sum.item() / chamfer_n, n=chamfer_n)

            batch_time.update(time.time() - end)
            end = time.time()

            self.time_elapsed_meter.update(time.time() - self.start_time + self.ckpt_time_elapsed)

            if torch.cuda.is_available():
                mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

        # ---- Aggregate Chamfer metric across ranks -----------------------
        def _all_reduce_meter(meter: AverageMeter) -> None:
            """Sum .sum and .count across ranks, recompute .avg."""
            stats = torch.tensor([meter.sum, meter.count], dtype=torch.float64, device=self.device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            meter.sum = stats[0].item()
            meter.count = int(stats[1].item())
            meter.avg = meter.sum / meter.count if meter.count > 0 else 0.0

        if dist.is_initialized() and world_size > 1:
            _all_reduce_meter(chamfer_meter)

        if self.rank == 0 and chamfer_meter.count > 0:
            self.writer.log(
                f"Metrics/{phase}/chamfer_distance_epoch",
                chamfer_meter.avg,
                self.epoch,
                phase=phase,
            )
            self.writer.commit(self.epoch, phase=phase)
            logging.info(f"{phase} Epoch [{self.epoch}] - Avg Chamfer Distance: {chamfer_meter.avg:.6f}")

        # Point-cloud viz for val / ood_val. The main val may run often (fast
        # val), so it is additionally gated by ``viz.val_epoch_freq``; ood_val
        # already runs only on ``ood_val_epoch_freq`` epochs, so it is visualized
        # whenever it runs. ``_log_point_clouds`` no-ops if the phase was not
        # primed (e.g. ood disabled).
        viz = self.logging_conf.viz
        if viz.enabled and self.rank == 0:
            if phase == "val" and viz.val_epoch_freq > 0 and self.epoch % viz.val_epoch_freq == 0:
                self._log_point_clouds("val", int(viz.num_val_scenes), self.epoch)
            elif phase == "ood_val":
                self._log_point_clouds("ood_val", int(viz.num_val_scenes), self.epoch)

        return True

    def train_epoch(self, train_loader):
        """Run one training epoch."""
        batch_time = AverageMeter("Batch Time", self.device, ":.4f")
        data_time = AverageMeter("Data Time", self.device, ":.4f")
        mem = AverageMeter("Mem (GB)", self.device, ":.4f")
        data_times = []
        phase = "train"

        loss_names = self._get_scalar_log_keys(phase)
        loss_names = [f"Loss/{phase}_{name}" for name in loss_names]
        loss_meters = {name: AverageMeter(name, self.device, ":.4f") for name in loss_names}
        # Ensure the objective meter always exists (used unconditionally in
        # _run_steps_on_batch_chunks).
        obj_key = f"Loss/{phase}_objective"
        if obj_key not in loss_meters:
            loss_meters[obj_key] = AverageMeter(obj_key, self.device, ":.4f")

        for config in self.gradient_clipper.configs:
            param_names = ",".join(config["module_names"])
            loss_meters[f"Grad/{param_names}"] = AverageMeter(f"Grad/{param_names}", self.device, ":.4f")

        # Per-batch view count N (constant within a batch: the multiview sampler
        # draws one N per iteration). Logged every step so loss can be correlated
        # with N offline -- see analyze_train_stability.py.
        n_views_meter = AverageMeter("Data/N_views", self.device, ":.1f")

        progress = ProgressMeter(
            num_batches=len(train_loader),
            meters=[
                batch_time,
                data_time,
                mem,
                self.time_elapsed_meter,
                n_views_meter,
                *loss_meters.values(),
            ],
            real_meters={},
            prefix="Train Epoch: [{}]".format(self.epoch),
        )

        self.model.train()
        end = time.time()

        iters_per_epoch = len(train_loader)
        limit_train_batches = iters_per_epoch if self.limit_train_batches is None else min(self.limit_train_batches, iters_per_epoch)

        if self.gradient_clipper is not None:
            self.gradient_clipper.setup_clipping(self.model)

        for data_iter, batch in enumerate(train_loader):
            if data_iter >= limit_train_batches:
                break

            data_time.update(time.time() - end)
            data_times.append(data_time.val)

            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            # --- Per-batch view count N (see n_views_meter) ---
            _fn = batch.get("frame_num", None)
            if _fn is not None:
                n_views = int(_fn.reshape(-1)[0]) if torch.is_tensor(_fn) else int(_fn)
            elif "ids" in batch and torch.is_tensor(batch["ids"]):
                n_views = int(batch["ids"].shape[1])
            else:
                n_views = -1
            n_views_meter.update(n_views)
            if self.rank == 0 and self.steps[phase] % self.logging_conf.log_freq == 0:
                self.writer.log("Data/N_views", float(n_views), self.steps[phase], phase=phase)

            accum_steps = self.accum_steps
            if accum_steps == 1:
                chunked_batches = [batch]
            else:
                chunked_batches = chunk_batch_for_accum_steps(batch, accum_steps)

            self._run_steps_on_batch_chunks(chunked_batches, phase, loss_meters)

            viz = self.logging_conf.viz
            if (viz.enabled and self.rank == 0 and viz.train_iter_freq > 0
                    and self.steps[phase] % viz.train_iter_freq == 0):
                self._log_point_clouds("train", int(viz.num_train_scenes), self.steps[phase])
                self.model.train()

            assert data_iter < limit_train_batches
            exact_epoch = self.epoch + float(data_iter) / limit_train_batches
            self.where = float(exact_epoch) / self.max_epochs

            assert self.where <= 1 + self.EPSILON
            if self.where < 1.0:
                for optim in self.optims:
                    optim.step_schedulers(self.where)
            else:
                logging.warning(f"Skipping scheduler update since the training is at the end, i.e, {self.where} of [0,1].")

            if self.steps[phase] % self.logging_conf.log_freq == 0:
                for i, optim in enumerate(self.optims):
                    for j, param_group in enumerate(optim.optimizer.param_groups):
                        for option in optim.schedulers[j]:
                            optim_prefix = (
                                f"{i}_" if len(self.optims) > 1 else ("" + f"{j}_" if len(optim.optimizer.param_groups) > 1 else "")
                            )
                            self.writer.log(
                                os.path.join("Optim", f"{optim_prefix}", option),
                                param_group[option],
                                self.steps[phase],
                                phase=phase,
                            )
                self.writer.log(
                    os.path.join("Optim", "where"),
                    self.where,
                    self.steps[phase],
                    phase=phase,
                )

            if self.gradient_clipper is not None:
                for optim in self.optims:
                    self.scaler.unscale_(optim.optimizer)

                grad_norm_dict = self.gradient_clipper(model=self.model)

                for key, grad_norm in grad_norm_dict.items():
                    loss_meters[f"Grad/{key}"].update(grad_norm)

            for optim in self.optims:
                self.scaler.step(optim.optimizer)
            self.scaler.update()

            if self.optim_conf.ema.enabled and self.ema_model is not None:
                self.ema_model.update()

            batch_time.update(time.time() - end)
            end = time.time()
            self.time_elapsed_meter.update(time.time() - self.start_time + self.ckpt_time_elapsed)
            mem.update(torch.cuda.max_memory_allocated() // 1e9)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

        return True

    def _run_steps_on_batch_chunks(
        self,
        chunked_batches: List[Any],
        phase: str,
        loss_meters: Dict[str, AverageMeter],
    ):
        """Run forward / backward once per chunk, accumulating gradients."""
        for optim in self.optims:
            optim.zero_grad(set_to_none=True)

        accum_steps = len(chunked_batches)
        amp_type = self._amp_dtype()

        for i, chunked_batch in enumerate(chunked_batches):
            ddp_context = self.model.no_sync() if i < accum_steps - 1 else contextlib.nullcontext()

            with ddp_context:
                with torch.amp.autocast('cuda', enabled=self.optim_conf.amp.enabled, dtype=amp_type):
                    loss_dict = self._step(chunked_batch, self.model, phase, loss_meters)

                loss = loss_dict["objective"]
                loss_key = f"Loss/{phase}_objective"
                batch_size = chunked_batch["extrinsics"].shape[0]

                if not loss.isfinite():
                    logging.error(f"Loss is {loss}, attempting to stop training")
                    return

                loss /= accum_steps
                self.scaler.scale(loss).backward()
                loss_meters[loss_key].update(loss, batch_size)

    def _preprocess_batch_inplace(self, batch, model: nn.Module):
        """Preprocess the batch on the original dict (outside DDP, which copies
        the input dict and would discard in-place modifications). Needed when
        code outside the model's forward needs the preprocessed batch (e.g.
        Chamfer evaluation).

        Only the cached (preprocessed) path is supported.
        """
        if batch.get("has_been_preprocessed", False):
            return
        unwrapped = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        with torch.no_grad():
            if "cached_aggregated_tokens" in batch:
                unwrapped.preprocess_from_cached(batch)
            else:
                raise NotImplementedError(
                    "Only the cached (preprocessed) data path is supported; run "
                    "scripts/preprocess.py to cache VGGT tokens."
                )

    def _step(
        self,
        batch,
        model: nn.Module,
        phase: str,
        loss_meters: dict,
        generator: torch.Generator = None,
        log_phase: Optional[str] = None,
    ):
        """Perform a single forward pass, compute loss, and log scalars.

        Args:
            phase: Phase key used for loss-key lookup and step counting.
            log_phase: Phase tag used for logging. Defaults to *phase* when ``None``.

        Returns:
            A dictionary containing the computed losses.
        """
        if log_phase is None:
            log_phase = phase

        # Forward pass
        y_hat = model(batch, generator=generator)

        # Loss computation
        loss_dict = self.loss(y_hat, batch, step=self.steps[phase])

        # Combine all data for logging
        log_data = {**y_hat, **loss_dict, **batch}

        if log_phase not in self.steps:
            self.steps[log_phase] = 0

        self._update_and_log_scalars(log_data, phase, self.steps[log_phase], loss_meters)

        self.writer.commit(self.steps[log_phase], phase=log_phase)
        self.steps[log_phase] += 1
        return loss_dict

    def _update_and_log_scalars(self, data: Mapping, phase: str, step: int, loss_meters: dict):
        """Updates average meters and logs scalar values."""
        keys_to_log = self._get_scalar_log_keys(phase)
        batch_size = data["extrinsics"].shape[0]

        for key in keys_to_log:
            if key in data:
                value = data[key]
                loss_meters[f"Loss/{phase}_{key}"].update(value, batch_size)
                if step % self.logging_conf.log_freq == 0 and self.rank == 0:
                    self.writer.log(f"Values/{phase}/{key}", value, step, phase=phase)

    @torch.no_grad()
    def _compute_val_chamfer(
        self,
        batch: dict,
        generator: torch.Generator = None,
        max_scenes: Optional[int] = None,
    ) -> tuple:
        """Run batched inference and return ``(chamfer_sum, num_scenes)``.

        Returns the *sum* (not mean) of per-scene Chamfer distances together with
        the number of scenes so callers can aggregate across ranks before averaging.
        """
        self._preprocess_batch_inplace(batch, self.model)
        if not batch.get("has_been_preprocessed", False):
            raise ValueError("Batch has not been preprocessed, please run the preprocess_batch function first")

        if self.optim_conf.ema.enabled and self.ema_model is not None:
            model, _orig_snet = self._get_ema_model()
        else:
            model = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
            _orig_snet = None

        gt_key = "chamfer_target_3d_points" if "chamfer_target_3d_points" in batch else "target_3d_points"
        batch_size = batch[gt_key].shape[0]
        if max_scenes is not None:
            batch_size = min(batch_size, max_scenes)

        num_steps = self.logging_conf.chamfer_eval["num_steps"]
        num_points_per_batch = self.logging_conf.chamfer_eval["num_points_per_batch"]
        guidance_scale = self.logging_conf.chamfer_eval.get("guidance_scale", 0.0)

        chamfer_values = []
        for scene_idx in range(batch_size):
            gt_points = batch[gt_key][scene_idx]
            num_chamfer_points = gt_points.shape[0]

            inference_kwargs = dict(
                num_steps=num_steps,
                num_query_points=num_chamfer_points,
                num_points_per_batch=num_points_per_batch,
                cull_radius=batch["cull_radius"][scene_idx] if "cull_radius" in batch else None,
                guidance_scale=guidance_scale,
                generator=generator,
                aggregated_tokens_list=[
                    t[scene_idx:scene_idx + 1] if t is not None else None
                    for t in batch["aggregated_tokens_list"]
                ],
                patch_start_idx=batch["patch_start_idx"],
            )
            vggt_wp = batch.get("vggt_world_points")
            inference_kwargs["world_points"] = vggt_wp[scene_idx:scene_idx + 1] if vggt_wp is not None else None

            surface_results = model.batched_inference(**inference_kwargs)
            if model.estimate_normals:
                pred_points, _ = surface_results
            else:
                pred_points = surface_results

            cd = compute_chamfer_distance(pred_points.float(), gt_points.float(), center_normalize_points2=True)
            chamfer_values.append(cd)

        self._restore_surface_net(model, _orig_snet)

        chamfer_sum = torch.stack(chamfer_values).sum() if chamfer_values else torch.tensor(0.0, device=self.device)
        return chamfer_sum, batch_size

    # ------------------------------------------------------------------
    # Periodic point-cloud visualization (wandb, rank 0, plain inference).
    # Deliberately kept to model.batched_inference so the Surflo rasterizer is
    # never imported -- see configs/logging/default.yaml::viz.
    # ------------------------------------------------------------------
    def _prime_viz_caches(self) -> None:
        """Build the fixed-scene viz caches ONCE, before the epoch loop.

        Each cache is the first batch of a seeded (epoch=0) loader, preprocessed
        and snapshotted to CPU. Building here -- not lazily mid-epoch -- is
        essential: ``get_loader`` mutates the shared sampler via ``set_epoch``,
        which would corrupt an in-progress training epoch. Rank 0 only. On any
        failure viz stays off rather than risking training.
        """
        viz = self.logging_conf.viz
        if not viz.enabled or self.rank != 0:
            return
        self._viz_cache = {}
        self._viz_gt_logged = set()
        try:
            for phase, dataset, n in [
                ("train", self.train_dataset, int(viz.num_train_scenes)),
                ("val", self.val_dataset, int(viz.num_val_scenes)),
                ("ood_val", self.ood_val_dataset, int(viz.num_val_scenes)),
            ]:
                if n <= 0 or dataset is None:
                    continue
                # num_workers=0: a single batch is read; the default pool would
                # fork many workers and duplicate memory for no benefit.
                loader = dataset.get_loader(epoch=0, num_workers=0)
                batch = next(iter(loader))
                batch = copy_data_to_device(batch, self.device, non_blocking=True)
                self._preprocess_batch_inplace(batch, self.model)
                self._viz_cache[phase] = copy_data_to_device(batch, "cpu")
                del loader, batch
        except Exception:
            logging.exception("Failed to prime viz caches; point-cloud viz disabled.")
            self._viz_cache = {}

    @torch.no_grad()
    def _log_point_clouds(self, phase: str, num_scenes: int, step: int) -> None:
        """Run plain inference on the fixed ``phase`` scenes and log clouds to wandb.

        Reads the primed cache only; no-ops if unprimed so it never builds a
        loader mid-epoch (see :meth:`_prime_viz_caches`).
        """
        if self.rank != 0 or num_scenes <= 0:
            return
        if not getattr(self, "_viz_cache", None) or phase not in self._viz_cache:
            return
        viz = self.logging_conf.viz

        batch = copy_data_to_device(self._viz_cache[phase], self.device, non_blocking=True)

        if self.optim_conf.ema.enabled and self.ema_model is not None:
            model, _orig_snet = self._get_ema_model()
        else:
            model = self.model.module if isinstance(self.model, torch.nn.parallel.DistributedDataParallel) else self.model
            _orig_snet = None
        model.eval()

        gt_key = "chamfer_target_3d_points" if "chamfer_target_3d_points" in batch else "target_3d_points"
        n = min(int(num_scenes), int(batch[gt_key].shape[0]))
        log_gt = phase not in self._viz_gt_logged
        # Fixed source noise so successive logs differ only by the model.
        generator = torch.Generator(device=self.device).manual_seed(int(viz.seed))

        for scene_idx in range(n):
            inference_kwargs = dict(
                num_steps=int(viz.num_steps),
                num_query_points=int(viz.num_query_points),
                num_points_per_batch=int(viz.num_query_points),
                cull_radius=batch["cull_radius"][scene_idx] if "cull_radius" in batch else None,
                guidance_scale=0.0,
                generator=generator,
                aggregated_tokens_list=[
                    t[scene_idx:scene_idx + 1] if t is not None else None
                    for t in batch["aggregated_tokens_list"]
                ],
                patch_start_idx=batch["patch_start_idx"],
            )
            vggt_wp = batch.get("vggt_world_points")
            inference_kwargs["world_points"] = vggt_wp[scene_idx:scene_idx + 1] if vggt_wp is not None else None

            # Same autocast context as the val-Chamfer inference: the cached
            # tokens are half precision, so without it the fp32 Linear weights
            # raise a Half/Float dtype mismatch.
            with torch.amp.autocast('cuda', enabled=self.optim_conf.amp.enabled, dtype=self._amp_dtype()):
                surface_results = model.batched_inference(**inference_kwargs)
            if model.estimate_normals:
                pred_points, pred_normals = surface_results
                colors = (1.0 - pred_normals.float()) / 2.0
            else:
                pred_points = surface_results
                colors = _normalize_points_to_unit_cube(pred_points.float())

            self.writer.log_point_cloud(
                f"Viz/{phase}/scene_{scene_idx}/pred", pred_points, colors, step, phase=phase,
            )
            if log_gt:
                gt_points = batch[gt_key][scene_idx].float()
                self.writer.log_point_cloud(
                    f"Viz/{phase}/scene_{scene_idx}/gt", gt_points,
                    _normalize_points_to_unit_cube(gt_points), step, phase=phase,
                )

        if log_gt:
            self._viz_gt_logged.add(phase)
        self._restore_surface_net(model, _orig_snet)
        self.model.train()
        self.writer.commit(step, phase=phase)


def _normalize_points_to_unit_cube(points: torch.Tensor) -> torch.Tensor:
    """Per-cloud min-max normalization of XYZ -> [0, 1] RGB (fallback colors)."""
    lo = points.min(dim=0, keepdim=True).values
    hi = points.max(dim=0, keepdim=True).values
    return (points - lo) / (hi - lo + 1e-8)


def chunk_batch_for_accum_steps(batch: Mapping, accum_steps: int) -> List[Mapping]:
    """Splits a batch into smaller chunks for gradient accumulation."""
    if accum_steps == 1:
        return [batch]
    return [get_chunk_from_data(batch, i, accum_steps) for i in range(accum_steps)]


def is_sequence_of_primitives(data: Any) -> bool:
    """Checks if data is a sequence of primitive types (str, int, float, bool)."""
    return isinstance(data, Sequence) and not isinstance(data, str) and len(data) > 0 and isinstance(data[0], (str, int, float, bool))


def get_chunk_from_data(data: Any, chunk_id: int, num_chunks: int) -> Any:
    """Recursively splits tensors and sequences within a data structure into chunks.

    Args:
        data: The data structure to split (e.g., a dictionary of tensors).
        chunk_id: The index of the chunk to retrieve.
        num_chunks: The total number of chunks to split the data into.

    Returns:
        A chunk of the original data structure.
    """
    if isinstance(data, torch.Tensor) or is_sequence_of_primitives(data):
        start = (len(data) // num_chunks) * chunk_id
        end = (len(data) // num_chunks) * (chunk_id + 1)
        return data[start:end]
    elif isinstance(data, Mapping):
        return {key: get_chunk_from_data(value, chunk_id, num_chunks) for key, value in data.items()}
    elif isinstance(data, str):
        return data
    elif isinstance(data, Sequence):
        return [get_chunk_from_data(value, chunk_id, num_chunks) for value in data]
    else:
        return data
