"""Adversarial training loop over the fusion relay and its two critics."""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.checkpoint import checkpoint

from ..config import ModelConfig, TrainingConfig
from ..datasets.nowcast import NowcastBatch
from ..evaluation.metrics import MetricAccumulator, higher_is_better
from ..models.dgmr.discriminators import DGMRDiscriminators
from ..models.fusion import IndraFusion
from .losses import (
    LossWeights,
    discriminator_objective,
    generator_objective,
    grid_cell_weights,
)
from .replay_buffer import ExperienceReplayBuffer

logger = logging.getLogger(__name__)

#: Consecutive fully-rejected cycles before ``fit`` gives up. A skipped cycle
#: does not advance the step counter, so without a limit an archive gap that
#: covers the whole split would spin indefinitely.
_MAX_CONSECUTIVE_SKIPS = 100


@dataclass
class TrainState:
    """Everything a resumed run needs that is not a tensor."""

    step: int = 0
    best_metric: float | None = None
    best_step: int | None = None
    #: Batches that arrived as ``None`` because every sample was rejected.
    skipped_batches: int = 0
    #: Individual samples dropped from otherwise usable batches. A rising
    #: count is a degrading archive, and it is only visible if it is counted.
    dropped_samples: int = 0
    replay_admitted: int = 0


class Trainer:
    """The adversarial loop over the fusion relay and its two critics."""

    def __init__(
        self,
        model: IndraFusion,
        discriminators: DGMRDiscriminators,
        config: TrainingConfig,
        train_loader: Any,
        validation_loader: Any | None = None,
        replay_buffer: ExperienceReplayBuffer | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        self.config = config
        self.model_config: ModelConfig = model.config
        self.device = torch.device(device or self._resolve_device(config))

        self.model = model.to(self.device)
        self.discriminators = discriminators.to(self.device)
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.replay = replay_buffer
        self.state = TrainState()

        self._configure_determinism()

        self.weights = LossWeights(
            adversarial_spatial=config.losses.adversarial_spatial,
            adversarial_temporal=config.losses.adversarial_temporal,
            grid_cell=config.losses.grid_cell,
            auxiliary=config.losses.auxiliary,
            grid_cell_max_weight=config.losses.grid_cell_max_weight,
        )

        # The heavy threshold comes from the replay policy rather than the
        # inference configuration, which IndraConfig has already required to
        # agree with it. Reading it here keeps the trainer from depending on
        # the serving configuration to know what "heavy" means.
        self.heavy_threshold = config.replay.policy.heavy_threshold_mm_h

        # Frozen from step 0 whenever the schedule owns the decision; a null
        # unfreeze step leaves model.fusion.freeze_backbone standing, which
        # IndraFusion has already applied.
        if config.schedule.backbone_unfreeze_step is not None:
            self.model.set_backbone_frozen(True)

        self._verify_replay_split()
        self.optimizer_g, self.optimizer_d = self._build_optimizers()

        # fp16 only. See the module docstring: under bf16 a scaler is a no-op
        # that looks like an active safeguard.
        self.scaler = (
            torch.amp.GradScaler(self.device.type)
            if config.run.precision == "fp16"
            else None
        )

        self.checkpoint_dir = config.checkpointing.resolved_dir()
        self._saved: list[Path] = []

        logger.info(
            "trainer ready on %s | precision %s | %s | est. activations %s",
            self.device,
            config.run.precision,
            self.model.parameter_summary(),
            _human_bytes(self.estimate_activation_bytes()),
        )

    # ------------------------------------------------------------------ setup
    @staticmethod
    def _resolve_device(config: TrainingConfig) -> str:
        if config.run.device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning(
                "%s requested but unavailable; falling back to %s",
                config.run.device,
                config.run.fallback_device,
            )
            return config.run.fallback_device
        return config.run.device

    def _configure_determinism(self) -> None:
        torch.manual_seed(self.config.run.seed)
        if not self.config.run.deterministic:
            return
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Raises on an operation with no deterministic kernel, which is the
        # point: silently falling back to a non-deterministic one would make
        # the run irreproducible while still claiming otherwise.
        torch.use_deterministic_algorithms(True, warn_only=False)

    def _verify_replay_split(self) -> None:
        """Refuse to admit windows from anything but the training split."""
        if self.replay is None or not self.config.replay.enabled:
            return
        if not self.config.replay.admission_only_from_training_split:
            logger.warning(
                "replay admission is not restricted to the training split; "
                "held-out windows can enter the reservoir"
            )
            return

        split = getattr(getattr(self.train_loader, "dataset", None), "split", None)
        if split is not None and split != "train":
            raise ValueError(
                f"the training loader carries the {split!r} split, but replay "
                "admission is restricted to 'train'. Feeding the buffer from "
                "this loader would replay held-out windows into training."
            )

    def _build_optimizers(self) -> tuple[torch.optim.Optimizer, torch.optim.Optimizer]:
        """Two optimisers, and two parameter groups inside the generator's."""
        gen = self.config.generator
        base_lr = gen.optimizer.lr

        head_params = list(self.model.adapter.parameters()) + list(
            self.model.generator.parameters()
        )
        backbone_params = list(self.model.earthformer.parameters())

        groups = [
            {
                "name": "generator_heads",
                "params": head_params,
                "lr": base_lr,
                "base_lr": base_lr,
            },
            {
                "name": "backbone",
                "params": backbone_params,
                "lr": base_lr * gen.backbone_lr_multiplier,
                "base_lr": base_lr * gen.backbone_lr_multiplier,
            },
        ]
        optimizer_g = _make_optimizer(gen.optimizer, groups)

        disc = self.config.discriminator
        optimizer_d = _make_optimizer(
            disc.optimizer,
            [
                {
                    "name": "critics",
                    "params": list(self.discriminators.parameters()),
                    "lr": disc.optimizer.lr,
                    "base_lr": disc.optimizer.lr,
                }
            ],
        )
        return optimizer_g, optimizer_d

    def estimate_activation_bytes(self) -> int:
        """Order-of-magnitude activation footprint of one generator step."""
        adapter = self.model_config.adapter
        per_step = sum(
            dim * scale * scale
            for dim, scale in zip(
                adapter.output_dims, adapter.output_scales, strict=False
            )
        )
        # A ConvGRU cell plus its GBlock retain on the order of half a dozen
        # tensors of the level's size per step.
        tensors_per_step = 6
        steps = self.model_config.output.lead_frames
        width = 2 if self.config.run.precision in ("bf16", "fp16") else 4

        members = self.config.generator.samples_per_step
        # Checkpointing collapses the member axis: only one member's
        # activations are live at a time during recomputation.
        live_members = 1 if self.config.run.gradient_checkpointing else members

        return (
            per_step
            * tensors_per_step
            * steps
            * width
            * live_members
            * self.config.data.batch_size
        )

    # ----------------------------------------------------------------- pieces
    def _prepare(
        self, batch: NowcastBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Move a batch to the device and sanitise the target."""
        batch = batch.to(self.device, non_blocking=True)
        x = batch.x
        mask = batch.target_validity
        target = torch.nan_to_num(batch.target_mm_h, nan=0.0)
        return x, target, mask

    def _conditioning(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Backbone and bridge, once per step regardless of member count."""
        latent = self.model.earthformer(x)
        return latent, self.model.adapter(latent)

    def _run_sampler(
        self, noise: torch.Tensor, *conditioning: torch.Tensor
    ) -> torch.Tensor:
        # The pyramid is unpacked into positional arguments so the checkpoint
        # boundary sees plain tensors and can track each one, rather than a
        # list it would have to introspect.
        return self.model.generator(list(conditioning), noise)

    def _sample_members(
        self,
        conditioning: list[torch.Tensor],
        members: int,
        seed: int,
        batch_size: int,
    ) -> torch.Tensor:
        """Draw ``members`` realisations, shaped ``(N, B, T, C, H, W)``."""
        noise = self.model.draw_noise(
            members=members,
            seed=seed,
            batch=batch_size,
            device=self.device,
            dtype=conditioning[0].dtype,
        )
        frames = []
        for index in range(members):
            if self.config.run.gradient_checkpointing:
                frames.append(
                    checkpoint(
                        self._run_sampler,
                        noise[index],
                        *conditioning,
                        use_reentrant=False,
                    )
                )
            else:
                frames.append(self.model.generator(conditioning, noise[index]))
        return torch.stack(frames)

    def _critic_scores(self, sequence: torch.Tensor) -> dict[str, torch.Tensor]:
        """Score a 12-frame sequence with both critics."""
        return self.discriminators(sequence, None, None)

    # ------------------------------------------------------------------ steps
    def _critic_step(self, batch: NowcastBatch) -> dict[str, float]:
        # The mask is unused here: the critics score whole sequences rather
        # than reducing over cells, so there is nothing for it to exclude.
        x, target, _ = self._prepare(batch)

        self.optimizer_d.zero_grad(set_to_none=True)
        with self.model.autocast(self.device.type):
            with torch.no_grad():
                # One member is all a critic step needs, and no generator
                # gradient flows from here.
                _, conditioning = self._conditioning(x)
                noise = self.model.draw_noise(
                    members=1,
                    seed=self.config.run.seed + self.state.step,
                    batch=x.shape[0],
                    device=self.device,
                    dtype=conditioning[0].dtype,
                )
                fake = self.model.generator(conditioning, noise[0])

            real_scores = self._critic_scores(target)
            fake_scores = self._critic_scores(fake.detach())
            breakdown = discriminator_objective(real_scores, fake_scores, self.weights)

        self._backward(breakdown.total, self.optimizer_d)
        return breakdown.detached()

    def _generator_step(self, batch: NowcastBatch) -> dict[str, float]:
        x, target, mask = self._prepare(batch)

        self.optimizer_g.zero_grad(set_to_none=True)
        with self.model.autocast(self.device.type):
            latent, conditioning = self._conditioning(x)
            samples = self._sample_members(
                conditioning,
                members=self.config.generator.samples_per_step,
                seed=self.config.run.seed + self.state.step,
                batch_size=x.shape[0],
            )

            # One member through the critics; all of them into R.
            fake_scores = self._critic_scores(samples[0])

            auxiliary = None
            if self.weights.auxiliary > 0:
                auxiliary = self.model.earthformer.auxiliary_precipitation(
                    latent, x.shape[-2], x.shape[-1]
                )

            breakdown = generator_objective(
                fake_scores,
                samples,
                target,
                weights=self.weights,
                # w(y) on the observation in mm h-1, which is exactly what the
                # target carries: it was never normalised, so there is nothing
                # to invert here and no chance of weighting a z-score.
                grid_weights=grid_cell_weights(
                    target, self.weights.grid_cell_max_weight
                ),
                mask=mask,
                auxiliary=auxiliary,
            )

        self._backward(breakdown.total, self.optimizer_g)
        return breakdown.detached()

    def _backward(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer) -> None:
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            if self.config.schedule.grad_clip_norm is not None:
                self.scaler.unscale_(optimizer)
                self._clip(optimizer)
            self.scaler.step(optimizer)
            self.scaler.update()
            return

        loss.backward()
        if self.config.schedule.grad_clip_norm is not None:
            self._clip(optimizer)
        optimizer.step()

    def _clip(self, optimizer: torch.optim.Optimizer) -> None:
        """Clip the global norm across every group, not each group separately."""
        limit = self.config.schedule.grad_clip_norm
        params = [p for group in optimizer.param_groups for p in group["params"]]
        torch.nn.utils.clip_grad_norm_(params, limit)

    # -------------------------------------------------------------- schedules
    def _apply_lr(self, step: int) -> None:
        """Linear warmup into the configured schedule."""
        warmup = self.config.schedule.warmup_steps
        scale = min(1.0, (step + 1) / warmup) if warmup else 1.0
        for optimizer in (self.optimizer_g, self.optimizer_d):
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * scale

    def _maybe_unfreeze(self, step: int) -> None:
        target = self.config.schedule.backbone_unfreeze_step
        if target is None or step != target or not self.model.backbone_frozen:
            return
        self.model.set_backbone_frozen(False)
        logger.info(
            "step %d: Earthformer released at lr %.2e",
            step,
            self.optimizer_g.param_groups[1]["lr"],
        )

    # ----------------------------------------------------------------- replay
    def _admit(self, batch: NowcastBatch) -> None:
        """Offer a batch's samples to the replay buffer."""
        if self.replay is None or not self.config.replay.enabled:
            return

        # The training-split restriction is structural rather than a check
        # here: this is only ever reached from the training iterator, and
        # ``validate`` never calls it. ``__init__`` verifies that the loader
        # this trainer was handed is in fact the training split, so the 2020
        # season cannot re-enter training through replay.
        for sample in batch.samples:
            decision = self.replay.consider(
                sample.input_window,
                sample.target_mm_h,
                sample.target_validity,
            )
            if decision.admitted:
                self.state.replay_admitted += 1

    def _next_batch(
        self, batches: Iterator[NowcastBatch | None]
    ) -> NowcastBatch | None:
        """One usable batch, admitting it to the buffer on the way past."""
        batch = next(batches, None)
        if batch is None:
            self.state.skipped_batches += 1
            return None
        self.state.dropped_samples += batch.dropped
        self._admit(batch)
        return batch

    # ------------------------------------------------------------------- loop
    def train_step(
        self, batches: Iterator[NowcastBatch | None]
    ) -> dict[str, float] | None:
        """One 2:1 cycle: two critic updates, then one generator update."""
        self._apply_lr(self.state.step)
        self._maybe_unfreeze(self.state.step)

        self.model.train()
        self.discriminators.train()

        metrics: dict[str, float] = {}
        last: NowcastBatch | None = None

        for _ in range(self.config.discriminator.steps_per_generator_step):
            batch = self._next_batch(batches)
            if batch is None:
                return None
            metrics.update(self._critic_step(batch))
            last = batch

        if last is None:
            return None
        # The generator reuses the final critic batch; see the module
        # docstring on why a third load buys nothing.
        metrics.update(self._generator_step(last))
        self.state.step += 1
        return metrics

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """A pooled pass over the held-out season."""
        if self.validation_loader is None:
            return {}

        settings = self.config.validation
        accumulator = MetricAccumulator(
            threshold=self.heavy_threshold,
            threshold_name="heavy",
            metrics=list(settings.metrics),
        )

        self.model.eval()
        self.discriminators.eval()
        seen = 0
        rejected = 0
        for batch in self.validation_loader:
            if seen >= settings.max_batches:
                break
            if batch is None:
                rejected += 1
                continue
            x, target, mask = self._prepare(batch)
            forecast = self.model.predict_ensemble(
                x,
                seed=settings.seed,
                members=settings.ensemble_members,
                reductions=(),
                return_members=True,
            )
            if forecast.members is None:
                # return_members=True above, so this cannot happen. Checked
                # rather than asserted because the alternative is scoring an
                # empty accumulator and reporting the result as a metric.
                raise RuntimeError(
                    "predict_ensemble returned no members during validation"
                )
            accumulator.update(forecast.members, target, mask)
            seen += 1

        results = accumulator.compute()
        logger.info(
            "validation over %d batches (%d rejected): %s", seen, rejected, results
        )
        return results

    def fit(self) -> TrainState:
        """Run until ``max_steps``, validating and checkpointing on schedule."""
        batches = _cycle(self.train_loader)
        started = time.monotonic()
        consecutive_skips = 0

        while self.state.step < self.config.schedule.max_steps:
            metrics = self.train_step(batches)
            step = self.state.step

            if metrics is None:
                # A skipped cycle does not advance the step, so an archive
                # that rejects everything would spin here forever, burning I/O
                # and logging nothing conclusive. Fail loudly instead: at this
                # point the data, not the model, is the problem.
                consecutive_skips += 1
                if consecutive_skips >= _MAX_CONSECUTIVE_SKIPS:
                    raise RuntimeError(
                        f"{consecutive_skips} consecutive batches were entirely "
                        f"rejected at step {step}. The archive, not the model, "
                        "needs attention; training would otherwise loop without "
                        "advancing."
                    )
                continue
            consecutive_skips = 0

            if step % self.config.logging.every_steps == 0:
                self._log_step(step, metrics, started)

            if (
                self.replay is not None
                and step % self.config.logging.replay_stats_every_steps == 0
            ):
                logger.info("replay: %s", self.replay.stats())

            if step % self.config.validation.every_steps == 0:
                self._maybe_checkpoint(self.validate())

        logger.info("training finished: %s", asdict(self.state))
        return self.state

    def _log_step(self, step: int, metrics: dict[str, float], started: float) -> None:
        if self.config.logging.log_loss_components:
            # Every term separately. A collapsed critic, swamping
            # regularisation and a stalled generator are three different
            # failures with three different fixes, and one scalar
            # distinguishes none of them.
            terms = " ".join(f"{k}={v:.4f}" for k, v in sorted(metrics.items()))
        else:
            terms = f"g_total={metrics.get('g_total', float('nan')):.4f}"
        logger.info(
            "step %d | %s | lr %.2e | skipped %d | %.1f s",
            step,
            terms,
            self.optimizer_g.param_groups[0]["lr"],
            self.state.skipped_batches,
            time.monotonic() - started,
        )

    # ------------------------------------------------------------ checkpoints
    def _maybe_checkpoint(self, metrics: dict[str, float]) -> None:
        monitor = self.config.checkpointing.monitor
        value = metrics.get(monitor)

        if value is None or math.isnan(value):
            # Undefined is not a score. Treating NaN as an improvement would
            # ship the checkpoint from the quietest validation window in the
            # record.
            logger.warning(
                "%s is unavailable this pass (%s); not considering a best "
                "checkpoint",
                monitor,
                value,
            )
            self.save_checkpoint(is_best=False)
            return

        ascending = self.config.checkpointing.mode == "max"
        if ascending != higher_is_better(monitor):
            raise ValueError(
                f"checkpointing.mode is {self.config.checkpointing.mode!r} but "
                f"{monitor!r} improves in the other direction. One of the two "
                "is wrong, and left alone this would ship the worst "
                "checkpoint of the run."
            )

        best = self.state.best_metric
        improved = best is None or (value > best if ascending else value < best)
        if improved:
            self.state.best_metric = value
            self.state.best_step = self.state.step
            logger.info("step %d: new best %s = %.4f", self.state.step, monitor, value)
        self.save_checkpoint(is_best=improved)

    def save_checkpoint(self, is_best: bool = False) -> Path:
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "model": self.model.state_dict(),
            "discriminators": self.discriminators.state_dict(),
            "state": asdict(self.state),
            "config_name": self.config.run.name,
        }
        if self.config.checkpointing.save_optimizer_state:
            # Resuming an adversarial run without the Adam moments restarts
            # both players from a standstill against opponents that are not.
            payload["optimizer_g"] = self.optimizer_g.state_dict()
            payload["optimizer_d"] = self.optimizer_d.state_dict()

        path = self.checkpoint_dir / f"{self.config.run.name}_{self.state.step:08d}.pt"
        _atomic_save(payload, path)
        self._saved.append(path)

        if is_best:
            # A separate file, not a symlink: the best checkpoint must survive
            # _prune removing the step file it was written from.
            _atomic_save(
                payload, self.checkpoint_dir / f"{self.config.run.name}_best.pt"
            )

        self._prune()
        return path

    def _prune(self) -> None:
        """Keep the most recent ``keep_last``. The best is a separate file."""
        keep = self.config.checkpointing.keep_last
        while len(self._saved) > keep:
            stale = self._saved.pop(0)
            try:
                stale.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("could not remove %s: %s", stale, exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_optimizer(spec: Any, groups: list[dict[str, Any]]) -> torch.optim.Optimizer:
    kwargs = {"betas": tuple(spec.betas), "eps": spec.eps}
    if spec.type == "adam":
        return torch.optim.Adam(groups, **kwargs)
    if spec.type == "adamw":
        return torch.optim.AdamW(groups, **kwargs)
    raise ValueError(f"unsupported optimizer {spec.type!r}")


def _atomic_save(payload: dict[str, Any], path: Path) -> None:
    """Write a checkpoint through a temporary name."""
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def _cycle(loader: Any) -> Iterator[Any]:
    """Endless iteration over a loader, counting epochs implicitly."""
    while True:
        yield from loader


def _human_bytes(count: int) -> str:
    value = float(count)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} GiB"


__all__ = ["TrainState", "Trainer"]
