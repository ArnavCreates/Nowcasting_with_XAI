"""Assembles Earthformer -> adapter -> DGMR into one model."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from ..config import CheckpointSpec, ModelConfig
from .adapter.bridge import AdapterBridge
from .dgmr.discriminators import DGMRDiscriminators
from .dgmr.generator import DGMRGeneratorModule
from .earthformer.backbone import EarthformerBackbone

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint plumbing
# ---------------------------------------------------------------------------


class MissingCheckpointError(FileNotFoundError):
    """Raised when weights a component depends on are not on disk."""

    TEMPLATE = (
        "Model weights not found at {path}. To download the fine-tuned "
        "Earthformer/DGMR weights from our cloud bucket, please run "
        "'bash scripts/fetch_weights.sh'."
    )

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        super().__init__(self.TEMPLATE.format(path=self.path))


# Configuration vocabulary -> ``state_dict`` prefix.
#
# ``reinit_modules`` in the YAML names components the way the architecture
# papers name them; the modules are attributes with their own names. Without
# this bridge a prefix match on "patch_embed" would find no key, reinitialise
# nothing, and load exactly the stem weights the configuration asked to
# discard -- a mismatch that raises no error and yields a subtly wrong model.
# Every resolved prefix is verified against the live state_dict in
# ``_resolve_reinit_prefixes``, so a future rename fails loudly rather than
# silently doing nothing.
_REINIT_ALIASES: dict[str, str] = {
    "patch_embed": "embed",
    "latent_conditioning_stack": "latent_stack",
}


@dataclass
class CheckpointReport:
    """What a load actually did, key by key."""

    component: str
    path: Path
    matched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    reinitialised: list[str] = field(default_factory=list)
    shape_mismatched: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = field(
        default_factory=list
    )

    @property
    def is_clean(self) -> bool:
        """True when every discrepancy was one the configuration asked for."""
        return not (self.missing or self.unexpected or self.shape_mismatched)

    def summary(self) -> str:
        return (
            f"{self.component}: {len(self.matched)} matched, "
            f"{len(self.reinitialised)} reinitialised, "
            f"{len(self.missing)} missing, "
            f"{len(self.unexpected)} unexpected, "
            f"{len(self.shape_mismatched)} shape-mismatched "
            f"(from {self.path})"
        )

    def log(self) -> None:
        logger.info("%s", self.summary())
        for key in self.reinitialised:
            logger.debug("%s: %s kept its initialisation", self.component, key)
        for key in self.missing:
            logger.warning(
                "%s: %s absent from the checkpoint and left at initialisation",
                self.component,
                key,
            )
        for key in self.unexpected:
            logger.warning(
                "%s: checkpoint carries %s, which this architecture has no "
                "parameter for",
                self.component,
                key,
            )
        for key, checkpoint_shape, own_shape in self.shape_mismatched:
            logger.warning(
                "%s: %s is %s in the checkpoint but %s here; not loaded",
                self.component,
                key,
                checkpoint_shape,
                own_shape,
            )


def _extract_state_dict(blob: Any) -> dict[str, Any]:
    """Unwrap the several shapes in which a saved checkpoint arrives."""
    if isinstance(blob, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            inner = blob.get(key)
            if isinstance(inner, dict):
                return inner
        return blob
    raise TypeError(
        f"checkpoint deserialised to {type(blob).__name__}, not a state dict"
    )


def _normalise_keys(state: dict[str, Any]) -> dict[str, torch.Tensor]:
    """Strip the ``module.`` prefix a DistributedDataParallel save leaves behind."""
    return {
        (key[len("module.") :] if key.startswith("module.") else key): value
        for key, value in state.items()
    }


def _has_prefix(key: str, prefixes: Sequence[str]) -> bool:
    return any(key == prefix or key.startswith(prefix + ".") for prefix in prefixes)


def _resolve_reinit_prefixes(module: nn.Module, names: Iterable[str]) -> list[str]:
    """Translate configured module names into verified ``state_dict`` prefixes."""
    own = list(module.state_dict())
    resolved: list[str] = []
    for name in names:
        prefix = _REINIT_ALIASES.get(name, name)
        if not _has_prefix_in(own, prefix):
            top_level = sorted({key.split(".", 1)[0] for key in own})
            raise ValueError(
                f"reinit_modules names '{name}' (resolved to prefix '{prefix}'), "
                f"which matches no parameter of {type(module).__name__}. Nothing "
                f"would be reinitialised and the checkpoint's weights for it "
                f"would be loaded instead. Available top-level modules: "
                f"{top_level}"
            )
        resolved.append(prefix)
    return resolved


def _has_prefix_in(keys: Sequence[str], prefix: str) -> bool:
    return any(key == prefix or key.startswith(prefix + ".") for key in keys)


def load_component_state(
    module: nn.Module,
    spec: CheckpointSpec,
    component: str,
    *,
    map_location: str | torch.device = "cpu",
) -> CheckpointReport:
    """Load one component's weights and report exactly what happened."""
    path = spec.resolved_path()
    if not path.is_file():
        raise MissingCheckpointError(path)

    # weights_only: a checkpoint is an untrusted pickle until proven otherwise,
    # and this one is mounted into the container from outside the repository.
    blob = torch.load(path, map_location=map_location, weights_only=True)
    incoming = _normalise_keys(_extract_state_dict(blob))
    own = module.state_dict()
    prefixes = _resolve_reinit_prefixes(module, spec.reinit_modules)

    accepted: dict[str, torch.Tensor] = {}
    report = CheckpointReport(component=component, path=path)

    for key, tensor in incoming.items():
        if _has_prefix(key, prefixes):
            report.reinitialised.append(key)
            continue
        if key not in own:
            report.unexpected.append(key)
            continue
        if tuple(tensor.shape) != tuple(own[key].shape):
            report.shape_mismatched.append(
                (key, tuple(tensor.shape), tuple(own[key].shape))
            )
            continue
        accepted[key] = tensor

    report.matched = sorted(accepted)
    report.missing = sorted(
        key for key in own if key not in accepted and not _has_prefix(key, prefixes)
    )
    # Both sides of a reinitialisation: the keys withheld from the checkpoint
    # and the parameters here that consequently keep their fresh values.
    report.reinitialised = sorted(
        set(report.reinitialised) | {key for key in own if _has_prefix(key, prefixes)}
    )

    module.load_state_dict(accepted, strict=False)
    report.log()

    if spec.strict and not report.is_clean:
        raise RuntimeError(
            f"strict checkpoint load failed for {component}. {report.summary()}. "
            "Set strict: false in the checkpoint block only if every "
            "discrepancy logged above is understood and intended."
        )
    return report


# ---------------------------------------------------------------------------
# Forward-pass results
# ---------------------------------------------------------------------------


@dataclass
class FusionOutput:
    """Everything one forward pass produced."""

    precipitation: torch.Tensor  # (B, 12, 1, 384, 384)
    latent: torch.Tensor  # (B, 12, 96, 96, 128)
    conditioning: list[torch.Tensor]  # pyramid, finest first
    auxiliary: torch.Tensor | None = None  # (B, 12, 1, 384, 384)
    attention: dict[str, torch.Tensor] | None = None


@dataclass
class EnsembleForecast:
    """A seeded ensemble and its reductions."""

    members: torch.Tensor | None  # (N, B, 12, 1, H, W)
    reductions: dict[str, torch.Tensor]  # each (B, 12, 1, H, W)
    seed: int
    member_seeds: list[int]

    @property
    def num_members(self) -> int:
        return len(self.member_seeds)


# Reductions the ensemble supports. ``std`` uses the unbiased estimator: with
# eight members the population form understates spread by roughly 6%, and
# spread feeds the severity banding, where understating uncertainty means
# under-warning.
_REDUCTIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "mean": lambda members: members.mean(dim=0),
    "max": lambda members: members.amax(dim=0),
    "min": lambda members: members.amin(dim=0),
    "median": lambda members: members.median(dim=0).values,
    "std": lambda members: members.std(dim=0, unbiased=True),
}


# ---------------------------------------------------------------------------
# The assembled model
# ---------------------------------------------------------------------------


class IndraFusion(nn.Module):
    """Earthformer backbone, adapter bridge and DGMR sampler as one model."""

    def __init__(self, config: ModelConfig, *, auxiliary_head: bool = True) -> None:
        super().__init__()
        self.config = config
        model_input, model_output = config.input, config.output

        if model_input.height != model_input.width:
            raise ValueError(
                f"the relay assumes a square grid; got {model_input.height}x"
                f"{model_input.width}. The DGMR sampler takes a single output "
                "extent and the cuboid partition is declared per axis, so a "
                "rectangular domain needs both revisited rather than silently "
                "squashed."
            )

        patch = config.earthformer.patch_embed.patch_size
        latent_size = (model_input.height // patch, model_input.width // patch)

        self.earthformer = EarthformerBackbone(
            config.earthformer,
            in_channels=model_input.channels,
            sequence_length=model_input.sequence_length,
            lead_frames=model_output.lead_frames,
            height=model_input.height,
            width=model_input.width,
            auxiliary_head=auxiliary_head,
        )
        self.adapter = AdapterBridge(config.adapter, latent_size=latent_size)
        self.generator = DGMRGeneratorModule(
            config.dgmr, output_size=model_input.height
        )

        self.sequence_length = model_input.sequence_length
        self.in_channels = model_input.channels
        self.grid = (model_input.height, model_input.width)
        self.lead_frames = model_output.lead_frames
        self.latent_size = latent_size

        # Held, not applied. Autocast and activation checkpointing change the
        # numerics and the memory profile of whatever loop is running; the
        # trainer and the serving layer each decide for themselves.
        self.precision = config.fusion.precision
        self.gradient_checkpointing = config.fusion.gradient_checkpointing

        if config.adapter.output_scales[0] != latent_size[0]:
            logger.warning(
                "backbone latent is %dx%d but the finest adapter scale is %d; "
                "the bridge will resample on every forward pass",
                latent_size[0],
                latent_size[1],
                config.adapter.output_scales[0],
            )

        self._backbone_frozen = False
        if config.fusion.freeze_backbone:
            self.set_backbone_frozen(True)

    # ------------------------------------------------------------- freezing
    def set_backbone_frozen(self, frozen: bool = True) -> None:
        """Freeze the Earthformer, and only the Earthformer."""
        self._backbone_frozen = frozen
        self.earthformer.requires_grad_(not frozen)
        if frozen:
            self.earthformer.eval()
            logger.info("Earthformer frozen; adapter and generator remain trainable")

    @property
    def backbone_frozen(self) -> bool:
        return self._backbone_frozen

    def train(self, mode: bool = True) -> IndraFusion:
        """Keep a frozen backbone in eval when the model is put in train mode."""
        super().train(mode)
        if self._backbone_frozen:
            self.earthformer.eval()
        return self

    def parameter_summary(self) -> dict[str, int]:
        """Counts for logging, read off the built modules."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    # ---------------------------------------------------------------- noise
    def noise_shape(self, batch: int = 1) -> tuple[int, int, int, int]:
        """The per-member noise shape the caller must supply."""
        return self.generator.noise_shape(batch)

    def draw_noise(
        self,
        *,
        members: int,
        seed: int,
        batch: int = 1,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        """Draw ``members`` reproducible latents, shaped ``(N, B, C, R, R)``."""
        if members < 1:
            raise ValueError(f"members must be positive; got {members}")

        _, channels, resolution, _ = self.generator.noise_shape(batch)
        draws: list[torch.Tensor] = []
        for index in range(members):
            rng = torch.Generator(device="cpu")
            rng.manual_seed(int(seed) + index)
            draws.append(
                torch.randn(
                    batch,
                    channels,
                    resolution,
                    resolution,
                    generator=rng,
                    dtype=torch.float32,
                )
            )
        noise = torch.stack(draws)
        if device is not None or dtype is not None:
            noise = noise.to(device=device, dtype=dtype)
        return noise

    # -------------------------------------------------------------- forward
    def _check_input(self, x: torch.Tensor) -> None:
        if x.ndim != 5:
            raise ValueError(f"expected (B, T, C, H, W); got {tuple(x.shape)}")
        _, t, c, h, w = x.shape
        if (t, c) != (self.sequence_length, self.in_channels):
            raise ValueError(
                f"expected {self.sequence_length} frames of {self.in_channels} "
                f"channels; got {t} of {c}. Channel order is load-bearing: a "
                "tensor of the right size with the wrong channel count means "
                "the assembled window did not come from this configuration."
            )
        if (h, w) != self.grid:
            raise ValueError(
                f"expected a {self.grid[0]}x{self.grid[1]} grid; got {h}x{w}"
            )

    def forward(
        self,
        x: torch.Tensor,
        noise: torch.Tensor,
        *,
        return_attention: bool = False,
        return_auxiliary: bool = False,
    ) -> FusionOutput:
        """Run the relay for one realisation."""
        self._check_input(x)
        expected = self.generator.noise_shape(x.shape[0])
        if tuple(noise.shape) != expected:
            raise ValueError(
                f"noise is {tuple(noise.shape)}, expected {expected}. The "
                "generator does not sample internally; draw with draw_noise() "
                "so the member stays reproducible from its seed."
            )

        attention: dict[str, torch.Tensor] | None = None
        if return_attention:
            latent, attention = self.earthformer(x, return_attention=True)
        else:
            latent = self.earthformer(x)

        conditioning = self.adapter(latent)
        precipitation = self.generator(conditioning, noise)

        auxiliary = None
        if return_auxiliary:
            auxiliary = self.earthformer.auxiliary_precipitation(
                latent, self.grid[0], self.grid[1]
            )

        return FusionOutput(
            precipitation=precipitation,
            latent=latent,
            conditioning=conditioning,
            auxiliary=auxiliary,
            attention=attention,
        )

    # ------------------------------------------------------------- ensemble
    @torch.no_grad()
    def predict_ensemble(
        self,
        x: torch.Tensor,
        *,
        seed: int,
        members: int | None = None,
        reductions: Sequence[str] = ("mean", "max", "std"),
        return_members: bool = True,
    ) -> EnsembleForecast:
        """Draw an ensemble and reduce it — the source of every hazard probability."""
        if members is None:
            members = self.config.dgmr.latent_conditioning_stack.ensemble_members

        unknown = [name for name in reductions if name not in _REDUCTIONS]
        if unknown:
            raise ValueError(
                f"unsupported reductions {unknown}; available: {sorted(_REDUCTIONS)}"
            )
        if "std" in reductions and members < 2:
            raise ValueError(
                "ensemble spread is undefined for a single member. Reporting "
                "zero spread would present an unsampled forecast as a certain "
                "one, which is the opposite of what the field means."
            )

        self._check_input(x)

        latent = self.earthformer(x)
        conditioning = self.adapter(latent)

        noise = self.draw_noise(
            members=members,
            seed=seed,
            batch=x.shape[0],
            device=conditioning[0].device,
            dtype=conditioning[0].dtype,
        )

        # Driven here rather than through ``generator.generate_ensemble`` so
        # the batch axis stays honest: that helper pairs one member's noise
        # with a whole batch of conditioning, which is only correct at batch
        # size 1.
        stack = torch.stack(
            [self.generator(conditioning, noise[index]) for index in range(members)]
        )

        reduced = {name: _REDUCTIONS[name](stack) for name in reductions}
        return EnsembleForecast(
            members=stack if return_members else None,
            reductions=reduced,
            seed=int(seed),
            member_seeds=[int(seed) + index for index in range(members)],
        )

    # ---------------------------------------------------------- checkpoints
    def load_pretrained(
        self, *, map_location: str | torch.device = "cpu"
    ) -> dict[str, CheckpointReport]:
        """Load the per-component checkpoints named in ``configs/model/fusion.yaml``."""
        reports: dict[str, CheckpointReport] = {}

        earthformer_spec = self.config.earthformer.checkpoint
        if earthformer_spec.load_pretrained:
            reports["earthformer"] = load_component_state(
                self.earthformer,
                earthformer_spec,
                "earthformer",
                map_location=map_location,
            )
        else:
            logger.info(
                "earthformer.checkpoint.load_pretrained is false; the backbone "
                "stays at initialisation"
            )

        dgmr_spec = self.config.dgmr.checkpoint
        if dgmr_spec.load_pretrained:
            reports["dgmr"] = load_component_state(
                self.generator,
                dgmr_spec,
                "dgmr_generator",
                map_location=map_location,
            )
        else:
            logger.info(
                "dgmr.checkpoint.load_pretrained is false; the sampler stays "
                "at initialisation"
            )

        return reports

    def load_fused(
        self,
        path: Path | str,
        *,
        strict: bool = True,
        map_location: str | torch.device = "cpu",
    ) -> CheckpointReport:
        """Load a single checkpoint covering the whole relay."""
        spec = CheckpointSpec(
            load_pretrained=True,
            path=str(path),
            strict=strict,
            reinit_modules=[],
        )
        return load_component_state(
            self, spec, "indra_fusion", map_location=map_location
        )

    # ------------------------------------------------------------ utilities
    def autocast(self, device_type: str = "cuda") -> Any:
        """Context manager for the configured precision."""
        if self.precision == "fp32":
            return contextlib.nullcontext()
        dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        return torch.autocast(device_type=device_type, dtype=dtype)

    def describe(self) -> dict[str, Any]:
        """The shape contract as this instance was actually built."""
        model_input, model_output = self.config.input, self.config.output
        return {
            "name": self.config.fusion.name,
            "input": (
                model_input.sequence_length,
                model_input.channels,
                model_input.height,
                model_input.width,
            ),
            "latent": (
                model_output.lead_frames,
                *self.latent_size,
                self.config.earthformer.decoder.dims[-1],
            ),
            "conditioning": self.adapter.describe(),
            "output": (
                model_output.lead_frames,
                model_output.channels,
                model_input.height,
                model_input.width,
            ),
            "units": model_output.units,
            "horizon_hours": model_output.horizon_hours,
            "precision": self.precision,
            "backbone_frozen": self._backbone_frozen,
            "parameters": self.parameter_summary(),
        }


# ---------------------------------------------------------------------------
# Factories
# ---------------------------------------------------------------------------


def build_from_config(
    config: ModelConfig,
    *,
    auxiliary_head: bool = True,
    load_weights: bool = True,
    map_location: str | torch.device = "cpu",
) -> IndraFusion:
    """Assemble the relay from a validated ``ModelConfig``."""
    model = IndraFusion(config, auxiliary_head=auxiliary_head)
    logger.info("built %s: %s", config.fusion.name, model.parameter_summary())

    if load_weights:
        model.load_pretrained(map_location=map_location)
    else:
        logger.warning(
            "%s built without weights; every parameter is at initialisation "
            "and any forecast it produces is meaningless",
            config.fusion.name,
        )
    return model


def build_discriminators(config: ModelConfig) -> DGMRDiscriminators:
    """Build the two critics for adversarial training."""
    return DGMRDiscriminators(
        config.dgmr.discriminators,
        input_channels=config.output.channels,
        forecast_steps=config.output.lead_frames,
    )


__all__ = [
    "CheckpointReport",
    "EnsembleForecast",
    "FusionOutput",
    "IndraFusion",
    "MissingCheckpointError",
    "build_discriminators",
    "build_from_config",
    "load_component_state",
]
