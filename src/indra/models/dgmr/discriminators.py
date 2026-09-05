"""DGMR critics — spatial and temporal discriminators."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as apply_spectral_norm

from ...config import Discriminators, DiscriminatorSpec

logger = logging.getLogger(__name__)


def _conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int | None = None,
    use_spectral_norm: bool = True,
) -> nn.Module:
    if padding is None:
        padding = kernel_size // 2
    layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    return apply_spectral_norm(layer) if use_spectral_norm else layer


def _conv3d(
    in_channels: int,
    out_channels: int,
    kernel_size: tuple[int, int, int],
    stride: tuple[int, int, int],
    padding: tuple[int, int, int],
    use_spectral_norm: bool = True,
) -> nn.Module:
    layer = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
    return apply_spectral_norm(layer) if use_spectral_norm else layer


def _linear(
    in_features: int, out_features: int, use_spectral_norm: bool = True
) -> nn.Module:
    layer = nn.Linear(in_features, out_features)
    return apply_spectral_norm(layer) if use_spectral_norm else layer


# ---------------------------------------------------------------------------
# Shared block
# ---------------------------------------------------------------------------


class DBlock(nn.Module):
    """Residual discriminator block, with optional 2x average-pool downsample."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        downsample: bool = True,
        first: bool = False,
        use_spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        self.downsample = downsample
        self.first = first

        self.conv1 = _conv2d(
            in_channels, out_channels, 3, use_spectral_norm=use_spectral_norm
        )
        self.conv2 = _conv2d(
            out_channels, out_channels, 3, use_spectral_norm=use_spectral_norm
        )
        self.skip = (
            _conv2d(in_channels, out_channels, 1, use_spectral_norm=use_spectral_norm)
            if in_channels != out_channels or downsample
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x if self.first else F.relu(x)
        h = self.conv2(F.relu(self.conv1(h)))
        if self.downsample:
            h = F.avg_pool2d(h, 2)

        if self.first:
            skip = F.avg_pool2d(x, 2) if self.downsample else x
            skip = self.skip(skip)
        else:
            skip = self.skip(x)
            if self.downsample:
                skip = F.avg_pool2d(skip, 2)
        return h + skip


def _build_stack(
    in_channels: int, base: int, num_layers: int, use_spectral_norm: bool
) -> tuple[nn.ModuleList, int]:
    """A ladder of ``DBlock``s, doubling width and halving resolution."""
    blocks = nn.ModuleList()
    channels = in_channels
    for layer in range(num_layers):
        out_channels = base * (2**layer)
        blocks.append(
            DBlock(
                channels,
                out_channels,
                downsample=True,
                first=(layer == 0),
                use_spectral_norm=use_spectral_norm,
            )
        )
        channels = out_channels
    return blocks, channels


# ---------------------------------------------------------------------------
# Spatial critic
# ---------------------------------------------------------------------------


class SpatialDiscriminator(nn.Module):
    """Judges individual frames for realistic spatial texture."""

    def __init__(
        self,
        spec: DiscriminatorSpec,
        input_channels: int = 1,
        forecast_steps: int = 12,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.forecast_steps = forecast_steps
        self.num_sampled_frames = min(
            spec.num_sampled_frames or forecast_steps, forecast_steps
        )

        # Space-to-depth halves resolution without discarding anything: the
        # four sub-pixels move into channels rather than being averaged away.
        # A pooling layer here would destroy fine texture, which is precisely
        # what this critic exists to judge.
        self.unshuffle = nn.PixelUnshuffle(2)
        stem_channels = input_channels * 4

        self.blocks, final_channels = _build_stack(
            stem_channels, spec.base_channels, spec.num_layers, spec.spectral_norm
        )
        self.final_block = DBlock(
            final_channels,
            final_channels,
            downsample=False,
            use_spectral_norm=spec.spectral_norm,
        )
        self.classifier = _linear(final_channels, 1, spec.spectral_norm)

    def sample_frames(
        self,
        sequence: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Pick ``num_sampled_frames`` frames without replacement."""
        n_time = sequence.shape[1]
        if self.num_sampled_frames >= n_time:
            return sequence
        indices = torch.randperm(n_time, generator=generator, device="cpu")
        indices = indices[: self.num_sampled_frames].to(sequence.device)
        return sequence.index_select(1, indices)

    def forward(
        self,
        sequence: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """``sequence``: ``(B, T, C, H, W)``. Returns ``(B,)`` scores."""
        if sequence.ndim != 5:
            raise ValueError(f"expected (B, T, C, H, W); got {tuple(sequence.shape)}")
        selected = self.sample_frames(sequence, generator)
        batch, n_time = selected.shape[0], selected.shape[1]

        x = selected.reshape(batch * n_time, *selected.shape[2:])
        x = self.unshuffle(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_block(x)

        # Sum, not mean: an averaged score is invariant to how much of the
        # domain is raining, so a small intense core and a broad weak field
        # would look identical to the critic.
        x = F.relu(x).sum(dim=(2, 3))
        scores = self.classifier(x).view(batch, n_time)
        return scores.mean(dim=1)


# ---------------------------------------------------------------------------
# Temporal critic
# ---------------------------------------------------------------------------


class TemporalDiscriminator(nn.Module):
    """Judges the sequence for realistic motion, growth and decay."""

    def __init__(
        self,
        spec: DiscriminatorSpec,
        input_channels: int = 1,
    ) -> None:
        super().__init__()
        self.spec = spec

        k = spec.stem_kernel or (4, 3, 3)
        kernel: tuple[int, int, int] = (int(k[0]), int(k[1]), int(k[2]))
        s = spec.stem_stride or (2, 1, 1)
        stride: tuple[int, int, int] = (int(s[0]), int(s[1]), int(s[2]))
        # Temporal padding stays zero so the stem genuinely contracts time;
        # spatial padding preserves the grid, which the 2-D ladder then
        # reduces.
        padding = (0, kernel[1] // 2, kernel[2] // 2)

        self.unshuffle = nn.PixelUnshuffle(2)
        stem_in = input_channels * 4

        self.stem = _conv3d(
            stem_in,
            spec.base_channels,
            kernel,
            stride,
            padding,
            spec.spectral_norm,
        )
        self.blocks, final_channels = _build_stack(
            spec.base_channels,
            spec.base_channels,
            spec.num_layers,
            spec.spectral_norm,
        )
        self.final_block = DBlock(
            final_channels,
            final_channels,
            downsample=False,
            use_spectral_norm=spec.spectral_norm,
        )
        self.classifier = _linear(final_channels, 1, spec.spectral_norm)

    def forward(
        self,
        sequence: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``sequence``: ``(B, T, C, H, W)``. Returns ``(B,)`` scores."""
        if sequence.ndim != 5:
            raise ValueError(f"expected (B, T, C, H, W); got {tuple(sequence.shape)}")
        if context is not None:
            if context.shape[2:] != sequence.shape[2:]:
                raise ValueError(
                    f"context {tuple(context.shape[2:])} and sequence "
                    f"{tuple(sequence.shape[2:])} must share C, H and W"
                )
            sequence = torch.cat([context, sequence], dim=1)

        batch, n_time = sequence.shape[0], sequence.shape[1]

        x = sequence.reshape(batch * n_time, *sequence.shape[2:])
        x = self.unshuffle(x)
        x = x.view(batch, n_time, *x.shape[1:])

        # (B, T, C, H, W) -> (B, C, T, H, W) for Conv3d, which expects the
        # channel axis second and treats the third as depth.
        x = x.permute(0, 2, 1, 3, 4)
        if x.shape[2] < self.stem.kernel_size[0]:
            raise ValueError(
                f"{x.shape[2]} frames is fewer than the temporal stem kernel "
                f"{self.stem.kernel_size[0]}; the critic cannot see motion in "
                "a sequence shorter than its receptive field"
            )
        x = self.stem(x)

        # Fold the contracted time axis into the batch so the 2-D ladder scores
        # each temporal window independently, then average the windows.
        x = x.permute(0, 2, 1, 3, 4)
        n_windows = x.shape[1]
        x = x.reshape(batch * n_windows, *x.shape[2:])

        for block in self.blocks:
            x = block(x)
        x = self.final_block(x)
        x = F.relu(x).sum(dim=(2, 3))

        scores = self.classifier(x).view(batch, n_windows)
        return scores.mean(dim=1)


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------


class DGMRDiscriminators(nn.Module):
    """Both critics, so the training loop holds one module and one optimiser."""

    def __init__(
        self,
        config: Discriminators,
        input_channels: int = 1,
        forecast_steps: int = 12,
    ) -> None:
        super().__init__()
        self.spatial = (
            SpatialDiscriminator(config.spatial, input_channels, forecast_steps)
            if config.spatial.enabled
            else None
        )
        self.temporal = (
            TemporalDiscriminator(config.temporal, input_channels)
            if config.temporal.enabled
            else None
        )
        if self.spatial is None and self.temporal is None:
            raise ValueError(
                "both discriminators are disabled; adversarial training needs "
                "at least one critic, or the generator has no signal beyond "
                "the reconstruction term and will regress to the blurred mean"
            )
        if self.temporal is None:
            logger.warning(
                "temporal discriminator disabled: nothing will penalise "
                "incoherent motion between forecast frames"
            )

    def forward(
        self,
        sequence: torch.Tensor,
        context: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        """Score a sequence with every enabled critic."""
        scores: dict[str, torch.Tensor] = {}
        if self.spatial is not None:
            scores["spatial"] = self.spatial(sequence, generator)
        if self.temporal is not None:
            scores["temporal"] = self.temporal(sequence, context)
        return scores


__all__ = [
    "DBlock",
    "DGMRDiscriminators",
    "SpatialDiscriminator",
    "TemporalDiscriminator",
]
