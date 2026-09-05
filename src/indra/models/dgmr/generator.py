"""DGMR generative refinement head."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as apply_spectral_norm

from ...config import DGMRConfig, DGMRGenerator, LatentConditioningStack

logger = logging.getLogger(__name__)


def _conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int | None = None,
    use_spectral_norm: bool = True,
) -> nn.Module:
    """Convolution, optionally spectrally normalised."""
    if padding is None:
        padding = kernel_size // 2
    layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
    return apply_spectral_norm(layer) if use_spectral_norm else layer


def _norm(channels: int, groups: int = 8) -> nn.Module:
    """GroupNorm, deliberately not BatchNorm."""
    return nn.GroupNorm(min(groups, channels), channels)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class GBlock(nn.Module):
    """Residual generator block, with optional 2x upsampling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        upsample: bool = False,
        use_spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        self.upsample = upsample

        self.norm1 = _norm(in_channels)
        self.conv1 = _conv(
            in_channels, out_channels, 3, use_spectral_norm=use_spectral_norm
        )
        self.norm2 = _norm(out_channels)
        self.conv2 = _conv(
            out_channels, out_channels, 3, use_spectral_norm=use_spectral_norm
        )

        self.skip: nn.Module = (
            _conv(in_channels, out_channels, 1, use_spectral_norm=use_spectral_norm)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.norm1(x))
        if self.upsample:
            # Nearest-neighbour before the convolution, the standard
            # arrangement: a transposed convolution at these widths produces
            # the regular checkerboard artefacts that a discriminator learns
            # to detect immediately, which destabilises training.
            h = F.interpolate(h, scale_factor=2, mode="nearest")
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        h = self.conv1(h)
        h = self.conv2(F.relu(self.norm2(h)))
        return h + self.skip(x)


class ConvGRUCell(nn.Module):
    """Convolutional GRU."""

    def __init__(
        self,
        input_channels: int,
        hidden_channels: int,
        kernel_size: int = 3,
        use_spectral_norm: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        padding = kernel_size // 2

        self.gates = _conv(
            input_channels + hidden_channels,
            hidden_channels * 2,
            kernel_size,
            padding=padding,
            use_spectral_norm=use_spectral_norm,
        )
        self.candidate = _conv(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size,
            padding=padding,
            use_spectral_norm=use_spectral_norm,
        )

    def forward(
        self, x: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> torch.Tensor:
        if hidden is None:
            hidden = torch.zeros(
                x.shape[0],
                self.hidden_channels,
                *x.shape[2:],
                device=x.device,
                dtype=x.dtype,
            )
        combined = torch.cat([x, hidden], dim=1)
        reset, update = torch.sigmoid(self.gates(combined)).chunk(2, dim=1)
        candidate = torch.tanh(self.candidate(torch.cat([x, reset * hidden], dim=1)))
        return (1.0 - update) * candidate + update * hidden


class SpatialAttention(nn.Module):
    """Self-attention over a small spatial grid."""

    def __init__(self, channels: int, use_spectral_norm: bool = True) -> None:
        super().__init__()
        self.query = _conv(
            channels, channels // 8, 1, use_spectral_norm=use_spectral_norm
        )
        self.key = _conv(
            channels, channels // 8, 1, use_spectral_norm=use_spectral_norm
        )
        self.value = _conv(channels, channels, 1, use_spectral_norm=use_spectral_norm)
        # Starts at zero so the block is an identity at initialisation and the
        # attention is introduced only as it earns its place.
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.query(x).view(b, -1, h * w).permute(0, 2, 1)
        k = self.key(x).view(b, -1, h * w)
        v = self.value(x).view(b, -1, h * w)

        attention = torch.softmax(q @ k, dim=-1)
        out = (v @ attention.permute(0, 2, 1)).view(b, c, h, w)
        return x + self.gamma * out


# ---------------------------------------------------------------------------
# Latent conditioning stack
# ---------------------------------------------------------------------------


class LatentConditioningStackModule(nn.Module):
    """Noise -> spatial latent matching the coarsest conditioning level."""

    def __init__(
        self, config: LatentConditioningStack, use_spectral_norm: bool = True
    ) -> None:
        super().__init__()
        self.config = config
        width = config.output_channels

        self.stem = _conv(
            config.noise_channels, width // 4, 3, use_spectral_norm=use_spectral_norm
        )
        self.block1 = GBlock(
            width // 4, width // 2, use_spectral_norm=use_spectral_norm
        )
        self.block2 = GBlock(width // 2, width, use_spectral_norm=use_spectral_norm)
        self.attention = SpatialAttention(width, use_spectral_norm)
        self.block3 = GBlock(width, width, use_spectral_norm=use_spectral_norm)

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        expected = (
            self.config.noise_channels,
            self.config.latent_resolution,
            self.config.latent_resolution,
        )
        if tuple(noise.shape[1:]) != expected:
            raise ValueError(
                f"noise must be (B, {expected[0]}, {expected[1]}, {expected[2]}); "
                f"got {tuple(noise.shape)}. Use generator.noise_shape() to build it."
            )
        h = self.stem(noise)
        h = self.block2(self.block1(h))
        return self.block3(self.attention(h))


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------


class DGMRGeneratorModule(nn.Module):
    """The conditional sampler."""

    def __init__(self, config: DGMRConfig, output_size: int = 384) -> None:
        super().__init__()
        generator: DGMRGenerator = config.generator
        self.config = config
        self.generator_config = generator
        self.forecast_steps = generator.forecast_steps
        self.output_size = output_size
        use_sn = generator.spectral_norm

        self.latent_stack = LatentConditioningStackModule(
            config.latent_conditioning_stack, use_sn
        )

        # The adapter returns the pyramid finest-first; the sampler consumes
        # it coarsest-first, since generation starts small and upsamples. The
        # reversal happens here, once, where it is visible.
        self.level_dims = list(reversed(generator.conditioning_dims))
        n_levels = len(self.level_dims)
        if n_levels != generator.num_upsample_blocks:
            raise ValueError(
                f"{n_levels} conditioning levels but num_upsample_blocks is "
                f"{generator.num_upsample_blocks}; each level owns one upsample"
            )

        latent_channels = config.latent_conditioning_stack.output_channels
        if latent_channels != self.level_dims[0]:
            raise ValueError(
                f"latent stack emits {latent_channels} channels but the "
                f"coarsest conditioning level is {self.level_dims[0]}"
            )

        self.cells = nn.ModuleList()
        self.blocks = nn.ModuleList()
        self.upsamplers = nn.ModuleList()

        width = latent_channels
        for level, dim in enumerate(self.level_dims):
            # Hidden state is initialised from the conditioning at this
            # resolution, so the recurrence starts from the backbone's
            # forecast rather than from zeros.
            self.cells.append(ConvGRUCell(width, dim, generator.convgru_kernel, use_sn))
            self.blocks.append(
                GBlock(dim, dim, upsample=False, use_spectral_norm=use_sn)
            )
            # Halve the width on the way up, as resolution grows, keeping the
            # activation footprint roughly constant across levels.
            next_width = (
                self.level_dims[level + 1] if level + 1 < n_levels else max(dim // 2, 8)
            )
            self.upsamplers.append(
                GBlock(dim, next_width, upsample=True, use_spectral_norm=use_sn)
            )
            width = next_width

        self.final_width = width

        # -- output head ------------------------------------------------------
        # After four upsamples the field is at output_size / 2; a pixel
        # shuffle of 2 completes the path to full resolution. Depth-to-space
        # rather than another interpolation keeps the last step learned.
        self.head_norm = _norm(width)
        self.head_conv = _conv(
            width, generator.output_channels * 4, 1, use_spectral_norm=use_sn
        )
        self.shuffle = nn.PixelShuffle(2)

    # ----------------------------------------------------------------- noise
    def noise_shape(self, batch: int = 1) -> tuple[int, int, int, int]:
        """The shape the caller must draw. Published, not assumed."""
        stack = self.config.latent_conditioning_stack
        return (
            batch,
            stack.noise_channels,
            stack.latent_resolution,
            stack.latent_resolution,
        )

    # --------------------------------------------------------------- forward
    def forward(
        self,
        conditioning: list[torch.Tensor],
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Generate one realisation."""
        if len(conditioning) != len(self.level_dims):
            raise ValueError(
                f"expected {len(self.level_dims)} conditioning levels, got "
                f"{len(conditioning)}"
            )

        # Reverse to coarsest-first for generation.
        pyramid = list(reversed(conditioning))
        for level, (tensor, dim) in enumerate(
            zip(pyramid, self.level_dims, strict=False)
        ):
            if tensor.shape[1] != dim:
                raise ValueError(
                    f"conditioning level {level} has {tensor.shape[1]} channels, "
                    f"expected {dim}. The pyramid must be passed finest-first; "
                    "the sampler reverses it."
                )

        batch = noise.shape[0]
        latent = self.latent_stack(noise)

        # Hidden states start as the conditioning itself: the recurrence is
        # seeded with the Earthformer forecast at every resolution.
        hidden = [tensor.clone() for tensor in pyramid]

        frames: list[torch.Tensor] = []
        for _ in range(self.forecast_steps):
            # The same latent enters at every step. Its role is to fix *which*
            # realisation this is, and that identity must not drift across the
            # six-hour horizon or the members would decorrelate from
            # themselves partway through.
            x = latent
            for level in range(len(self.level_dims)):
                hidden[level] = self.cells[level](x, hidden[level])
                h = self.blocks[level](hidden[level])
                x = self.upsamplers[level](h)

            out = self.head_conv(F.relu(self.head_norm(x)))
            frames.append(self.shuffle(out))

        stacked = torch.stack(frames, dim=1)  # (B, T, C, H, W)

        if stacked.shape[-1] != self.output_size:
            stacked = F.interpolate(
                stacked.flatten(0, 1),
                size=(self.output_size, self.output_size),
                mode="bilinear",
                align_corners=False,
            ).view(batch, self.forecast_steps, -1, self.output_size, self.output_size)
        return stacked

    # -------------------------------------------------------------- ensemble
    def generate_ensemble(
        self,
        conditioning: list[torch.Tensor],
        noise_batch: torch.Tensor,
    ) -> torch.Tensor:
        """Generate several members from a stack of pre-drawn noise."""
        return torch.stack(
            [
                self.forward(conditioning, noise_batch[i : i + 1])
                for i in range(noise_batch.shape[0])
            ]
        )


__all__ = [
    "ConvGRUCell",
    "DGMRGeneratorModule",
    "GBlock",
    "LatentConditioningStackModule",
    "SpatialAttention",
]
