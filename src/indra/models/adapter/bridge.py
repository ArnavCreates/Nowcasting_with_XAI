"""Spatio-temporal adapter bridge — Earthformer latents to DGMR conditioning."""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...config import AdapterConfig

logger = logging.getLogger(__name__)

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "leaky_relu": nn.LeakyReLU,
}


def _make_norm(kind: str, channels: int, groups: int) -> nn.Module:
    """Normalisation for the conditioning path."""
    if kind == "group_norm":
        if channels % groups:
            raise ValueError(f"{channels} channels is not divisible by groups={groups}")
        return nn.GroupNorm(groups, channels)
    if kind == "batch_norm":
        return nn.BatchNorm2d(channels)
    if kind == "instance_norm":
        return nn.InstanceNorm2d(channels, affine=True)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"unknown normalisation {kind!r}")


class ConditioningBlock(nn.Module):
    """Residual convolution block for one pyramid level."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str,
        groups: int,
        activation: str,
        residual_projection: bool,
    ) -> None:
        super().__init__()
        act = _ACTIVATIONS.get(activation)
        if act is None:
            raise ValueError(f"unknown activation {activation!r}")

        self.norm1 = _make_norm(norm, in_channels, min(groups, in_channels))
        self.act1 = act()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)

        self.norm2 = _make_norm(norm, out_channels, min(groups, out_channels))
        self.act2 = act()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        if in_channels != out_channels:
            # Mandatory when the width changes: without it the residual
            # addition is a shape error, so the configuration flag does not
            # get a say here.
            self.skip: nn.Module = nn.Conv2d(in_channels, out_channels, 1)
        elif residual_projection:
            # Widths already match, so this is the flag's actual effect: a
            # learned 1x1 on the skip lets the block attenuate or reweight
            # what it passes through instead of being forced to carry the
            # input unchanged.
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.skip(x)


class AdapterBridge(nn.Module):
    """Earthformer latent -> DGMR multi-scale conditioning pyramid."""

    def __init__(
        self, config: AdapterConfig, latent_size: tuple[int, int] | None = None
    ) -> None:
        super().__init__()
        self.config = config
        self.scales = list(config.output_scales)
        self.dims = list(config.output_dims)
        self.latent_size = latent_size

        if len(self.scales) != len(self.dims):
            raise ValueError(
                f"output_scales has {len(self.scales)} entries but output_dims "
                f"has {len(self.dims)}"
            )

        in_channels = (
            config.in_dim * config.in_sequence
            if config.temporal_to_channel
            else config.in_dim
        )
        self.in_channels = in_channels

        # -- finest level -----------------------------------------------------
        self.stem = ConditioningBlock(
            in_channels,
            self.dims[0],
            config.norm,
            config.groups,
            config.activation,
            config.residual_projection,
        )

        # -- coarser levels ---------------------------------------------------
        # Each step reduces resolution and widens. Strided convolution rather
        # than pooling: the reduction is learned, and average pooling at the
        # first step would blunt exactly the fine-scale gradients that tell the
        # sampler where a convective core sits.
        self.downsample = nn.ModuleList()
        self.blocks = nn.ModuleList()

        for level in range(1, len(self.scales)):
            previous, current = self.scales[level - 1], self.scales[level]
            if previous % current == 0 and previous // current == 2:
                self.downsample.append(
                    nn.Conv2d(
                        self.dims[level - 1],
                        self.dims[level - 1],
                        kernel_size=3,
                        stride=2,
                        padding=1,
                    )
                )
            else:
                # Non-halving ratios cannot be expressed by a stride-2
                # convolution; fall back to the configured interpolation,
                # which reaches any target size exactly.
                logger.info(
                    "pyramid step %d -> %d is not a halving; using %s "
                    "interpolation for this level",
                    previous,
                    current,
                    config.interpolation,
                )
                self.downsample.append(nn.Identity())

            self.blocks.append(
                ConditioningBlock(
                    self.dims[level - 1],
                    self.dims[level],
                    config.norm,
                    config.groups,
                    config.activation,
                    config.residual_projection,
                )
            )

    # -------------------------------------------------------------- helpers
    def _resize(self, x: torch.Tensor, size: int) -> torch.Tensor:
        if x.shape[-1] == size and x.shape[-2] == size:
            return x
        return F.interpolate(
            x,
            size=(size, size),
            mode=self.config.interpolation,
            align_corners=(
                self.config.align_corners
                if self.config.interpolation in ("bilinear", "bicubic", "linear")
                else None
            ),
        )

    def fold_time(self, latent: torch.Tensor) -> torch.Tensor:
        """``(B, T, H, W, C)`` -> ``(B, T*C, H, W)``."""
        if latent.ndim != 5:
            raise ValueError(
                f"expected (B, T, H, W, C) latent; got {tuple(latent.shape)}"
            )
        b, t, h, w, c = latent.shape

        if self.config.temporal_to_channel:
            if t != self.config.in_sequence:
                raise ValueError(
                    f"adapter is configured for {self.config.in_sequence} lead "
                    f"frames but received {t}"
                )
            x = latent.permute(0, 1, 4, 2, 3).reshape(b, t * c, h, w)
        else:
            # Time folded into the batch: each lead time is conditioned
            # independently.
            x = latent.permute(0, 1, 4, 2, 3).reshape(b * t, c, h, w)
        return x

    # -------------------------------------------------------------- forward
    def forward(self, latent: torch.Tensor) -> list[torch.Tensor]:
        """Build the conditioning pyramid."""
        x = self.fold_time(latent)

        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"adapter expected {self.in_channels} folded channels "
                f"({self.config.in_dim} x {self.config.in_sequence}) but got "
                f"{x.shape[1]}"
            )

        # Align to the finest pyramid scale once, at the entrance, so no later
        # level inherits a size mismatch.
        if x.shape[-1] != self.scales[0] or x.shape[-2] != self.scales[0]:
            logger.debug(
                "latent is %dx%d but the finest pyramid scale is %d; resizing",
                x.shape[-2],
                x.shape[-1],
                self.scales[0],
            )
            x = self._resize(x, self.scales[0])

        pyramid: list[torch.Tensor] = [self.stem(x)]

        for level, (reduce, block) in enumerate(
            zip(self.downsample, self.blocks, strict=False), start=1
        ):
            h = reduce(pyramid[-1])
            # Guarantees the exact declared size regardless of how the
            # reduction was performed, including the identity path.
            h = self._resize(h, self.scales[level])
            pyramid.append(block(h))

        for level, tensor in enumerate(pyramid):
            expected = (self.dims[level], self.scales[level], self.scales[level])
            actual = tuple(tensor.shape[1:])
            if actual != expected:
                raise RuntimeError(
                    f"pyramid level {level} is {actual}, expected {expected}"
                )
        return pyramid

    def describe(self) -> list[dict[str, int]]:
        """Static description of the pyramid, for logging and documentation."""
        return [
            {"level": index, "channels": dim, "height": scale, "width": scale}
            for index, (scale, dim) in enumerate(
                zip(self.scales, self.dims, strict=False)
            )
        ]


__all__ = ["AdapterBridge", "ConditioningBlock"]
