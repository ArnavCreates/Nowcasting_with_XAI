"""Earthformer backbone — hierarchical cuboid-transformer encoder/decoder."""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn as nn

from ...config import EarthformerConfig
from .cuboid_attention import CuboidAttentionBlock

logger = logging.getLogger(__name__)


def adapt_cuboid(
    cuboid: tuple[int, int, int], shape: tuple[int, int, int]
) -> tuple[int, int, int]:
    """Clamp a cuboid to the extents actually available at a stage."""
    return tuple(min(c, s) for c, s in zip(cuboid, shape, strict=False))  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Stem and resolution changes
# ---------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    """Frame-wise convolutional stem. ``(B,T,C,H,W) -> (B,T,H/p,W/p,D)``."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c, h, w = x.shape
        if h % self.patch_size or w % self.patch_size:
            raise ValueError(
                f"input grid {h}x{w} is not divisible by patch_size "
                f"{self.patch_size}"
            )
        x = self.proj(x.reshape(b * t, c, h, w))
        _, d, ph, pw = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, t, ph, pw, d)
        return self.norm(x)


class PatchMerge(nn.Module):
    """Halve H and W, widen channels. ``(B,T,H,W,C) -> (B,T,H/2,W/2,C_out)``."""

    def __init__(self, dim: int, out_dim: int) -> None:
        super().__init__()
        self.reduce = nn.Linear(dim * 4, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, h, w, c = x.shape
        if h % 2 or w % 2:
            x = nn.functional.pad(x, (0, 0, 0, w % 2, 0, h % 2))
            h, w = x.shape[2], x.shape[3]
        x = torch.cat(
            [
                x[:, :, 0::2, 0::2],
                x[:, :, 1::2, 0::2],
                x[:, :, 0::2, 1::2],
                x[:, :, 1::2, 1::2],
            ],
            dim=-1,
        )
        return self.norm(self.reduce(x))


class PatchExpand(nn.Module):
    """Double H and W, narrow channels. ``(B,T,H,W,C) -> (B,T,2H,2W,C_out)``."""

    def __init__(self, dim: int, out_dim: int) -> None:
        super().__init__()
        self.expand = nn.Linear(dim, out_dim * 4, bias=False)
        self.out_dim = out_dim
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, h, w, _ = x.shape
        x = self.expand(x).view(b, t, h, w, 2, 2, self.out_dim)
        x = x.permute(0, 1, 2, 4, 3, 5, 6).reshape(b, t, h * 2, w * 2, self.out_dim)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Cross-attention
# ---------------------------------------------------------------------------


class TemporalCrossAttention(nn.Module):
    """Decoder queries read the encoder memory along time, position by position."""

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} is not divisible by num_heads {num_heads}")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim)
        self.to_kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        memory: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """``query``: ``(B,Tq,H,W,C)``. ``memory``: ``(B,Tk,H,W,C)``."""
        b, tq, h, w, c = query.shape
        bm, tk, hm, wm, cm = memory.shape
        if (h, w, c) != (hm, wm, cm):
            raise ValueError(
                f"cross-attention needs matching spatial extent and width; "
                f"query is {(h, w, c)}, memory is {(hm, wm, cm)}"
            )

        q_in = self.norm_q(query)
        kv_in = self.norm_kv(memory)

        # Fold space into the batch so each spatial cell attends over its own
        # temporal column independently.
        q = self.to_q(q_in).permute(0, 2, 3, 1, 4).reshape(b * h * w, tq, c)
        kv = self.to_kv(kv_in).permute(0, 2, 3, 1, 4).reshape(b * h * w, tk, 2 * c)

        q = q.view(b * h * w, tq, self.num_heads, self.head_dim).transpose(1, 2)
        k, v = kv.view(b * h * w, tk, 2, self.num_heads, self.head_dim).permute(
            2, 0, 3, 1, 4
        )

        weights = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        out = (weights @ v).transpose(1, 2).reshape(b * h * w, tq, c)
        out = out.view(b, h, w, tq, c).permute(0, 3, 1, 2, 4)
        out = query + self.drop(self.proj(out))

        if return_attention:
            # Mean over heads and over output steps: how much each *input*
            # timestep was consulted at each location. Reshaped to
            # (B, Tk, H, W), this is the temporal half of the XAI attention
            # map and is what ranks the evidence frames.
            received = weights.mean(dim=1).mean(dim=1)  # (B*H*W, Tk)
            received = received.view(b, h, w, tk).permute(0, 3, 1, 2)
            return out, received
        return out


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


class CuboidStage(nn.Module):
    """One resolution level: ``depth`` sweeps of the full cuboid pattern."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        depth: int,
        cuboid_blocks: Any,
        shape: tuple[int, int, int],
        config: EarthformerConfig,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            for spec in cuboid_blocks:
                self.blocks.append(
                    CuboidAttentionBlock(
                        dim=dim,
                        num_heads=num_heads,
                        cuboid=adapt_cuboid(tuple(spec.size), shape),
                        strategy=spec.strategy,
                        shift=tuple(spec.shift),
                        num_global=(
                            config.global_vectors.num_vectors
                            if config.global_vectors.enabled
                            else 0
                        ),
                        mlp_ratio=float(config.ffn_expansion),
                        attn_dropout=config.attn_dropout,
                        proj_dropout=config.proj_dropout,
                        ffn_dropout=config.ffn_dropout,
                        activation=config.activation,
                        use_relative_position_bias=config.relative_position_bias,
                    )
                )

    def forward(
        self,
        x: torch.Tensor,
        global_vectors: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor | None]
        | tuple[torch.Tensor, torch.Tensor | None, Any]
    ):
        attention = None
        for index, block in enumerate(self.blocks):
            if return_attention and index == len(self.blocks) - 1:
                x, global_vectors, attention = block(
                    x, global_vectors, return_attention=True
                )
            else:
                x, global_vectors = block(x, global_vectors)
        if return_attention:
            return x, global_vectors, attention
        return x, global_vectors


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------


class EarthformerBackbone(nn.Module):
    """The deterministic space-time backbone of the fusion relay."""

    def __init__(
        self,
        config: EarthformerConfig,
        in_channels: int,
        sequence_length: int,
        lead_frames: int,
        height: int,
        width: int,
        auxiliary_head: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.sequence_length = sequence_length
        self.lead_frames = lead_frames

        patch = config.patch_embed.patch_size
        embed_dim = config.patch_embed.embed_dim
        self.embed = PatchEmbed(in_channels, embed_dim, patch)

        stem_h, stem_w = height // patch, width // patch
        self.stem_shape = (stem_h, stem_w)

        # -- learned positional encoding ------------------------------------
        # Temporal and spatial codes are separate parameters, per
        # ``separate_temporal_spatial``. A single joint code would need
        # T*H*W entries and would not transfer between the encoder's 13 steps
        # and the decoder's 12.
        self.pos_time_enc = nn.Parameter(
            torch.zeros(1, sequence_length, 1, 1, embed_dim)
        )
        self.pos_space_enc = nn.Parameter(torch.zeros(1, 1, stem_h, stem_w, embed_dim))
        nn.init.trunc_normal_(self.pos_time_enc, std=0.02)
        nn.init.trunc_normal_(self.pos_space_enc, std=0.02)

        # -- global vectors --------------------------------------------------
        self.use_globals = config.global_vectors.enabled
        if self.use_globals:
            self.global_vectors = nn.Parameter(
                torch.zeros(1, config.global_vectors.num_vectors, embed_dim)
            )
            nn.init.trunc_normal_(self.global_vectors, std=0.02)

        # -- encoder ---------------------------------------------------------
        enc = config.encoder
        self.encoder_stages = nn.ModuleList()
        self.encoder_downsample = nn.ModuleList()
        self.global_resize_down = nn.ModuleList()

        dim = embed_dim
        h, w = stem_h, stem_w
        self.encoder_shapes: list[tuple[int, int, int]] = []

        for _stage, (depth, factor, stage_dim) in enumerate(
            zip(enc.depths, enc.downsample_factors, enc.dims, strict=False)
        ):
            if factor > 1:
                self.encoder_downsample.append(PatchMerge(dim, stage_dim))
                h, w = h // factor, w // factor
            else:
                self.encoder_downsample.append(
                    nn.Identity() if dim == stage_dim else nn.Linear(dim, stage_dim)
                )
            # Global vectors live in the stage's width, so they are projected
            # whenever the width changes rather than being reinitialised.
            self.global_resize_down.append(
                nn.Identity() if dim == stage_dim else nn.Linear(dim, stage_dim)
            )
            dim = stage_dim
            shape = (sequence_length, h, w)
            self.encoder_shapes.append(shape)
            self.encoder_stages.append(
                CuboidStage(
                    dim,
                    config.num_heads,
                    depth,
                    config.cuboid_blocks,
                    shape,
                    config,
                )
            )

        self.memory_dim = dim
        self.memory_shape = (h, w)

        # -- decoder ---------------------------------------------------------
        dec = config.decoder
        self.decoder_stages = nn.ModuleList()
        self.decoder_upsample = nn.ModuleList()
        self.cross_attention = nn.ModuleList()
        self.global_resize_up = nn.ModuleList()
        self.decoder_shapes: list[tuple[int, int, int]] = []

        # Output steps are seeded from the final encoder state and separated
        # by their own temporal codes; the cross-attention then differentiates
        # them by what each lead time needs from the history.
        self.query_time_enc = nn.Parameter(torch.zeros(1, lead_frames, 1, 1, dim))
        nn.init.trunc_normal_(self.query_time_enc, std=0.02)

        for stage, (depth, factor, stage_dim) in enumerate(
            zip(dec.depths, dec.upsample_factors, dec.dims, strict=False)
        ):
            self.decoder_shapes.append((lead_frames, h, w))

            self.global_resize_up.append(
                nn.Identity() if dim == stage_dim else nn.Linear(dim, stage_dim)
            )
            self.decoder_stages.append(
                nn.ModuleDict(
                    {
                        "project": (
                            nn.Identity()
                            if dim == stage_dim
                            else nn.Linear(dim, stage_dim)
                        ),
                        "stage": CuboidStage(
                            stage_dim,
                            dec.cross_attn_heads,
                            depth,
                            config.cuboid_blocks,
                            (lead_frames, h, w),
                            config,
                        ),
                    }
                )
            )
            dim = stage_dim

            # Cross-attend to the encoder stage of matching resolution: the
            # deepest decoder stage reads the deepest encoder stage, forming a
            # U. Matching resolutions is what makes the temporal cross-
            # attention well defined.
            encoder_index = len(enc.dims) - 1 - stage
            self.cross_attention.append(
                TemporalCrossAttention(dim, dec.cross_attn_heads, config.attn_dropout)
                if dec.cross_attention
                and 0 <= encoder_index < len(enc.dims)
                and enc.dims[encoder_index] == dim
                else nn.Identity()
            )

            if factor > 1:
                next_dim = dec.dims[stage + 1] if stage + 1 < len(dec.dims) else dim
                self.decoder_upsample.append(PatchExpand(dim, next_dim))
                h, w = h * factor, w * factor
                dim = next_dim
            else:
                self.decoder_upsample.append(nn.Identity())

        self.latent_dim = dim
        self.latent_shape = (h, w)
        self.out_norm = nn.LayerNorm(dim)

        # -- auxiliary head ---------------------------------------------------
        # Named ``output_head`` to match the checkpoint's module naming, and
        # listed in ``reinit_modules`` because the published backbone was
        # trained for a different channel count and cannot transfer here.
        self.output_head = nn.Linear(dim, patch * patch) if auxiliary_head else None

    # ------------------------------------------------------------------ util
    def _expand_globals(self, batch: int) -> torch.Tensor | None:
        if not self.use_globals:
            return None
        return self.global_vectors.expand(batch, -1, -1)

    # --------------------------------------------------------------- forward
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """``x``: ``(B, T, C, H, W)``."""
        if x.ndim != 5:
            raise ValueError(f"expected (B, T, C, H, W); got {tuple(x.shape)}")
        batch, t = x.shape[0], x.shape[1]
        if t != self.sequence_length:
            raise ValueError(f"expected {self.sequence_length} input frames, got {t}")

        h = self.embed(x)
        h = h + self.pos_time_enc + self.pos_space_enc
        globals_ = self._expand_globals(batch)

        # -- encode ----------------------------------------------------------
        skips: list[torch.Tensor] = []
        attention: dict[str, torch.Tensor] = {}

        for index, (resize, stage) in enumerate(
            zip(self.encoder_downsample, self.encoder_stages, strict=False)
        ):
            h = resize(h)
            if globals_ is not None:
                globals_ = self.global_resize_down[index](globals_)

            want = return_attention and index == len(self.encoder_stages) - 1
            if want:
                h, globals_, spatial = stage(h, globals_, return_attention=True)
                if spatial is not None:
                    attention["spatial"] = spatial
            else:
                h, globals_ = stage(h, globals_)
            skips.append(h)

        memory = h

        # -- decode ----------------------------------------------------------
        # Seed every lead time with the final encoded state. The temporal
        # codes and the cross-attention then differentiate them, rather than
        # an autoregressive chain that would accumulate drift across the
        # six-hour horizon.
        d = memory[:, -1:].expand(-1, self.lead_frames, -1, -1, -1)
        d = d + self.query_time_enc

        for index, (block, cross, upsample) in enumerate(
            zip(
                self.decoder_stages,
                self.cross_attention,
                self.decoder_upsample,
                strict=False,
            )
        ):
            d = block["project"](d)
            if globals_ is not None:
                globals_ = self.global_resize_up[index](globals_)

            encoder_index = len(skips) - 1 - index
            if not isinstance(cross, nn.Identity) and 0 <= encoder_index < len(skips):
                if return_attention and "temporal" not in attention:
                    d, temporal = cross(d, skips[encoder_index], return_attention=True)
                    attention["temporal"] = temporal
                else:
                    d = cross(d, skips[encoder_index])

            d, globals_ = block["stage"](d, globals_)
            d = upsample(d)

        latent = self.out_norm(d)
        if return_attention:
            return latent, attention
        return latent

    def auxiliary_precipitation(
        self, latent: torch.Tensor, height: int, width: int
    ) -> torch.Tensor:
        """Coarse deterministic estimate, for the auxiliary training loss only."""
        if self.output_head is None:
            raise RuntimeError("backbone was built without an auxiliary head")

        b, t, h, w, _ = latent.shape
        patch = self.config.patch_embed.patch_size
        out = self.output_head(latent)  # (B,T,h,w,p*p)
        out = out.view(b, t, h, w, patch, patch)
        out = out.permute(0, 1, 2, 4, 3, 5).reshape(b, t, h * patch, w * patch)
        return out[:, :, :height, :width].unsqueeze(2)  # (B,T,1,H,W)


__all__ = [
    "CuboidStage",
    "EarthformerBackbone",
    "PatchEmbed",
    "PatchExpand",
    "PatchMerge",
    "TemporalCrossAttention",
    "adapt_cuboid",
]
