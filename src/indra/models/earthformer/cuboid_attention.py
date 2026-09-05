"""Space-time cuboid attention — the Earthformer primitive."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

Cuboid = tuple[int, int, int]
Shift = tuple[int, int, int]

# Additive mask value for disallowed attention pairs. Large and negative rather
# than -inf: -inf produces NaN when an entire row is masked, which happens for
# a fully padded token, and one NaN propagates through the whole batch.
_MASK_VALUE = -1.0e4


# ---------------------------------------------------------------------------
# Decomposition
# ---------------------------------------------------------------------------


def pad_to_multiple(x: torch.Tensor, cuboid: Cuboid) -> tuple[torch.Tensor, Cuboid]:
    """Right-pad ``(B, T, H, W, C)`` so each axis divides its cuboid extent."""
    _, t, h, w, _ = x.shape
    ct, ch, cw = cuboid
    pt, ph, pw = (-t) % ct, (-h) % ch, (-w) % cw
    if pt or ph or pw:
        # F.pad consumes dimensions from the last backwards: C, W, H, T.
        x = F.pad(x, (0, 0, 0, pw, 0, ph, 0, pt))
    return x, (pt, ph, pw)


def cuboid_partition(
    x: torch.Tensor, cuboid: Cuboid, strategy: str
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """``(B, T, H, W, C)`` -> ``(B * n_cuboids, L, C)``."""
    b, t, h, w, c = x.shape
    ct, ch, cw = cuboid
    nt, nh, nw = t // ct, h // ch, w // cw

    if strategy == "local":
        x = x.view(b, nt, ct, nh, ch, nw, cw, c)
        x = x.permute(0, 1, 3, 5, 2, 4, 6, 7)
    elif strategy == "dilated":
        x = x.view(b, ct, nt, ch, nh, cw, nw, c)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)
    else:
        raise ValueError(
            f"unknown cuboid strategy {strategy!r}; expected 'local' or 'dilated'"
        )

    return x.reshape(b * nt * nh * nw, ct * ch * cw, c), (nt, nh, nw)


def cuboid_reverse(
    x: torch.Tensor,
    grid: tuple[int, int, int],
    cuboid: Cuboid,
    strategy: str,
    batch: int,
) -> torch.Tensor:
    """Inverse of :func:`cuboid_partition`."""
    nt, nh, nw = grid
    ct, ch, cw = cuboid
    c = x.shape[-1]
    x = x.view(batch, nt, nh, nw, ct, ch, cw, c)

    if strategy == "local":
        x = x.permute(0, 1, 4, 2, 5, 3, 6, 7)
    else:  # dilated
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7)

    return x.reshape(batch, nt * ct, nh * ch, nw * cw, c)


# ---------------------------------------------------------------------------
# Position bias
# ---------------------------------------------------------------------------


def relative_position_index(cuboid: Cuboid) -> torch.Tensor:
    """``(L, L)`` index into the relative position bias table."""
    ct, ch, cw = cuboid
    coords = torch.stack(
        torch.meshgrid(
            torch.arange(ct), torch.arange(ch), torch.arange(cw), indexing="ij"
        )
    ).flatten(1)  # (3, L)

    relative = coords[:, :, None] - coords[:, None, :]  # (3, L, L)
    relative = relative.permute(1, 2, 0).contiguous()  # (L, L, 3)

    relative[..., 0] += ct - 1
    relative[..., 1] += ch - 1
    relative[..., 2] += cw - 1

    relative[..., 0] *= (2 * ch - 1) * (2 * cw - 1)
    relative[..., 1] *= 2 * cw - 1
    return relative.sum(-1)  # (L, L)


def relative_position_table_size(cuboid: Cuboid) -> int:
    ct, ch, cw = cuboid
    return (2 * ct - 1) * (2 * ch - 1) * (2 * cw - 1)


# ---------------------------------------------------------------------------
# Masks
# ---------------------------------------------------------------------------


def build_padding_mask(
    shape: tuple[int, int, int],
    padding: Cuboid,
    cuboid: Cuboid,
    strategy: str,
    device: torch.device,
) -> torch.Tensor | None:
    """``(n_cuboids, L)`` mask marking real positions, or ``None`` if unpadded."""
    if not any(padding):
        return None

    t, h, w = shape
    pt, ph, pw = padding
    real = torch.zeros(1, t + pt, h + ph, w + pw, 1, device=device)
    real[:, :t, :h, :w, :] = 1.0

    tokens, _ = cuboid_partition(real, cuboid, strategy)
    return tokens.squeeze(-1) > 0.5  # (n_cuboids, L)


def build_shift_mask(
    shape: tuple[int, int, int],
    cuboid: Cuboid,
    shift: Shift,
    strategy: str,
    device: torch.device,
) -> torch.Tensor | None:
    """``(n_cuboids, L, L)`` mask forbidding attention across a cyclic wrap."""
    if not any(shift):
        return None

    t, h, w = shape
    st, sh, sw = shift
    labels = torch.zeros(1, t, h, w, 1, device=device)

    counter = 0
    t_slices = (slice(0, -st), slice(-st, None)) if st else (slice(None),)
    h_slices = (slice(0, -sh), slice(-sh, None)) if sh else (slice(None),)
    w_slices = (slice(0, -sw), slice(-sw, None)) if sw else (slice(None),)
    for ts in t_slices:
        for hs in h_slices:
            for ws in w_slices:
                labels[:, ts, hs, ws, :] = counter
                counter += 1

    tokens, _ = cuboid_partition(labels, cuboid, strategy)  # (n, L, 1)
    tokens = tokens.squeeze(-1)  # (n, L)
    return tokens.unsqueeze(1) == tokens.unsqueeze(2)  # (n, L, L)


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------


class CuboidSelfAttention(nn.Module):
    """Multi-head self-attention restricted to cuboids, with global vectors."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        cuboid: Cuboid,
        strategy: str = "local",
        shift: Shift = (0, 0, 0),
        num_global: int = 8,
        qkv_bias: bool = True,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        use_relative_position_bias: bool = True,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} is not divisible by num_heads {num_heads}")
        if strategy not in ("local", "dilated"):
            raise ValueError(f"unknown strategy {strategy!r}")
        if any(s < 0 for s in shift):
            raise ValueError(f"shift {shift} must be non-negative")
        for axis, (extent, offset) in enumerate(zip(cuboid, shift, strict=False)):
            if offset >= extent:
                # A shift of a whole cuboid is the identity partition, so it
                # buys nothing while still paying for the wrap mask.
                raise ValueError(
                    f"shift {offset} on axis {axis} is not smaller than the "
                    f"cuboid extent {extent}; it would reproduce the unshifted "
                    "decomposition"
                )

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.cuboid = cuboid
        self.strategy = strategy
        self.shift = shift
        self.num_global = num_global
        self.use_relative_position_bias = use_relative_position_bias

        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

        if use_relative_position_bias:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros(relative_position_table_size(cuboid), num_heads)
            )
            self.register_buffer(
                "relative_position_index",
                relative_position_index(cuboid),
                persistent=False,
            )
            nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        if num_global > 0:
            # Globals are read by every cuboid and then refreshed from the
            # cuboid summaries, so they need their own projections; reusing the
            # token qkv would tie two different roles to one set of weights.
            self.global_norm = nn.LayerNorm(dim)
            self.global_kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
            self.global_q = nn.Linear(dim, dim, bias=qkv_bias)
            self.global_proj = nn.Linear(dim, dim)

    # -- helpers -----------------------------------------------------------
    def _position_bias(self, length: int) -> torch.Tensor:
        """``(1, heads, L, L)`` learned bias over within-cuboid offsets."""
        bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(length, length, self.num_heads)
        return bias.permute(2, 0, 1).unsqueeze(0).contiguous()

    def _attend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        bias: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Scaled dot-product attention over ``(N, heads, L, head_dim)``."""
        scores = (q @ k.transpose(-2, -1)) * self.scale
        if bias is not None:
            scores = scores + bias
        if mask is not None:
            scores = scores.masked_fill(~mask, _MASK_VALUE)
        weights = self.attn_drop(scores.softmax(dim=-1))
        return weights @ v, weights

    # -- forward -----------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        global_vectors: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor | None]
        | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]
    ):
        """``x``: ``(B, T, H, W, C)``. ``global_vectors``: ``(B, G, C)``."""
        batch, t, h, w, _ = x.shape
        shortcut = x
        x = self.norm(x)

        if any(self.shift):
            x = torch.roll(
                x,
                shifts=(-self.shift[0], -self.shift[1], -self.shift[2]),
                dims=(1, 2, 3),
            )

        x, padding = pad_to_multiple(x, self.cuboid)
        tokens, grid = cuboid_partition(x, self.cuboid, self.strategy)
        n_cuboids, length, _ = tokens.shape

        qkv = self.qkv(tokens).reshape(
            n_cuboids, length, 3, self.num_heads, self.head_dim
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        bias = self._position_bias(length) if self.use_relative_position_bias else None

        # -- masks ---------------------------------------------------------
        padded = build_padding_mask(
            (t, h, w), padding, self.cuboid, self.strategy, x.device
        )
        wrapped = build_shift_mask(
            (t + padding[0], h + padding[1], w + padding[2]),
            self.cuboid,
            self.shift,
            self.strategy,
            x.device,
        )

        mask: torch.Tensor | None = None
        if padded is not None:
            # A padded key is invisible to every query.
            mask = padded.unsqueeze(1).expand(-1, length, -1)
        if wrapped is not None:
            per_batch = wrapped.repeat(n_cuboids // wrapped.shape[0], 1, 1)
            mask = per_batch if mask is None else (mask & per_batch)
        if mask is not None:
            mask = mask.unsqueeze(1)  # (n, 1, L, L)

        # -- global vectors as extra keys and values ------------------------
        if global_vectors is not None and self.num_global > 0:
            repeats_per_batch = n_cuboids // batch
            g = self.global_norm(global_vectors)
            g = g.repeat_interleave(repeats_per_batch, dim=0)  # (n, G, C)
            gkv = self.global_kv(g).reshape(
                n_cuboids, self.num_global, 2, self.num_heads, self.head_dim
            )
            gk, gv = gkv.permute(2, 0, 3, 1, 4)
            k = torch.cat([k, gk], dim=2)
            v = torch.cat([v, gv], dim=2)

            if bias is not None:
                # Globals have no spatial position, so no relative bias
                # applies to them.
                bias = F.pad(bias, (0, self.num_global))
            if mask is not None:
                # Every query may read every global.
                allow = torch.ones(
                    n_cuboids,
                    1,
                    length,
                    self.num_global,
                    dtype=torch.bool,
                    device=x.device,
                )
                mask = torch.cat([mask, allow], dim=-1)

        out, weights = self._attend(q, k, v, bias, mask)
        out = out.transpose(1, 2).reshape(n_cuboids, length, self.dim)
        out = self.proj_drop(self.proj(out))

        out = cuboid_reverse(out, grid, self.cuboid, self.strategy, batch)
        out = out[:, : t + padding[0], : h + padding[1], : w + padding[2], :]

        if any(self.shift):
            out = torch.roll(
                out,
                shifts=(self.shift[0], self.shift[1], self.shift[2]),
                dims=(1, 2, 3),
            )
        out = out[:, :t, :h, :w, :]
        x_out = shortcut + out

        # -- refresh the globals -------------------------------------------
        new_globals = global_vectors
        if global_vectors is not None and self.num_global > 0:
            # Summarise each cuboid, then let the globals attend over those
            # summaries. This is the return leg: without it the globals are
            # written once at initialisation and never learn what the field
            # currently contains.
            summaries = tokens.mean(dim=1).view(batch, -1, self.dim)
            gq = self.global_q(self.global_norm(global_vectors))
            gq = gq.view(
                batch, self.num_global, self.num_heads, self.head_dim
            ).transpose(1, 2)
            sk = summaries.view(batch, -1, self.num_heads, self.head_dim).transpose(
                1, 2
            )
            updated, _ = self._attend(gq, sk, sk)
            updated = updated.transpose(1, 2).reshape(batch, self.num_global, self.dim)
            new_globals = global_vectors + self.global_proj(updated)

        if return_attention:
            # Attention *received* by each token: averaged over heads and over
            # queries, then restored to the volume's shape. This is the raw
            # material for the XAI attention map, and it is the received view
            # rather than the emitted one because the question being asked is
            # "which inputs did the model look at".
            received = weights.mean(dim=1).mean(dim=1)[:, :length]
            received = received.unsqueeze(-1)
            spatial = cuboid_reverse(received, grid, self.cuboid, self.strategy, batch)
            spatial = spatial[:, :t, :h, :w, 0]
            return x_out, new_globals, spatial
        return x_out, new_globals


class CuboidAttentionBlock(nn.Module):
    """Cuboid attention followed by a position-wise MLP, both pre-norm."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        cuboid: Cuboid,
        strategy: str = "local",
        shift: Shift = (0, 0, 0),
        num_global: int = 8,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        ffn_dropout: float = 0.0,
        activation: str = "gelu",
        use_relative_position_bias: bool = True,
    ) -> None:
        super().__init__()
        self.attn = CuboidSelfAttention(
            dim=dim,
            num_heads=num_heads,
            cuboid=cuboid,
            strategy=strategy,
            shift=shift,
            num_global=num_global,
            attn_dropout=attn_dropout,
            proj_dropout=proj_dropout,
            use_relative_position_bias=use_relative_position_bias,
        )
        self.norm = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        activations = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}
        if activation not in activations:
            raise ValueError(f"unknown activation {activation!r}")
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            activations[activation](),
            nn.Dropout(ffn_dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(ffn_dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        global_vectors: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> (
        tuple[torch.Tensor, torch.Tensor | None]
        | tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]
    ):
        if return_attention:
            x, global_vectors, attention = self.attn(
                x, global_vectors, return_attention=True
            )
            return x + self.mlp(self.norm(x)), global_vectors, attention
        x, global_vectors = self.attn(x, global_vectors)
        return x + self.mlp(self.norm(x)), global_vectors


__all__ = [
    "CuboidAttentionBlock",
    "CuboidSelfAttention",
    "build_padding_mask",
    "build_shift_mask",
    "cuboid_partition",
    "cuboid_reverse",
    "pad_to_multiple",
    "relative_position_index",
    "relative_position_table_size",
]
