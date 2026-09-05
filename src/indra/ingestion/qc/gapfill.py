"""Temporal gap filling by dense optical flow."""

from __future__ import annotations

import logging

import numpy as np

from ...config import GapFilling, OpticalFlowParams, SplineParams
from ...types import FloatArray, InterpolationKind, QCFlag

logger = logging.getLogger(__name__)

#: Fraction of a frame that must be finite for it to anchor a flow estimate.
#: Below this the motion field is fitted to fragments and extrapolates wildly.
_MIN_ANCHOR_COVERAGE = 0.25


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_uint8(field: FloatArray) -> np.ndarray:
    """Render a float field to 8-bit for flow estimation only."""
    finite = np.isfinite(field)
    if not finite.any():
        return np.zeros(field.shape, dtype=np.uint8)

    values = field[finite]
    lo, hi = np.percentile(values, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return np.zeros(field.shape, dtype=np.uint8)

    scaled = (field - lo) / (hi - lo)
    scaled = np.where(finite, scaled, 0.0)
    return (np.clip(scaled, 0.0, 1.0) * 255.0).astype(np.uint8)


def _coverage(field: FloatArray) -> float:
    size = field.size
    return float(np.count_nonzero(np.isfinite(field))) / size if size else 0.0


def _estimate_flow(
    frame_a: FloatArray, frame_b: FloatArray, params: OpticalFlowParams
) -> np.ndarray | None:
    """Dense motion field from ``frame_a`` to ``frame_b``, or ``None``."""
    try:
        import cv2
    except ImportError:
        logger.error("opencv-python-headless is required for optical-flow gap filling")
        return None

    try:
        return cv2.calcOpticalFlowFarneback(
            _to_uint8(frame_a),
            _to_uint8(frame_b),
            None,
            params.pyramid_scale,
            params.levels,
            params.window_size,
            params.iterations,
            params.poly_n,
            params.poly_sigma,
            0,
        )
    except Exception as exc:
        logger.error("optical flow estimation failed: %s", exc)
        return None


def _warp(field: FloatArray, flow: np.ndarray, fraction: float) -> FloatArray | None:
    """Advect a field along a fraction of a motion field."""
    try:
        import cv2
    except ImportError:
        return None

    height, width = field.shape
    grid_x, grid_y = np.meshgrid(
        np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32)
    )
    map_x = (grid_x + fraction * flow[..., 0]).astype(np.float32)
    map_y = (grid_y + fraction * flow[..., 1]).astype(np.float32)

    finite = np.isfinite(field)
    filled = np.where(finite, field, 0.0).astype(np.float32)

    warped = cv2.remap(
        filled,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    warped_mask = cv2.remap(
        finite.astype(np.float32),
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )

    out = np.where(warped_mask >= 0.5, warped, np.nan).astype(np.float32)
    return out


def _spline_between(
    frame_a: FloatArray | None,
    frame_b: FloatArray | None,
    fraction: float,
    params: SplineParams,
) -> FloatArray | None:
    """Fallback when a motion field cannot be estimated."""
    if frame_a is not None and frame_b is not None:
        return ((1.0 - fraction) * frame_a + fraction * frame_b).astype(np.float32)
    if frame_a is not None:
        return frame_a.astype(np.float32, copy=True)
    if frame_b is not None:
        return frame_b.astype(np.float32, copy=True)
    return None


def _runs_of_missing(observed: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous runs of missing frames as ``(start, length)``."""
    missing = ~np.asarray(observed, dtype=bool)
    if not missing.any():
        return []
    padded = np.concatenate(([False], missing, [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [
        (int(edges[i]), int(edges[i + 1] - edges[i])) for i in range(0, len(edges), 2)
    ]


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def assess(observed: np.ndarray, config: GapFilling) -> tuple[bool, str]:
    """Decide whether a sequence is reconstructable at all."""
    if not observed.any():
        return False, "no observed frames in the sequence"

    runs = _runs_of_missing(observed)
    if not runs:
        return True, "complete"

    longest = max(length for _, length in runs)
    if (
        config.reject_sequence_if_exceeded
        and longest > config.max_consecutive_missing_frames
    ):
        return False, (
            f"longest gap is {longest} frames, over the limit of "
            f"{config.max_consecutive_missing_frames}"
        )
    total = sum(length for _, length in runs)
    return True, f"{total} frames missing, longest run {longest}"


def fill_stack(
    stack: np.ndarray,
    observed: np.ndarray,
    config: GapFilling,
    reference_channel: int = 0,
) -> tuple[np.ndarray, list[QCFlag], dict]:
    """Reconstruct missing frames in an assembled lookback stack."""
    single_channel = stack.ndim == 3
    work = stack[:, None] if single_channel else stack
    _n_time, n_chan = work.shape[0], work.shape[1]

    observed = np.asarray(observed, dtype=bool)
    flags = [QCFlag.OK if ok else QCFlag.MISSING_FILE for ok in observed]
    report: dict = {"filled": [], "failed": [], "method": {}}

    acceptable, reason = assess(observed, config)
    report["assessment"] = reason
    if not config.enabled:
        report["assessment"] = "gap filling disabled"
        return stack, flags, report
    if not acceptable:
        logger.warning("sequence rejected for gap filling: %s", reason)
        report["rejected"] = True
        for index in np.flatnonzero(~observed):
            flags[index] |= QCFlag.GAP_FILL_FAILED
        return stack, flags, report

    filled = work.copy()
    valid_indices = np.flatnonzero(observed)

    for index in np.flatnonzero(~observed):
        before = valid_indices[valid_indices < index]
        after = valid_indices[valid_indices > index]
        left = int(before[-1]) if before.size else None
        right = int(after[0]) if after.size else None

        method = InterpolationKind.NONE
        succeeded = False

        if left is not None and right is not None:
            fraction = (index - left) / (right - left)
            ref_a = work[left, reference_channel]
            ref_b = work[right, reference_channel]

            use_flow = (
                config.strategy is not None
                and config.strategy.value == "optical_flow"
                and _coverage(ref_a) >= _MIN_ANCHOR_COVERAGE
                and _coverage(ref_b) >= _MIN_ANCHOR_COVERAGE
            )

            if use_flow:
                # Both directions, so each anchor is advected toward the gap
                # rather than one being dragged the whole way. A single
                # direction accumulates error across the longer leg.
                flow_ab = _estimate_flow(ref_a, ref_b, config.optical_flow)
                flow_ba = _estimate_flow(ref_b, ref_a, config.optical_flow)

                if flow_ab is not None and flow_ba is not None:
                    for channel in range(n_chan):
                        forward = _warp(work[left, channel], flow_ab, fraction)
                        backward = _warp(work[right, channel], flow_ba, 1.0 - fraction)
                        if forward is None and backward is None:
                            continue
                        if forward is None:
                            blended = backward
                        elif backward is None:
                            blended = forward
                        else:
                            # Weight each anchor by its temporal proximity;
                            # where only one advected cleanly, take it whole.
                            both = np.isfinite(forward) & np.isfinite(backward)
                            blended = np.where(
                                both,
                                (1.0 - fraction) * np.nan_to_num(forward)
                                + fraction * np.nan_to_num(backward),
                                np.where(np.isfinite(forward), forward, backward),
                            ).astype(np.float32)
                        filled[index, channel] = blended
                    method = InterpolationKind.OPTICAL_FLOW
                    succeeded = True

            if not succeeded:
                for channel in range(n_chan):
                    blended = _spline_between(
                        work[left, channel],
                        work[right, channel],
                        fraction,
                        config.spline,
                    )
                    if blended is not None:
                        filled[index, channel] = blended
                method = InterpolationKind.SPLINE
                succeeded = True

        elif left is not None or right is not None:
            # One-sided: hold the nearest observation. There is no motion
            # field to estimate from a single frame, and extrapolating one
            # would be inventing a future.
            anchor = left if left is not None else right
            for channel in range(n_chan):
                filled[index, channel] = work[anchor, channel]
            method = InterpolationKind.HOLD
            succeeded = True

        if succeeded:
            flags[index] = (
                QCFlag.GAP_FILLED_OPTICAL_FLOW
                if method is InterpolationKind.OPTICAL_FLOW
                else QCFlag.GAP_FILLED_SPLINE
            )
            report["filled"].append(index)
            report["method"][index] = method.value
            logger.info(
                "frame %d reconstructed by %s from frames %s/%s",
                index,
                method.value,
                left,
                right,
            )
        else:
            flags[index] |= QCFlag.GAP_FILL_FAILED
            report["failed"].append(index)
            logger.warning("frame %d could not be reconstructed", index)

    out = filled[:, 0] if single_channel else filled
    report["n_filled"] = len(report["filled"])
    report["n_failed"] = len(report["failed"])
    return out, flags, report


__all__ = ["assess", "fill_stack"]
