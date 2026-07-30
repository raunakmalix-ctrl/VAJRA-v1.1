"""
vision — frame-level helpers for windowed lip-sync.

Counterpart to dsp/: where dsp/ makes a regenerated audio span indistinguishable
from its neighbours, this makes a regenerated video window indistinguishable from
the surrounding footage -- by regenerating as little of it as possible.

Only the time windows that actually changed are lip-synced, and the result is
composited into the ORIGINAL frames through a feathered mouth mask with a
temporal ramp at each window edge. Everything outside the mask and outside the
windows is original pixels.

Pure numpy; no model and no GPU.
"""
from vision.composite import (
    frame_range, merge_windows, coverage,
    feathered_box_mask, feathered_ellipse_mask,
    mouth_box, mouth_box_from_landmarks, changed_region_box,
    temporal_weight, blend, composite_window,
    DEFAULT_MOUTH_TOP, DEFAULT_FEATHER_FRAC, DEFAULT_RAMP_FRAMES,
)

__all__ = [
    "frame_range", "merge_windows", "coverage",
    "feathered_box_mask", "feathered_ellipse_mask",
    "mouth_box", "mouth_box_from_landmarks", "changed_region_box",
    "temporal_weight", "blend", "composite_window",
    "DEFAULT_MOUTH_TOP", "DEFAULT_FEATHER_FRAC", "DEFAULT_RAMP_FRAMES",
]
