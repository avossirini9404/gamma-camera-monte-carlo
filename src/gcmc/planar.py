"""Projection-domain analysis.

What a planar acquisition of this phantom legitimately supports: energy
spectra, window counts and sensitivity, scatter fraction, system spatial
resolution from a line or point source, and projection profiles.

What it does not support: per-sphere image-quality metrics. The six inserts lie
on a circle in the transaxial plane and a planar projection integrates along
one of those axes, so the inserts superimpose. Any ROI placed on a projection
of this phantom measures a superposition of inserts and background, and a
"recovery" computed from it is a number about the ROI pattern, not about a
sphere. :func:`gcmc.metrics.analyse_reconstructed_slice` raises rather than
letting that happen by accident.
"""

from __future__ import annotations

import numpy as np


def line_profile(projection, index: int, axis: int = 0) -> np.ndarray:
    """One row or column through a projection image."""
    p = np.asarray(projection, dtype=float)
    return p[index, :] if axis == 0 else p[:, index]


def profile_positions_mm(n_bins: int, pixel_size_mm: float) -> np.ndarray:
    return (np.arange(n_bins) - (n_bins - 1) / 2.0) * pixel_size_mm


def total_window_counts(projection) -> float:
    """Counts accepted into the image; the input to a sensitivity figure."""
    return float(np.asarray(projection, dtype=float).sum())


def radial_profile(projection, n_bins: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Azimuthally averaged profile about the image centre."""
    p = np.asarray(projection, dtype=float)
    cy, cx = (p.shape[0] - 1) / 2.0, (p.shape[1] - 1) / 2.0
    yy, xx = np.mgrid[0:p.shape[0], 0:p.shape[1]]
    r = np.hypot(yy - cy, xx - cx)
    edges = np.linspace(0, r.max(), n_bins + 1)
    idx = np.clip(np.digitize(r, edges) - 1, 0, n_bins - 1)
    means = np.array([p[idx == b].mean() if (idx == b).any() else np.nan
                      for b in range(n_bins)])
    return 0.5 * (edges[:-1] + edges[1:]), means


def uniformity_nema(projection, mask=None) -> float:
    """Integral uniformity, (max - min) / (max + min), as used in camera QC."""
    p = np.asarray(projection, dtype=float)
    v = p.ravel() if mask is None else p[np.asarray(mask, dtype=bool)]
    hi, lo = v.max(), v.min()
    return float((hi - lo) / (hi + lo)) if (hi + lo) else float("nan")
