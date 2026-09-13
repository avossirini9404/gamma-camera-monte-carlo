"""Region-of-interest placement for the image-quality analysis.

ROI geometry is part of the measurement, not a display choice: NEMA-style
sphere ROIs have the physical diameter of the sphere, and background ROIs have
the same diameter as the sphere they are compared against. Using one fixed ROI
size for all six spheres changes every recovery coefficient.
"""

from __future__ import annotations

import numpy as np


def circular_roi(shape, centre_px, diameter_px: float) -> np.ndarray:
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    r2 = (yy - centre_px[0]) ** 2 + (xx - centre_px[1]) ** 2
    return r2 <= (diameter_px / 2.0) ** 2


def sphere_centres_px(shape, centre_circle_diameter_mm: float, pixel_size_mm: float,
                      n_spheres: int = 6) -> list[tuple[float, float]]:
    cy, cx = (shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0
    r_px = centre_circle_diameter_mm / 2.0 / pixel_size_mm
    angles = np.deg2rad(np.arange(n_spheres) * 360.0 / n_spheres)
    # Row index runs along the phantom x axis and the column index along y
    # (NumPy "ij" indexing), matching gcmc.phantom. Swapping the two silently
    # places every ROI on the wrong sphere and produces a recovery curve that
    # looks plausible but is not ordered by diameter.
    return [(cy + r_px * np.cos(a), cx + r_px * np.sin(a)) for a in angles]


def background_rois(shape, diameter_px: float, n_rois: int = 12,
                    radius_fraction: float = 0.55, seed: int = 42) -> list[np.ndarray]:
    """Background ROIs on a ring, avoiding the sphere positions.

    NEMA places background ROIs across the phantom; here they are distributed
    on a ring offset from the sphere ring so that background variability is not
    measured inside the spheres' partial-volume halo.
    """
    cy, cx = (shape[0] - 1) / 2.0, (shape[1] - 1) / 2.0
    r_px = radius_fraction * min(shape) / 2.0
    offset = np.deg2rad(360.0 / n_rois / 2.0)
    angles = np.deg2rad(np.arange(n_rois) * 360.0 / n_rois) + offset
    return [
        circular_roi(shape, (cy + r_px * np.sin(a), cx + r_px * np.cos(a)), diameter_px)
        for a in angles
    ]
