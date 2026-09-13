"""Analytic attenuated forward projector — a SURROGATE, never a result.

Purpose: let the entire Python chain (detector response, projections,
reconstruction, ROI analysis, metrics, uncertainty) run and be unit tested
without a Geant4/GAMOS installation, and give the metrics a case with a known
answer.

What it does model: geometric projection, photon attenuation along the ray,
Poisson counting statistics.
What it does NOT model: scattered photons, septal penetration, collimator
scatter, backscatter from the light guide and PMTs, depth of interaction.
Those are exactly the effects the Monte Carlo exists to quantify, which is why
no number produced here may be reported as a simulation result. Every figure
generated from this module is watermarked as a surrogate by
``scripts/analyse.py``.
"""

from __future__ import annotations

import numpy as np


def _rotate(volume: np.ndarray, angle_deg: float) -> np.ndarray:
    from scipy.ndimage import rotate

    if angle_deg % 360 == 0:
        return volume
    return rotate(volume, angle_deg, axes=(0, 1), reshape=False, order=1,
                  mode="constant", cval=0.0)


def attenuated_projection(activity, mu, voxel_size_mm, angle_deg=0.0) -> np.ndarray:
    """Parallel-beam projection along +y after rotating by ``angle_deg``.

    The attenuation integral is taken from each voxel to the detector side
    only, which is what a real projection measures; integrating over the whole
    ray is a classic sign error that makes small structures look deeper than
    they are.
    """
    a = _rotate(np.asarray(activity, dtype=np.float64), angle_deg)
    m = _rotate(np.asarray(mu, dtype=np.float64), angle_deg)

    # Detector sits at the +y end; accumulate mu from each voxel outwards.
    mu_beyond = np.cumsum(m[:, ::-1, :], axis=1)[:, ::-1, :] - m
    attenuation = np.exp(-mu_beyond * voxel_size_mm)
    return (a * attenuation).sum(axis=1)


def sinogram(activity, mu, voxel_size_mm, angles_deg) -> np.ndarray:
    """Stack of projections, shape (n_angles, nx, nz)."""
    return np.stack(
        [attenuated_projection(activity, mu, voxel_size_mm, ang) for ang in angles_deg]
    )
