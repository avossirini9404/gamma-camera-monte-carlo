"""Detector response model.

Design decision: transport and energy deposition happen in GAMOS; the detector
response is applied here, in Python, from the per-hit output.

Three reasons this split is deliberate rather than convenient:

1. Sweeping the energy resolution, the intrinsic spatial resolution or the
   acquisition window no longer requires re-running the Monte Carlo. One
   expensive transport run supports an entire parameter study.
2. Every response parameter becomes an explicit, versioned, testable value
   instead of a constant buried in a simulation macro.
3. The analysis chain can be exercised in continuous integration on a machine
   with no Geant4 installation.

The cost is that depth-of-interaction effects on the spatial response are not
captured beyond what the scored interaction position already contains; this is
stated in docs/model_card.md.
"""

from __future__ import annotations

import numpy as np

FWHM_TO_SIGMA = 1.0 / 2.3548200450309493


def energy_fwhm(energy_keV, fwhm_ref_pct: float, e_ref_keV: float = 140.5):
    """Detector energy FWHM in keV, scaling as 1/sqrt(E).

    Statistical fluctuation of the scintillation photon yield dominates the
    energy resolution of a NaI(Tl) camera, giving FWHM/E proportional to
    1/sqrt(E) and therefore FWHM proportional to sqrt(E).
    """
    e = np.asarray(energy_keV, dtype=float)
    fwhm_ref_keV = fwhm_ref_pct / 100.0 * e_ref_keV
    return fwhm_ref_keV * np.sqrt(np.maximum(e, 0.0) / e_ref_keV)


def blur_energy(energies, fwhm_ref_pct: float, e_ref_keV: float = 140.5, seed: int = 42):
    """Apply the energy resolution to ideal deposited energies."""
    e = np.asarray(energies, dtype=float)
    sigma = energy_fwhm(e, fwhm_ref_pct, e_ref_keV) * FWHM_TO_SIGMA
    return e + np.random.default_rng(seed).normal(0.0, np.maximum(sigma, 1e-12))


def window_bounds(centre_keV: float, width_pct: float) -> tuple[float, float]:
    half = centre_keV * width_pct / 200.0
    return centre_keV - half, centre_keV + half


def in_window(energies, centre_keV: float, width_pct: float) -> np.ndarray:
    lo, hi = window_bounds(centre_keV, width_pct)
    e = np.asarray(energies, dtype=float)
    return (e >= lo) & (e <= hi)


def collimator_fwhm(distance_mm: float, hole_diameter_mm: float,
                    hole_length_mm: float, mu_septa_per_mm: float = 2.6) -> float:
    """Geometric collimator resolution for a parallel-hole collimator.

    R_g = d (l_eff + b) / l_eff, with l_eff = l - 2/mu the effective hole
    length corrected for septal penetration. The correction matters: ignoring
    it underestimates the system resolution by several per cent.
    """
    l_eff = hole_length_mm - 2.0 / mu_septa_per_mm
    if l_eff <= 0:
        raise ValueError("effective hole length is non-positive")
    return hole_diameter_mm * (l_eff + distance_mm) / l_eff


def system_fwhm(collimator_fwhm_mm: float, intrinsic_fwhm_mm: float) -> float:
    """Quadrature sum of the collimator and intrinsic contributions."""
    return float(np.hypot(collimator_fwhm_mm, intrinsic_fwhm_mm))


def apply_spatial_resolution(projection, fwhm_mm: float, pixel_size_mm: float):
    """Blur a projection with a Gaussian of the given FWHM."""
    from scipy.ndimage import gaussian_filter

    sigma_px = fwhm_mm * FWHM_TO_SIGMA / pixel_size_mm
    return gaussian_filter(np.asarray(projection, dtype=float), sigma=sigma_px)


def add_poisson_noise(projection, total_counts: float, seed: int = 42):
    """Scale a projection to a total count level and sample Poisson noise.

    Count level is an acquisition parameter, not a detail: contrast recovery
    and background variability move in opposite directions with it, so no
    image-quality metric in this repository is reported without it.
    """
    p = np.asarray(projection, dtype=float)
    if p.sum() <= 0:
        return np.zeros_like(p)
    scaled = p * (total_counts / p.sum())
    return np.random.default_rng(seed).poisson(scaled).astype(float)
