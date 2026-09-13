"""Image-quality metrics for the six-sphere phantom.

Two scope rules, both enforced in code rather than left to the reader.

**Domain.** These figures are defined on reconstructed tomographic images.
``analyse_reconstructed_slice`` refuses any other domain. On a planar
projection of this phantom the inserts overlap one another along the
integration axis, so no fixed ROI pattern separates them and the metrics are
not merely biased — they are undefined.

**Terminology.** The literature uses "recovery coefficient" for several
different quantities (RC_max, RC_mean, RC_peak, contrast recovery
coefficient). This module names the two it computes explicitly:

``contrast_recovery_coefficient_pct``
    NEMA NU 2 hot-sphere form, Q = [(C_s/C_b) - 1] / [(a_s/a_b) - 1].
``relative_activity_recovery``
    the measured-to-true concentration ratio, (C_s/C_b) / (a_s/a_b).

Neither is called simply "RC", because that name does not identify a
quantity.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SphereResult:
    diameter_mm: float
    domain: str                       # "projection" or "reconstructed"
    sphere_mean: float
    background_mean: float
    background_sd: float
    activity_ratio_true: float
    contrast_recovery_coefficient_pct: float
    relative_activity_recovery: float
    background_variability_pct: float
    contrast_to_noise: float = float("nan")
    counts_in_roi: float = float("nan")
    extra: dict = field(default_factory=dict)


def contrast_recovery_coefficient(sphere_mean: float, background_mean: float,
                                  activity_ratio_true: float) -> float:
    """Hot-sphere contrast recovery coefficient, in per cent, NEMA NU 2 form.

    Q = [(C_sphere / C_background) - 1] / [(a_sphere / a_background) - 1]

    Perfect recovery is 100%. The denominator is the *built* activity ratio,
    read from the digital phantom, not the nominal one from the config file:
    voxelisation of a 10 mm sphere on a 4 mm grid does not reproduce the
    nominal ratio exactly.
    """
    if background_mean == 0 or activity_ratio_true == 1:
        return float("nan")
    return 100.0 * ((sphere_mean / background_mean) - 1.0) / (activity_ratio_true - 1.0)


def relative_activity_recovery(sphere_mean: float, background_mean: float,
                               activity_ratio_true: float) -> float:
    """Measured-to-true activity concentration ratio (1.0 = full recovery).

    Deliberately not named "recovery coefficient": that term is used in the
    literature for several distinct quantities, and an unqualified RC in a
    results table is not reproducible.
    """
    if background_mean == 0 or activity_ratio_true == 0:
        return float("nan")
    return float((sphere_mean / background_mean) / activity_ratio_true)


def background_variability(background_roi_means) -> float:
    """Coefficient of variation of the background ROIs, in per cent."""
    m = np.asarray(background_roi_means, dtype=float)
    if m.size < 2 or m.mean() == 0:
        return float("nan")
    return float(100.0 * m.std(ddof=1) / m.mean())


def contrast_to_noise(sphere_mean: float, background_mean: float,
                      background_sd: float) -> float:
    if background_sd == 0:
        return float("nan")
    return float((sphere_mean - background_mean) / background_sd)


def analyse_reconstructed_slice(image, sphere_centres_px, sphere_diameters_mm,
                                pixel_size_mm, activity_ratio_true: float,
                                domain: str = "reconstructed",
                                n_background_rois: int = 12) -> list[SphereResult]:
    """Six-sphere analysis of one reconstructed transaxial slice.

    Raises on any other domain. The guard is not defensive programming: on a
    planar projection of this phantom the six inserts overlap along the
    integration axis, so a per-sphere ROI analysis there measures a
    superposition, not a sphere.
    """
    from gcmc.rois import background_rois, circular_roi

    if domain != "reconstructed":
        raise ValueError(
            "sphere image-quality metrics are defined on reconstructed images; "
            f"got domain={domain!r}. Use gcmc.planar for projection-domain "
            "analysis (profiles, sensitivity, scatter fraction, resolution)."
        )

    image = np.asarray(image, dtype=float)
    results = []
    for centre, diameter_mm in zip(sphere_centres_px, sphere_diameters_mm):
        diameter_px = diameter_mm / pixel_size_mm
        sphere_mask = circular_roi(image.shape, centre, diameter_px)
        if sphere_mask.sum() == 0:                    # sphere smaller than a pixel
            sphere_mask = circular_roi(image.shape, centre, 1.0)
        bg_masks = background_rois(image.shape, diameter_px, n_background_rois)
        bg_means = [image[m].mean() for m in bg_masks if m.sum() > 0]
        bg_mean = float(np.mean(bg_means)) if bg_means else float("nan")
        bg_sd = float(np.std(bg_means, ddof=1)) if len(bg_means) > 1 else float("nan")
        sphere_mean = float(image[sphere_mask].mean())
        results.append(
            SphereResult(
                diameter_mm=diameter_mm,
                domain=domain,
                sphere_mean=sphere_mean,
                background_mean=bg_mean,
                background_sd=bg_sd,
                activity_ratio_true=activity_ratio_true,
                contrast_recovery_coefficient_pct=contrast_recovery_coefficient(
                    sphere_mean, bg_mean, activity_ratio_true),
                relative_activity_recovery=relative_activity_recovery(
                    sphere_mean, bg_mean, activity_ratio_true),
                background_variability_pct=background_variability(bg_means),
                contrast_to_noise=contrast_to_noise(sphere_mean, bg_mean, bg_sd),
                counts_in_roi=float(image[sphere_mask].sum()),
            )
        )
    return results
