"""Validation of the camera model itself, before any image-quality claim.

The order matters. Recovery coefficients from a model whose energy resolution,
sensitivity and spatial resolution have not been checked are numbers without a
referent. These four checks are the acceptance tests of the simulated camera,
and they mirror the acceptance tests done on a physical one.
"""

from __future__ import annotations

import numpy as np


def energy_resolution_pct(centres, counts, reference_keV: float = 140.5) -> float:
    """Recovered energy resolution, in per cent of the photopeak energy."""
    from gcmc.spectra import photopeak_fwhm, photopeak_position

    fwhm = photopeak_fwhm(centres, counts)
    position = photopeak_position(centres, counts)
    if not np.isfinite(fwhm) or position <= 0:
        return float("nan")
    return float(100.0 * fwhm / position)


def sensitivity_cps_per_MBq(window_counts: float, live_time_s: float,
                            activity_MBq: float) -> float:
    """Planar sensitivity for a point source, the standard QC figure."""
    if live_time_s <= 0 or activity_MBq <= 0:
        return float("nan")
    return float(window_counts / live_time_s / activity_MBq)


def profile_fwhm(positions_mm, values) -> float:
    """FWHM of a line-source profile by linear interpolation."""
    x = np.asarray(positions_mm, dtype=float)
    y = np.asarray(values, dtype=float)
    if y.max() <= 0:
        return float("nan")
    peak = int(np.argmax(y))
    half = y[peak] / 2.0

    def crossing(indices):
        for i in indices:
            j = i + 1 if i < peak else i - 1
            if (y[i] - half) * (y[j] - half) <= 0:
                if y[j] == y[i]:
                    return x[i]
                return x[i] + (half - y[i]) * (x[j] - x[i]) / (y[j] - y[i])
        return float("nan")

    return float(crossing(range(peak, len(y) - 1)) - crossing(range(peak, 0, -1)))


def compare_to_analytical(measured_mm: float, analytical_mm: float) -> dict:
    """Percentage difference between a simulated and an analytical value.

    Agreement is reported as a number with a sign, not as two curves plotted on
    top of each other.
    """
    if analytical_mm == 0:
        return {"measured": measured_mm, "analytical": analytical_mm,
                "difference_pct": float("nan")}
    return {"measured": float(measured_mm), "analytical": float(analytical_mm),
            "difference_pct": float(100.0 * (measured_mm - analytical_mm)
                                    / analytical_mm)}
