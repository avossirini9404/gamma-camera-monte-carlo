"""Energy-spectrum analysis, including the primary/scatter decomposition.

Separating primary from scattered counts inside the acquisition window is the
main reason to run a Monte Carlo at all: no measurement can do it. Every hit
carries its Compton interaction count, so the decomposition is exact by
construction rather than estimated.
"""

from __future__ import annotations

import numpy as np


def energy_histogram(energies, bin_width_keV: float = 1.0, e_max_keV: float = 400.0):
    edges = np.arange(0.0, e_max_keV + bin_width_keV, bin_width_keV)
    counts, _ = np.histogram(np.asarray(energies, dtype=float), bins=edges)
    return 0.5 * (edges[:-1] + edges[1:]), counts


def decompose(energies, n_compton, bin_width_keV: float = 1.0, e_max_keV: float = 400.0):
    """Return (centres, total, primary, scattered) histograms."""
    e = np.asarray(energies, dtype=float)
    n = np.asarray(n_compton)
    centres, total = energy_histogram(e, bin_width_keV, e_max_keV)
    _, primary = energy_histogram(e[n == 0], bin_width_keV, e_max_keV)
    _, scattered = energy_histogram(e[n > 0], bin_width_keV, e_max_keV)
    return centres, total, primary, scattered


def scatter_fraction(energies, n_compton, centre_keV: float, width_pct: float) -> float:
    """Scattered fraction of the counts accepted by the acquisition window."""
    from gcmc.detector import in_window

    accepted = in_window(energies, centre_keV, width_pct)
    if accepted.sum() == 0:
        return float("nan")
    scattered = accepted & (np.asarray(n_compton) > 0)
    return float(scattered.sum() / accepted.sum())


def photopeak_fwhm(centres, counts) -> float:
    """FWHM of the photopeak by linear interpolation at half maximum."""
    centres = np.asarray(centres, dtype=float)
    counts = np.asarray(counts, dtype=float)
    if counts.max() <= 0:
        return float("nan")
    peak = int(np.argmax(counts))
    half = counts[peak] / 2.0

    def crossing(indices):
        for i in indices:
            j = i + 1 if i < peak else i - 1
            if (counts[i] - half) * (counts[j] - half) <= 0:
                x0, x1, y0, y1 = centres[i], centres[j], counts[i], counts[j]
                return x0 if y1 == y0 else x0 + (half - y0) * (x1 - x0) / (y1 - y0)
        return float("nan")

    return float(crossing(range(peak, len(counts) - 1)) - crossing(range(peak, 0, -1)))


def photopeak_position(centres, counts) -> float:
    counts = np.asarray(counts, dtype=float)
    return float(np.asarray(centres, dtype=float)[int(np.argmax(counts))])
