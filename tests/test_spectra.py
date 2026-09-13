import numpy as np
import pytest

from gcmc.detector import blur_energy
from gcmc.spectra import (
    decompose,
    energy_histogram,
    photopeak_fwhm,
    photopeak_position,
    scatter_fraction,
)


def test_histogram_conserves_counts():
    e = np.random.default_rng(0).uniform(0, 300, 1000)
    _, counts = energy_histogram(e, bin_width_keV=2.0)
    assert counts.sum() == 1000


def test_recovered_fwhm_matches_the_applied_energy_resolution():
    """Round trip: blur a monoenergetic line, recover the resolution."""
    e = blur_energy(np.full(300000, 140.5), fwhm_ref_pct=10.0, seed=1)
    centres, counts = energy_histogram(e, bin_width_keV=0.5, e_max_keV=250.0)
    assert photopeak_position(centres, counts) == pytest.approx(140.5, abs=1.0)
    assert photopeak_fwhm(centres, counts) == pytest.approx(14.05, rel=0.10)


def test_primary_and_scattered_histograms_sum_to_the_total():
    rng = np.random.default_rng(2)
    e = rng.uniform(50, 200, 5000)
    n = rng.integers(0, 3, 5000)
    _, total, primary, scattered = decompose(e, n)
    assert np.array_equal(total, primary + scattered)


def test_scatter_fraction_counts_only_in_window_events():
    e = np.array([120.0, 130.0, 140.0, 141.0, 200.0])
    n = np.array([1, 1, 0, 0, 1])
    assert scatter_fraction(e, n, 140.5, 20) == pytest.approx(1 / 3)


def test_scatter_fraction_is_nan_when_the_window_is_empty():
    assert np.isnan(scatter_fraction([300.0], [0], 140.5, 20))
