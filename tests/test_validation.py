import numpy as np
import pytest

from gcmc.detector import blur_energy, collimator_fwhm, system_fwhm
from gcmc.spectra import energy_histogram
from gcmc.validation import (
    compare_to_analytical,
    energy_resolution_pct,
    profile_fwhm,
    sensitivity_cps_per_MBq,
)


def test_energy_resolution_round_trip():
    e = blur_energy(np.full(300000, 140.5), fwhm_ref_pct=9.5, seed=3)
    centres, counts = energy_histogram(e, bin_width_keV=0.5, e_max_keV=250.0)
    assert energy_resolution_pct(centres, counts) == pytest.approx(9.5, rel=0.12)


def test_sensitivity_is_counts_per_time_per_activity():
    assert sensitivity_cps_per_MBq(3.0e5, 300.0, 10.0) == pytest.approx(100.0)


def test_profile_fwhm_matches_a_known_gaussian():
    x = np.linspace(-50, 50, 1001)
    fwhm_true = 10.0
    sigma = fwhm_true / 2.3548
    y = np.exp(-0.5 * (x / sigma) ** 2)
    assert profile_fwhm(x, y) == pytest.approx(fwhm_true, rel=0.02)


def test_comparison_reports_a_signed_difference():
    out = compare_to_analytical(measured_mm=11.0, analytical_mm=10.0)
    assert out["difference_pct"] == pytest.approx(10.0)


def test_system_resolution_is_dominated_by_the_collimator_at_distance():
    """The intrinsic term adds in quadrature, so its weight falls with distance."""
    near = collimator_fwhm(50.0, 1.5, 25.0)
    far = collimator_fwhm(300.0, 1.5, 25.0)
    penalty_near = system_fwhm(near, 3.8) / near
    penalty_far = system_fwhm(far, 3.8) / far
    assert penalty_far < penalty_near
    assert penalty_far < 1.02
