import numpy as np
import pytest

from gcmc.detector import (
    add_poisson_noise,
    apply_spatial_resolution,
    blur_energy,
    collimator_fwhm,
    energy_fwhm,
    in_window,
    system_fwhm,
    window_bounds,
)


def test_energy_fwhm_scales_as_sqrt_energy():
    assert energy_fwhm(140.5, 9.5) == pytest.approx(0.095 * 140.5)
    assert energy_fwhm(4 * 140.5, 9.5) == pytest.approx(2 * energy_fwhm(140.5, 9.5))


def test_energy_blurring_preserves_the_mean():
    e = np.full(50000, 140.5)
    blurred = blur_energy(e, fwhm_ref_pct=9.5)
    assert blurred.mean() == pytest.approx(140.5, abs=0.3)
    assert blurred.std() > 4.0


def test_window_bounds_and_acceptance():
    lo, hi = window_bounds(140.5, 20)
    assert (lo, hi) == pytest.approx((126.45, 154.55))
    accepted = in_window([120.0, 130.0, 140.5, 200.0], 140.5, 20)
    assert list(accepted) == [False, True, True, False]


def test_collimator_resolution_degrades_with_distance():
    near = collimator_fwhm(0.0, 1.5, 25.0)
    far = collimator_fwhm(100.0, 1.5, 25.0)
    assert far > near > 0


def test_effective_hole_length_correction_worsens_resolution():
    """Ignoring septal penetration underestimates the collimator FWHM."""
    with_correction = collimator_fwhm(100.0, 1.5, 25.0, mu_septa_per_mm=2.6)
    no_correction = 1.5 * (25.0 + 100.0) / 25.0
    assert with_correction > no_correction


def test_system_resolution_is_a_quadrature_sum():
    assert system_fwhm(3.0, 4.0) == pytest.approx(5.0)


def test_spatial_blur_conserves_total_counts():
    img = np.zeros((32, 32))
    img[16, 16] = 100.0
    blurred = apply_spatial_resolution(img, fwhm_mm=8.0, pixel_size_mm=4.0)
    assert blurred.sum() == pytest.approx(100.0, rel=1e-3)
    assert blurred.max() < 100.0


def test_poisson_noise_hits_the_requested_count_level():
    img = np.ones((16, 16))
    noisy = add_poisson_noise(img, total_counts=1e5, seed=0)
    assert noisy.sum() == pytest.approx(1e5, rel=0.02)
