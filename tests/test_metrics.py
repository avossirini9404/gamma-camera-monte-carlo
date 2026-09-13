import numpy as np
import pytest

from gcmc.metrics import (
    analyse_reconstructed_slice,
    background_variability,
    contrast_recovery_coefficient,
    contrast_to_noise,
    relative_activity_recovery,
)
from gcmc.rois import background_rois, circular_roi, sphere_centres_px


def test_perfect_recovery_is_100_percent():
    assert contrast_recovery_coefficient(sphere_mean=8.0, background_mean=1.0,
                             activity_ratio_true=8.0) == pytest.approx(100.0)
    assert relative_activity_recovery(8.0, 1.0, 8.0) == pytest.approx(1.0)


def test_no_contrast_gives_zero_recovery():
    assert contrast_recovery_coefficient(1.0, 1.0, 8.0) == pytest.approx(0.0)


def test_partial_recovery_is_between_zero_and_one():
    rc = relative_activity_recovery(sphere_mean=4.0, background_mean=1.0,
                                    activity_ratio_true=8.0)
    assert 0.0 < rc < 1.0


def test_background_variability_is_zero_for_a_uniform_background():
    assert background_variability([5.0] * 12) == pytest.approx(0.0)


def test_contrast_to_noise_scales_with_the_background_spread():
    assert contrast_to_noise(8.0, 1.0, 1.0) > contrast_to_noise(8.0, 1.0, 2.0)


def test_roi_diameter_controls_the_number_of_pixels():
    small = circular_roi((64, 64), (32, 32), 4.0)
    large = circular_roi((64, 64), (32, 32), 10.0)
    assert 0 < small.sum() < large.sum()


def test_background_rois_do_not_all_coincide():
    rois = background_rois((64, 64), diameter_px=5.0, n_rois=12)
    assert len(rois) == 12
    assert not np.array_equal(rois[0], rois[6])


def test_analysis_returns_one_result_per_sphere():
    image = np.ones((64, 64))
    centres = sphere_centres_px(image.shape, 114.4, 4.0)
    results = analyse_reconstructed_slice(image, centres, (10, 13, 17, 22, 28, 37),
                                          pixel_size_mm=4.0,
                                          activity_ratio_true=8.0,
                                          domain="reconstructed")
    assert len(results) == 6
    # A uniform image has no contrast, so recovery must be zero everywhere.
    assert all(r.contrast_recovery_coefficient_pct == pytest.approx(0.0, abs=1e-6)
               for r in results)


def test_projection_domain_is_refused():
    image = np.ones((64, 64))
    centres = sphere_centres_px(image.shape, 114.4, 4.0)
    with pytest.raises(ValueError, match="reconstructed"):
        analyse_reconstructed_slice(image, centres, (10,), 4.0, 8.0,
                                    domain="projection")
