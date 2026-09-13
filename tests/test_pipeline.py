"""End-to-end tests on the analytic surrogate.

These protect the scientific claims of the repository. The recovery curve test
is the regression test for the whole chain: phantom geometry, projector
convention, ROI placement, the adjoint property of the reconstruction operator
and the metric definition all have to be right at once for recovery to be
ordered by sphere diameter. Three of them were wrong at some point during
development and this test is what caught them.
"""

import numpy as np
import pytest

from gcmc.config import AcquisitionConfig, PhantomConfig, ReconstructionConfig
from gcmc.detector import add_poisson_noise
from gcmc.metrics import analyse_reconstructed_slice
from gcmc.phantom import build_phantom
from gcmc.pipeline import planar_pipeline, tomographic_pipeline
from gcmc.planar import radial_profile, total_window_counts
from gcmc.projector import Projector
from gcmc.reconstruction import log_likelihood, mlem
from gcmc.rois import sphere_centres_px
from gcmc.uncertainty import poisson_relative_uncertainty, replica_statistics

TOMO = AcquisitionConfig(mode="tomographic", n_angles=60, angular_range_deg=360.0)
RECON = ReconstructionConfig(iterations=40)
PLANAR = AcquisitionConfig(mode="planar")


@pytest.fixture(scope="module")
def phantom():
    return build_phantom(PhantomConfig())


@pytest.fixture(scope="module")
def reconstructed(phantom):
    return tomographic_pipeline(phantom, TOMO, RECON, system_fwhm_mm=12.0)


def _analyse(image, phantom):
    centres = sphere_centres_px(image.shape, 114.4, phantom.voxel_size_mm)
    return analyse_reconstructed_slice(
        image, centres, phantom.sphere_diameters_mm,
        pixel_size_mm=phantom.voxel_size_mm,
        activity_ratio_true=phantom.true_concentration_ratio(),
        domain="reconstructed")


def test_recovery_increases_with_sphere_size(phantom, reconstructed):
    from scipy import stats

    rar = [r.relative_activity_recovery for r in _analyse(reconstructed["image"],
                                                          phantom)]
    assert rar[-1] > rar[0]
    # Rank correlation, not Pearson: recovery rises with diameter but
    # saturates, so a linear correlation understates a monotonic curve.
    assert stats.spearmanr(rar, phantom.sphere_diameters_mm).statistic > 0.9
    assert all(0.0 < value < 1.0 for value in rar)


def test_sphere_metrics_are_refused_in_the_projection_domain(phantom):
    """The six inserts overlap along the integration axis of a planar view.

    No ROI pattern separates them there, so the metric is undefined rather
    than merely biased, and the code refuses instead of returning a number.
    """
    planar = planar_pipeline(phantom, PLANAR, system_fwhm_mm=12.0)
    centres = sphere_centres_px(planar["image"].shape, 114.4, 4.0)
    with pytest.raises(ValueError, match="reconstructed"):
        analyse_reconstructed_slice(planar["image"], centres,
                                    phantom.sphere_diameters_mm, 4.0, 8.0,
                                    domain=planar["domain"])


def test_planar_supports_the_quantities_it_is_valid_for(phantom):
    planar = planar_pipeline(phantom, PLANAR, system_fwhm_mm=12.0)
    assert total_window_counts(planar["image"]) > 0
    radii, profile = radial_profile(planar["image"])
    assert len(radii) == len(profile)
    assert np.isfinite(profile).any()


def test_mlem_increases_the_poisson_log_likelihood(phantom):
    """MLEM is monotone in the likelihood only if the operator pair is adjoint."""
    activity, _ = (phantom.activity[:, :, 24].astype(float), None)
    angles = np.arange(0, 180, 20.0)
    projector = Projector(activity.shape, angles)
    sinogram = projector.forward(activity)
    likelihoods = [log_likelihood(sinogram, projector,
                                  mlem(sinogram, projector, n_iterations=n))
                   for n in (1, 5, 15, 30)]
    from itertools import pairwise

    assert all(b >= a - 1e-6 for a, b in pairwise(likelihoods))


def test_reconstruction_metadata_travels_with_the_image(reconstructed):
    """Recovery is a property of the whole chain, not of the camera."""
    for key in ("system_fwhm_mm", "n_iterations", "attenuation_compensated",
                "scatter_compensated", "domain", "seed"):
        assert key in reconstructed
    assert reconstructed["attenuation_compensated"] is False


def test_more_mlem_iterations_increase_recovery(phantom):
    acquisition = AcquisitionConfig(mode="tomographic", n_angles=36)
    low = tomographic_pipeline(phantom, acquisition, ReconstructionConfig(iterations=5),
                               system_fwhm_mm=12.0)
    high = tomographic_pipeline(phantom, acquisition, ReconstructionConfig(iterations=40),
                                system_fwhm_mm=12.0)
    assert _analyse(high["image"], phantom)[-1].relative_activity_recovery > \
           _analyse(low["image"], phantom)[-1].relative_activity_recovery


def test_noise_increases_background_variability(phantom):
    clean = tomographic_pipeline(phantom, TOMO, RECON, system_fwhm_mm=12.0)
    noisy_acq = AcquisitionConfig(mode="tomographic", n_angles=60,
                                  total_counts=2e5)
    noisy = tomographic_pipeline(phantom, noisy_acq, RECON, system_fwhm_mm=12.0)
    assert _analyse(noisy["image"], phantom)[0].background_variability_pct > \
           _analyse(clean["image"], phantom)[0].background_variability_pct


def test_replica_variability_is_the_uncertainty_on_a_derived_metric(phantom):
    """Counting statistics do not give the uncertainty of a recovery figure.

    A recovery value is a ratio of correlated ROI means passed through a
    reconstruction, so its uncertainty is estimated from independent replicas
    with different seeds, not from 1/sqrt(N) in the sphere ROI.
    """
    acquisition = AcquisitionConfig(mode="tomographic", n_angles=36,
                                    total_counts=1e6)
    values = [
        _analyse(tomographic_pipeline(phantom, acquisition,
                                      ReconstructionConfig(iterations=20),
                                      system_fwhm_mm=12.0, seed=seed)["image"],
                 phantom)[-1].contrast_recovery_coefficient_pct
        for seed in range(4)
    ]
    stats = replica_statistics(values)
    assert stats["n_replicas"] == 4
    assert stats["ci_low"] < stats["mean"] < stats["ci_high"]


def test_poisson_uncertainty_is_a_planning_metric_only():
    u = poisson_relative_uncertainty([100.0, 10000.0])
    assert u[0] == pytest.approx(0.1)
    assert u[1] == pytest.approx(0.01)


def test_noise_free_projection_has_no_negative_counts(phantom):
    planar = planar_pipeline(phantom, PLANAR, system_fwhm_mm=12.0)
    assert planar["image"].min() >= 0.0


def test_add_poisson_noise_preserves_the_count_level(phantom):
    planar = planar_pipeline(phantom, PLANAR, system_fwhm_mm=12.0)
    noisy = add_poisson_noise(planar["image"], total_counts=1e6, seed=0)
    assert noisy.sum() == pytest.approx(1e6, rel=0.01)
