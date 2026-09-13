"""Operator tests.

The adjoint test is the one that licenses the word "MLEM" in this repository.
"""

import numpy as np
import pytest

from gcmc.projector import Projector, attenuation_weights


@pytest.fixture(scope="module")
def projector():
    return Projector(shape=(32, 32), angles_deg=np.arange(0, 180, 15.0))


def test_forward_and_back_are_exact_adjoints(projector):
    """<A x, y> == <x, A^T y> to machine precision, for random x and y."""
    rng = np.random.default_rng(0)
    x = rng.random(projector.shape)
    y = rng.random(projector.sinogram_shape)
    lhs = float(np.sum(projector.forward(x) * y))
    rhs = float(np.sum(x * projector.back(y)))
    assert lhs == pytest.approx(rhs, rel=1e-12, abs=1e-12)


def test_adjoint_holds_with_attenuation_weights():
    mu = np.full((24, 24), 0.015)
    angles = np.arange(0, 360, 45.0)
    w = attenuation_weights(mu, voxel_size_mm=4.0, angles_deg=angles)
    p = Projector((24, 24), angles, weights=w)
    rng = np.random.default_rng(1)
    x, y = rng.random(p.shape), rng.random(p.sinogram_shape)
    assert float(np.sum(p.forward(x) * y)) == pytest.approx(
        float(np.sum(x * p.back(y))), rel=1e-12, abs=1e-12)


def test_projection_conserves_mass_for_a_centred_object(projector):
    image = np.zeros(projector.shape)
    image[12:20, 12:20] = 1.0
    per_angle = projector.forward(image).sum(axis=1)
    assert np.allclose(per_angle, image.sum(), rtol=1e-6)


def test_attenuation_weights_are_survival_probabilities():
    mu = np.full((16, 16), 0.02)
    w = attenuation_weights(mu, voxel_size_mm=4.0, angles_deg=[0.0])
    assert w.min() > 0.0
    assert w.max() <= 1.0


def test_deeper_sources_are_attenuated_more():
    """A voxel further from the detector must survive less."""
    mu = np.full((32, 32), 0.02)
    w = attenuation_weights(mu, voxel_size_mm=4.0, angles_deg=[0.0])[0]
    near_detector = w[16, 28]
    far_from_detector = w[16, 3]
    assert far_from_detector < near_detector


def test_attenuation_reduces_the_projected_signal():
    activity = np.zeros((24, 24))
    activity[8:16, 8:16] = 1.0
    mu = np.full((24, 24), 0.02)
    angles = np.arange(0, 180, 30.0)
    plain = Projector((24, 24), angles).forward(activity).sum()
    attenuated = Projector((24, 24), angles,
                           weights=attenuation_weights(mu, 4.0, angles)
                           ).forward(activity).sum()
    assert attenuated < plain
