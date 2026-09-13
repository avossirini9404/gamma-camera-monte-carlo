"""Maximum-likelihood expectation maximisation, on an adjoint-exact operator.

MLEM is only MLEM when the back-projector is the adjoint of the forward
projector: the multiplicative update is derived from the Poisson likelihood
under exactly that assumption. This module therefore takes a
:class:`gcmc.projector.Projector`, whose back-projection is the transpose of
its own matrix, rather than building a projector pair of its own.

The reconstruction deliberately uses the *non-attenuated* operator while the
simulated data are generated with the attenuated one. That mismatch is the
absence of attenuation compensation, stated as a modelling choice rather than
hidden inside an interpolation.
"""

from __future__ import annotations

import numpy as np

from gcmc.projector import Projector


def mlem(sinogram, projector: Projector, n_iterations: int, eps: float = 1e-9,
         initial=None) -> np.ndarray:
    """Poisson MLEM.

    The iteration number is a reconstruction parameter, not a default: recovery
    rises monotonically with it before convergence, so it is reported with
    every result and has no value here that a caller can silently inherit.
    """
    y = np.asarray(sinogram, dtype=float).ravel()
    if y.size != projector.matrix.shape[0]:
        raise ValueError("sinogram does not match the projector geometry")

    x = np.ones(projector.matrix.shape[1]) if initial is None else \
        np.asarray(initial, dtype=float).ravel().copy()
    sensitivity = np.maximum(projector.matrix.T @ np.ones(y.size), eps)

    for _ in range(int(n_iterations)):
        projected = np.maximum(projector.matrix @ x, eps)
        x = x / sensitivity * (projector.matrix.T @ (y / projected))
    return x.reshape(projector.shape)


def log_likelihood(sinogram, projector: Projector, image, eps: float = 1e-9) -> float:
    """Poisson log-likelihood, up to a constant independent of the image.

    Used to check that the iteration is doing what its name says: MLEM is
    monotonically non-decreasing in this quantity, and a violation is the
    signature of a non-adjoint operator pair.
    """
    y = np.asarray(sinogram, dtype=float).ravel()
    projected = np.maximum(projector.matrix @ np.asarray(image, float).ravel(), eps)
    return float(np.sum(y * np.log(projected) - projected))
