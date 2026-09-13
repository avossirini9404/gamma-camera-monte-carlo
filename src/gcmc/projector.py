"""Explicit system matrix and its exact adjoint.

Why this module replaces the rotation-based projector pair
----------------------------------------------------------
The previous implementation built the forward projection with
``scipy.ndimage.rotate`` and the back-projection with a rotation by the
opposite angle. Those two operations are *not* adjoint: bilinear interpolation
on the way out and on the way back are different linear maps, and the residual
mismatch is not small enough to ignore. An EM iteration built on a
non-adjoint pair is a plausible-looking iterative scheme, not maximum-likelihood
expectation maximisation, and its fixed point is not the ML estimate.

Here the projector is a sparse matrix ``A`` built once, and the
back-projector is literally ``A.T``. The adjoint identity

    <A x, y> = <x, A^T y>

then holds to machine precision, and ``tests/test_projector.py`` asserts it.

Discretisation: pixel-driven, with each pixel's projected centre split linearly
between the two nearest detector bins. Attenuation enters as a per-pixel,
per-angle weight, so the *same* code builds the attenuated forward model used
by the surrogate and the non-attenuated operator used for reconstruction. The
difference between them is precisely "no attenuation compensation", which is
therefore explicit rather than implied.
"""

from __future__ import annotations

import numpy as np


def _pixel_coordinates(shape):
    n0, n1 = shape
    x = np.arange(n0) - (n0 - 1) / 2.0
    y = np.arange(n1) - (n1 - 1) / 2.0
    return np.meshgrid(x, y, indexing="ij")


def attenuation_weights(mu, voxel_size_mm: float, angles_deg) -> np.ndarray:
    """Per-pixel survival probability towards the detector, one map per angle.

    Returns an array of shape (n_angles, n0, n1) with values in (0, 1].
    """
    from scipy.ndimage import rotate

    mu = np.asarray(mu, dtype=float)
    out = np.empty((len(angles_deg), *mu.shape), dtype=float)
    for i, angle in enumerate(angles_deg):
        rotated = mu if angle % 360 == 0 else rotate(
            mu, angle, reshape=False, order=1, mode="constant", cval=0.0)
        # Detector on the +axis1 side: accumulate mu strictly beyond each voxel.
        beyond = np.cumsum(rotated[:, ::-1], axis=1)[:, ::-1] - rotated
        survival = np.exp(-beyond * voxel_size_mm)
        out[i] = survival if angle % 360 == 0 else rotate(
            survival, -angle, reshape=False, order=1, mode="constant", cval=1.0)
    return np.clip(out, 0.0, 1.0)


def build_system_matrix(shape, angles_deg, n_bins: int | None = None,
                        weights=None):
    """Sparse parallel-beam projector of shape (n_angles * n_bins, n0 * n1).

    ``weights`` is an optional (n_angles, n0, n1) array of per-pixel, per-angle
    multipliers — attenuation, in practice. Passing ``None`` gives the plain
    geometric operator used for reconstruction.
    """
    from scipy import sparse

    n0, n1 = shape
    n_bins = int(n_bins or max(shape))
    xx, yy = _pixel_coordinates(shape)
    pixel_index = np.arange(n0 * n1).reshape(n0, n1)
    centre = (n_bins - 1) / 2.0

    rows, cols, vals = [], [], []
    for a_i, angle in enumerate(angles_deg):
        theta = np.deg2rad(angle)
        t = xx * np.cos(theta) + yy * np.sin(theta) + centre
        lower = np.floor(t).astype(int)
        frac = t - lower
        w = np.ones(shape) if weights is None else np.asarray(weights[a_i])
        for shift, share in ((0, 1.0 - frac), (1, frac)):
            b = lower + shift
            valid = (b >= 0) & (b < n_bins) & (share > 0)
            if not valid.any():
                continue
            rows.append(a_i * n_bins + b[valid])
            cols.append(pixel_index[valid])
            vals.append((share * w)[valid])

    return sparse.csr_matrix(
        (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
        shape=(len(angles_deg) * n_bins, n0 * n1),
    )


class Projector:
    """A system matrix together with the geometry it was built for.

    Carrying the geometry with the operator removes a whole class of silent
    errors: a sinogram can no longer be reshaped against the wrong number of
    detector bins, and the reconstruction cannot be run against angles other
    than the ones the matrix encodes.
    """

    def __init__(self, shape, angles_deg, n_bins=None, weights=None):
        self.shape = tuple(shape)
        self.angles_deg = np.asarray(angles_deg, dtype=float)
        self.n_bins = int(n_bins or max(self.shape))
        self.matrix = build_system_matrix(self.shape, self.angles_deg,
                                          self.n_bins, weights)

    @property
    def sinogram_shape(self):
        return (len(self.angles_deg), self.n_bins)

    def forward(self, image) -> np.ndarray:
        """A x, shaped (n_angles, n_bins)."""
        flat = self.matrix @ np.asarray(image, dtype=float).ravel()
        return flat.reshape(self.sinogram_shape)

    def back(self, sinogram) -> np.ndarray:
        """A^T y, shaped like the image. Exactly the adjoint of forward."""
        flat = self.matrix.T @ np.asarray(sinogram, dtype=float).ravel()
        return flat.reshape(self.shape)
