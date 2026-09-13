"""End-to-end paths, each restricted to the domain its metrics are valid in.

The split is not stylistic. The six inserts of this phantom lie on a circle in
the transaxial plane, all at the same axial position. A planar projection
integrates along one transaxial axis, so the inserts overlap one another in the
projection image and no fixed ROI pattern recovers them individually. Sphere
image-quality metrics are therefore **not defined** on a planar projection of
this phantom, and this module does not offer a way to compute them there.

    planar        spectra, sensitivity, scatter fraction, spatial resolution,
                  projection profiles
    tomographic   contrast recovery, relative activity recovery, background
                  variability, contrast-to-noise, sphere-size dependence
"""

from __future__ import annotations

import numpy as np

from gcmc.config import AcquisitionConfig, ReconstructionConfig
from gcmc.detector import add_poisson_noise
from gcmc.phantom import DigitalPhantom
from gcmc.projector import Projector, attenuation_weights
from gcmc.reconstruction import mlem

FWHM_TO_SIGMA = 1.0 / 2.3548200450309493


def central_slice(phantom: DigitalPhantom) -> tuple[np.ndarray, np.ndarray]:
    """The transaxial slice through the sphere centres."""
    k = phantom.activity.shape[2] // 2
    return (phantom.activity[:, :, k].astype(float),
            phantom.mu[:, :, k].astype(float))


def _blur_sinogram(sinogram, system_fwhm_mm: float, bin_size_mm: float):
    """System resolution applied where it physically acts: before detection."""
    from scipy.ndimage import gaussian_filter1d

    sigma = system_fwhm_mm * FWHM_TO_SIGMA / bin_size_mm
    return gaussian_filter1d(np.asarray(sinogram, dtype=float), sigma, axis=1)


def tomographic_pipeline(phantom: DigitalPhantom, acquisition: AcquisitionConfig,
                         reconstruction: ReconstructionConfig,
                         system_fwhm_mm: float, seed: int = 42) -> dict:
    """Attenuated acquisition of one slice, then MLEM without compensation.

    Every parameter comes from the configuration objects: nothing scientific is
    defaulted inside this function, so a result cannot silently depend on a
    value that appears in no config file.
    """
    activity, mu = central_slice(phantom)
    angles = acquisition.angles_deg()

    weights = attenuation_weights(mu, phantom.voxel_size_mm, angles)
    acquisition_operator = Projector(activity.shape, angles, weights=weights)
    reconstruction_operator = Projector(activity.shape, angles)  # no compensation

    sinogram = acquisition_operator.forward(activity)
    sinogram = _blur_sinogram(sinogram, system_fwhm_mm, phantom.voxel_size_mm)
    if acquisition.total_counts:
        sinogram = add_poisson_noise(sinogram, acquisition.total_counts, seed=seed)

    image = mlem(sinogram, reconstruction_operator, reconstruction.iterations)
    return {
        "image": image,
        "sinogram": sinogram,
        "angles_deg": angles,
        "domain": "reconstructed",
        "system_fwhm_mm": system_fwhm_mm,
        "n_iterations": reconstruction.iterations,
        "total_counts": acquisition.total_counts,
        "attenuation_compensated": False,
        "scatter_compensated": False,
        "seed": seed,
    }


def planar_pipeline(phantom: DigitalPhantom, acquisition: AcquisitionConfig,
                    system_fwhm_mm: float, seed: int = 42) -> dict:
    """A single projection, for spectra, sensitivity, scatter and resolution.

    The returned image is explicitly labelled ``projection``; the image-quality
    functions in :mod:`gcmc.metrics` refuse it.
    """
    from gcmc.detector import apply_spatial_resolution
    from gcmc.surrogate import attenuated_projection

    projection = attenuated_projection(phantom.activity, phantom.mu,
                                       phantom.voxel_size_mm)
    projection = apply_spatial_resolution(projection, system_fwhm_mm,
                                          phantom.voxel_size_mm)
    if acquisition.total_counts:
        projection = add_poisson_noise(projection, acquisition.total_counts, seed=seed)
    return {"image": projection, "domain": "projection",
            "system_fwhm_mm": system_fwhm_mm,
            "total_counts": acquisition.total_counts, "seed": seed}
