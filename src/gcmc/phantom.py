"""Voxelised NEMA-IEC-inspired digital phantom.

The same object is used twice: it is exported to a GAMOS geometry/source
description for the Monte Carlo run, and it is used directly in Python as the
ground truth against which recovery coefficients are computed. Having one
definition removes the most common source of wrong recovery coefficients,
which is a ground-truth activity that does not match the simulated one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gcmc.config import PhantomConfig

# Linear attenuation coefficients at 140.5 keV, in mm^-1, from public
# NIST XCOM tabulations (see docs/data_sources.md).
MU_WATER_140KEV = 0.01536
MU_LUNG_140KEV = 0.00461
MU_AIR = 0.0


@dataclass
class DigitalPhantom:
    activity: np.ndarray          # relative activity concentration per voxel
    mu: np.ndarray                # linear attenuation coefficient, 1/mm
    sphere_masks: list[np.ndarray]
    background_mask: np.ndarray
    sphere_diameters_mm: tuple[float, ...]
    voxel_size_mm: float

    @property
    def voxel_volume_mL(self) -> float:
        return (self.voxel_size_mm ** 3) / 1000.0

    def true_concentration_ratio(self) -> float:
        """Sphere-to-background activity concentration ratio actually built."""
        bg = self.activity[self.background_mask].mean()
        sp = self.activity[self.sphere_masks[-1]].mean()
        return float(sp / bg)


def _grid(shape, voxel_mm):
    axes = [(np.arange(n) - (n - 1) / 2.0) * voxel_mm for n in shape]
    return np.meshgrid(*axes, indexing="ij")


def build_phantom(cfg: PhantomConfig | None = None) -> DigitalPhantom:
    cfg = cfg or PhantomConfig()
    x, y, z = _grid(cfg.grid_shape, cfg.voxel_size_mm)

    inside_ellipse = (x / cfg.background_semiaxis_x_mm) ** 2 + (
        y / cfg.background_semiaxis_y_mm
    ) ** 2 <= 1.0
    inside_height = np.abs(z) <= cfg.background_height_mm / 2.0
    body = inside_ellipse & inside_height

    lung = (np.sqrt(x**2 + y**2) <= cfg.lung_insert_diameter_mm / 2.0) & inside_height

    r_centres = cfg.sphere_centre_circle_diameter_mm / 2.0
    angles = np.deg2rad(np.arange(len(cfg.sphere_diameters_mm)) * 60.0)
    sphere_masks = []
    for diameter, angle in zip(cfg.sphere_diameters_mm, angles):
        cx, cy = r_centres * np.cos(angle), r_centres * np.sin(angle)
        d2 = (x - cx) ** 2 + (y - cy) ** 2 + z**2
        sphere_masks.append(d2 <= (diameter / 2.0) ** 2)

    any_sphere = np.any(sphere_masks, axis=0)
    background = body & ~lung & ~any_sphere

    activity = np.zeros(cfg.grid_shape, dtype=np.float32)
    activity[background] = 1.0
    for mask in sphere_masks:
        activity[mask] = cfg.sphere_to_background_ratio

    mu = np.full(cfg.grid_shape, MU_AIR, dtype=np.float32)
    mu[body] = MU_WATER_140KEV
    mu[lung] = MU_LUNG_140KEV

    return DigitalPhantom(
        activity=activity,
        mu=mu,
        sphere_masks=sphere_masks,
        background_mask=background,
        sphere_diameters_mm=cfg.sphere_diameters_mm,
        voxel_size_mm=cfg.voxel_size_mm,
    )


def to_gamos_source_table(phantom: DigitalPhantom) -> str:
    """Export the activity map as a plain text table for the GAMOS source.

    One line per emitting voxel: x y z (mm) and relative activity. Text is used
    on purpose — a reviewer can diff it, and it makes the simulated activity
    distribution auditable rather than implicit.
    """
    idx = np.argwhere(phantom.activity > 0)
    centre = (np.array(phantom.activity.shape) - 1) / 2.0
    lines = ["# x_mm y_mm z_mm relative_activity"]
    for i, j, k in idx:
        pos = (np.array([i, j, k]) - centre) * phantom.voxel_size_mm
        lines.append(f"{pos[0]:.2f} {pos[1]:.2f} {pos[2]:.2f} "
                     f"{phantom.activity[i, j, k]:.4f}")
    return "\n".join(lines) + "\n"
