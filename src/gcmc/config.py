"""Configuration loading.

Every experiment in this repository is defined by YAML files under ``config/``.
Nothing is configured in a notebook, and no default lives in two places: the
same file drives the GAMOS geometry generation and the Python analysis, so the
simulated camera and the analysed camera cannot silently diverge.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Any

import yaml


def load_yaml(path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


@dataclass
class CameraConfig:
    """Generic parallel-hole gamma camera.

    All values are illustrative and generic. No parameter here is taken from
    manufacturer documentation of any commercial system.
    """

    hole_diameter_mm: float = 1.5
    septal_thickness_mm: float = 0.2
    hole_length_mm: float = 25.0
    collimator_material: str = "lead"
    crystal_material: str = "NaI(Tl)"
    crystal_thickness_mm: float = 9.5
    intrinsic_energy_resolution_pct: float = 9.5
    energy_resolution_reference_keV: float = 140.5
    intrinsic_spatial_resolution_mm: float = 3.8
    pixel_size_mm: float = 4.0
    matrix: tuple[int, int] = (64, 64)

    @classmethod
    def from_yaml(cls, path) -> CameraConfig:
        d = load_yaml(path)
        d = {**d.get("collimator", {}), **d.get("crystal", {}), **d.get("detector", {})}
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class PhantomConfig:
    """NEMA-IEC-inspired digital phantom with six spherical inserts.

    Sphere diameters follow the NEMA NU 2 image-quality set. The background
    compartment is an elliptical cylinder of generic dimensions: this is an
    *inspired* phantom, not a claim to reproduce a certified physical one.
    """

    sphere_diameters_mm: tuple[float, ...] = (10.0, 13.0, 17.0, 22.0, 28.0, 37.0)
    sphere_centre_circle_diameter_mm: float = 114.4
    background_semiaxis_x_mm: float = 115.0
    background_semiaxis_y_mm: float = 90.0
    background_height_mm: float = 180.0
    lung_insert_diameter_mm: float = 50.0
    sphere_to_background_ratio: float = 8.0
    voxel_size_mm: float = 4.0
    grid_shape: tuple[int, int, int] = (64, 64, 48)

    @classmethod
    def from_yaml(cls, path) -> PhantomConfig:
        d = load_yaml(path).get("phantom", {})
        known = {f for f in cls.__dataclass_fields__}
        out = {k: v for k, v in d.items() if k in known}
        for k in ("sphere_diameters_mm", "grid_shape"):
            if k in out:
                out[k] = tuple(out[k])
        return cls(**out)


@dataclass
class SourceConfig:
    """Radionuclide emission data.

    Photon yields are per decay. Values are taken from public decay-data
    libraries and are cited in ``docs/data_sources.md``.
    """

    radionuclide: str = "Tc-99m"
    photopeaks_keV: tuple[float, ...] = (140.5,)
    yields: tuple[float, ...] = (0.885,)
    acquisition_windows: tuple[dict, ...] = field(default_factory=tuple)

    @classmethod
    def from_yaml(cls, path) -> SourceConfig:
        d = load_yaml(path).get("source", {})
        return cls(
            radionuclide=d.get("radionuclide", "Tc-99m"),
            photopeaks_keV=tuple(d.get("photopeaks_keV", (140.5,))),
            yields=tuple(d.get("yields", (0.885,))),
            acquisition_windows=tuple(d.get("acquisition_windows", ())),
        )


@dataclass
class AcquisitionConfig:
    """Acquisition parameters. No scientific default lives anywhere else."""

    mode: str = "planar"
    n_angles: int = 1
    angular_range_deg: float = 360.0
    distance_to_collimator_mm: float = 100.0
    total_counts: float | None = None
    live_time_s: float = 300.0
    replicas: int = 5

    def angles_deg(self):
        import numpy as np

        if self.mode == "planar":
            return np.array([0.0])
        return np.arange(0.0, self.angular_range_deg,
                         self.angular_range_deg / self.n_angles)

    @classmethod
    def from_yaml(cls, path) -> AcquisitionConfig:
        d = load_yaml(path).get("acquisition", {})
        known = set(cls.__dataclass_fields__)
        out = {k: v for k, v in d.items() if k in known}
        if out.get("total_counts") is not None:
            out["total_counts"] = float(out["total_counts"])
        return cls(**out)


@dataclass
class ReconstructionConfig:
    algorithm: str = "mlem"
    iterations: int = 30
    post_filter_fwhm_mm: float = 0.0

    @classmethod
    def from_yaml(cls, path) -> ReconstructionConfig:
        d = load_yaml(path).get("reconstruction", {})
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class RunConfig:
    """Everything one run depends on, assembled from config/ and nothing else."""

    camera: CameraConfig
    phantom: PhantomConfig
    source: SourceConfig
    acquisition: AcquisitionConfig
    reconstruction: ReconstructionConfig
    seed: int = 42
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_directory(cls, cfg_dir, camera="camera_generic_lehr.yaml",
                       phantom="phantom_nema_iec_6spheres.yaml",
                       source="source_tc99m.yaml",
                       acquisition="acquisition_planar.yaml",
                       seed: int = 42) -> RunConfig:
        cfg_dir = pathlib.Path(cfg_dir)
        acquisition_path = cfg_dir / acquisition
        return cls(
            camera=CameraConfig.from_yaml(cfg_dir / camera),
            phantom=PhantomConfig.from_yaml(cfg_dir / phantom),
            source=SourceConfig.from_yaml(cfg_dir / source),
            acquisition=AcquisitionConfig.from_yaml(acquisition_path),
            reconstruction=ReconstructionConfig.from_yaml(acquisition_path),
            seed=seed,
        )

    def provenance(self) -> dict:
        """Flat record of every parameter a result depends on."""
        from dataclasses import asdict

        return {"camera": asdict(self.camera), "phantom": asdict(self.phantom),
                "source": asdict(self.source), "acquisition": asdict(self.acquisition),
                "reconstruction": asdict(self.reconstruction), "seed": self.seed}
