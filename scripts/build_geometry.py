#!/usr/bin/env python3
"""Generate the GAMOS geometry and source description from config/.

Everything the simulation needs is derived here from the same YAML files the
analysis reads, so the simulated camera and the analysed camera cannot diverge.
Nothing physical is hard-coded in this script or in the macros.

Outputs, written to gamos/generated/ and not committed:

    collimator_geometry.txt   lead body with one air hole per lattice site
    detector_geometry.txt     crystal and backscatter compartment
    source_activity.txt       voxelised activity map of the phantom
    source_emission.txt       photon energies and yields of the radionuclide
"""

from __future__ import annotations

import argparse
import json
import pathlib

from gcmc.collimator import (
    hexagonal_lattice,
    minimum_septal_thickness,
    open_area_fraction,
    to_gamos_geometry,
)
from gcmc.config import RunConfig
from gcmc.phantom import build_phantom, to_gamos_source_table


def detector_block(cam, collimator_length_mm: float) -> str:
    z = collimator_length_mm + cam.crystal_thickness_mm / 2.0
    return "\n".join([
        "// Generated from config/ — do not edit by hand.",
        f":VOLU crystal BOX 250.0 250.0 {cam.crystal_thickness_mm / 2:.4f} NaI",
        f":PLACE crystal 1 world 0. 0. {z:.4f}",
        ":VOLU backscatter BOX 250.0 250.0 25.0 G4_Pyrex_Glass",
        (":PLACE backscatter 1 world 0. 0. "
         f"{z + cam.crystal_thickness_mm / 2 + 25.0:.4f}"),
    ]) + "\n"


def emission_block(source) -> str:
    lines = [f"# {source.radionuclide}: energy_keV yield_per_decay"]
    lines += [f"{e:.3f} {y:.6f}" for e, y in zip(source.photopeaks_keV, source.yields)]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--source", default="source_tc99m.yaml",
                        help="radionuclide configuration file in --config-dir")
    parser.add_argument("--collimator-fov-mm", type=float, default=100.0,
                        help="side of the collimator region modelled hole by hole")
    parser.add_argument("--out", default="gamos/generated")
    args = parser.parse_args()

    cfg = RunConfig.from_directory(args.config_dir, source=args.source)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cam, fov = cfg.camera, args.collimator_fov_mm

    centres = hexagonal_lattice(fov, fov, cam.hole_diameter_mm,
                                cam.septal_thickness_mm)
    (out / "collimator_geometry.txt").write_text(
        to_gamos_geometry(centres, cam.hole_diameter_mm, cam.hole_length_mm,
                          fov, fov))
    (out / "detector_geometry.txt").write_text(
        detector_block(cam, cam.hole_length_mm))
    (out / "source_activity.txt").write_text(
        to_gamos_source_table(build_phantom(cfg.phantom)))
    (out / "source_emission.txt").write_text(emission_block(cfg.source))

    summary = {
        "radionuclide": cfg.source.radionuclide,
        "collimator_fov_mm": fov,
        "n_holes": len(centres),
        "open_area_fraction": open_area_fraction(centres, cam.hole_diameter_mm,
                                                 fov, fov),
        "minimum_septal_thickness_mm": minimum_septal_thickness(
            centres, cam.hole_diameter_mm),
        "config": cfg.provenance(),
    }
    (out / "geometry_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
