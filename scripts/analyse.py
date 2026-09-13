#!/usr/bin/env python3
"""Analysis entry point. Every scientific parameter comes from config/.

    python scripts/analyse.py --events results/run001/events.csv \
        --acquisition acquisition_tomographic.yaml
    python scripts/analyse.py --surrogate --acquisition acquisition_tomographic.yaml

There is no scientific default in this file: window widths, count levels,
angles, iteration numbers and resolutions are read from the YAML files, so a
result cannot depend on a value that appears in no configuration.
"""

from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import asdict

import numpy as np

from gcmc.config import RunConfig
from gcmc.detector import collimator_fwhm, in_window, system_fwhm
from gcmc.hits import aggregate_steps_to_events, load_events, to_projection
from gcmc.metrics import analyse_reconstructed_slice
from gcmc.phantom import build_phantom
from gcmc.pipeline import planar_pipeline, tomographic_pipeline
from gcmc.planar import radial_profile, total_window_counts
from gcmc.rois import sphere_centres_px
from gcmc.spectra import decompose, scatter_fraction
from gcmc.uncertainty import replica_statistics
from gcmc.validation import sensitivity_cps_per_MBq


def system_resolution_mm(cfg: RunConfig) -> float:
    """System FWHM at the configured source-to-collimator distance."""
    geometric = collimator_fwhm(cfg.acquisition.distance_to_collimator_mm,
                                cfg.camera.hole_diameter_mm,
                                cfg.camera.hole_length_mm)
    return system_fwhm(geometric, cfg.camera.intrinsic_spatial_resolution_mm)


def analyse_events(events, cfg: RunConfig, out: pathlib.Path) -> dict:
    """Spectral and planar quantities, which are defined on the projection."""
    window = cfg.source.acquisition_windows[0]
    accepted = in_window(events["energy_keV"], window["centre_keV"],
                         window["width_pct"])
    centres, total, primary, scattered = decompose(events["energy_keV"],
                                                   events["n_compton_history"])
    np.savez(out / "spectrum.npz", centres=centres, total=total,
             primary=primary, scattered=scattered)

    projection = to_projection(events.loc[accepted, "x_mm"],
                               events.loc[accepted, "y_mm"],
                               cfg.camera.matrix, cfg.camera.pixel_size_mm)
    counts = total_window_counts(projection)
    return {
        "window": window,
        "events_detected": len(events),
        "events_in_window": int(accepted.sum()),
        "scatter_fraction_in_window": scatter_fraction(
            events["energy_keV"], events["n_compton_history"],
            window["centre_keV"], window["width_pct"]),
        "counts_in_window": counts,
        "sensitivity_cps_per_MBq": sensitivity_cps_per_MBq(
            counts, cfg.acquisition.live_time_s, activity_MBq=1.0),
        "radial_profile_bins": len(radial_profile(projection)[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=pathlib.Path,
                        help="event-level hit table from the simulation")
    parser.add_argument("--steps", type=pathlib.Path,
                        help="step-level table, reduced to events before analysis")
    parser.add_argument("--surrogate", action="store_true")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--source", default="source_tc99m.yaml")
    parser.add_argument("--acquisition", default="acquisition_planar.yaml")
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("results/latest"))
    args = parser.parse_args()

    if not any((args.events, args.steps, args.surrogate)):
        parser.error("pass --events, --steps or --surrogate")

    cfg = RunConfig.from_directory(args.config_dir, source=args.source,
                                   acquisition=args.acquisition)
    phantom = build_phantom(cfg.phantom)
    args.out.mkdir(parents=True, exist_ok=True)
    fwhm = system_resolution_mm(cfg)

    report = {"provenance": cfg.provenance(),
              "system_fwhm_mm": fwhm,
              "activity_ratio_true": phantom.true_concentration_ratio()}

    if args.events or args.steps:
        import pandas as pd

        events = (load_events(args.events) if args.events else
                  aggregate_steps_to_events(pd.read_csv(args.steps, comment="#")))
        report["planar"] = analyse_events(events, cfg, args.out)
    else:
        report["provenance"]["warning"] = (
            "SURROGATE — analytic projector: no scatter, no septal penetration, "
            "no backscatter. Not a Monte Carlo result.")
        planar = planar_pipeline(phantom, cfg.acquisition, fwhm, seed=cfg.seed)
        report["planar"] = {"counts_in_window": total_window_counts(planar["image"])}

    if cfg.acquisition.mode == "tomographic":
        per_replica = []
        for seed in range(cfg.acquisition.replicas):
            run = tomographic_pipeline(phantom, cfg.acquisition, cfg.reconstruction,
                                       system_fwhm_mm=fwhm, seed=seed)
            centres = sphere_centres_px(run["image"].shape,
                                        cfg.phantom.sphere_centre_circle_diameter_mm,
                                        cfg.phantom.voxel_size_mm)
            per_replica.append(analyse_reconstructed_slice(
                run["image"], centres, phantom.sphere_diameters_mm,
                cfg.phantom.voxel_size_mm, phantom.true_concentration_ratio()))

        spheres = []
        for j, diameter in enumerate(phantom.sphere_diameters_mm):
            spheres.append({
                "diameter_mm": diameter,
                # Replica variability, not 1/sqrt(N): these are ratios of
                # correlated ROI means passed through a reconstruction, so
                # counting statistics do not give their uncertainty.
                "contrast_recovery_coefficient_pct": replica_statistics(
                    [r[j].contrast_recovery_coefficient_pct for r in per_replica]),
                "relative_activity_recovery": replica_statistics(
                    [r[j].relative_activity_recovery for r in per_replica]),
                "background_variability_pct": replica_statistics(
                    [r[j].background_variability_pct for r in per_replica]),
            })
        report["reconstructed"] = {
            "spheres": spheres,
            "reconstruction": asdict(cfg.reconstruction),
            "attenuation_compensated": False,
            "scatter_compensated": False,
        }

    (args.out / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"wrote {args.out / 'report.json'}")


if __name__ == "__main__":
    main()
