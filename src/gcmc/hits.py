"""The simulation to analysis interface: one row per *detected event*.

Why not one row per step
------------------------
A 140 keV photon absorbed in the crystal deposits its energy through a
secondary electron that takes several Geant4 steps. Writing one row per step
produces a table whose rows are 20, 30 and 90 keV where the physics of the
measurement produced a single 140 keV detection. A camera integrates the
scintillation light of the whole interaction and reports one event with one
energy and one position.

Every downstream quantity depends on getting this right: the energy spectrum,
the photopeak, the scatter fraction, the sensitivity and the projection counts
all assume that a row is a detected event.

Event schema
------------
``event_id``               Geant4 event identifier
``energy_keV``             total energy deposited in the crystal by the event
``x_mm``, ``y_mm``         energy-weighted interaction centroid, which is what
                           an Anger position circuit estimates
``n_compton_history``      Compton interactions undergone by the photon history
                           that deposited most of the energy (0 = primary)
``origin_volume``          volume in which the emitting decay occurred
``n_steps``                number of steps merged into the event, for auditing

The C++ user action writes this table directly. ``aggregate_steps_to_events``
performs the same reduction in Python, so the rule is testable in CI on a
machine with no Geant4, and so that step-level tables from other tools can be
brought into the same schema.
"""

from __future__ import annotations

import pathlib

import numpy as np

EVENT_COLUMNS = ("event_id", "energy_keV", "x_mm", "y_mm",
                 "n_compton_history", "origin_volume")
STEP_COLUMNS = ("event_id", "edep_keV", "x_mm", "y_mm",
                "n_compton_history", "origin_volume")


def load_events(path):
    """Load an event-level hit table (CSV or Parquet)."""
    import pandas as pd

    path = pathlib.Path(path)
    df = (pd.read_parquet(path) if path.suffix == ".parquet"
          else pd.read_csv(path, comment="#"))
    missing = [c for c in EVENT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"not an event-level hit table, missing {missing}. If this is a "
            "step-level table, reduce it with aggregate_steps_to_events first."
        )
    return df


def aggregate_steps_to_events(steps):
    """Reduce a step-level table to one row per detected event.

    Energy is summed. Position is the energy-weighted centroid, matching what
    an Anger position circuit computes from the scintillation light. The
    scatter order is taken from the photon history contributing the most
    energy: an event is classified by the photon that produced it, not by
    whichever step happened to be written last.
    """
    import pandas as pd

    steps = pd.DataFrame(steps)
    missing = [c for c in STEP_COLUMNS if c not in steps.columns]
    if missing:
        raise ValueError(f"step table is missing required columns: {missing}")

    steps = steps[steps["edep_keV"] > 0]
    if steps.empty:
        return pd.DataFrame(columns=[*EVENT_COLUMNS, "n_steps"])

    rows = []
    for event_id, group in steps.groupby("event_id", sort=True):
        energy = float(group["edep_keV"].sum())
        weights = group["edep_keV"].to_numpy(dtype=float)
        dominant = group.groupby("n_compton_history")["edep_keV"].sum().idxmax()
        origin = group.loc[group["edep_keV"].idxmax(), "origin_volume"]
        rows.append({
            "event_id": event_id,
            "energy_keV": energy,
            "x_mm": float(np.average(group["x_mm"], weights=weights)),
            "y_mm": float(np.average(group["y_mm"], weights=weights)),
            "n_compton_history": int(dominant),
            "origin_volume": origin,
            "n_steps": len(group),
        })
    return pd.DataFrame(rows)


def to_projection(x_mm, y_mm, matrix, pixel_size_mm, weights=None) -> np.ndarray:
    """Bin accepted events into a projection image."""
    ny, nx = matrix
    half_x, half_y = nx * pixel_size_mm / 2.0, ny * pixel_size_mm / 2.0
    counts, _, _ = np.histogram2d(
        np.asarray(y_mm, dtype=float), np.asarray(x_mm, dtype=float),
        bins=[ny, nx], range=[[-half_y, half_y], [-half_x, half_x]],
        weights=None if weights is None else np.asarray(weights, dtype=float),
    )
    return counts
