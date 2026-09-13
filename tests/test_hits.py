"""The step-to-event reduction, which every spectral quantity depends on."""

import pandas as pd
import pytest

from gcmc.hits import aggregate_steps_to_events, load_events


def _steps():
    """One 140 keV primary absorbed in three steps, one 90 keV scattered event."""
    return pd.DataFrame([
        {"event_id": 1, "edep_keV": 20.0, "x_mm": 0.0, "y_mm": 0.0,
         "n_compton_history": 0, "origin_volume": "phantom"},
        {"event_id": 1, "edep_keV": 30.0, "x_mm": 1.0, "y_mm": 0.0,
         "n_compton_history": 0, "origin_volume": "phantom"},
        {"event_id": 1, "edep_keV": 90.0, "x_mm": 2.0, "y_mm": 0.0,
         "n_compton_history": 0, "origin_volume": "phantom"},
        {"event_id": 2, "edep_keV": 90.0, "x_mm": 10.0, "y_mm": 5.0,
         "n_compton_history": 2, "origin_volume": "phantom"},
    ])


def test_steps_are_merged_into_one_event_each():
    events = aggregate_steps_to_events(_steps())
    assert len(events) == 2
    assert events.loc[events.event_id == 1, "n_steps"].item() == 3


def test_event_energy_is_the_sum_of_its_steps():
    """The photopeak exists only if this holds: 20 + 30 + 90 = 140, not three rows."""
    events = aggregate_steps_to_events(_steps())
    assert events.loc[events.event_id == 1, "energy_keV"].item() == pytest.approx(140.0)


def test_position_is_the_energy_weighted_centroid():
    events = aggregate_steps_to_events(_steps())
    expected = (20 * 0 + 30 * 1 + 90 * 2) / 140
    assert events.loc[events.event_id == 1, "x_mm"].item() == pytest.approx(expected)


def test_scatter_order_comes_from_the_dominant_history():
    events = aggregate_steps_to_events(_steps())
    assert events.loc[events.event_id == 1, "n_compton_history"].item() == 0
    assert events.loc[events.event_id == 2, "n_compton_history"].item() == 2


def test_zero_energy_steps_are_dropped():
    steps = _steps()
    steps.loc[len(steps)] = {"event_id": 3, "edep_keV": 0.0, "x_mm": 0.0,
                             "y_mm": 0.0, "n_compton_history": 0,
                             "origin_volume": "phantom"}
    assert len(aggregate_steps_to_events(steps)) == 2


def test_a_step_table_is_rejected_by_the_event_loader(tmp_path):
    path = tmp_path / "steps.csv"
    _steps().to_csv(path, index=False)
    with pytest.raises(ValueError, match="event-level"):
        load_events(path)


def test_an_event_table_round_trips(tmp_path):
    path = tmp_path / "events.csv"
    aggregate_steps_to_events(_steps()).to_csv(path, index=False)
    assert len(load_events(path)) == 2
