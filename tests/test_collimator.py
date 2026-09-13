"""The collimator must actually have holes, with the septa it claims."""

import numpy as np
import pytest

from gcmc.collimator import (
    hexagonal_lattice,
    minimum_septal_thickness,
    open_area_fraction,
    pitch_mm,
    to_gamos_geometry,
)

D, S, FOV = 1.5, 0.2, 40.0


@pytest.fixture(scope="module")
def lattice():
    return hexagonal_lattice(FOV, FOV, D, S)


def test_lattice_is_not_empty(lattice):
    assert len(lattice) > 100


def test_septal_thickness_is_respected(lattice):
    """The lead between neighbouring holes must be at least the configured value."""
    assert minimum_septal_thickness(lattice, D) == pytest.approx(S, abs=1e-6)


def test_packing_is_hexagonal_not_square():
    """Hexagonal packing gives a higher open area than square at equal pitch."""
    hexagonal = hexagonal_lattice(FOV, FOV, D, S)
    p = pitch_mm(D, S)
    n_square = int(FOV // p) ** 2
    assert len(hexagonal) > n_square


def test_open_area_is_close_to_the_theoretical_hexagonal_value(lattice):
    theoretical = np.pi / (2 * np.sqrt(3)) * (D / pitch_mm(D, S)) ** 2
    assert open_area_fraction(lattice, D, FOV, FOV) == pytest.approx(
        theoretical, rel=0.15)


def test_finer_septa_pack_more_holes():
    assert len(hexagonal_lattice(FOV, FOV, D, 0.1)) > len(
        hexagonal_lattice(FOV, FOV, D, 0.6))


def test_generated_geometry_places_every_hole(lattice):
    text = to_gamos_geometry(lattice, D, 25.0, FOV, FOV)
    assert text.count(":PLACE hole") == len(lattice)
    assert ":VOLU collimator BOX" in text
    assert "POLYHEDRA" in text


def test_holes_stay_inside_the_modelled_face(lattice):
    assert np.all(np.abs(lattice) <= FOV / 2 - D / 2 + 1e-9)
