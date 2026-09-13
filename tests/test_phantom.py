import numpy as np
import pytest

from gcmc.config import PhantomConfig
from gcmc.phantom import build_phantom, to_gamos_source_table


def test_six_spheres_are_built_and_distinct():
    p = build_phantom()
    assert len(p.sphere_masks) == 6
    assert all(m.sum() > 0 for m in p.sphere_masks)
    overlap = sum(np.logical_and(a, b).sum()
                  for i, a in enumerate(p.sphere_masks)
                  for b in p.sphere_masks[i + 1:])
    assert overlap == 0


def test_sphere_volumes_increase_with_diameter():
    p = build_phantom()
    volumes = [m.sum() for m in p.sphere_masks]
    assert volumes == sorted(volumes)


def test_voxelised_sphere_volume_is_close_to_analytical():
    cfg = PhantomConfig(voxel_size_mm=2.0, grid_shape=(128, 128, 96))
    p = build_phantom(cfg)
    largest = p.sphere_masks[-1].sum() * cfg.voxel_size_mm ** 3
    analytical = 4 / 3 * np.pi * (37.0 / 2) ** 3
    assert largest == pytest.approx(analytical, rel=0.05)


def test_activity_ratio_matches_the_configuration():
    p = build_phantom(PhantomConfig(sphere_to_background_ratio=4.0))
    assert p.true_concentration_ratio() == pytest.approx(4.0, rel=1e-6)


def test_lung_insert_is_less_attenuating_than_water():
    p = build_phantom()
    assert p.mu.max() > p.mu[p.mu > 0].min()


def test_source_table_is_plain_text_and_non_empty():
    table = to_gamos_source_table(build_phantom(PhantomConfig(grid_shape=(16, 16, 12),
                                                             voxel_size_mm=16.0)))
    assert table.startswith("# x_mm")
    assert len(table.splitlines()) > 1
