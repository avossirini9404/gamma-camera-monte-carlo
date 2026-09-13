"""Statistical uncertainty on Monte Carlo results.

A Monte Carlo number without an uncertainty is not a measurement of anything.
Two sources are tracked separately because they behave differently: counting
statistics inside a region of interest, and run-to-run variability estimated
from independent replicas with different seeds.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def poisson_relative_uncertainty(counts) -> np.ndarray:
    """1/sqrt(N), the relative standard uncertainty of a count."""
    c = np.asarray(counts, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(c > 0, 1.0 / np.sqrt(np.maximum(c, 1e-12)), np.inf)
    return out


def replica_statistics(values, alpha: float = 0.05) -> dict:
    """Mean and normal-theory CI over independent simulation replicas."""
    v = np.asarray(values, dtype=float)
    n = v.size
    if n < 2:
        return {"mean": float(v.mean()) if n else float("nan"),
                "sem": float("nan"), "ci_low": float("nan"),
                "ci_high": float("nan"), "n_replicas": int(n)}
    sem = v.std(ddof=1) / np.sqrt(n)
    from scipy import stats

    t = stats.t.ppf(1 - alpha / 2, df=n - 1)
    return {"mean": float(v.mean()), "sem": float(sem),
            "ci_low": float(v.mean() - t * sem), "ci_high": float(v.mean() + t * sem),
            "n_replicas": int(n)}


def bootstrap_ci(values, statistic: Callable = np.mean, n_resamples: int = 2000,
                 alpha: float = 0.05, seed: int = 42):
    v = np.asarray(values, dtype=float)
    if v.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_resamples, v.size))
    boot = np.array([statistic(v[i]) for i in idx])
    lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(statistic(v)), float(lo), float(hi)


def required_histories(relative_uncertainty_target: float, counts_per_history: float):
    """Histories needed to reach a target relative uncertainty in a ROI.

    Useful before launching a run: it turns "simulate a lot" into a number that
    can be justified in a methods section.
    """
    if counts_per_history <= 0 or relative_uncertainty_target <= 0:
        return float("inf")
    return float(1.0 / (relative_uncertainty_target**2 * counts_per_history))
