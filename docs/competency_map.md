# What each part of this repository demonstrates

A mapping from files to the competencies that doctoral positions in
computational medical imaging and medical AI typically ask for. It is written
in terms of competencies rather than the wording of any particular call, so it
stays honest as calls change; paste the actual requirement list from a specific
call to align the wording.

| Competency | Where it is demonstrated |
|---|---|
| Monte Carlo radiation transport | `gamos/macros/`, `gamos/plugins/`, physics list and region-dependent production cuts |
| C++ in a scientific codebase | `gamos/plugins/src/PhotonHistoryUA.cc`, CMake build |
| Scientific Python | `src/gcmc/` — typed configuration, NumPy/SciPy numerics, no notebook-only logic |
| 3D medical image processing | `phantom.py` (voxelised 3D phantom), `pipeline.py`, `reconstruction.py` |
| Quantitative imaging and physics of image formation | `detector.py` (collimator resolution with septal-penetration correction, energy resolution scaling), `spectra.py` |
| Tomographic reconstruction | `projector.py` (explicit sparse system matrix, exact adjoint) and `reconstruction.py` (MLEM + Poisson log-likelihood) |
| Numerical linear algebra | the adjoint identity and likelihood-monotonicity tests in `tests/test_projector.py` and `tests/test_pipeline.py` |
| Statistical evaluation | `uncertainty.py` — Poisson and replica statistics, sample-size planning |
| Evaluation methodology | `metrics.py` — image-domain tracking, NEMA definitions, ROI conventions |
| Software engineering | `pyproject.toml`, CI, 44 tests, configuration-driven experiments |
| Reproducibility | `config/` as single source of truth, per-run `environment.json`, archivable hit tables |
| Scientific judgement | `docs/model_card.md` and `docs/validation.md` — stating what the model cannot support |

## The two things worth pointing at in an application

**The domain rule, enforced in code.** The six inserts superimpose on one
another in a planar projection of this phantom, so per-sphere image-quality
metrics are undefined there. `metrics.analyse_reconstructed_slice` raises
rather than returning a number, and a test asserts that it does. It is a test
of a methodological claim, not of a function.

**The adjoint test.** `tests/test_projector.py` asserts
<A x, y> = <x, A^T y> to machine precision. It is what licenses calling the
reconstruction MLEM rather than "an iterative scheme resembling MLEM".

**The exact scatter decomposition.** `PhotonHistoryUA.cc` scores one integer per
detected photon that no measurement can provide, and `spectra.decompose` turns
it into an exact primary/scatter separation. That is the whole argument for
running a Monte Carlo instead of acquiring an image.
