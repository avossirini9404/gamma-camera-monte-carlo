# gamma-camera-monte-carlo

Monte Carlo model of a gamma camera: radiation transport, detector response,
energy spectra and quantitative projection imaging.

[![ci](https://github.com/avossirini9404/gamma-camera-monte-carlo/actions/workflows/ci.yml/badge.svg)](https://github.com/avossirini9404/gamma-camera-monte-carlo/actions)
[![licence](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)

> **Generic model.** Every camera parameter in `config/` is generic and
> illustrative, chosen to be physically sensible and typical of published
> low-energy high-resolution designs. Nothing is taken from manufacturer
> documentation of a commercial system, and no institutional or patient data is
> used. The phantom is *inspired by* the NEMA/IEC image-quality design; it is
> not a reproduction of a certified physical phantom.

---

## Research question

For a generic parallel-hole gamma camera imaging a six-sphere
NEMA-IEC-inspired phantom, how do scattered photons accepted by the
acquisition window, the system spatial resolution and the count level each
propagate into the quantitative accuracy of sphere activity concentration —
and how large is the run-to-run uncertainty on that answer?

## Why a Monte Carlo rather than a measurement

A physical acquisition gives the total counts in the energy window. It cannot
say which of those counts were scattered, where they scattered, or what the
projection would look like without them. The simulation can: the photon lineage
is propagated from parent track to secondary, so every detected event carries
the scatter order of the photon history that produced it. That gives an exact
primary/scattered classification **within the simulated photon history** —
which is not the same claim as a perfect representation of a real detector, and
the model card says where the two differ.

---

## Architecture

```
                        CONFIG (config/*.yaml)
                              │
          ┌───────────────────┼───────────────────┐
       CAMERA              PHANTOM              SOURCE
          └───────────────────┼───────────────────┘
                              ▼
                       GAMOS / GEANT4
                    ┌─────────┴─────────┐
                 geometry            physics
              (hole lattice,      (low-energy EM,
               crystal, phantom)   region cuts)
                    └─────────┬─────────┘
                              ▼
                      photon histories
                    ┌─────────┴─────────┐
                 ancestry        energy deposition
                    └─────────┬─────────┘
                              ▼
                   detector-event table
                 (one row per detected event)
                              ▼
                           PYTHON
          ┌───────────────────┼───────────────────┐
       spectra           projections          uncertainty
          │                   │              (replicas, Poisson)
          │                   ▼
          │             reconstruction (MLEM, adjoint-exact)
          │                   │
          │                   ▼
          │             image metrics
          ▼
   primary / scatter
   energy analysis
```

Scientific output: energy resolution · sensitivity · scatter fraction ·
spatial resolution · contrast recovery · background variability · uncertainty.

### Three design decisions worth stating

**The detector response lives in Python.** Transport is expensive; energy
resolution, intrinsic resolution and window width are cheap. Keeping them on
the analysis side means one transport run supports a whole parameter study,
every response parameter is an explicit versioned value, and the chain can be
tested in CI on a machine with no Geant4.

**A row is a detected event, not a Geant4 step.** A 140 keV photon absorbed in
the crystal deposits its energy over several steps through a secondary
electron. A step-level table would record 20, 30 and 90 keV where the
measurement produced a single 140 keV detection, and the photopeak, the scatter
fraction, the sensitivity and the projection counts would all be wrong.
Energy is summed per event and the position is the energy-weighted centroid,
which is what an Anger position circuit estimates.

**`config/` is the only source of truth.** No script, pipeline function or
macro carries a scientific default. The system resolution used in the analysis
is computed from the collimator dimensions and the source-to-collimator
distance in the config; the iteration count, angles and count levels come from
the acquisition file. A result cannot depend on a value that appears in no
configuration file.

---

## Which metric belongs in which domain

The six inserts lie on a circle in the transaxial plane, all at the same axial
position. A planar projection integrates along one of those transaxial axes, so
**the inserts superimpose on one another in the projection image**. No fixed ROI
pattern separates them there: per-sphere image-quality metrics on a planar view
of this phantom are not merely biased, they are undefined.

| Planar | Tomographic (reconstructed) |
|---|---|
| energy spectrum, photopeak | contrast recovery coefficient |
| sensitivity, window counts | relative activity recovery |
| scatter fraction | background variability |
| system spatial resolution | contrast-to-noise |
| projection profiles, uniformity | sphere-size dependence |

`gcmc.metrics.analyse_reconstructed_slice` raises on any other domain, and
`tests/test_pipeline.py` asserts that it does. The rule is enforced in code
rather than left to the reader.

## Terminology

"Recovery coefficient" names several different quantities in the literature
(RC_max, RC_mean, RC_peak, contrast recovery coefficient), so the two computed
here are named explicitly:

| Name | Definition |
|---|---|
| `contrast_recovery_coefficient_pct` | NEMA NU 2 hot-sphere form, `[(C_s/C_b) − 1] / [(a_s/a_b) − 1]` |
| `relative_activity_recovery` | measured-to-true concentration ratio, `(C_s/C_b) / (a_s/a_b)` |

Neither is called simply "RC": an unqualified RC in a results table is not
reproducible.

## Reconstruction

MLEM is derived from the Poisson likelihood on the assumption that the
back-projector is the **adjoint** of the forward projector. A rotation-based
projector paired with a rotation by the opposite angle does not satisfy that:
bilinear interpolation out and back are different linear maps, and the
iteration is then a plausible-looking scheme whose fixed point is not the ML
estimate.

`gcmc.projector` therefore builds an explicit sparse system matrix `A`, and the
back-projector is `A.T`. Two tests license the name: the adjoint identity
`⟨Ax, y⟩ = ⟨x, Aᵀy⟩` to machine precision on random `x` and `y`, and the
monotonic increase of the Poisson log-likelihood across iterations.

The acquisition operator carries attenuation weights; the reconstruction
operator does not. That difference *is* the absence of attenuation
compensation, expressed as two matrices rather than hidden in an
interpolation.

<sub>Illustrative recovery curve from the analytic surrogate, 8:1 sphere-to-background,
system FWHM computed from `config/` at 100 mm, 60 angles, 30 MLEM iterations, no
attenuation or scatter compensation: relative activity recovery rises from about
0.25 (10 mm) to about 0.44 (37 mm). Monte Carlo results replace it.</sub>

---

## Running without Geant4

The analysis layer ships with an **analytic surrogate projector**
(`gcmc.surrogate`, `gcmc.projector`): geometric projection, attenuation,
Poisson statistics. It models no scatter, no septal penetration and no
backscatter — exactly the effects the Monte Carlo exists to quantify — so no
number it produces may be reported as a simulation result, and its outputs are
labelled `SURROGATE`.

It earns its place because reviewers can run the pipeline without installing
Geant4, CI covers the analysis chain on every commit, and the metrics can be
tested against a case whose answer is known in advance.

```bash
conda env create -f environment.yml && conda activate gamma-camera-monte-carlo
pip install -e .
pytest -q                                  # 64 tests, no Geant4 required
python scripts/analyse.py --surrogate --acquisition acquisition_tomographic.yaml \
       --out results/surrogate_demo
```

`pyproject.toml` sets `pythonpath = ["src"]`, so `pytest -q` also works on a
clean checkout with no install and no network.

With Geant4 + GAMOS (see [`docs/install.md`](docs/install.md)):

```bash
bash scripts/run_simulation.sh config results/run001 42 source_tc99m.yaml \
     acquisition_tomographic.yaml 100000000
```

Switching radionuclide is a configuration change: `source_lu177.yaml` is read
by the geometry generator and the analysis, and the macro takes its emission
spectrum from the generated file rather than a hard-coded energy. The 177Lu
path is functional but **not yet validated**, and is described as such.

---

## Repository layout

```
config/                    camera · phantom · source · acquisition + reconstruction
gamos/
  macros/                  GAMOS macros with no physical parameter in them
  plugins/                 PhotonHistoryUA (lineage + event-level scoring) + CMake
  generated/               geometry, activity map and emission tables (untracked)
src/gcmc/
  config.py                typed configuration, including provenance export
  phantom.py               voxelised six-sphere phantom + GAMOS source export
  collimator.py            hexagonal hole lattice and its GAMOS geometry
  projector.py             sparse system matrix, exact adjoint, attenuation weights
  reconstruction.py        MLEM on that operator + Poisson log-likelihood
  pipeline.py              planar and tomographic paths, config-driven
  detector.py              energy/spatial response, windows, collimator physics
  spectra.py               spectra and primary/scatter decomposition
  planar.py                projection-domain analysis (profiles, uniformity)
  rois.py                  ROI placement
  metrics.py               contrast recovery, relative activity recovery, CV, CNR
  uncertainty.py           replica statistics (primary) and Poisson (planning)
  validation.py            camera performance checks
  hits.py                  event schema and the step-to-event reduction
scripts/                   build_geometry · run_simulation · analyse
tests/                     64 tests, analysis layer only
docs/                      model card · validation · data sources · install · competencies
```

## Validation

| Check | Compared against | Where |
|---|---|---|
| Energy resolution | the value applied in the response model (round trip) | `validation.energy_resolution_pct` |
| Photopeak position | radionuclide decay data | `spectra.photopeak_position` |
| Planar sensitivity | published LEHR values, order of magnitude | `validation.sensitivity_cps_per_MBq` |
| Spatial resolution vs distance | analytical collimator formula with septal-penetration correction | `validation.compare_to_analytical` |
| Scatter fraction vs depth | Monte Carlo SPECT literature | `spectra.scatter_fraction` |
| Collimator lattice | septal thickness and open area recomputed from the generated geometry | `tests/test_collimator.py` |

These checks are chosen to mirror key measurable performance characteristics of
a physical gamma camera. They are not an acceptance procedure and confer no
accreditation.

Agreement is reported as a signed percentage difference with an uncertainty,
never as two curves plotted on top of each other.

## Uncertainty

Two sources, tracked separately because they answer different questions.

- **Replica variability** is the uncertainty reported on every derived result.
  A recovery figure is a ratio of correlated ROI means passed through a
  reconstruction; counting statistics in the sphere ROI do not give its
  uncertainty.
- **Poisson `1/√N`** is used for run planning: `uncertainty.required_histories`
  turns a target relative uncertainty into a number of histories, so run length
  is justified rather than guessed.

## Limitations

- The generic geometry is illustrative; quantitative agreement with a specific
  camera requires that camera's real specifications, which are not published
  here.
- The collimator lattice is modelled hole by hole over a configurable region of
  the face (`--collimator-fov-mm`), not the full field of view: a full FOV at
  this pitch is of the order of 10⁵ holes. A G4 replica or parameterised volume
  is the scalable route and is not implemented yet.
- No pile-up, dead time or position-dependent energy response, so high
  count-rate behaviour is not modelled.
- MLEM has no attenuation or scatter compensation. Background variability is
  consequently dominated by the uncompensated attenuation gradient and is
  **not** comparable with published NEMA values obtained with compensated
  reconstructions.
- The 177Lu configuration runs but is not validated.
- **The C++ layer has not been compiled yet.** It is reviewed and passes
  formatting and static checks, but no Geant4 build has been run against it, so
  it is a documented design for the scoring layer rather than verified code.
  CI cannot change this: a Geant4 build takes hours on a hosted runner. A green
  badge means the analysis chain is sound, not that the simulation built.

## Future work

Attenuation-compensated reconstruction and dual-energy-window scatter
correction, so residual quantification error can be attributed to partial
volume alone; validation of the 177Lu path, where quantification error
propagates directly into absorbed dose.

## Data and provenance

No input dataset is required: the phantom is generated. Public reference data
for materials, attenuation coefficients and decay yields are cited in
[`docs/data_sources.md`](docs/data_sources.md). Simulation outputs are intended
for archival release with a DOI, so results can be re-analysed — different
window, different detector response, different metric — without re-running the
transport.

## Citation

See [`CITATION.cff`](CITATION.cff).

## Licence

MIT.
