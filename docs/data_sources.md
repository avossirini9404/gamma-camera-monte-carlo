# Reference data and provenance

This repository requires no input dataset: the phantom is generated. The public
reference data it does rely on are listed here so that every constant can be
traced.

| Quantity | Source |
|---|---|
| Sphere diameters (10, 13, 17, 22, 28, 37 mm) | NEMA NU 2 image-quality phantom specification |
| Linear attenuation coefficients at 140.5 keV | NIST XCOM photon cross-section tabulations |
| Tissue and material compositions | ICRU/ICRP published compositions |
| Tc-99m and Lu-177 photon energies and yields | public decay-data libraries (e.g. NNDC/DDEP) |
| Collimator resolution expression | standard parallel-hole formulation, medical physics literature |
| Camera dimensions in `config/` | **none — generic illustrative values chosen by the author** |

The last row is the important one. No parameter in `config/camera_generic_lehr.yaml`
is taken from manufacturer documentation of a commercial system.

## Scientific precedents

The design follows two lines of published work:

1. **Monte Carlo SPECT modelling with NEMA/IEC image-quality phantoms**, which
   establishes the practice of validating a simulated camera against energy
   spectra, sensitivity and spatial resolution before reporting recovery
   coefficients.
2. **Dedicated gamma-camera simulation codes** such as SIMIND, used with
   Jaszczak-type and manufacturer-specific camera models, as a complementary
   precedent for how a camera model is validated and reported.

This repository does not reproduce either; it applies their methodology to a
generic camera with a fully open, reproducible analysis chain.

## Output data

Simulation outputs (hit tables and derived spectra) are intended for archival
release with a DOI. Publishing the hit table matters more than publishing the
figures: it lets a third party re-derive every result — different energy
window, different detector response, different metric — without re-running the
transport.
