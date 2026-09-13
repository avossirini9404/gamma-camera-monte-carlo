# Validation plan

The camera model is validated before any image-quality result is reported. The
order is deliberate: image quality measured on an unvalidated camera model is a
number without a referent.

These checks are chosen to mirror key measurable performance characteristics of
a physical gamma camera. They are not an acceptance procedure, and passing them
confers no accreditation.

## 1. Energy response

- **Photopeak position** against radionuclide decay data. Tolerance: 1 keV.
- **Energy resolution round trip.** Blur a monoenergetic line with a known
  FWHM, recover it from the simulated spectrum. Tolerance: 10% relative. This
  is the check that catches a broken histogram binning or a wrong sigma/FWHM
  conversion, and it runs in CI.
- **Scatter tail shape** against Monte Carlo SPECT literature, qualitatively,
  with the primary/scatter decomposition shown separately.

## 2. Sensitivity

Planar sensitivity for a point source in air, in counts per second per MBq,
compared against published values for LEHR collimators as an order-of-magnitude
check. An exact match is not expected and is not claimed: the geometry is
generic.

## 3. Spatial resolution

System resolution against the analytical parallel-hole expression

    R_sys = sqrt(R_g^2 + R_i^2),   R_g = d (l_eff + b) / l_eff

with `l_eff = l - 2/mu` the septal-penetration-corrected hole length. Measured
at several source-to-collimator distances; agreement is reported as a signed
percentage difference at each distance, with its Monte Carlo uncertainty.

## 4. Scatter fraction

Scattered fraction of the counts accepted by the acquisition window, as a
function of source depth in the scattering medium and of window width. Compared
against the range reported in the Monte Carlo SPECT literature. This is the
quantity that most directly justifies the model: it cannot be measured
directly, and the simulation gives it exactly.

## 5. Statistical adequacy

Before a run is accepted, the relative Poisson uncertainty in the smallest
sphere ROI must be below the target set in the analysis configuration.
`gcmc.uncertainty.required_histories` turns that target into the number of
histories to simulate, so the run length is justified rather than guessed.

## 6. Operator validation

Before any reconstructed result is reported, the reconstruction operator itself
is checked:

- **Adjoint identity.** <A x, y> = <x, A^T y> to machine precision on random
  x and y. MLEM is derived from the Poisson likelihood on exactly this
  assumption; without it the iteration is not maximum-likelihood expectation
  maximisation and its fixed point is not the ML estimate.
- **Likelihood monotonicity.** The Poisson log-likelihood must not decrease
  across iterations. A violation is the signature of a non-adjoint pair.

Both run in CI.

## 7. Geometry validation

The collimator lattice is checked against the parameters it claims: the minimum
lead thickness between neighbouring holes is recomputed from the generated
geometry and must equal the configured septal thickness, and the open-area
fraction must match the theoretical hexagonal-packing value.

## Reporting rules

- Every comparison is a signed percentage difference with an uncertainty.
- Agreement is never claimed by superimposing two curves.
- Run-to-run variability is estimated from independent replicas with different
  seeds, not from a single run, and it — not 1/sqrt(N) — is the uncertainty
  quoted on derived metrics such as recovery.
- Sphere image-quality figures are reported only in the reconstructed domain.
