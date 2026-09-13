# Model card

What this model is, what it is not, and under which assumptions each number it
produces is valid.

## Intended use

Research on the quantitative accuracy of gamma-camera imaging: the propagation
of scatter, spatial resolution and counting statistics into activity
concentration estimates. It is a teaching and research model, not a device, not
a substitute for acceptance testing of a physical camera, and not a source of
clinical values.

## What is modelled

| Component | Treatment |
|---|---|
| Photon transport | Geant4/GAMOS, low-energy electromagnetic physics |
| Collimator | hexagonal parallel-hole lattice generated hole by hole from hole diameter, septal thickness and length, over a configurable region of the face |
| Crystal | NaI(Tl), fixed thickness, energy deposition scored per interaction |
| Scatter | explicit, with the Compton interaction count scored per photon |
| Energy resolution | applied in post-processing, FWHM proportional to sqrt(E) |
| Intrinsic spatial resolution | Gaussian blur in the projection domain |
| Attenuation | explicit in the phantom materials |
| Reconstruction | MLEM on an explicit sparse system matrix whose back-projector is its exact transpose; parallel-beam; no attenuation or scatter compensation |
| Scatter classification | photon lineage propagated from parent track to secondary, so fluorescence and bremsstrahlung photons are never scored as primaries |
| Detection | one event per Geant4 event: energy summed over steps, position from the energy-weighted centroid |

## What is not modelled

- Pile-up, dead time and count-rate dependent behaviour.
- Position-dependent energy response and linearity/uniformity corrections.
- Depth-of-interaction effects on the spatial response beyond the scored
  interaction position.
- Light collection, photomultiplier statistics and their contribution to the
  intrinsic resolution: the intrinsic resolution is an input, not an outcome.
- Attenuation and scatter compensation in reconstruction.
- The full field of view of the collimator: the lattice is modelled hole by
  hole over a configurable region, since a full face at this pitch is of the
  order of 10^5 holes. A G4 replica or parameterised volume is the scalable
  route and is not implemented.
- Pulse pile-up between events: each Geant4 event is treated as one detection,
  which is the zero-dead-time limit.

## Assumptions that change the reported numbers

Every image-quality figure is a property of the whole chain. Changing any of
the following changes the recovery coefficients, and each is recorded with each
result:

1. System spatial resolution (collimator design and source-to-collimator
   distance).
2. Total counts in the acquisition.
3. MLEM iteration number — recovery rises monotonically before convergence.
4. Sphere-to-background activity ratio.
5. ROI diameter convention.
6. Whether attenuation is compensated (it is not, in this version).
7. The uncertainty quoted: replica variability across seeds, not 1/sqrt(N) in
   the ROI. A recovery figure is a ratio of correlated ROI means passed through
   a reconstruction, so counting statistics do not give its uncertainty.

## Known behaviour worth flagging

Without attenuation compensation, background variability in the reconstructed
slice is dominated by the attenuation gradient rather than by counting noise.
The CV values are therefore high and are **not** comparable with published NEMA
background variability figures obtained with compensated reconstructions.

## Validity of the surrogate projector

`gcmc.surrogate` models geometry, attenuation and Poisson statistics only. It
exists to test the analysis chain and to let the pipeline run without Geant4.
Any figure derived from it is labelled SURROGATE and must not be presented as a
simulation result.
