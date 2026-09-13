# Simulation layer

> **Status: written, not yet compiled.** The plug-in sources here have been
> reviewed and pass a formatting and static-analysis check, but they have not
> been built against a Geant4 installation. Treat them as a documented design
> for the scoring layer until this notice is removed. The Python analysis layer
> is fully tested and does not depend on them.

Everything under this directory needs a working Geant4 + GAMOS installation and
is therefore **not covered by continuous integration**; CI exercises the Python
analysis layer only. This is stated explicitly rather than hidden behind a green
badge.

```
macros/      GAMOS macro templates (no hard-coded camera parameters)
plugins/     C++ user actions compiled against Geant4/GAMOS
generated/   geometry and source tables written by scripts/build_geometry.py
             (not committed)
```

The one thing the C++ layer adds that macros cannot: `PhotonHistoryUA` tags every
detected photon with the number of Compton interactions it underwent, which is
what makes the primary/scatter decomposition in `gcmc.spectra` exact.

Build:

```bash
cd gamos/plugins && cmake -S . -B build && cmake --build build
export GAMOS_USER_LIBS=$PWD/build/libPhotonHistoryUA.so
```

See `docs/install.md`.
