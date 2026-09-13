"""gcmc — analysis layer of a Monte Carlo gamma-camera model.

The package is deliberately independent of Geant4/GAMOS: it consumes hit
files produced by the simulation, and it can also run end to end on an
analytic surrogate projector (``gcmc.surrogate``) so that the whole analysis
chain is testable, reviewable and reproducible without a Geant4 installation.
"""

__version__ = "0.1.0"
