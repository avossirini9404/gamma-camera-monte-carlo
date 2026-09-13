# Installation

## Analysis layer only (no Geant4)

Sufficient to run the whole pipeline on the analytic surrogate, to run the test
suite, and to re-analyse hit tables produced elsewhere.

```bash
conda env create -f environment.yml
conda activate gamma-camera-monte-carlo
pip install -e .
pytest -q
```

`pyproject.toml` sets `pythonpath = ["src"]`, so `pytest -q` also works on a
clean checkout without `pip install -e .` — useful on a machine with no network
access to fetch build dependencies. The `scripts/` entry points do need the
package on the path, either through the editable install or via
`PYTHONPATH=src`.

## Simulation layer (Geant4 + GAMOS)

Not installed by the conda environment and not exercised by CI.

1. Install Geant4 with the low-energy electromagnetic data files.
2. Install GAMOS against that Geant4 build; set `GAMOS_INSTALL`.
3. Build the user-action plug-in:

```bash
cd gamos/plugins
cmake -S . -B build && cmake --build build
export GAMOS_USER_LIBS=$PWD/build/libPhotonHistoryUA.so
```

4. Generate the geometry and source tables from the configuration, then run:

```bash
python scripts/build_geometry.py --config-dir config --out gamos/generated
bash scripts/run_simulation.sh config results/run001 42 source_tc99m.yaml \
     acquisition_tomographic.yaml 100000000
```

Version numbers of Geant4, GAMOS and the physics data files are recorded in
each run's `environment.json`; results from different versions are not pooled.
