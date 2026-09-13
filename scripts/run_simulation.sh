#!/usr/bin/env bash
# Launch a GAMOS run. Requires a working Geant4 + GAMOS installation; see
# docs/install.md. Not exercised by CI.
set -euo pipefail

CONFIG_DIR=${1:-config}
OUT=${2:-results/run_$(date +%Y%m%d_%H%M%S)}
SEED=${3:-42}
SOURCE=${4:-source_tc99m.yaml}
ACQUISITION=${5:-acquisition_planar.yaml}
HISTORIES=${6:-100000000}

mkdir -p "$OUT"
python scripts/build_geometry.py --config-dir "$CONFIG_DIR" \
       --source "$SOURCE" --out gamos/generated

cp -r "$CONFIG_DIR" "$OUT/config"          # the run carries its own configuration
gamos gamos/macros/projection.in \
      -DSEED="$SEED" \
      -DOUTPUT="$OUT/events.csv" \
      -DHISTORIES="$HISTORIES" \
      | tee "$OUT/simulation.log"

python - <<PY
import json, subprocess, sys
meta = {"seed": $SEED,
        "git_commit": subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip(),
        "python": sys.version.split()[0]}
open("$OUT/environment.json","w").write(json.dumps(meta, indent=2))
PY

python scripts/analyse.py --events "$OUT/events.csv" --config-dir "$CONFIG_DIR" \
       --source "$SOURCE" --acquisition "$ACQUISITION" --out "$OUT"
