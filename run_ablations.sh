#!/usr/bin/env bash
# Adds the techniques one at a time, then runs Muon on its own against the baseline.
# Extra flags are passed through, e.g.:  ./run_ablations.sh --tokens 5e6   (quick pass)
set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

"$PY" train.py --name 1_baseline                                  "$@"
"$PY" train.py --name 2_rope              --rope                  "$@"
"$PY" train.py --name 3_rope_qknorm       --rope --qknorm         "$@"
"$PY" train.py --name 4_rope_qknorm_relu2 --rope --qknorm --relu2 "$@"
"$PY" train.py --name 5_all_muon          --rope --qknorm --relu2 --muon "$@"
"$PY" train.py --name 6_baseline_muon     --muon                  "$@"

"$PY" compare.py
