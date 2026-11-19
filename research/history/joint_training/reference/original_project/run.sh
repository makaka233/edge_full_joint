#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$ROOT:$ROOT/r12:$ROOT/r9:$ROOT/r41:$ROOT/audit:$ROOT/baseline/edge_round31/source${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$ROOT/baseline/edge_round31/source"
case "${1:-}" in
 train) shift; python "$ROOT/r12/joint.py" --arm A "$@" ;;
 s-only) shift; python "$ROOT/r12/s_only.py" "$@" ;;
 eval) shift; python "$ROOT/r12/eval_min.py" "$@" ;;
 d120) shift; python "$ROOT/r12/d120_min.py" "$@" ;;
 plot) shift; python "$ROOT/r12/plot_min.py" "$@" ;;
 *) echo "Usage: bash run.sh {train|s-only|eval|d120|plot} [arguments]" >&2; exit 2 ;;
esac
