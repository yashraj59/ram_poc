#!/usr/bin/env bash
# Wrap the user's canonical cell-eval invocation.
#
# Usage:
#   scripts/run_cell_eval.sh <pred.h5ad> <real.h5ad> [num_threads]
#
# Defaults to 64 threads, profile vcc.
# Exits non-zero if cell-eval fails or the PDS field cannot be parsed.

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <pred.h5ad> <real.h5ad> [num_threads]" >&2
    exit 2
fi

PRED="$1"
REAL="$2"
THREADS="${3:-64}"

for f in "$PRED" "$REAL"; do
    if [[ ! -f "$f" ]]; then
        echo "Missing file: $f" >&2
        exit 2
    fi
done

if ! command -v cell-eval >/dev/null 2>&1; then
    echo "cell-eval not on PATH. Install Arc Institute's cell-eval package first." >&2
    exit 2
fi

cell-eval run \
    -ap "$PRED" \
    -ar "$REAL" \
    --num-threads "$THREADS" \
    --profile vcc
