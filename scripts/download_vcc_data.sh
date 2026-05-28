#!/usr/bin/env bash
# Fetch the Virtual Cell Challenge data from the official Arc Institute GCS bucket.
#
# Usage:
#   scripts/download_vcc_data.sh [destination_dir]
#
# Default destination: data/vcc/
# Prerequisites:
#   - gcloud CLI installed and authenticated
#     (gcloud auth application-default login)
#   - gsutil available
#
# Note: the agent is free to inspect the bucket structure first to figure out
# the actual file layout. The Arc bucket structure may change, so this script
# does a recursive copy of the whole challenge directory.

set -euo pipefail

DEST="${1:-data/vcc}"
SRC="gs://arc-institute-virtual-cell-atlas/virtual-cell-challenge"

mkdir -p "$DEST"

if ! command -v gsutil >/dev/null 2>&1; then
    echo "gsutil not on PATH. Install the Google Cloud SDK first." >&2
    echo "https://cloud.google.com/sdk/docs/install" >&2
    exit 2
fi

echo "Fetching VCC data from $SRC into $DEST ..."
gsutil -m cp -r "$SRC/*" "$DEST/"
echo "Done. Listing top-level contents:"
ls -lh "$DEST"
