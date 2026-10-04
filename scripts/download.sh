#!/usr/bin/env bash
# Download CAMELYON16 slides + ASAP XML annotations from the public AWS bucket.
# Usage: scripts/download.sh tumor_031 [tumor_016 ...]
set -euo pipefail
BASE=https://camelyon-dataset.s3.amazonaws.com/CAMELYON16
OUT="$(dirname "$0")/../data/raw"
mkdir -p "$OUT"
for id in "$@"; do
  curl -fL -C - -o "$OUT/$id.xml" "$BASE/annotations/$id.xml"
  curl -fL -C - -o "$OUT/$id.tif" "$BASE/images/$id.tif"   # -C - resumes interrupted downloads
done
