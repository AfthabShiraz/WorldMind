#!/usr/bin/env bash
# Download the room5 example splat.ply from the project's GitHub release.
#
# The .ply (~150 MB) is too large for the git repo, so it's distributed as a
# release asset. This script fetches it once into outputs/room5/ply_full/
# where the viewer + Makefile expect it.
#
# Usage:
#   scripts/fetch_example.sh                    # default URL + path
#   EXAMPLE_PLY_URL=<url> scripts/fetch_example.sh
#
# Idempotent: if the .ply already exists locally, does nothing.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST_DIR="${REPO}/outputs/room5/ply_full"
DEST="${DEST_DIR}/splat.ply"
URL="${EXAMPLE_PLY_URL:-https://github.com/AfthabShiraz/WorldMind/releases/download/v0.1-room5/splat.ply}"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

if [[ -f "$DEST" ]]; then
  size=$(stat -c %s "$DEST" 2>/dev/null || stat -f %z "$DEST")
  log "splat.ply already present (${size} bytes) — skipping download"
  log "remove $DEST to force re-download"
  exit 0
fi

mkdir -p "$DEST_DIR"
log "downloading example scene from:"
log "  $URL"
log "  -> $DEST"

if command -v curl >/dev/null; then
  curl --fail --location --progress-bar --retry 3 --output "$DEST" "$URL"
elif command -v wget >/dev/null; then
  wget --tries=3 -O "$DEST" "$URL"
else
  echo "ERROR: need curl or wget on PATH" >&2
  exit 2
fi

size=$(stat -c %s "$DEST" 2>/dev/null || stat -f %z "$DEST")
log "done. ${size} bytes -> $DEST"
