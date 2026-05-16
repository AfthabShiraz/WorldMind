#!/usr/bin/env bash
# Download the room5 example assets from the project's GitHub release:
#
#   - splat.ply              (~150 MB) -> outputs/room5/ply_full/splat.ply
#   - scene_inventory.json   (~12 KB)  -> semantics/room5/scene_inventory.json
#
# Both are too large / not-versioned to live in the git repo. This script
# fetches them once where the viewer + Makefile expect them. Idempotent: if
# a file is already present locally, that download is skipped.
#
# Usage:
#   scripts/fetch_example.sh                              # default URLs
#   EXAMPLE_PLY_URL=<url> EXAMPLE_INVENTORY_URL=<url> scripts/fetch_example.sh

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PLY_DIR="${REPO}/outputs/room5/ply_full"
PLY_DEST="${PLY_DIR}/splat.ply"
PLY_URL="${EXAMPLE_PLY_URL:-https://github.com/AfthabShiraz/WorldMind/releases/download/v0.1-room5/splat.ply}"

INV_DIR="${REPO}/semantics/room5"
INV_DEST="${INV_DIR}/scene_inventory.json"
INV_URL="${EXAMPLE_INVENTORY_URL:-https://github.com/AfthabShiraz/WorldMind/releases/download/v0.1-room5/scene_inventory.json}"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

# fetch <dest> <url> <human_label>
fetch() {
  local dest="$1" url="$2" label="$3"
  if [[ -f "$dest" ]]; then
    local size
    size=$(stat -c %s "$dest" 2>/dev/null || stat -f %z "$dest")
    log "${label} already present (${size} bytes) — skipping"
    return 0
  fi
  mkdir -p "$(dirname "$dest")"
  log "downloading ${label}:"
  log "  $url"
  log "  -> $dest"
  if command -v curl >/dev/null; then
    curl --fail --location --progress-bar --retry 3 --output "$dest" "$url"
  elif command -v wget >/dev/null; then
    wget --tries=3 -O "$dest" "$url"
  else
    echo "ERROR: need curl or wget on PATH" >&2
    exit 2
  fi
  local size
  size=$(stat -c %s "$dest" 2>/dev/null || stat -f %z "$dest")
  log "done. ${size} bytes -> $dest"
}

fetch "$PLY_DEST" "$PLY_URL" "splat.ply"

# Inventory is optional — the viewer works without it, just no panel.
# Don't fail the whole script if it 404s (e.g. user is on an older release).
if ! fetch "$INV_DEST" "$INV_URL" "scene_inventory.json"; then
  log "WARN: scene_inventory.json download failed; viewer will run without "
  log "      the inventory panel. (splat.ply is fine.)"
fi
