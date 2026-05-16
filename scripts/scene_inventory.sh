#!/usr/bin/env bash
# What does the VLM see in this room? Samples keyframes from a scene and asks
# Qwen2.5-VL to list significant objects. No SAM, no 3D — just a sanity check.
#
# Usage:
#   scripts/scene_inventory.sh --scene <scene_id> [--n-keyframes 15] [--show-raw]

source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
activate_env

SCENE=""
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene) SCENE="$2"; shift 2 ;;
    -h|--help) grep -E '^# ' "$0" | sed 's/^# //'; exit 0 ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done
[[ -n "$SCENE" ]] || die "--scene <id> is required"

SCENE_DIR="${REPO_ROOT}/data/scenes/${SCENE}"
[[ -f "${SCENE_DIR}/transforms.json" ]] || die "no transforms.json at ${SCENE_DIR}"

log "scene-inventory: scene=${SCENE}"
python "${REPO_ROOT}/scripts/scene_inventory.py" --scene "${SCENE}" "${EXTRA[@]}"
