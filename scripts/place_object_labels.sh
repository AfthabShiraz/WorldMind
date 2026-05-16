#!/usr/bin/env bash
# Anchor 3D floating labels for the strongly-seen objects in
# scene_inventory.json. Uses Qwen2.5-VL grounding + splat z-buffer depth
# proxy with density/depth/AABB safety filters to prevent labels landing
# far in the distance through window holes.
#
# Usage:
#   scripts/place_object_labels.sh --scene <scene_id>
#     [--n-keyframes N] [--min-frames K] [--cluster-eps E]
#     [--max-depth-frac F] [--show-raw] [--force]
#
# Requires: make scene-inventory SCENE=<scene>  to have been run first.

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

INV="${REPO_ROOT}/semantics/${SCENE}/scene_inventory.json"
[[ -f "$INV" ]] || die "no scene_inventory.json at $INV — run: make scene-inventory SCENE=$SCENE"

log "place-object-labels: scene=${SCENE}"
python "${REPO_ROOT}/scripts/place_object_labels.py" --scene "${SCENE}" "${EXTRA[@]}"
