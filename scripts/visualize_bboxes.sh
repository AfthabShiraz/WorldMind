#!/usr/bin/env bash
# Render the bounding boxes Qwen drew on each keyframe, colour-coded by
# whether they were accepted or which safety filter rejected them.
# Reads semantics/<scene>/object_anchors.json (from place-labels), writes
# JPEGs to semantics/<scene>/bbox_audit/.
#
# Usage:
#   scripts/visualize_bboxes.sh --scene <scene_id> [--accepted-only]

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

log "visualize-bboxes: scene=${SCENE}"
python "${REPO_ROOT}/scripts/visualize_bboxes.py" --scene "${SCENE}" "${EXTRA[@]}"
