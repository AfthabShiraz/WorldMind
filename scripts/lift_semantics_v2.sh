#!/usr/bin/env bash
# Robust per-Gaussian semantics: depth-aware voting, cross-view mask
# association, one VLM label per instance, sidecar files only — splat.ply is
# never modified. See scripts/lift_semantics_v2.py for stage details.
#
# Usage:
#   scripts/lift_semantics_v2.sh --scene <scene_id>
#       [--n-keyframes N]
#       [--only keyframes|masks|voters|associate|label|lift|export]
#       [--force]

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
PLY="${REPO_ROOT}/outputs/${SCENE}/ply_full/splat.ply"
[[ -f "${SCENE_DIR}/transforms.json" ]] || die "no transforms.json at ${SCENE_DIR}"
[[ -f "${PLY}" ]] || die "no splat.ply at ${PLY} (run train-full + export-splat first)"

log "lift-semantics-v2: scene=${SCENE}"
python "${REPO_ROOT}/scripts/lift_semantics_v2.py" --scene "${SCENE}" "${EXTRA[@]}"
log "lift-semantics-v2: done. sidecars under semantics/${SCENE}/ (splat.ply untouched)"
