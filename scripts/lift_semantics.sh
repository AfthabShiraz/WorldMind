#!/usr/bin/env bash
# Phase 8 of docs/IMPLEMENTATION_PLAN.md.
#
# SAM masks + Qwen2.5-VL labels lifted to Gaussians for a trained scene.
# Outputs land in semantics/<scene>/ — see lift_semantics.py for the artifact
# layout. The Python script does all the work; this wrapper just standardises
# logging and env activation with the rest of the repo.
#
# Usage:
#   scripts/lift_semantics.sh --scene <scene_id>
#       [--n-keyframes N]
#       [--only keyframes|masks|labels|lift]
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

log "lift-semantics: scene=${SCENE}"
python "${REPO_ROOT}/scripts/lift_semantics.py" --scene "${SCENE}" "${EXTRA[@]}"
log "lift-semantics: done. outputs under semantics/${SCENE}/"
