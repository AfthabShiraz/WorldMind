#!/usr/bin/env bash
# Launch the viser viewer with the splat + semantic-overlay sidecars.
# Requires that lift-semantics-v2 has been run first.
#
# Usage:
#   scripts/view_semantics.sh --scene <scene_id> [--port 8080] [--max-points N]

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

log "view-semantics: scene=${SCENE}"
python "${REPO_ROOT}/scripts/view_semantics.py" --scene "${SCENE}" "${EXTRA[@]}"
