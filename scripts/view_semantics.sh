#!/usr/bin/env bash
# Launch the in-browser splat viewer (viser). Semantic-overlay sidecars are
# optional — without them the viewer just shows the raw splat.
#
# Usage:
#   scripts/view_semantics.sh --scene <scene_id> [--port 8080] [--max-points N] [--share]
#
# Activates the project conda env when present, otherwise falls back to the
# current python as long as viser+plyfile+numpy are importable. That lets
# viewer-only users skip the heavy training-stack install:
#   python -m venv .venv && source .venv/bin/activate
#   pip install -r requirements-viewer.txt

source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

# Try the project conda env first; otherwise tolerate a lightweight pip env
# as long as the viewer's deps are importable in whatever python is on PATH.
WORLDMIND_PREFIX="${WORLDMIND_CONDA_PREFIX:-$HOME/miniforge3}"
WORLDMIND_ENV_NAME="${WORLDMIND_ENV:-worldmind}"
if [[ -d "${WORLDMIND_PREFIX}/envs/${WORLDMIND_ENV_NAME}" ]]; then
  activate_env
elif python3 -c 'import viser, plyfile, numpy' 2>/dev/null; then
  log "no conda env at ${WORLDMIND_PREFIX}/envs/${WORLDMIND_ENV_NAME}; using current python"
else
  die "no conda env and current python lacks viewer deps. Install one of:
  - full env (training+viewer):    make install
  - viewer-only (lightweight):     pip install -r requirements-viewer.txt"
fi

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

log "view: scene=${SCENE}"
# Prefer python3, fall back to python (handles minimal venv setups).
PY=$(command -v python3 || command -v python)
"$PY" "${REPO_ROOT}/scripts/view_semantics.py" --scene "${SCENE}" "${EXTRA[@]}"
