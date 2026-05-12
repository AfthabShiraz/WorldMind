#!/usr/bin/env bash
# Phase 4 (smoke) and Phase 5 (full) of docs/IMPLEMENTATION_PLAN.md.
#
# Trains a 3D Gaussian splat (splatfacto) on data/scenes/<scene>/, writes
# outputs to outputs/<scene>/, and captures run metadata.
#
# Usage:
#   scripts/train_splat.sh --scene <scene_id>
#       [--profile smoke|full]               (smoke=5k iters [Phase 4], full=30k [Phase 5])
#       [--max-num-iterations N]             (overrides profile default)
#       [--extra "additional ns-train args"] (e.g. "--pipeline.model.use-bilateral-grid True")
#       [--force]                            (delete outputs/<scene>/splatfacto/ first)

source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
activate_env

SCENE=""
PROFILE="smoke"
MAX_ITERS=""
EXTRA=""
FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene)              SCENE="$2"; shift 2 ;;
    --profile)            PROFILE="$2"; shift 2 ;;
    --max-num-iterations) MAX_ITERS="$2"; shift 2 ;;
    --extra)              EXTRA="$2"; shift 2 ;;
    --force)              FORCE=1; shift ;;
    -h|--help) grep -E '^# ' "$0" | sed 's/^# //'; exit 0 ;;
    *) die "unknown arg: $1" ;;
  esac
done
[[ -n "$SCENE" ]] || die "--scene <id> is required"

# Profile -> iteration budget.
case "$PROFILE" in
  smoke) MAX_ITERS="${MAX_ITERS:-$(read_cfg train.smoke_iters 5000)}" ;;
  full)  MAX_ITERS="${MAX_ITERS:-$(read_cfg train.full_iters 30000)}" ;;
  *)     die "--profile must be 'smoke' or 'full', got '$PROFILE'" ;;
esac

DATA_SCENES="${REPO_ROOT}/$(read_cfg data_scenes data/scenes)"
OUTPUTS="${REPO_ROOT}/$(read_cfg outputs outputs)"
SCENE_DIR="${DATA_SCENES}/${SCENE}"
OUT_DIR="${OUTPUTS}/${SCENE}"

[[ -d "$SCENE_DIR" ]] || die "scene dir not found: $SCENE_DIR  (run scripts/process_data.sh first)"
[[ -f "${SCENE_DIR}/transforms.json" ]] || die "no transforms.json — re-run process_data, this scene isn't packaged correctly"

# nerfstudio writes outputs/<exp>/<method>/<timestamp>/, so isolate this run
# under the scene's output dir with a clean experiment name we control.
EXP_DIR="${OUT_DIR}/splatfacto_${PROFILE}"
if [[ -d "$EXP_DIR" ]]; then
  if [[ $FORCE -eq 1 ]]; then
    log "removing existing $EXP_DIR (--force)"
    rm -rf "$EXP_DIR"
  else
    die "experiment dir already exists: $EXP_DIR  (use --force, or pick another --profile)"
  fi
fi
mkdir -p "$EXP_DIR"

log "scene:           ${SCENE}"
log "profile:         ${PROFILE}"
log "iterations:      ${MAX_ITERS}"
log "data dir:        ${SCENE_DIR}"
log "output dir:      ${EXP_DIR}"

# --vis tensorboard: no interactive viewer (SSH-friendly).
# --viewer.quit-on-train-completion: process exits when training is done.
START_TS=$(date +%s)
set -x
ns-train splatfacto \
  --data "$SCENE_DIR" \
  --output-dir "$OUT_DIR" \
  --experiment-name "splatfacto_${PROFILE}" \
  --method-name "splatfacto" \
  --max-num-iterations "$MAX_ITERS" \
  --vis tensorboard \
  --viewer.quit-on-train-completion True \
  --logging.local-writer.enable True \
  --steps-per-save 5000 \
  ${EXTRA} \
  2>&1 | tee "${EXP_DIR}/train.log"
NS_EXIT=${PIPESTATUS[0]}
set +x
END_TS=$(date +%s)
[[ $NS_EXIT -eq 0 ]] || die "ns-train exited ${NS_EXIT}; see ${EXP_DIR}/train.log"

DURATION=$(( END_TS - START_TS ))

# The actual run dir is outputs/<scene>/splatfacto_<profile>/splatfacto/<timestamp>/
# Find the most recently created one for downstream wrappers.
LATEST_RUN=$(find "$EXP_DIR" -maxdepth 3 -name 'config.yml' -printf '%T@ %p\n' 2>/dev/null \
             | sort -rn | head -1 | cut -d' ' -f2-)
[[ -n "$LATEST_RUN" ]] || die "could not find config.yml after training"
log "trained config: $LATEST_RUN"

python "${REPO_ROOT}/scripts/_run_meta.py" \
  --stage "train_splat_${PROFILE}" \
  --scene "$SCENE" \
  --out "${EXP_DIR}/run_meta.json" \
  --arg=ns-train \
  --arg=splatfacto \
  "--arg=--data" "--arg=${SCENE_DIR}" \
  "--arg=--max-num-iterations" "--arg=${MAX_ITERS}" \
  --extra-json "$(printf '{"duration_seconds": %d, "profile": "%s", "max_iterations": %d, "config_path": "%s"}' \
                   "$DURATION" "$PROFILE" "$MAX_ITERS" "$LATEST_RUN")"

log ""
log "training finished in ${DURATION}s"
log "config:  $LATEST_RUN"
log "next:    scripts/export_splat.sh --scene ${SCENE} --profile ${PROFILE}"
