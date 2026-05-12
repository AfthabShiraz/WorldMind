#!/usr/bin/env bash
# Phase 1 + Phase 2 of docs/IMPLEMENTATION_PLAN.md.
#
# A single `ns-process-data video` invocation extracts frames AND runs COLMAP
# (sequential matching for video). Both phases are produced by this one
# command; the per-phase QC happens *after* via the exit-criteria report.
#
# Usage:
#   scripts/process_data.sh --scene <scene_id>
#       [--num-frames-target N]   (default from configs/paths.yaml; plan suggests 150-300)
#       [--matching-method sequential|exhaustive|vocab_tree]   (default sequential)
#       [--force]                 (delete existing data/scenes/<scene>/ before running)
#
# Input:  data/raw/<scene_id>.mp4   (or .mov / .MP4 — auto-detected)
# Output: data/scenes/<scene_id>/
#           images/                 (Phase 1: undistorted frames)
#           colmap/sparse/0/        (Phase 2: SfM poses + sparse cloud)
#           transforms.json         (Nerfstudio-format poses for splatfacto)
#           run_meta.json           (this script's metadata: CLI, versions, timing)

source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
activate_env

# -------- arg parsing --------
SCENE=""
NUM_FRAMES_TARGET=""
MATCHING_METHOD=""
FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene)              SCENE="$2"; shift 2 ;;
    --num-frames-target)  NUM_FRAMES_TARGET="$2"; shift 2 ;;
    --matching-method)    MATCHING_METHOD="$2"; shift 2 ;;
    --force)              FORCE=1; shift ;;
    -h|--help)
      grep -E '^# ' "$0" | sed 's/^# //'; exit 0 ;;
    *) die "unknown arg: $1" ;;
  esac
done
[[ -n "$SCENE" ]] || die "--scene <id> is required"

NUM_FRAMES_TARGET="${NUM_FRAMES_TARGET:-$(read_cfg process_data.num_frames_target 300)}"
MATCHING_METHOD="${MATCHING_METHOD:-$(read_cfg process_data.matching_method sequential)}"

# -------- resolve input video --------
DATA_RAW="${REPO_ROOT}/$(read_cfg data_raw data/raw)"
INPUT=""
for ext in mp4 MP4 mov MOV m4v; do
  candidate="${DATA_RAW}/${SCENE}.${ext}"
  if [[ -f "$candidate" ]]; then INPUT="$candidate"; break; fi
done
[[ -n "$INPUT" ]] || die "no input video found at ${DATA_RAW}/${SCENE}.{mp4,MP4,mov,MOV,m4v}"

# -------- resolve output dir --------
DATA_SCENES="${REPO_ROOT}/$(read_cfg data_scenes data/scenes)"
OUTDIR="${DATA_SCENES}/${SCENE}"
if [[ -d "$OUTDIR" ]]; then
  if [[ $FORCE -eq 1 ]]; then
    log "removing existing ${OUTDIR} (--force)"
    rm -rf "$OUTDIR"
  else
    die "output dir already exists: ${OUTDIR}  (rerun with --force to overwrite)"
  fi
fi
mkdir -p "$OUTDIR"

# -------- show the plan, then run --------
log "scene:             ${SCENE}"
log "input video:       ${INPUT}"
log "output dir:        ${OUTDIR}"
log "num-frames-target: ${NUM_FRAMES_TARGET}"
log "matching-method:   ${MATCHING_METHOD}"

START_TS=$(date +%s)
set -x
ns-process-data video \
  --data "$INPUT" \
  --output-dir "$OUTDIR" \
  --num-frames-target "$NUM_FRAMES_TARGET" \
  --matching-method "$MATCHING_METHOD" \
  2>&1 | tee "${OUTDIR}/ns_process_data.log"
NS_EXIT=${PIPESTATUS[0]}
set +x
END_TS=$(date +%s)
[[ $NS_EXIT -eq 0 ]] || die "ns-process-data exited ${NS_EXIT}; see ${OUTDIR}/ns_process_data.log"

DURATION=$(( END_TS - START_TS ))

# -------- write run_meta.json --------
# Use --arg=VAL form so values starting with '-' aren't mistaken for new flags.
python "${REPO_ROOT}/scripts/_run_meta.py" \
  --stage process_data \
  --scene "$SCENE" \
  --out "${OUTDIR}/run_meta.json" \
  --arg=ns-process-data \
  --arg=video \
  "--arg=--data" "--arg=${INPUT}" \
  "--arg=--output-dir" "--arg=${OUTDIR}" \
  "--arg=--num-frames-target" "--arg=${NUM_FRAMES_TARGET}" \
  "--arg=--matching-method" "--arg=${MATCHING_METHOD}" \
  --extra-json "$(printf '{"duration_seconds": %d, "num_frames_target": %d, "matching_method": "%s"}' \
                   "$DURATION" "$NUM_FRAMES_TARGET" "$MATCHING_METHOD")"

log ""
log "process_data finished in ${DURATION}s"
log "next: scripts/qc_process_data.sh --scene ${SCENE}   (Phase 1 + Phase 2 exit-criteria checks)"
