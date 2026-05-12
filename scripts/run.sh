#!/usr/bin/env bash
# WorldMind one-shot pipeline: video -> trained .ply + previews.
#
# Drop a video anywhere on disk, hand its path to this script, get back a
# Gaussian-splat .ply you can drag into https://playcanvas.com/supersplat.
# No hardcoded paths, no copying things into data/raw manually.
#
# Usage:
#   scripts/run.sh <video_path> [scene_id]
#   scripts/run.sh --video <path> [--scene <id>] [--profile smoke|full]
#                  [--focal-length N] [--num-frames-target N]
#
# Defaults:
#   scene_id  = basename of video (sanitized to [a-z0-9_-])
#   profile   = full  (~30k iters, ~10 min on Blackwell GB10)
#
# What happens:
#   1. copy your video -> data/raw/<scene_id>.<ext>
#   2. extract frames + run COLMAP with focal length seeded for typical phone cameras
#   3. train splatfacto (smoke or full)
#   4. export the .ply and render 6 training-view previews for offline QC
#
# Output:
#   outputs/<scene_id>/ply_<profile>/splat.ply   <- drag into SuperSplat
#   outputs/<scene_id>/_qc/renders_<profile>/    <- preview JPGs

source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"

VIDEO=""
SCENE=""
PROFILE="full"
EXTRA_FLAGS=()
SKIP_TRAIN=0

usage() {
  grep -E '^# ' "$0" | sed 's/^# \?//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --video|-v)            VIDEO="$2"; shift 2 ;;
    --scene|-s)            SCENE="$2"; shift 2 ;;
    --profile)             PROFILE="$2"; shift 2 ;;
    --focal-length)        EXTRA_FLAGS+=(--focal-length "$2"); shift 2 ;;
    --num-frames-target)   EXTRA_FLAGS+=(--num-frames-target "$2"); shift 2 ;;
    --matching-method)     EXTRA_FLAGS+=(--matching-method "$2"); shift 2 ;;
    --skip-train)          SKIP_TRAIN=1; shift ;;  # only run frame+COLMAP stage
    -h|--help)             usage; exit 0 ;;
    --*)                   die "unknown flag: $1" ;;
    *)
      if [[ -z "$VIDEO" ]]; then
        VIDEO="$1"; shift
      elif [[ -z "$SCENE" ]]; then
        SCENE="$1"; shift
      else
        die "unknown positional arg: $1 (already have video=$VIDEO scene=$SCENE)"
      fi
      ;;
  esac
done

[[ -n "$VIDEO" ]] || { usage; echo; die "video path required"; }
[[ -f "$VIDEO" ]] || die "no file at: $VIDEO"

# Resolve absolute path so cp/symlink isn't sensitive to cwd.
VIDEO_ABS="$(readlink -f "$VIDEO")"

# Derive scene_id from filename if user didn't specify.
if [[ -z "$SCENE" ]]; then
  base="$(basename "$VIDEO_ABS")"
  base="${base%.*}"  # strip extension
  # Sanitize to a portable identifier: lowercase, only [a-z0-9_-].
  SCENE="$(printf '%s' "$base" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9_-' '_' | sed 's/_*$//;s/^_*//')"
fi
[[ -n "$SCENE" ]] || die "could not derive scene_id from filename"

case "$PROFILE" in smoke|full) ;; *) die "--profile must be 'smoke' or 'full', got '$PROFILE'";; esac

activate_env

# Stage videos into data/raw/. If the file is already inside data/raw/ at the
# right name, do nothing. Otherwise copy (cheap on local disk; we keep the user's
# original where it was).
DATA_RAW="${REPO_ROOT}/data/raw"
mkdir -p "$DATA_RAW"
ext="${VIDEO_ABS##*.}"
DEST="${DATA_RAW}/${SCENE}.${ext,,}"
if [[ "$VIDEO_ABS" != "$DEST" ]]; then
  log "staging: ${VIDEO_ABS}"
  log "     -> ${DEST}"
  cp -f "$VIDEO_ABS" "$DEST"
fi

log ""
log "===================== WorldMind run ====================="
log "scene_id:        $SCENE"
log "input video:     $DEST"
log "profile:         $PROFILE"
log "[skip-train:     ${SKIP_TRAIN}]"
log "========================================================="

# ----------------------------------------------------------------------------
# Stage 1 — frames + COLMAP (with focal-length seed for phone cameras).
# Always uses scripts/colmap_pipeline.sh, not ns-process-data, because phone
# videos without EXIF focal length silently break the latter — see commit log.
# ----------------------------------------------------------------------------
log ""
log "=== [1/3] frames + COLMAP ==="
bash "${REPO_ROOT}/scripts/colmap_pipeline.sh" \
  --scene "$SCENE" --force "${EXTRA_FLAGS[@]}"

if [[ $SKIP_TRAIN -eq 1 ]]; then
  log ""
  log "--skip-train set; stopping after process-data."
  log "to QC: bash scripts/qc_process_data.sh --scene $SCENE"
  log "to render sparse cloud: python scripts/qc_render_sparse.py --scene $SCENE"
  exit 0
fi

# ----------------------------------------------------------------------------
# Stage 2 — train splatfacto.
# ----------------------------------------------------------------------------
log ""
log "=== [2/3] train splatfacto (profile=$PROFILE) ==="
bash "${REPO_ROOT}/scripts/train_splat.sh" \
  --scene "$SCENE" --profile "$PROFILE" --force

# ----------------------------------------------------------------------------
# Stage 3 — export .ply + previews.
# ----------------------------------------------------------------------------
log ""
log "=== [3/3] export .ply + render previews ==="
bash "${REPO_ROOT}/scripts/export_splat.sh" \
  --scene "$SCENE" --profile "$PROFILE"

# ----------------------------------------------------------------------------
# Summary.
# ----------------------------------------------------------------------------
PLY="${REPO_ROOT}/outputs/${SCENE}/ply_${PROFILE}/splat.ply"
QC_DIR="${REPO_ROOT}/outputs/${SCENE}/_qc/renders_${PROFILE}"
SPARSE_QC="${REPO_ROOT}/outputs/${SCENE}/_qc"

log ""
log "======================== DONE ========================="
log "scene_id:    $SCENE"
if [[ -f "$PLY" ]]; then
  log "splat.ply:   $PLY ($(du -h "$PLY" | cut -f1))"
else
  log "splat.ply:   MISSING ($PLY)"
fi
log "renders:     $QC_DIR"
log "sparse QC:   $SPARSE_QC/{top_down,side}.png  (re-run: python scripts/qc_render_sparse.py --scene $SCENE)"
log ""
log "View: drag $PLY into https://playcanvas.com/supersplat"
log "======================================================="
