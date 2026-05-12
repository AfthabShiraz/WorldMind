#!/usr/bin/env bash
# Phase 6 of docs/IMPLEMENTATION_PLAN.md.
#
# Exports the trained splatfacto checkpoint to a standard 3DGS .ply
# (SuperSplat-readable), then renders a handful of training-set views to
# PNG so we can do the Phase 4 "render sanity" check over SSH (no
# interactive viewer needed).
#
# Usage:
#   scripts/export_splat.sh --scene <scene_id>
#       [--profile smoke|full]   (which training run to export; default: smoke if exists else full)
#       [--n-renders N]          (number of training-pose renders; default 6)

source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
activate_env

SCENE=""
PROFILE=""
N_RENDERS=6
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene)    SCENE="$2"; shift 2 ;;
    --profile)  PROFILE="$2"; shift 2 ;;
    --n-renders) N_RENDERS="$2"; shift 2 ;;
    -h|--help) grep -E '^# ' "$0" | sed 's/^# //'; exit 0 ;;
    *) die "unknown arg: $1" ;;
  esac
done
[[ -n "$SCENE" ]] || die "--scene <id> is required"

OUTPUTS="${REPO_ROOT}/$(read_cfg outputs outputs)"
SCENE_OUT="${OUTPUTS}/${SCENE}"

# Pick profile if not specified — prefer 'full' if present, else 'smoke'.
if [[ -z "$PROFILE" ]]; then
  if [[ -d "${SCENE_OUT}/splatfacto_full" ]]; then PROFILE=full
  elif [[ -d "${SCENE_OUT}/splatfacto_smoke" ]]; then PROFILE=smoke
  else die "no trained checkpoint at ${SCENE_OUT}/splatfacto_{smoke,full} (run scripts/train_splat.sh)"
  fi
fi

EXP_DIR="${SCENE_OUT}/splatfacto_${PROFILE}"
[[ -d "$EXP_DIR" ]] || die "expected experiment dir not found: $EXP_DIR"

# Find the latest config.yml under the experiment dir.
CONFIG=$(find "$EXP_DIR" -maxdepth 3 -name 'config.yml' -printf '%T@ %p\n' 2>/dev/null \
         | sort -rn | head -1 | cut -d' ' -f2-)
[[ -n "$CONFIG" ]] || die "no config.yml under $EXP_DIR"
log "scene:    ${SCENE}"
log "profile:  ${PROFILE}"
log "config:   ${CONFIG}"

PLY_DIR="${SCENE_OUT}/ply_${PROFILE}"
mkdir -p "$PLY_DIR"

log "1/2 ns-export gaussian-splat -> ${PLY_DIR}"
ns-export gaussian-splat \
  --load-config "$CONFIG" \
  --output-dir "$PLY_DIR" \
  2>&1 | tee "${PLY_DIR}/export.log"
PLY_FILE=$(find "$PLY_DIR" -maxdepth 1 -name '*.ply' | head -1)
[[ -n "$PLY_FILE" ]] || die "ns-export wrote no .ply under $PLY_DIR"
PLY_SIZE=$(du -h "$PLY_FILE" | cut -f1)
log "  wrote $(basename "$PLY_FILE")  (${PLY_SIZE})"

log "2/2 rendering ${N_RENDERS} training-pose views to PNG (for SSH eyeball check)"
RENDER_DIR="${SCENE_OUT}/_renders_${PROFILE}"
rm -rf "$RENDER_DIR"
ns-render dataset \
  --load-config "$CONFIG" \
  --output-path "$RENDER_DIR" \
  --rendered-output-names rgb \
  --split train 2>&1 | tee "${RENDER_DIR}.log" | tail -5

# ns-render dataset writes outputs/<exp>/_renders/<split>/<output_name>/<NNNNN>.png
# (or similar). Find the rgb pngs and downsample to N_RENDERS evenly-spaced ones.
RGB_DIR=$(find "$RENDER_DIR" -maxdepth 4 -type d -name 'rgb' | head -1)
if [[ -z "$RGB_DIR" ]]; then
  RGB_DIR=$(find "$RENDER_DIR" -maxdepth 4 -type d | tail -1)
fi
log "  render dir: $RGB_DIR"
N_TOTAL=$(find "$RGB_DIR" -maxdepth 1 -name '*.png' 2>/dev/null | wc -l)
log "  rendered ${N_TOTAL} training views"

# Pick N_RENDERS evenly spaced and copy them out to a flat dir for easy reading.
QC_DIR="${SCENE_OUT}/_qc/renders_${PROFILE}"
mkdir -p "$QC_DIR"
"${CONDA_PREFIX}/bin/python" - "$RGB_DIR" "$QC_DIR" "$N_RENDERS" <<'PY'
import sys, pathlib, shutil
src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
n = int(sys.argv[3])
# ns-render writes .jpg in 1.1.5; older versions wrote .png — handle both.
imgs = sorted([*src.glob('*.png'), *src.glob('*.jpg')])
if not imgs:
    sys.exit("no .png/.jpg in render dir")
indices = [int(i * (len(imgs) - 1) / max(n - 1, 1)) for i in range(n)]
for k, i in enumerate(indices):
    out = dst / f"sample_{k:02d}_{imgs[i].name}"
    shutil.copy2(imgs[i], out)
    print(f"  picked {imgs[i].name} -> {out.name}")
PY

log ""
log "export + render complete."
log "  .ply:     $PLY_FILE  (${PLY_SIZE})"
log "  renders:  ${QC_DIR}/  (${N_RENDERS} evenly-spaced training views)"
log ""
log "Open the .ply in https://playcanvas.com/supersplat by drag-and-drop."
