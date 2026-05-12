#!/usr/bin/env bash
# Manual COLMAP pipeline for scenes where ns-process-data's defaults fail.
#
# The reason we need this: ns-process-data lets COLMAP estimate focal length
# from EXIF, but most phone videos (especially transcoded clips) lack EXIF
# focal length. COLMAP then falls back to f = 1.2 * max(W, H), which for a
# phone wide-angle camera is ~2x too long. Wrong focal length -> PnP fails
# for every candidate frame -> 0-1% registration rate.
#
# This wrapper runs feature_extractor with an explicit --ImageReader.camera_params
# baked in (estimated from typical phone wide-cam HFoV ~70-75°), then runs
# matcher + mapper + image_undistorter, then synthesizes a Nerfstudio-compatible
# transforms.json. The matcher and mapper consume the correct intrinsics from
# the start, so matches and verified pairs aren't tainted by wrong projections.
#
# Usage:
#   scripts/colmap_pipeline.sh --scene <id>
#       [--focal-length <px>]                 (default: 0.71 * max(W,H) ≈ HFoV 70°)
#       [--matching-method sequential|exhaustive]
#       [--num-frames-target N]
#       [--force]

source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
activate_env

SCENE=""
FOCAL=""
MATCHING="sequential"
NUM_FRAMES="$(read_cfg process_data.num_frames_target 300)"
FORCE=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene)              SCENE="$2"; shift 2 ;;
    --focal-length)       FOCAL="$2"; shift 2 ;;
    --matching-method)    MATCHING="$2"; shift 2 ;;
    --num-frames-target)  NUM_FRAMES="$2"; shift 2 ;;
    --force)              FORCE=1; shift ;;
    -h|--help) grep -E '^# ' "$0" | sed 's/^# //'; exit 0 ;;
    *) die "unknown arg: $1" ;;
  esac
done
[[ -n "$SCENE" ]] || die "--scene <id> is required"

DATA_RAW="${REPO_ROOT}/$(read_cfg data_raw data/raw)"
DATA_SCENES="${REPO_ROOT}/$(read_cfg data_scenes data/scenes)"

# Resolve input
INPUT=""
for ext in mp4 MP4 mov MOV m4v; do
  cand="${DATA_RAW}/${SCENE}.${ext}"
  [[ -f "$cand" ]] && INPUT="$cand" && break
done
[[ -n "$INPUT" ]] || die "no input video at ${DATA_RAW}/${SCENE}.{mp4,MP4,mov,MOV,m4v}"

OUTDIR="${DATA_SCENES}/${SCENE}"
if [[ -d "$OUTDIR" ]]; then
  if [[ $FORCE -eq 1 ]]; then rm -rf "$OUTDIR"
  else die "output dir exists: ${OUTDIR}  (use --force)"; fi
fi
mkdir -p "$OUTDIR/images" "$OUTDIR/colmap/sparse"

# Probe dimensions, derive focal length default if not given.
W=$(ffprobe -v error -select_streams v:0 -show_entries stream=width  -of default=nw=1:nk=1 "$INPUT")
H=$(ffprobe -v error -select_streams v:0 -show_entries stream=height -of default=nw=1:nk=1 "$INPUT")
if [[ -z "$FOCAL" ]]; then
  # 0.71 ≈ 1 / (2 tan(35°)); HFoV ≈ 70° matches typical phone wide cameras.
  FOCAL=$(python -c "print(round(0.71 * max($W, $H), 1))")
fi
CX=$(python -c "print(round($W / 2, 1))")
CY=$(python -c "print(round($H / 2, 1))")
log "scene:       ${SCENE}"
log "input:       ${INPUT}"
log "outdir:      ${OUTDIR}"
log "dimensions:  ${W} x ${H}"
log "focal:       ${FOCAL} px  (cx=${CX}, cy=${CY})"
log "matching:    ${MATCHING}"
log "frames tgt:  ${NUM_FRAMES}"

# ------------------------------------------------------------------
# 1. Extract frames evenly with ffmpeg. ns-process-data does this with
#    `select` filter; we do the equivalent ourselves so we don't
#    fight its CLI.
# ------------------------------------------------------------------
log "1/5 extracting frames"
NB_FRAMES=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=nw=1:nk=1 "$INPUT")
STRIDE=$(python -c "print(max(1, $NB_FRAMES // $NUM_FRAMES))")
log "  nb_frames=$NB_FRAMES stride=$STRIDE  (will yield ~$((NB_FRAMES / STRIDE)) frames)"
ffmpeg -loglevel error -y -i "$INPUT" \
  -vf "select='not(mod(n\,${STRIDE}))',setpts=N/FRAME_RATE/TB" \
  -vsync vfr -q:v 2 -start_number 1 \
  "${OUTDIR}/images/frame_%05d.png" 2>&1 | tail -3
N_EXTRACTED=$(find "$OUTDIR/images" -name 'frame_*.png' | wc -l)
log "  extracted ${N_EXTRACTED} frames"

# ------------------------------------------------------------------
# 2. Feature extraction WITH explicit camera_params. This is the
#    fix for the wrong-focal-length failure mode.
# ------------------------------------------------------------------
log "2/5 colmap feature_extractor (camera_params seeded)"
colmap feature_extractor \
  --database_path "${OUTDIR}/colmap/database.db" \
  --image_path "${OUTDIR}/images" \
  --ImageReader.single_camera 1 \
  --ImageReader.camera_model OPENCV \
  --ImageReader.camera_params "${FOCAL},${FOCAL},${CX},${CY},0.0,0.0,0.0,0.0" \
  --FeatureExtraction.use_gpu 1 \
  > "${OUTDIR}/colmap_feature_extractor.log" 2>&1
log "  done"

# ------------------------------------------------------------------
# 3. Matching
# ------------------------------------------------------------------
log "3/5 colmap ${MATCHING}_matcher"
colmap "${MATCHING}_matcher" \
  --database_path "${OUTDIR}/colmap/database.db" \
  --FeatureMatching.use_gpu 1 \
  > "${OUTDIR}/colmap_matcher.log" 2>&1
log "  done"

# ------------------------------------------------------------------
# 4. Mapper — incremental reconstruction
# ------------------------------------------------------------------
log "4/5 colmap mapper (this is the slow stage)"
colmap mapper \
  --database_path "${OUTDIR}/colmap/database.db" \
  --image_path "${OUTDIR}/images" \
  --output_path "${OUTDIR}/colmap/sparse" \
  --Mapper.ba_global_function_tolerance=1e-6 \
  > "${OUTDIR}/colmap_mapper.log" 2>&1
log "  done"

# pick the largest sub-model (mapper writes sparse/0, sparse/1, ... when fragmented).
LARGEST_MODEL=""
LARGEST_N=0
for d in "${OUTDIR}/colmap/sparse"/*/; do
  [[ -d "$d" ]] || continue
  N=$(python -c "
import struct
with open('${d}images.bin', 'rb') as f:
    print(struct.unpack('<Q', f.read(8))[0])
")
  log "  model $(basename "$d"): ${N} images"
  if (( N > LARGEST_N )); then LARGEST_N=$N; LARGEST_MODEL="$d"; fi
done
[[ -n "$LARGEST_MODEL" ]] || die "mapper produced no sub-models"
RATE=$(python -c "print(f'{${LARGEST_N}/${N_EXTRACTED}*100:.1f}')")
log "  largest model: $(basename "$LARGEST_MODEL")  ${LARGEST_N}/${N_EXTRACTED} frames  (${RATE}%)"

# Normalise to the standard COLMAP layout: the largest model lives at
# colmap/sparse/0/. Downstream tools (the QC script, splatfacto, etc.)
# assume sparse/0/ is the canonical reconstruction. Move secondary models
# out of the way so they don't confuse loaders.
LARGEST_NAME=$(basename "$LARGEST_MODEL")
if [[ "$LARGEST_NAME" != "0" ]]; then
  log "  promoting $(basename "$LARGEST_MODEL") -> 0"
  TMP_BACKUP="${OUTDIR}/colmap/sparse/_other_models"
  mkdir -p "$TMP_BACKUP"
  for d in "${OUTDIR}/colmap/sparse"/*/; do
    [[ -d "$d" ]] || continue
    name=$(basename "$d")
    [[ "$name" == "_other_models" ]] && continue
    if [[ "$name" == "$LARGEST_NAME" ]]; then continue; fi
    mv "$d" "${TMP_BACKUP}/${name}"
  done
  mv "${OUTDIR}/colmap/sparse/${LARGEST_NAME}" "${OUTDIR}/colmap/sparse/0"
  LARGEST_MODEL="${OUTDIR}/colmap/sparse/0/"
fi

# ------------------------------------------------------------------
# 5. Image undistortion + transforms.json (Nerfstudio-format)
#    Use nerfstudio's helper directly rather than re-deriving the conversion.
# ------------------------------------------------------------------
log "5/5 image_undistorter + transforms.json"
# Standard COLMAP image_undistorter produces images/ + sparse/ in OpenCV undistorted form.
# But splatfacto reads the original layout fine; we just need transforms.json.
python - "$OUTDIR" <<'PY'
import sys, pathlib
out_dir = pathlib.Path(sys.argv[1])
sparse_path = out_dir / "colmap" / "sparse" / "0"  # canonical location after promotion
from nerfstudio.process_data.colmap_utils import colmap_to_json
n_frames = colmap_to_json(
    recon_dir=sparse_path,
    output_dir=out_dir,
    image_id_to_depth_path=None,
    image_rename_map=None,
)
print(f"  wrote {out_dir/'transforms.json'} with {n_frames} frames")
PY

START_TS=$(date +%s)
END_TS=$START_TS  # placeholder so meta has the field
python "${REPO_ROOT}/scripts/_run_meta.py" \
  --stage process_data_manual \
  --scene "$SCENE" \
  --out "${OUTDIR}/run_meta.json" \
  --arg=colmap-pipeline \
  "--arg=--scene" "--arg=${SCENE}" \
  "--arg=--focal-length" "--arg=${FOCAL}" \
  "--arg=--matching-method" "--arg=${MATCHING}" \
  "--arg=--num-frames-target" "--arg=${NUM_FRAMES}" \
  --extra-json "$(printf '{"focal_length_px": %s, "image_size": "%dx%d", "matching_method": "%s", "registered_frames": %d, "extracted_frames": %d, "registration_rate_pct": %s}' \
                   "$FOCAL" "$W" "$H" "$MATCHING" "$LARGEST_N" "$N_EXTRACTED" "$RATE")"

log ""
log "manual COLMAP pipeline finished. registration rate: ${RATE}%"
log "next: scripts/qc_process_data.sh --scene ${SCENE}"
