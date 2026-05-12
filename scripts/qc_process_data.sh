#!/usr/bin/env bash
# Exit-criteria checks for Phase 1 (frame extraction) and Phase 2 (COLMAP poses)
# per docs/IMPLEMENTATION_PLAN.md. Run after scripts/process_data.sh.
#
# Phase 1 (mechanical):
#   - Count sanity:     ~150-300 frames in images/
#   - Order test:       prints first/last filenames so user can visually compare
#                       to clip start/end
#   - Quality probe:    5 random frame paths printed for user to open
#
# Phase 2 (mechanical + visual):
#   - Process success:  ns-process-data exited 0 (we check the marker files exist)
#   - Registration rate: registered cameras / extracted frames (from transforms.json)
#   - Sparse cloud existence + point count
#   - Trajectory & sparse-cloud visual sanity: user must open in COLMAP / CloudCompare

source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
activate_env

SCENE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --scene) SCENE="$2"; shift 2 ;;
    -h|--help) grep -E '^# ' "$0" | sed 's/^# //'; exit 0 ;;
    *) die "unknown arg: $1" ;;
  esac
done
[[ -n "$SCENE" ]] || die "--scene <id> is required"

DATA_SCENES="${REPO_ROOT}/$(read_cfg data_scenes data/scenes)"
DIR="${DATA_SCENES}/${SCENE}"
[[ -d "$DIR" ]] || die "scene dir not found: $DIR  (run scripts/process_data.sh first)"

fail=0
section() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
ok()   { printf '  \033[32mOK\033[0m   %s\n' "$*"; }
warn() { printf '  \033[33mWARN\033[0m %s\n' "$*"; }
bad()  { printf '  \033[31mFAIL\033[0m %s\n' "$*"; fail=1; }

# -------- Phase 1: frame extraction --------
section "Phase 1 — Frame extraction"

IMG_DIR="${DIR}/images"
if [[ ! -d "$IMG_DIR" ]]; then
  bad "no images/ directory at $IMG_DIR"
  exit 1
fi

# Count
N_IMG=$(find "$IMG_DIR" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.png' -o -iname '*.jpeg' \) | wc -l)
echo "  frame count: $N_IMG"
if (( N_IMG >= 150 && N_IMG <= 350 )); then
  ok "count in plan band [150, 300] (+10% headroom for sampling)"
elif (( N_IMG < 150 )); then
  bad "count $N_IMG below plan minimum 150 — clip too short or sampled too sparsely"
else
  warn "count $N_IMG above plan recommendation (~300); not a failure but redundancy will slow training"
fi

# Order test — print first and last filenames; user compares to clip start/end visually.
# Use a temp file so `head`/`tail` closing the pipe early doesn't trip `set -o pipefail`.
TMP_LIST=$(mktemp)
find "$IMG_DIR" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.png' \) | sort > "$TMP_LIST"
FIRST=$(head -1 "$TMP_LIST")
LAST=$( tail -1 "$TMP_LIST")
echo "  first frame:  $FIRST"
echo "  last frame:   $LAST"
echo "  (open these and confirm they match the clip's start/end)"

# Quality spot check — 5 random frames.
echo "  spot-check frames:"
shuf -n 5 "$TMP_LIST" | sed 's/^/    /'
rm -f "$TMP_LIST"

# Auto sharpness probe (variance of Laplacian) — flags suspicious blur.
python - "$IMG_DIR" <<'PY'
import sys, pathlib, random
try:
    import cv2, numpy as np
except ImportError:
    print("  (skip auto-sharpness probe: opencv not installed in env)")
    sys.exit(0)
imgs = sorted(p for p in pathlib.Path(sys.argv[1]).iterdir()
              if p.suffix.lower() in {'.jpg','.jpeg','.png'})
sample = random.sample(imgs, min(30, len(imgs)))
sharps = []
for p in sample:
    im = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    if im is None: continue
    sharps.append((cv2.Laplacian(im, cv2.CV_64F).var(), p.name))
sharps.sort()
print(f"  sharpness (var-of-laplacian) on {len(sharps)} random frames:")
print(f"    min:  {sharps[0][0]:8.1f}  ({sharps[0][1]})")
print(f"    p25:  {sharps[len(sharps)//4][0]:8.1f}")
print(f"    med:  {sharps[len(sharps)//2][0]:8.1f}")
print(f"    p75:  {sharps[3*len(sharps)//4][0]:8.1f}")
print(f"    max:  {sharps[-1][0]:8.1f}  ({sharps[-1][1]})")
blurry = [s for s in sharps if s[0] < 50]
if blurry:
    print(f"  WARN: {len(blurry)}/{len(sharps)} frames have var<50 (likely motion-blurred):")
    for s,n in blurry[:5]:
        print(f"    {s:8.1f}  {n}")
else:
    print("  no frames with var-of-laplacian < 50 (no obvious motion blur)")
PY

# -------- Phase 2: COLMAP poses --------
section "Phase 2 — COLMAP poses + sparse cloud"

SPARSE_DIR="${DIR}/colmap/sparse/0"
TRANSFORMS="${DIR}/transforms.json"
if [[ ! -d "$SPARSE_DIR" ]]; then
  bad "no colmap/sparse/0/ at $SPARSE_DIR  (mapper failed)"
fi
if [[ ! -f "$TRANSFORMS" ]]; then
  bad "no transforms.json at $TRANSFORMS  (ns-process-data didn't write it)"
fi

if [[ -f "$TRANSFORMS" ]]; then
  python - "$TRANSFORMS" "$N_IMG" <<'PY'
import json, sys, pathlib
data = json.load(open(sys.argv[1]))
total_extracted = int(sys.argv[2])
frames = data.get("frames", [])
n_reg = len(frames)
print(f"  transforms.json frames: {n_reg}")
print(f"  extracted images:       {total_extracted}")
rate = (n_reg / total_extracted * 100) if total_extracted else 0
print(f"  registration rate:      {rate:.1f}%")
if rate >= 80:
    print(f"  OK   registration rate >= 80% (plan: 'a high fraction' for textured rooms)")
elif rate >= 50:
    print(f"  WARN registration rate {rate:.1f}% — many frames dropped; investigate blur/blank-walls")
    sys.exit(2)
else:
    print(f"  FAIL registration rate {rate:.1f}% — most frames dropped; refilm or change frames-target")
    sys.exit(3)
# Camera intrinsics sanity
keys_present = sorted(k for k in data if k != "frames")
print(f"  top-level keys: {keys_present}")
PY
  rc=$?
  [[ $rc -eq 0 ]] && ok "registration rate looks good"
  [[ $rc -eq 2 ]] && warn "registration rate marginal"
  [[ $rc -eq 3 ]] && bad "registration rate too low"
fi

if [[ -d "$SPARSE_DIR" ]]; then
  # COLMAP sparse outputs are points3D.{bin,txt}, images.{bin,txt}, cameras.{bin,txt}
  pts_file=""
  for f in "$SPARSE_DIR"/points3D.bin "$SPARSE_DIR"/points3D.txt; do
    [[ -f "$f" ]] && pts_file="$f" && break
  done
  if [[ -n "$pts_file" ]]; then
    SIZE=$(stat -c%s "$pts_file")
    echo "  sparse points file: $pts_file ($SIZE bytes)"
    if (( SIZE > 10000 )); then
      ok "sparse cloud non-trivial ($SIZE bytes)"
    else
      bad "sparse cloud suspiciously small ($SIZE bytes)"
    fi
  else
    bad "no points3D.bin / points3D.txt in $SPARSE_DIR"
  fi
fi

section "Visual sanity (manual)"
cat <<EOF
  The plan requires a human eyeball check before training. Open the sparse cloud:

    A) COLMAP GUI:  colmap gui --import_path "$SPARSE_DIR" --image_path "$IMG_DIR"
    B) CloudCompare or MeshLab:  load $SPARSE_DIR/points3D.bin
    C) ns-viewer (after training)

  Pass: rough room shape (walls / floor / furniture) recognizable;
        camera path is smooth, similar to how you filmed.
  Fail: random ball, duplicated structure, twisted cloud, big jumps in trajectory.
EOF

echo
if [[ $fail -eq 0 ]]; then
  printf '\033[32mphase 1/2 mechanical checks: PASS  (still do the visual sanity above)\033[0m\n'
else
  printf '\033[31mphase 1/2 checks: FAIL — see above\033[0m\n'
  exit 1
fi
