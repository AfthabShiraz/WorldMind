#!/usr/bin/env bash
# Idempotent post-install patch for nerfstudio 1.1.5.
#
# nerfstudio 1.1.5's colmap_utils.py shells out using the old COLMAP option
# names `--SiftExtraction.use_gpu` and `--SiftMatching.use_gpu`. Conda-forge's
# COLMAP 3.13+ (the only build available on linux-aarch64 with CUDA support
# as of 2026-05) renamed these to `--FeatureExtraction.use_gpu` and
# `--FeatureMatching.use_gpu`. Without this patch ns-process-data fails with:
#
#   unrecognised option '--SiftExtraction.use_gpu'
#
# This script edits the installed file in place. It's safe to re-run.

set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
activate_env

FILE="${CONDA_PREFIX}/lib/python3.10/site-packages/nerfstudio/process_data/colmap_utils.py"
[[ -f "$FILE" ]] || die "expected file not found: $FILE"

patched=0
if grep -q 'SiftExtraction.use_gpu' "$FILE"; then
  sed -i 's/SiftExtraction\.use_gpu/FeatureExtraction.use_gpu/g' "$FILE"
  patched=1
fi
if grep -q 'SiftMatching.use_gpu' "$FILE"; then
  sed -i 's/SiftMatching\.use_gpu/FeatureMatching.use_gpu/g' "$FILE"
  patched=1
fi

if [[ $patched -eq 1 ]]; then
  log "patched colmap option names in $FILE"
else
  log "colmap option names already patched (or never present); no changes"
fi

# Verify no stale references remain.
if grep -nE 'SiftExtraction\.use_gpu|SiftMatching\.use_gpu' "$FILE"; then
  die "colmap-options patch did not stick — manual investigation needed"
fi

# ---- patch 2: torch.load(weights_only=...) ----
# PyTorch 2.6+ flipped torch.load's `weights_only` default to True. Nerfstudio
# 1.1.5's checkpoints carry numpy scalars not in torch's default safe-globals
# list, so loading fails with:
#   _pickle.UnpicklingError: Weights only load failed
# Nerfstudio writes the checkpoint itself — trust is fine — so force weights_only=False.
EVAL_FILE="${CONDA_PREFIX}/lib/python3.10/site-packages/nerfstudio/utils/eval_utils.py"
[[ -f "$EVAL_FILE" ]] || die "expected file not found: $EVAL_FILE"
if grep -q 'torch.load(load_path, map_location="cpu")' "$EVAL_FILE"; then
  sed -i 's|torch.load(load_path, map_location="cpu")|torch.load(load_path, map_location="cpu", weights_only=False)|g' "$EVAL_FILE"
  log "patched torch.load weights_only=False in $EVAL_FILE"
fi
if ! grep -q 'weights_only=False' "$EVAL_FILE"; then
  die "torch.load patch did not stick — manual investigation needed"
fi
