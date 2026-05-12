#!/usr/bin/env bash
# Phase 0 — DGX bring-up. The plan's exit-criteria checks for Phase 0, mechanised.
# Exits non-zero on any failure so this can gate CI / Makefile targets.

# shellcheck source=./_lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/_lib.sh"
activate_env

fail=0
section() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

# check LABEL CMD...  — runs CMD, swallows its stdout/stderr, prints OK/FAIL with LABEL.
# Using an explicit label keeps the script readable and prevents redirects on the
# caller side from silencing the OK line (a bug in the earlier `check cmd >/dev/null` form).
check() {
  local label="$1"; shift
  if "$@" >/dev/null 2>&1; then
    printf '  \033[32mOK\033[0m   %s\n' "$label"
  else
    printf '  \033[31mFAIL\033[0m %s\n' "$label"
    fail=1
  fi
}

section "GPU visibility (nvidia-smi)"
if command -v nvidia-smi >/dev/null; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
  echo "  FAIL nvidia-smi not on PATH"; fail=1
fi

section "Python + torch + CUDA"
python - <<'PY'
import sys, torch
print(f"  python      {sys.version.split()[0]}")
print(f"  torch       {torch.__version__}")
print(f"  cuda build  {torch.version.cuda}")
print(f"  cuda avail  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  gpu[{i}]      {p.name}  sm_{p.major}{p.minor}  {p.total_memory/1e9:.1f} GB")
    # The real proof: do an actual CUDA op, not just is_available().
    x = torch.randn(1024, 1024, device='cuda')
    y = (x @ x).sum().item()
    print(f"  cuda matmul OK (sum={y:.2f})")
else:
    sys.exit(1)
PY
[[ $? -eq 0 ]] || fail=1

section "gsplat import (compiles CUDA on first run — may take a minute)"
python - <<'PY'
import gsplat, torch
print(f"  gsplat      {gsplat.__version__}")
# Touch a CUDA-using API so any lazy compile happens here, not at training time.
means     = torch.randn(8, 3, device='cuda')
scales    = torch.rand(8, 3, device='cuda')
quats     = torch.zeros(8, 4, device='cuda'); quats[:, 0] = 1.0
opacities = torch.rand(8, device='cuda')
colors    = torch.rand(8, 3, device='cuda')
viewmats  = torch.eye(4, device='cuda')[None]
Ks        = torch.tensor([[100., 0, 64], [0, 100., 64], [0, 0, 1.]], device='cuda')[None]
out = gsplat.rasterization(means, quats, scales, opacities, colors, viewmats, Ks, 128, 128)
print(f"  rasterization smoke OK (output type: {type(out).__name__})")
PY
[[ $? -eq 0 ]] || fail=1

section "Nerfstudio CLI"
check "ns-train --help"        ns-train --help
check "ns-process-data --help" ns-process-data --help
check "ns-export --help"       ns-export --help
# nerfstudio doesn't expose __version__ on the module; ask pip's installed metadata.
ns_ver=$(python -c "from importlib.metadata import version; print(version('nerfstudio'))" 2>/dev/null || echo "unknown")
echo "  nerfstudio  ${ns_ver}"

section "COLMAP + CUDA support"
check "colmap -h" colmap -h
# The CUDA-build banner is in the first line of `colmap -h`. Hard-require it —
# the whole point of pinning the cuda_129 conda-forge build is to get GPU SIFT.
if colmap -h 2>&1 | head -2 | grep -qi "with CUDA"; then
  echo "  OK   colmap reports 'with CUDA'"
else
  echo "  FAIL colmap does NOT report 'with CUDA' — wrong build pinned?"; fail=1
fi
# Confirm the actual link target (catches libfaiss-style missing-so regressions).
if ldd "$(command -v colmap)" 2>/dev/null | grep -q libcudart; then
  echo "  OK   colmap is dynamically linked to libcudart"
else
  echo "  FAIL colmap is NOT linked to libcudart"; fail=1
fi

section "ffmpeg (needed by ns-process-data video)"
check "ffmpeg -version" ffmpeg -version

section "Path resolution"
echo "  REPO_ROOT = $REPO_ROOT"
echo "  data_raw    = $(read_cfg data_raw)"
echo "  data_scenes = $(read_cfg data_scenes)"
echo "  outputs     = $(read_cfg outputs)"

echo
if [[ $fail -eq 0 ]]; then
  printf '\033[32mPhase 0 bring-up: PASS\033[0m\n'
else
  printf '\033[31mPhase 0 bring-up: FAIL — fix issues above before extracting frames\033[0m\n'
  exit 1
fi
