#!/usr/bin/env bash
# Idempotent installer for the WorldMind env. Safe to re-run.
#
# Why this script deviates from docs/IMPLEMENTATION_PLAN.md's pinned stack:
#
#   The plan pins amd64 + the official Nerfstudio Docker image + torch 2.1.2/cu118.
#   This box is aarch64 + Blackwell GB10 (compute capability sm_121). Two blockers:
#     1) ghcr.io/nerfstudio-project/nerfstudio:latest is amd64-only (won't run).
#     2) torch cu126 wheels ship only sm_80/sm_90 kernels → cudaErrorNoKernelImageForDevice.
#
#   The deviation: install via miniforge (single-user, no sudo, no docker), pin to
#   torch 2.9.1+cu129 — the earliest stable torch wheel whose arch_list includes
#   sm_120 (PTX forward-compatible to GB10's sm_121). Vision pinned to 0.24.1
#   to match torch 2.9.1 ABI. colmap pulled from conda-forge's CUDA 12.9 build.

set -euo pipefail

ENV_NAME="${WORLDMIND_ENV:-worldmind}"
MINIFORGE_DIR="${WORLDMIND_CONDA_PREFIX:-$HOME/miniforge3}"
PY_VER="3.10"
TORCH_VER="2.9.1+cu129"
TORCHVISION_VER="0.24.1"
TORCH_INDEX="https://download.pytorch.org/whl/cu129"
COLMAP_SPEC="colmap=4.0.4=cuda_129*"
# colmap 4.0.4 dynamically links libfaiss; on linux-aarch64 conda-forge does not
# pull it transitively as of 2026-05, so we add it explicitly. Without this you
# get: `colmap: error while loading shared libraries: libfaiss.so`
LIBFAISS_SPEC="libfaiss"
HOST_CUDA="/usr/local/cuda-13.0"

log() { printf '[install_env] %s\n' "$*" >&2; }

# 1. Miniforge (aarch64 installer is correct for this box).
if [[ ! -x "${MINIFORGE_DIR}/bin/conda" ]]; then
  log "installing miniforge -> ${MINIFORGE_DIR}"
  arch="$(uname -m)"
  tmp="$(mktemp -d)"
  curl -fsSL -o "${tmp}/miniforge.sh" "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${arch}.sh"
  bash "${tmp}/miniforge.sh" -b -p "${MINIFORGE_DIR}"
  rm -rf "${tmp}"
else
  log "miniforge already present at ${MINIFORGE_DIR}"
fi

MAMBA="${MINIFORGE_DIR}/bin/mamba"
ENV_PREFIX="${MINIFORGE_DIR}/envs/${ENV_NAME}"
PIP="${ENV_PREFIX}/bin/pip"
PY="${ENV_PREFIX}/bin/python"

# 2. Conda env: Python + COLMAP (CUDA 12.9 build) + ffmpeg.
if [[ ! -d "${ENV_PREFIX}" ]]; then
  log "creating conda env ${ENV_NAME}"
  "${MAMBA}" create -n "${ENV_NAME}" -y -c conda-forge \
    "python=${PY_VER}" "${COLMAP_SPEC}" "${LIBFAISS_SPEC}" ffmpeg pip
else
  log "conda env ${ENV_NAME} already exists; ensuring colmap + libfaiss + ffmpeg present"
  "${MAMBA}" install -n "${ENV_NAME}" -y -c conda-forge "${COLMAP_SPEC}" "${LIBFAISS_SPEC}" ffmpeg
fi

# 3. Modern pip / wheel / setuptools.
"${PIP}" install --upgrade pip wheel setuptools

# 4. PyTorch with Blackwell-capable kernel set (sm_120 + PTX → sm_121 hw).
log "installing torch ${TORCH_VER} + torchvision ${TORCHVISION_VER} (aarch64 / cu129)"
# Use --no-deps because the cu129 index only has +cu129-tagged torch but stock
# torchvision dist tag — pip resolver gets confused otherwise.
"${PIP}" install --no-cache-dir --no-deps \
  "torch==${TORCH_VER}" "torchvision==${TORCHVISION_VER}" \
  --index-url "${TORCH_INDEX}"
# Bring back the deps torch needs (sympy, networkx, jinja2, etc.) from PyPI.
"${PIP}" install --no-cache-dir \
  filelock fsspec jinja2 networkx sympy typing-extensions

# 5. Build tooling for gsplat's JIT CUDA extension.
log "installing ninja (required by torch.utils.cpp_extension for gsplat compile)"
"${PIP}" install --no-cache-dir ninja

# Sanity-check torch can actually see and use the GPU before going further.
"${PY}" - <<'PY'
import torch, sys
print(f"torch={torch.__version__} cuda_built={torch.version.cuda} avail={torch.cuda.is_available()}")
print(f"arch_list={torch.cuda.get_arch_list()}")
if not torch.cuda.is_available():
    sys.exit("torch.cuda.is_available() is False — abort before pip install nerfstudio")
x = torch.randn(1024, 1024, device='cuda')
y = (x @ x).sum().item()
print(f"matmul OK sum={y:.2e}")
PY

# 6. Nerfstudio (latest 1.1.x). Pulls gsplat==1.4.0, viser==0.2.7, etc.
log "installing nerfstudio (latest 1.1.x)"
"${PIP}" install --no-cache-dir nerfstudio

# nerfstudio's `ns-export` import-graph reaches `pymeshlab` (mesh/texture export)
# but the dep isn't declared. Without this, even `ns-export --help` raises
# ModuleNotFoundError before any subcommand runs — including `gaussian-splat`,
# which we do need. pymeshlab 2025.7+ ships an aarch64 wheel on PyPI.
log "installing pymeshlab (undeclared transitive dep of ns-export)"
"${PIP}" install --no-cache-dir pymeshlab

# 7. Pre-compile gsplat's CUDA extension. This avoids a 2–5 min cold compile on
#    the very first training step (when the user is waiting for a smoke run).
#    The compile is keyed on TORCH_CUDA_ARCH_LIST, so set it explicitly to the
#    Blackwell PTX so the resulting kernels JIT-promote to GB10 sm_121.
log "pre-compiling gsplat CUDA extension (one-time, ~3 min)"
PATH="${ENV_PREFIX}/bin:${HOST_CUDA}/bin:${PATH}" \
CUDA_HOME="${HOST_CUDA}" \
TORCH_CUDA_ARCH_LIST="12.0+PTX" \
"${PY}" -c "
import torch, gsplat
m = torch.randn(8,3,device='cuda'); s = torch.rand(8,3,device='cuda')
q = torch.zeros(8,4,device='cuda'); q[:,0]=1.0
o = torch.rand(8,device='cuda'); c = torch.rand(8,3,device='cuda')
V = torch.eye(4,device='cuda')[None]
K = torch.tensor([[100.,0,64],[0,100.,64],[0,0,1.]],device='cuda')[None]
out = gsplat.rasterization(m, q, s, o, c, V, K, 128, 128)
print(f'gsplat OK: img shape {tuple(out[0].shape)}')
"

# 8. Freeze the working env so it's reproducible.
log "freezing environment.lock.txt"
"${PIP}" freeze > "$(dirname "$0")/../environment.lock.txt"

log ""
log "install complete."
log "Activate manually:  source ${MINIFORGE_DIR}/etc/profile.d/conda.sh && conda activate ${ENV_NAME}"
log "Or use wrappers (auto-activate):  scripts/env_check.sh, scripts/process_data.sh, ..."
