# shellcheck shell=bash
# Common helpers sourced by every wrapper script.
# Sets REPO_ROOT, activates the conda env, exposes log/die.

set -euo pipefail

# Resolve REPO_ROOT from this file's own location so scripts work from any cwd.
_lib_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${_lib_dir}/.." && pwd)"
export REPO_ROOT

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die() { log "ERROR: $*"; exit 1; }

# Activate the project conda env. Falls back to absolute-path binaries if
# `conda activate` isn't available in this shell (e.g. non-interactive).
activate_env() {
  local env_name="${WORLDMIND_ENV:-worldmind}"
  local prefix="${WORLDMIND_CONDA_PREFIX:-$HOME/miniforge3}"
  local host_cuda="${WORLDMIND_HOST_CUDA:-/usr/local/cuda-13.0}"
  if [[ ! -d "${prefix}/envs/${env_name}" ]]; then
    die "conda env '${env_name}' not found at ${prefix}/envs/${env_name}. Run: scripts/install_env.sh"
  fi
  # Prepend env bin to PATH (ns-*, colmap, python, ninja) then host CUDA (nvcc).
  # nvcc is needed because gsplat 1.4.0 JIT-compiles its CUDA extension on first
  # import unless a prebuilt csrc is present (which it isn't, on aarch64).
  export PATH="${prefix}/envs/${env_name}/bin:${host_cuda}/bin:${PATH}"
  export CONDA_PREFIX="${prefix}/envs/${env_name}"
  export CUDA_HOME="${host_cuda}"
  # GB10 reports as sm_121; nearest arch torch knows is sm_120, +PTX JITs forward.
  # Without this, gsplat compiles for the entire torch arch_list (8.0..12.0),
  # which is wasteful and breaks if any arch fails on host nvcc.
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0+PTX}"
}

# Read a single scalar from configs/paths.yaml (or paths.example.yaml fallback).
# Usage: read_cfg KEY [DEFAULT]
read_cfg() {
  local key="$1" default="${2-}"
  local cfg="${REPO_ROOT}/configs/paths.yaml"
  [[ -f "$cfg" ]] || cfg="${REPO_ROOT}/configs/paths.example.yaml"
  python3 - "$cfg" "$key" "$default" <<'PY'
import sys, pathlib
cfg, key, default = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    import yaml
    data = yaml.safe_load(pathlib.Path(cfg).read_text()) or {}
except Exception:
    data = {}
cur = data
for part in key.split("."):
    if isinstance(cur, dict) and part in cur:
        cur = cur[part]
    else:
        cur = default
        break
print(cur if cur is not None else default)
PY
}

require_scene() {
  [[ "${SCENE:-}" ]] || die "SCENE not set. Usage: SCENE=<scene_id> $0   (or pass --scene <id>)"
}
