# WorldMind — RGB video → 3D Gaussian splatting

Reproducible pipeline: a phone-shot video walks through a room → frames + camera poses (COLMAP) → trained 3D Gaussian splat → viewable in the browser (SuperSplat). Optional semantics layer attaches object labels to Gaussians.

Full design rationale and phase-by-phase exit criteria live in [`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md).

---

## Hardware this repo is pinned to

**`spark-cefe`** — NVIDIA **DGX Spark**: Ubuntu 24.04 aarch64, single NVIDIA **GB10** GPU (Grace Blackwell, compute capability **sm_121**, ~128 GB unified memory), CUDA 13.0 / driver 580.142 on host.

This is **not** the stack the plan pins (the plan assumes amd64 + the official Nerfstudio Docker image + PyTorch 2.1.2/cu118). The plan's stack physically cannot run on this box for two reasons: the published Nerfstudio image is amd64-only, and PyTorch 2.1.2 wheels do not ship Blackwell kernels. The deviation is documented inline below per the plan's "Deviate only with cause" clause.

## Pinned stack (this repo)

| Layer | Pin | Why this exact version |
|---|---|---|
| Python | **3.10** | Nerfstudio's best-tested version |
| Torch  | **2.9.1+cu129** (aarch64 manylinux_2_28 wheel) | Earliest stable torch whose `arch_list` includes `sm_120` (PTX forward-compatible to the GB10's `sm_121`). cu126 wheels only target sm_80/sm_90 and will fail `cudaErrorNoKernelImageForDevice` on this GPU. |
| Torchvision | **0.24.1** | Matches torch 2.9.1 ABI |
| Nerfstudio | **1.1.5** (latest) | `splatfacto` trainer + `ns-process-data` + `ns-export gaussian-splat` |
| gsplat | **1.4.0** (Nerfstudio pulls it) | Compiles its CUDA extension JIT against the host's nvcc on first import |
| COLMAP | **4.0.4** (conda-forge `cuda_129` build for `linux-aarch64`) | CUDA SIFT matching; `ns-process-data` shells out to this binary |
| ffmpeg | conda-forge | Frame extraction inside `ns-process-data video` |
| Container | **None — miniforge user-install in `$HOME/miniforge3`** | The user account on this box is not in the `docker` group, so the Docker-first path the plan prefers is not available. Single-user conda env is the equivalent of the plan's "manual conda env" fallback. |

`scripts/install_env.sh` encodes this exactly. `environment.lock.txt` is the `pip freeze` from a known-good install.

## Repo layout

| Path | Purpose | Tracked? |
|---|---|---|
| `docs/IMPLEMENTATION_PLAN.md` | Phase-by-phase plan + gates | ✅ |
| `scripts/` | Wrapper shell scripts (one per phase target) | ✅ |
| `configs/paths.example.yaml` | Local-machine path overrides — copy to `paths.yaml` | ✅ template only |
| `Makefile` | Phase entry points (`make process-data SCENE=…`) | ✅ |
| `data/raw/<scene>.mp4` | Input videos | ⛔ gitignored — ~GB each |
| `data/scenes/<scene>/` | `ns-process-data` output: `images/` + `colmap/sparse/0/` + `transforms.json` | ⛔ |
| `outputs/<scene>/` | `ns-train` run dirs, exported `.ply`, `run_meta.json` | ⛔ |
| `semantics/<scene>/` | Phase 8 keyframe masks + per-Gaussian labels | ⛔ |

## Quickstart

```bash
# Phase 0 — one-time. Idempotent; safe to re-run.
make install      # installs miniforge + env at ~/miniforge3
make env-check    # proves GPU + torch + gsplat + colmap + nerfstudio all work on this box

# Per scene, drop the input video then run the wrappers in order.
cp ~/some_video.mp4 data/raw/living_room.mp4
make process-data SCENE=living_room   # Phase 1+2: frames + COLMAP poses
make train-smoke  SCENE=living_room   # Phase 4: 5k-iter smoke; exports a .ply
# Inspect outputs/living_room/*.ply in https://playcanvas.com/supersplat — Phase 4 viewer probe.
make train-full   SCENE=living_room   # Phase 5: full ~30k-iter production run
make export-splat SCENE=living_room   # Phase 6: final submission .ply
```

## Phase status

| Phase | Status | Output |
|---|---|---|
| 0 — env, scaffold, DGX bring-up | **in progress** | env at `~/miniforge3/envs/worldmind`, `scripts/env_check.sh` |
| 1 — frame extraction | not started | `data/scenes/<scene>/images/` |
| 2 — COLMAP poses + sparse cloud QC | not started | `data/scenes/<scene>/colmap/sparse/0/`, `transforms.json` |
| 3 — dataset packaging verification | not started | loader check |
| 4 — splat training smoke | not started | first `.ply` |
| 5 — full splat training | not started | submission `.ply` |
| 6 — export + viewer | not started | viewer-ready artifact |
| 7 — README + canonical example | not started | reproducible third-party run |
| 8 — semantics (optional) | not started | per-Gaussian labels |

## Filming tips (affect every phase downstream)

- **Slow, steady pan.** Motion blur is the single most common reason COLMAP drops frames.
- **Overlap.** Each spot in the room should appear in many frames from many angles.
- **Texture matters.** Blank walls break feature matching — keep posters, furniture, clutter in frame.
- **Single sweep.** ~30–60 s of footage is plenty for a small room at `--num-frames-target 300`.
- **No moving people / pets** in frame (or be ready to mask them in Phase 8).
