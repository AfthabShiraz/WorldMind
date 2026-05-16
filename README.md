# WorldMind — phone video → 3D Gaussian splat

Drop in a phone-shot video of a room, get back a trained 3D Gaussian splat
you can open in a web viewer or drop into [PlayCanvas SuperSplat](https://playcanvas.com/supersplat).
No manual frame extraction, no COLMAP wrestling, no Nerfstudio CLI to learn.

![Example scene rendered from the trained splat — bedroom-with-workspace `room5`](splatscreenshot.png)

```
video.mp4
   ↓  make run VIDEO=video.mp4
   ↓  (frames → COLMAP poses → splatfacto training → .ply export)
outputs/<scene>/ply_full/splat.ply        ←  drag into SuperSplat
                                              or:  make view SCENE=<scene>
```

---

## 🎥 View the example scene (`room5`)

The repo ships with one finished example: `room5`, a bedroom-with-workspace
captured on a phone and trained for ~30k iterations. The trained splat is
hosted as a [GitHub release asset](https://github.com/AfthabShiraz/WorldMind/releases/tag/v0.1-room5)
(~150 MB) — the source code is in this repo, the artifact is one click away.

### Option A — Zero-install (drag-and-drop)

1. Download [`splat.ply` from the v0.1-room5 release](https://github.com/AfthabShiraz/WorldMind/releases/download/v0.1-room5/splat.ply).
2. Open [PlayCanvas SuperSplat](https://playcanvas.com/supersplat/editor) in any browser.
3. Drag `splat.ply` onto the page. Mouse-drag to orbit, scroll to zoom, right-click-drag to pan.

No git clone, no Python, no GPU required.

### Option B — The custom in-browser viewer in this repo

For seeing the actual viewer code in action, with the (optional) scene-inventory panel + tint-by-instance toggle when semantic sidecars are present. Two commands:

```bash
# Lightweight: just enough to view (viser + plyfile + numpy)
make install-viewer

# Auto-downloads the .ply from the release, opens localhost:8080
make view-example
```

That's it — open the printed URL (`http://localhost:8080` if local, or use
`make view-example EXTRA=--share` to get a public `*.share.viser.studio`
link a remote reviewer can click).

> `make install-viewer` runs `pip install -r requirements-viewer.txt` in
> whatever python you have on PATH. If you'd rather use the full conda
> training env, run `make install` instead — `make view-example` works
> with either.

---

## 🚀 Run the pipeline on your own video

**Prerequisites:** an NVIDIA GPU (the repo is pinned to a Blackwell GB10 — see
*Stack* below). For amd64 + recent CUDA the conda env should just work; for
other architectures expect to retune `TORCH_CUDA_ARCH_LIST`.

### One-time setup (~10 min for the training env, ~30 s for viewer-only)

```bash
# Full env — required for training your own scene
make install      # installs miniforge + conda env at ~/miniforge3
make env-check    # proves GPU + torch + gsplat + colmap + nerfstudio work

# Lightweight env — sufficient for just viewing .ply files
make install-viewer
```

### Per-video (~10 min on GB10)

```bash
make run VIDEO=/path/to/your_clip.mp4
```

That single command:

1. Copies your video to `data/raw/<scene>.mp4`.
2. Extracts frames + runs COLMAP (focal-length seeded — works on phone videos with no EXIF).
3. Trains splatfacto for ~30 000 iterations.
4. Exports the result to `outputs/<scene>/ply_full/splat.ply`.
5. Renders 6 preview JPGs to `outputs/<scene>/_qc/renders_full/` so you can sanity-check the result over SSH without a browser.

Scene name defaults to the video filename. Override with `SCENE=`:

```bash
make run VIDEO=~/clip.mp4 SCENE=living_room        # custom name
make run VIDEO=~/clip.mp4 PROFILE=smoke            # 5k iters (~1 min) instead of 30k
```

### View your scene

```bash
make view SCENE=<your_scene>                       # localhost:8080
make view SCENE=<your_scene> EXTRA=--share         # public URL, anyone can open
```

Or download `outputs/<your_scene>/ply_full/splat.ply` and drop into SuperSplat.

---

## 📋 All commands

```bash
make help                                          # list every target
make install                                       # full env (training + viewer)
make install-viewer                                # lightweight env (viewer only)
make env-check                                     # verify full env

# Main pipeline
make run VIDEO=path/to/clip.mp4 [SCENE=name]       # video → .ply (full)
make view SCENE=<name> [EXTRA=--share]             # in-browser viewer
make fetch-example                                 # download room5 .ply from release
make view-example                                  # auto-fetch + view room5

# Per-stage (debugging / re-running one step)
make process-data SCENE=<name> [EXTRA=--force]     # frames + COLMAP only
make qc-process-data SCENE=<name>                  # exit-criteria check
make train-full SCENE=<name>                       # training only
make train-smoke SCENE=<name>                      # 5k-iter quick check
make export-splat SCENE=<name>                     # checkpoint → .ply

make clean-scene SCENE=<name>                      # wipe derived artifacts (keeps data/raw)
```

### Filming tips (affect every stage downstream)

- **Slow, steady pan.** Motion blur is the #1 reason COLMAP drops frames.
- **Overlap.** Every spot in the room should appear in many frames from many angles.
- **Texture matters.** Blank walls break feature matching — keep furniture, posters, clutter in frame.
- **One smooth sweep.** ~30–60 s of footage is plenty for a small room.
- **No moving people / pets.** They cause floaters in the splat.

---

## Where things end up

```
data/raw/<scene>.<ext>                   your input video (gitignored)
data/scenes/<scene>/
    images/                              extracted frames
    colmap/sparse/0/                     COLMAP reconstruction
    transforms.json                      Nerfstudio dataset descriptor
outputs/<scene>/
    splatfacto_full/.../                 training run dir (checkpoints)
    ply_full/splat.ply                   ← the artifact you want
    _qc/renders_full/                    6 preview JPGs for SSH sanity-checks
```

`data/` and `outputs/` are gitignored — they're regenerated from the input video.

---

## Stack

This box: a single **NVIDIA DGX Spark** (Ubuntu 24.04 aarch64, **GB10 GPU**,
compute capability `sm_121`, CUDA 13.0). The env is pinned for that target.

| Layer | Version | Reason |
|---|---|---|
| Python | 3.10 | Nerfstudio's best-tested version |
| Torch | 2.9.1+cu129 (aarch64) | Earliest stable torch whose `arch_list` covers `sm_120` (PTX-forward to GB10's `sm_121`) |
| Torchvision | 0.24.1 | ABI-matched to torch |
| Nerfstudio | 1.1.5 | `splatfacto` + `ns-process-data` + `.ply` export |
| gsplat | 1.4.0 | Nerfstudio's rasteriser; JIT-compiles against host nvcc on first import |
| COLMAP | 4.0.4 (conda-forge `cuda_129/linux-aarch64`) | CUDA SIFT matching |
| viser | (pip) | In-browser viewer (`make view`) |

`scripts/install_env.sh` encodes all of this; `environment.lock.txt` is the
`pip freeze` from a known-good install. Full design rationale in
[`docs/IMPLEMENTATION_PLAN.md`](docs/IMPLEMENTATION_PLAN.md).

---

## Optional: experimental semantics layer

There's an in-progress add-on that attaches VLM-derived object labels to
Gaussians (SAM mask association across views + Qwen2.5-VL labelling, with
all outputs in sidecar files so `splat.ply` is never modified). **Not part
of the standard pipeline** and not run by `make run`.

```bash
make scene-inventory SCENE=<name>                  # VLM-only: list significant objects
make lift-semantics-v2 SCENE=<name>                # 3D instance lifting
```

If you've run either of these, the viewer (`make view`) picks up the
sidecars automatically and shows a "Scene inventory" GUI panel + a
tint-by-instance toggle. With no sidecars present, the viewer just shows
the raw splat — which is the default.
