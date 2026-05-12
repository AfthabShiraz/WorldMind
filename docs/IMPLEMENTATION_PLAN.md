# Implementation plan: video → 3D Gaussian splatting

This document breaks the project into **implementation phases**. Each phase ends with a **test phase**: concrete checks you run before starting the next phase. The stack assumed here is **RGB video → frame extraction → structure-from-motion (SfM), e.g. COLMAP → 3D Gaussian splatting training → exported splat → viewer**, with **optional** semantic labeling after the geometric pipeline is stable.

**Hardware note:** develop scripts and configs on your laptop; run **COLMAP at full scale** and **splat training** on the DGX (or any CUDA machine). Tests marked **(DGX)** should be run there.

---

## Stack decisions — pinned (lock these before Phase 1)

Most concrete failures in this pipeline come from version mismatches between COLMAP, PyTorch, CUDA, and the splat trainer. The stack below is chosen so those pieces are already known to agree. **Deviate only with cause, and if you do, Phase 0 must prove the build works on the DGX before anything downstream starts.**

### Why this shape

| Decision | Choice | Why |
|----------|--------|-----|
| **Splat trainer** | **Nerfstudio `splatfacto`** (Nerfstudio's 3DGS implementation, built on `gsplat`) | one toolchain that also wraps COLMAP (`ns-process-data video`) and exports a standard `.ply` (`ns-export gaussian-splat`); avoids hand-assembling a training loop or compiling the Inria `gaussian-splatting` submodules (`diff-gaussian-rasterization` / `simple-knn`) — that repo is the classic "won't build on the box" trap |
| **SfM tool** | **COLMAP**, CUDA-enabled, driven via `ns-process-data video` | Nerfstudio shells out to it; you get undistorted `images/` + `sparse/0/` in the layout `splatfacto` expects, no hand-rolled `transforms.json` (COLMAP is OpenCV camera convention; getting the convention wrong yields floaters with no error message) |
| **Orchestration language** | Python 3.10 | Nerfstudio's best-tested Python; one language for wrappers, the semantics lift, and QC |
| **Cold-open viewer** | **SuperSplat** (web app, no install) | a grader opens your `.ply` by dragging it into a browser; building Inria's SIBR desktop viewer is its own yak-shave |

### Pinned environment

**Primary install — the official Nerfstudio Docker image.** This eliminates the CUDA / PyTorch / gsplat / COLMAP version dance entirely; the image already contains a matching set. DGX boxes ship `nvidia-container-toolkit`, so this Just Works:

```
docker pull ghcr.io/nerfstudio-project/nerfstudio:latest      # or pin a specific tag once you've picked one
docker run --gpus all -it --shm-size=12gb \
  -v /abs/path/to/WorldMind:/workspace \
  ghcr.io/nerfstudio-project/nerfstudio:latest
```

(The image does **not** include the semantics-layer libraries — `pip install` those inside the container, or run the semantics stage in a separate env.)

**Fallback — manual conda env** (only if Docker isn't available on the DGX). Install order is load-bearing: **conda env → CUDA-matched PyTorch → Nerfstudio → COLMAP + ffmpeg via conda.**

| Component | Pin | Landmines |
|---|---|---|
| Python | **3.10** | avoid 3.12+ — Nerfstudio is not well-tested there |
| CUDA toolkit | **11.8** (12.1 also OK) | `nvcc --version` must match the CUDA your PyTorch wheel was built for — `gsplat` compiles CUDA against `nvcc` on first import, so they have to agree |
| PyTorch | **2.1.2 + cu118** | `pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118` — the most-tested Nerfstudio combo |
| Nerfstudio | **latest 1.1.x** — `pip install nerfstudio` | install **after** torch; check its install docs if you used a different CUDA |
| `gsplat` | whatever Nerfstudio pulls | compiles CUDA on first import — needs `nvcc` on PATH matching torch's CUDA build |
| `tinycudann` | **skip it** | only NeRF methods need it; `splatfacto` (3DGS) uses `gsplat`, not tcnn — installing tcnn is the #1 Nerfstudio install pain and you do not need it |
| COLMAP | `conda install -c conda-forge colmap` | install via conda, **not** pip (no working pip build); confirm `colmap -h` reports CUDA support |
| ffmpeg | `conda install -c conda-forge ffmpeg` | `ns-process-data video` needs it for frame extraction |

### Core geometric pipeline — the actual commands

```
ns-process-data video --data scene.mp4 --output-dir data/scenes/<id>            # extracts frames + runs COLMAP
ns-train splatfacto --data data/scenes/<id>                                     # trains 3DGS (~30k iters default)
ns-export gaussian-splat --load-config outputs/<id>/.../config.yml \
  --output-dir outputs/<id>                                                     # writes the 3DGS .ply
```

### Viewer

| Use | Tool | Notes |
|---|---|---|
| Submission / cold open | **SuperSplat** — `https://playcanvas.com/supersplat` | web, zero install; reads the standard 3DGS `.ply` from `ns-export gaussian-splat` |
| During development | `ns-viewer --load-config ...` | built into Nerfstudio (viser-based) |
| Custom viewer for the semantics layer (Tier 1/2) | **`@mkkellogg/gaussian-splats-3d`** (three.js) | easiest to bolt a UI onto — label toggle, text-query box; loads `.ply` / `.splat` |

### Semantics layer (creativity on top — pull in only the tiers you reach)

| Purpose | Library / model | Pin & landmines |
|---|---|---|
| 2D masks on keyframes (Tier 0) | **Meta SAM 1** — `segment-anything`, ViT-H checkpoint | `pip install git+https://github.com/facebookresearch/segment-anything.git`; download `sam_vit_h_4b8939.pth` (~2.4 GB); SAM 1 over SAM 2 for static keyframes — simpler, rock-solid |
| (optional) box prompts for named regions | `groundingdino` (IDEA-Research) | has a CUDA op but falls back to CPU; if it won't build, **skip it** — use SAM auto-masks + VLM naming instead |
| Open-vocabulary naming (Tier 1) | `transformers` + `accelerate`; model `Qwen/Qwen2.5-VL-7B-Instruct` (or `Qwen/Qwen2-VL-7B-Instruct`) | needs a recent `transformers` (≥4.49 for Qwen2.5-VL; ≥4.45 for Qwen2-VL) — follow the model card; **local** model so a grader needs no API key; runs fine on a DGX |
| Per-Gaussian text query (Tier 2) | `open_clip_torch` | stable; lift CLIP image embeddings onto Gaussians the same ray-cast way, match typed text at render time |
| Parse / write the Gaussian `.ply`; the lift itself | `plyfile`, `numpy` | the lift is just: load Gaussians from the `.ply`, load camera poses from Nerfstudio's `transforms.json`, ray-cast each labeled pixel to the nearest Gaussians, vote |

> If you go the Docker route for the core, run the semantics stage either with `pip install` inside the container or in a separate conda env that only needs `torch` + the table above — it does not need Nerfstudio or COLMAP.

---

## Conventions

| Path (example) | Purpose |
|----------------|---------|
| `data/raw/<scene_id>.mp4` | Input videos |
| `data/scenes/<scene_id>/` | What `ns-process-data video --output-dir` creates: `images/` (undistorted frames), `colmap/sparse/0/` (SfM output), `transforms.json` (poses `splatfacto` reads) |
| `outputs/<scene_id>/` | `ns-train` run dirs (config, checkpoints, logs), exported `.ply`, `run_meta.json` |
| `semantics/<scene_id>/` | keyframe masks, per-Gaussian labels, the label-tinted `.ply` |

Use **environment variables** or a single `config.yaml` for roots so paths differ between machines without code edits.

**`.gitignore` `data/` and `outputs/`** — one scene is tens of GB (frames + COLMAP + checkpoints) and a trained 3DGS `.ply` is typically **100 MB – 1 GB+**. Commit configs and scripts, not artifacts; reference artifacts by path/URL/hash in the README.

---

## Phase 0 — Repository, environment, and run contract

### Goal

One reproducible way to install dependencies and invoke each stage (no notebook-only “works on my machine” steps).

### Implementation

- **Pin the environment, and test it on the DGX, before anything else.** Use the pinned stack in **"Stack decisions — pinned"** above: Docker-first (the official Nerfstudio image) so CUDA / PyTorch / gsplat / COLMAP already agree; the manual conda recipe is the fallback. Whichever you use, capture exactly what worked (image tag, or exported `environment.yml`) so the README is reproducible.
- Add **thin wrapper scripts** or a `Makefile` / `justfile` with explicit targets, for example:
  - `process-data` (video → frames + COLMAP, via `ns-process-data`)
  - `train-splat` (smoke + full profiles)
  - `export-splat`
  - `lift-semantics` (keyframes → masks → per-Gaussian labels → tinted `.ply`)
- Record **CUDA driver / PyTorch / COLMAP / Nerfstudio** versions in the root `README.md` and in `run_meta.json` for every run.

### Test phase (exit criteria)

- [ ] **DGX bring-up (DGX):** in the pinned env on the DGX — `nvidia-smi` shows the GPUs; `python -c "import torch; print(torch.cuda.is_available())"` prints `True`; `python -c "import gsplat"` succeeds **without triggering a CUDA compile** (or the compile completes cleanly once); `ns-train --help` runs; `colmap -h` runs and reports CUDA support. **If this isn't green within ~45 min on the manual path, fall back to the Docker image rather than wrestling versions.** This is the gate that prevents the "doesn't build on the box" surprise after you've already extracted frames.
- [ ] **Fresh clone test:** from a clean directory, follow install docs; `import` of your Python entrypoints succeeds.
- [ ] **Path resolution test:** run a `--dry-run` or print resolved absolute paths for `data/` and `outputs/`; no hardcoded user-specific paths in committed scripts.
- [ ] **Smoke command test:** a no-op or `--help` on every CLI entrypoint exits `0`.

**Gate:** Do not extract frames at production scale until Phase 0 passes — including the DGX bring-up check.

---

## Phase 1 — Frames (via `ns-process-data`)

### Goal

Turn an input video into the **ordered, undistorted frame set** that COLMAP and `splatfacto` consume — with the right *number* of frames and no garbage ones.

### Implementation

- **Primary path — one command:** `ns-process-data video --data data/raw/<scene_id>.mp4 --output-dir data/scenes/<scene_id> --num-frames-target 300`. The `--num-frames-target` flag is the important one: it samples roughly N evenly-spaced frames, which *is* the "target a count, not a fixed stride" idea — for a small room, **~150–300 frames** is the band you want (enough overlap, not redundant). Don't hand-roll an extractor; this also feeds straight into the COLMAP step (Phase 2 is the same command).
- **Optional safeguard — only if a take has visible motion blur:** dump frames yourself (`ffmpeg -i clip.mp4 -vf fps=N frames/%06d.jpg`), drop the bottom X% by sharpness (variance of Laplacian), then run `ns-process-data images --data frames/ --output-dir data/scenes/<scene_id>` instead. ~20 lines; skip it entirely if you filmed cleanly (slow pan, bright room, steady hands — far more effective than post-hoc culling).
- Either way, the output is `data/scenes/<scene_id>/images/` (zero-padded, temporal order) — Nerfstudio handles naming.

### Test phase (exit criteria)

- [ ] **Count sanity:** `images/` holds roughly the target count (~150–300); if far below, the clip is too short or too fast — re-film before COLMAP, this is the cheapest place to catch it.
- [ ] **Order test:** first and last images in `images/` visually match the start/end of the clip.
- [ ] **Quality spot check:** open ~5 random frames; no all-black / all-white frames; if many are motion-blurred, take the optional-safeguard route or re-film.

**Gate:** Don't run COLMAP at full scale on a bad frame set (blurry / too few) — bad frames waste all downstream time.

---

## Phase 2 — Camera poses (COLMAP, run by `ns-process-data`)

### Goal

Get **intrinsics** (lens model) + **extrinsics** (per-frame camera pose) + a **sparse 3D point cloud** consistent with the room. Same `ns-process-data` invocation as Phase 1 produces all of this — this phase is the **quality check on the poses**, not a separate run.

### Implementation

- COLMAP runs *inside* `ns-process-data video|images`. For video-derived frames pass **`--matching-method sequential`** (frames are temporally ordered, so don't pay for exhaustive matching); add vocab-tree / loop-closure options if it offers them and you have time. It writes `data/scenes/<scene_id>/colmap/sparse/0/` + `transforms.json` + an undistorted `images/` — the layout `splatfacto` reads directly.
- Keep the `ns-process-data` console log (registration counts, COLMAP messages) — that's your debugging trail.
- **If you instead went the raw-COLMAP route** (your own `colmap feature_extractor` / `sequential_matcher` / `mapper`): make sure GPU matching is actually on (`SiftMatching.use_gpu=1`) — CPU matching on a few hundred frames is painfully slow — and archive the raw text logs.

### Test phase (exit criteria)

- [ ] **Process success:** `ns-process-data` exits `0`; `data/scenes/<scene_id>/colmap/sparse/0/` and `transforms.json` exist and are non-empty.
- [ ] **Registration rate:** read how many frames COLMAP registered vs. total (in the `ns-process-data` log / the `transforms.json` frame count). For a slow pan of a textured room, expect a high fraction; many drops → motion blur, blank walls, or too few frames.
- [ ] **Sparse cloud sanity (visual):** open `colmap/sparse/0/` in the COLMAP GUI, Meshlab, or CloudCompare. **Pass:** rough room shape (walls/floor/furniture) recognizable. **Fail:** random ball, duplicated structures, or a "twisted" cloud.
- [ ] **Trajectory sanity (visual):** camera path is smooth and similar to how you moved; huge jumps without corresponding motion are a red flag.

**Gate:** Do not start splat training until the sparse cloud and trajectory pass visual sanity (a tiny smoke subset to test wiring only is fine — just label that run non-production).

---

## Phase 3 — Dataset packaging for Gaussian splatting

### Goal

Make sure what `ns-process-data` produced is in the layout `splatfacto` reads — which, on the primary path, it already is. This phase is mostly a **verification step**, not a conversion step.

### Implementation

- **Primary path: nothing to convert.** `ns-process-data video|images` already emits an undistorted `images/` + `colmap/sparse/0/` + `transforms.json` in exactly the layout `ns-train splatfacto --data data/scenes/<scene_id>` expects. You point `ns-train` at that directory and you're done. Do **not** hand-roll your own `transforms.json` — COLMAP is the OpenCV camera convention (x right, y down, z forward), Nerfstudio's JSON is not, and `ns-process-data` already handled the conversion correctly. Re-doing it by hand is the classic silent-floaters bug.
- **Only if you went the raw-COLMAP route** (your own `colmap mapper`, no `ns-process-data`): run `colmap image_undistorter` to get the standard undistorted `images/` + `sparse/0/` layout, then either feed that to a converter or use the Inria-style dataset layout — and treat the coordinate convention as load-bearing.
- Record **COLMAP / Nerfstudio versions**, the `ns-process-data` command, and the resulting frame count in `outputs/<scene_id>/run_meta.json`.

### Test phase (exit criteria)

- [ ] **Layout check:** `data/scenes/<scene_id>/` contains `images/`, `colmap/sparse/0/`, and `transforms.json`; `transforms.json` lists the expected number of frames.
- [ ] **Loader validation:** `ns-train splatfacto --data ...` (run it for a few steps) prints the **correct image count**; no missing-file errors; not "0 frames".
- [ ] **Coordinate spot check (optional but strong):** pick one 3D point from `colmap/sparse/0/` and verify its projection into a frame lands near the expected image feature — the single best check that nothing got mis-converted. If you used the primary path this should pass for free; if it fails, suspect a manual `transforms.json` step you shouldn't have done.

**Gate:** If the loader can't read the directory, fix it before any long GPU job.

---

## Phase 4 — Gaussian splat training (smoke)

### Goal

Prove **end-to-end GPU training** works: images + poses → optimization steps → checkpoint → export path—using **short iterations** (the full frame set is fine; it's the iteration count that's small here, not the data).

### Implementation

- Run: `ns-train splatfacto --data data/scenes/<scene_id> --max-num-iterations 5000` (full frame set is fine — 5k iters on a DGX is minutes; no need to subset). The point is a fast end-to-end pass, not quality.
- Iterations: **~3k–7k**, not a few hundred. 3DGS densification runs through roughly the first half of the default ~30k schedule; under ~3k iters the model is still translucent blobs **even when everything is wired correctly**, so a too-short smoke fails its own render check for the wrong reason.
- Then `ns-export gaussian-splat --load-config outputs/<scene_id>/.../config.yml --output-dir outputs/<scene_id>` to produce the `.ply`.
- Watch: iteration time, loss, **peak VRAM** (sizes the full run).

### Test phase (exit criteria) **(DGX)**

- [ ] **Stability:** no CUDA OOM; training hits `--max-num-iterations` without crash. Note peak VRAM vs. card capacity.
- [ ] **Loss trend:** the photometric loss in the `ns-train` output **decreases** (monotonicity not required, but flat-from-step-1 is suspicious).
- [ ] **Render sanity:** open `ns-viewer --load-config ...` — the scene is **spatially structured and recognizably the room** (blurry is fine; uniform fog at 3k+ iters → suspect poses/packaging, not training length).
- [ ] **Artifact IO:** `ns-export gaussian-splat` writes a `.ply`; size plausible (not empty, not absurdly small).
- [ ] **Viewer load probe (brings Phase 6 forward):** that `.ply` opens in **SuperSplat** (drag-and-drop) without any manual conversion. Cheap here; prevents discovering a format mismatch *after* the full run.

**Gate:** Do not launch full training until the smoke `.ply` both renders something room-shaped and opens in SuperSplat.

---

## Phase 5 — Gaussian splat training (full)

### Goal

Train on the **full** frame set (or your chosen production stride) with your production hyperparameters until quality plateaus or a scheduled iteration budget ends.

### Implementation

- Use **tmux** / **screen** / cluster scheduler so long jobs survive SSH drops.
- Save periodic checkpoints (if supported) and the **final** export.
- Optional: hold out every *N*th frame from training for a simple **novel-view** evaluation folder (move those images out of the training set but keep poses for metrics only if your tooling supports evaluation mode).

### Test phase (exit criteria) **(DGX)**

- [ ] **Training-view quality:** renders at known poses match photos: minimal “double edges” on textures, stable walls.
- [ ] **Novel-view quality (if holdout exists):** report **PSNR / SSIM / LPIPS** on the holdout set (most trainers compute these in eval mode) so the submission has numbers, not just adjectives. Holdout renders should be **coherent** — acceptable blur OK; catastrophic warp or floaters = investigate poses or coverage.
- [ ] **Coverage audit:** if large screen-facing regions are never observed, expect holes; document as limitation rather than chasing infinite quality.
- [ ] **Reproducibility note:** save full CLI args + git commit hash + COLMAP/trainer/CUDA versions + final `.ply` size in `run_meta.json`.

**Gate:** Freeze this export as the **submission artifact** before starting optional semantics.

---

## Phase 6 — Export and viewer

### Goal

Produce a **submission-friendly** splat file (e.g. `.ply`) and a **repeatable** way to view it (desktop and/or web).

### Implementation

- A 3DGS `.ply` is not a generic point cloud — it carries per-Gaussian **position, opacity, anisotropic scale, rotation quaternion, and SH color coefficients** in a specific property layout. Document which exporter produced it and which viewer reads that layout; not every viewer reads every variant.
- If the chosen web viewer wants a compressed format (`.splat`, `.spz`), include the conversion step (`.ply` → `.splat`) in the export wrapper, not as a manual afterthought.
- Pin a **known-good** viewer commit or vendor a static build; document browser(s) tested (Chrome, Safari, …).

### Test phase (exit criteria)

- [ ] **Cold open:** on a machine that did **not** run training, open the export in the documented viewer with **zero manual conversion** — the wrapper produces whatever the viewer needs.
- [ ] **Navigation:** 60–120 seconds of interaction without crashes; no NaN-explosion visuals.
- [ ] **Evidence bundle:** capture **3–5 screenshots** + optional short screen recording for `README.md`; note the `.ply` file size.

**Gate:** README “example outputs” should point at these artifacts.

---

## Phase 7 — Documentation and frozen example

### Goal

A stranger (or future you) can reproduce your **canonical** result on the DGX with your repo.

### Implementation

- Root `README.md`: prerequisites, install, **exact commands**, expected output tree, example input (hosted or script to download), example outputs (images or links), short **design choices** section.
- Include **filming tips** (slow pan, overlap, avoid motion blur) because they affect all stages.

### Test phase (exit criteria)

- [ ] **Third-party simulation:** follow your own README in a clean environment; fix gaps found.
- [ ] **Artifact checklist:** input video hash or URL, output PLY path, viewer instructions, runtime ballpark (COLMAP minutes, training GPU-hours order-of-magnitude).

**Gate:** Treat README completion as a **release candidate** for the internship submission.

---

## Phase 8 (optional) — Semantic understanding fused to geometry

### Goal

Attach **labels** (e.g. chair, table) to 3D in a way that **respects** cameras and surfaces—not only 2D overlays.

### Implementation (suggested minimal path)

- Run 2D segmentation on a **sparse** set of keyframes (same indices as training images or a subset).
- Back-project or intersect rays using **training cameras** and **rendered depth** or mesh proxy from your geometric representation; **fuse** labels with voting / confidence.
- Store per-Gaussian or per-voxel label + confidence; visualize in viewer (tint or filter).

### Test phase (exit criteria)

- [ ] **2D sanity:** masks on sample frames are visually correct for a few classes you care about.
- [ ] **3D alignment:** labels **cling to surfaces** when orbiting; not a smeared cloud offset from walls.
- [ ] **Multi-view consistency:** same physical object receives the **same** top label from most viewpoints (report simple agreement %).
- [ ] **Honest limitations:** document failure cases (mirrors, glass, people, motion).

**Gate:** If agreement is poor, ship geometry + README first; present semantics as **experimental** with screenshots, not as the main claim.

---

## Cross-phase integration gates (summary)

| Gate | When | What it prevents |
|------|------|------------------|
| **0** | After Phase 0 (incl. DGX bring-up) | Discovering the trainer/COLMAP/CUDA stack doesn't build *after* you've extracted frames |
| **A** | After Phase 2 | Long jobs on bad poses |
| **B** | After Phase 3 | Training on broken camera packaging / wrong coordinate convention |
| **C** | After Phase 4 (incl. viewer load probe) | Burning full GPU budget on unwired code or an export format the viewer can't read |
| **D** | After Phase 5 | Building the semantics layer (or writing the README) on top of a splat result you might still re-train — freeze the `.ply` as the submission artifact first |
| **E** | After Phase 6 | Submission with no viewable result |

---

## Suggested parallel tracks (non-blocking)

These can happen alongside early phases without changing order above:

- **CI-lite:** format check + “dataset loader only” test on a **tiny** synthetic or public sample (no full training).
- **Linting / typing:** keep orchestration code readable for reviewers skimming the repo.

---

## Risk register (short)

| Risk | Mitigation |
|------|------------|
| Plain walls break feature matching | More angular change while filming; keep posters/clutter in frame; raise `--num-frames-target` so overlap is denser |
| Motion blur | Shorter shutter in camera app; slower motion; if a take is already blurry, use the optional ffmpeg + sharpness-floor route in Phase 1 |
| Dynamic objects (people) | Film empty room or mask dynamic regions |
| COLMAP drops frames | Read the registration rate from the `ns-process-data` log; use `--matching-method sequential` (+ loop-closure / vocab-tree if offered); raise `--num-frames-target`; drop blurry takes |
| Trainer can't build CUDA ext on DGX | Phase 0 gate: prefer pip-wheel trainers (`gsplat`/`splatfacto`); if using Inria repo, prove the build in Phase 0 |
| Wrong coordinate convention → floaters | Use trainer's native COLMAP dataparser; don't hand-roll `transforms.json`; do the projection spot-check in Phase 3 |
| OOM on DGX | Reduce resolution or image batch; cap Gaussian count; use trainer’s memory guidance; size from Phase 4 peak VRAM |
| Export format viewer can't read | Phase 4 viewer load probe; bake `.ply`→`.splat` conversion into the export wrapper |

---

## Definition of done (whole project)

- [ ] One **canonical** scene runs **commands-only** from `README.md` on the DGX through export.
- [ ] Example **input** and **output** artifacts are referenced or included.
- [ ] Viewer demonstrates **coherent** geometry for that example.
- [ ] Design note explains **why** COLMAP (or alternative) + **why** Gaussian splatting, and major tradeoffs (frame stride, runtime, failure modes).
- [ ] Optional semantics, if present, include **alignment** evidence; if absent, geometry is still submission-complete.
