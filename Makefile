# WorldMind — phase wrappers. Every target is a thin shim around scripts/*.sh
# so the README can read as plain commands. SCENE is required for all per-scene
# targets; pass it as: make process-data SCENE=living_room
#
# These targets follow the phase ordering in docs/IMPLEMENTATION_PLAN.md.

SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

# Required for any per-scene target.
SCENE ?=
require-scene:
	@test -n "$(SCENE)" || { echo "ERROR: SCENE= is required (e.g. make $(MAKECMDGOALS) SCENE=living_room)"; exit 2; }

.PHONY: help install install-viewer env-check run view view-example fetch-example process-data process-data-manual qc-process-data train-smoke train-full export-splat lift-semantics-v2 view-semantics scene-inventory clean-scene require-scene require-video

help:  ## list targets
	@awk 'BEGIN{FS=":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

VIDEO ?=
require-video:
	@test -n "$(VIDEO)" || { echo "ERROR: VIDEO= is required. e.g.  make run VIDEO=~/myclip.mp4"; exit 2; }

run: require-video ## ONE-SHOT: video -> .ply. Usage: make run VIDEO=/path/to/clip.mp4 [SCENE=name] [PROFILE=full|smoke]
	bash scripts/run.sh --video "$(VIDEO)" $(if $(SCENE),--scene "$(SCENE)") $(if $(PROFILE),--profile "$(PROFILE)") $(EXTRA)

install: ## Full install: miniforge + conda env + torch + nerfstudio. Needed to TRAIN scenes.
	bash scripts/install_env.sh

install-viewer: ## Lightweight install: just viser+plyfile+numpy via pip. Enough to VIEW .ply files.
	@python3 -m pip install -r requirements-viewer.txt

env-check: ## Phase 0 gate: prove the env works end-to-end on the GPU
	bash scripts/env_check.sh

process-data: require-scene ## Phase 1+2: frames + COLMAP, focal-length-seeded (works on phone videos without EXIF)
	bash scripts/colmap_pipeline.sh --scene "$(SCENE)" $(EXTRA)

process-data-nerfstudio: require-scene ## Phase 1+2 via ns-process-data — fails silently on videos without EXIF focal length; kept for debugging
	bash scripts/process_data.sh --scene "$(SCENE)" $(EXTRA)

qc-process-data: require-scene ## Phase 1+2 exit-criteria checks (run after process-data)
	bash scripts/qc_process_data.sh --scene "$(SCENE)"

train-smoke: require-scene ## Phase 4: 5k-iter smoke run, exports a .ply for viewer probe
	bash scripts/train_splat.sh --scene "$(SCENE)" --profile smoke $(EXTRA)

train-full: require-scene  ## Phase 5: full production run (~30k iters), exports submission .ply
	bash scripts/train_splat.sh --scene "$(SCENE)" --profile full $(EXTRA)

export-splat: require-scene ## Phase 6: export latest checkpoint to outputs/<scene>/<scene>.ply
	bash scripts/export_splat.sh --scene "$(SCENE)" $(EXTRA)

view: require-scene ## Launch in-browser viewer for a scene (localhost:8080). Add EXTRA=--share for public URL.
	bash scripts/view_semantics.sh --scene "$(SCENE)" $(EXTRA)

fetch-example: ## Download the room5 example .ply (~150 MB) from the GitHub release if not already present
	bash scripts/fetch_example.sh

view-example: fetch-example ## Auto-download + view the bundled room5 example scene on localhost:8080
	bash scripts/view_semantics.sh --scene room5 $(EXTRA)

# --- Optional: experimental semantics layer (not part of the standard pipeline)
lift-semantics-v2: require-scene ## (Optional) depth-aware instance lifting, sidecar files. splat.ply untouched.
	bash scripts/lift_semantics_v2.sh --scene "$(SCENE)" $(EXTRA)

scene-inventory: require-scene ## (Optional) VLM-only: ask Qwen what objects are in the scene, print a list
	bash scripts/scene_inventory.sh --scene "$(SCENE)" $(EXTRA)

place-labels: require-scene ## (Optional) anchor 3D floating labels for inventory objects via Qwen grounding
	bash scripts/place_object_labels.sh --scene "$(SCENE)" $(EXTRA)

audit-bboxes: require-scene ## (Debug) overlay Qwen's bboxes on each keyframe, colour-coded by accept/reject
	bash scripts/visualize_bboxes.sh --scene "$(SCENE)" $(EXTRA)

clean-scene: require-scene ## remove derived artifacts for a scene (keeps data/raw)
	rm -rf "data/scenes/$(SCENE)" "outputs/$(SCENE)" "semantics/$(SCENE)"
