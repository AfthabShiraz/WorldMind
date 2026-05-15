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

.PHONY: help install env-check run process-data process-data-manual qc-process-data train-smoke train-full export-splat lift-semantics lift-semantics-v2 view-semantics clean-scene require-scene require-video

help:  ## list targets
	@awk 'BEGIN{FS=":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

VIDEO ?=
require-video:
	@test -n "$(VIDEO)" || { echo "ERROR: VIDEO= is required. e.g.  make run VIDEO=~/myclip.mp4"; exit 2; }

run: require-video ## ONE-SHOT: video -> .ply. Usage: make run VIDEO=/path/to/clip.mp4 [SCENE=name] [PROFILE=full|smoke]
	bash scripts/run.sh --video "$(VIDEO)" $(if $(SCENE),--scene "$(SCENE)") $(if $(PROFILE),--profile "$(PROFILE)") $(EXTRA)

install: ## Phase 0: install miniforge + conda env + torch + nerfstudio (idempotent)
	bash scripts/install_env.sh

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

lift-semantics: require-scene ## Phase 8 (v1, legacy): bakes labels into splat.ply colours
	bash scripts/lift_semantics.sh --scene "$(SCENE)" $(EXTRA)

lift-semantics-v2: require-scene ## Phase 8 (v2): depth-aware instance lifting, sidecar files, splat.ply untouched
	bash scripts/lift_semantics_v2.sh --scene "$(SCENE)" $(EXTRA)

view-semantics: require-scene ## Launch viser viewer with floating labels (needs lift-semantics-v2 outputs)
	bash scripts/view_semantics.sh --scene "$(SCENE)" $(EXTRA)

clean-scene: require-scene ## remove derived artifacts for a scene (keeps data/raw)
	rm -rf "data/scenes/$(SCENE)" "outputs/$(SCENE)" "semantics/$(SCENE)"
