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

.PHONY: help install env-check process-data train-smoke train-full export-splat lift-semantics clean-scene require-scene

help:  ## list targets
	@awk 'BEGIN{FS=":.*##"} /^[a-zA-Z_-]+:.*##/ {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Phase 0: install miniforge + conda env + torch + nerfstudio (idempotent)
	bash scripts/install_env.sh

env-check: ## Phase 0 gate: prove the env works end-to-end on the GPU
	bash scripts/env_check.sh

process-data: require-scene ## Phase 1+2: video -> frames + COLMAP (ns-process-data video)
	bash scripts/process_data.sh --scene "$(SCENE)" $(EXTRA)

train-smoke: require-scene ## Phase 4: 5k-iter smoke run, exports a .ply for viewer probe
	bash scripts/train_splat.sh --scene "$(SCENE)" --profile smoke $(EXTRA)

train-full: require-scene  ## Phase 5: full production run (~30k iters), exports submission .ply
	bash scripts/train_splat.sh --scene "$(SCENE)" --profile full $(EXTRA)

export-splat: require-scene ## Phase 6: export latest checkpoint to outputs/<scene>/<scene>.ply
	bash scripts/export_splat.sh --scene "$(SCENE)" $(EXTRA)

lift-semantics: require-scene ## Phase 8: SAM masks + Qwen2.5-VL labels lifted to Gaussians
	bash scripts/lift_semantics.sh --scene "$(SCENE)" $(EXTRA)

clean-scene: require-scene ## remove derived artifacts for a scene (keeps data/raw)
	rm -rf "data/scenes/$(SCENE)" "outputs/$(SCENE)" "semantics/$(SCENE)"
