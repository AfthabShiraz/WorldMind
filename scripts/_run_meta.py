"""Write a run_meta.json next to each output so runs are reproducible.

Recorded for every stage that produces an artifact (frames/COLMAP, training,
export). Captures git hash, exact CLI, durations, and tool versions. The plan
calls this out as a hard requirement (Phase 3 + Phase 5: "save full CLI args
+ git commit hash + COLMAP/trainer/CUDA versions").
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> str | None:
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True)
        return out.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def collect(stage: str, scene: str, args: list[str], extra: dict | None = None) -> dict:
    meta = {
        "stage": stage,
        "scene": scene,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "platform": f"{platform.system()} {platform.release()} {platform.machine()}",
        "cli": args,
        "git": {
            "commit": _run(["git", "rev-parse", "HEAD"]),
            "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
            "dirty": bool(_run(["git", "status", "--porcelain"])),
        },
        "python": sys.version.split()[0],
        "versions": {
            "torch": _import_version("torch"),
            "torch_cuda_build": _torch_cuda(),
            "gsplat": _import_version("gsplat"),
            "nerfstudio": _import_version("nerfstudio"),
            "colmap": _colmap_version(),
            "ffmpeg": _ffmpeg_version(),
            "driver": _run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        },
    }
    if extra:
        meta.update(extra)
    return meta


def _import_version(mod: str) -> str | None:
    try:
        m = __import__(mod)
    except Exception:
        return None
    return getattr(m, "__version__", None)


def _torch_cuda() -> str | None:
    try:
        import torch
        return torch.version.cuda
    except Exception:
        return None


def _colmap_version() -> str | None:
    if not shutil.which("colmap"):
        return None
    out = _run(["colmap", "-h"]) or ""
    for line in out.splitlines():
        if "colmap" in line.lower() and any(c.isdigit() for c in line):
            return line.strip()
    return "unknown"


def _ffmpeg_version() -> str | None:
    if not shutil.which("ffmpeg"):
        return None
    out = _run(["ffmpeg", "-version"]) or ""
    first = out.splitlines()[0] if out else ""
    return first or None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--stage", required=True, help="phase/stage name, e.g. process_data, train_smoke")
    p.add_argument("--scene", required=True, help="scene_id")
    p.add_argument("--out", required=True, type=Path, help="path to run_meta.json")
    p.add_argument("--arg", action="append", default=[], help="raw CLI tokens, repeated")
    p.add_argument("--extra-json", default=None, help="extra fields to merge (JSON string)")
    ns = p.parse_args()

    extra = json.loads(ns.extra_json) if ns.extra_json else None
    meta = collect(ns.stage, ns.scene, ns.arg, extra)
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    ns.out.write_text(json.dumps(meta, indent=2, sort_keys=True))
    print(f"wrote {ns.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
