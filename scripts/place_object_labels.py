#!/usr/bin/env python3
"""
place_object_labels.py — anchor 3D floating labels for the strongly-seen
objects in scene_inventory.json.

Method (Option A in the design):
  For each strongly-seen object name from scene_inventory.json, ask
  Qwen2.5-VL to draw a bounding box of that object in each keyframe.
  Back-project the bbox center into 3D using a depth proxy built from
  projected Gaussian z-values, with three safety filters to avoid the
  classic 'label flies through a window into the distance' failure:

    1. Local Gaussian density. The pixel must have >= MIN_DENSITY
       projected Gaussians inside a small patch — otherwise it's over
       a void/hole and the depth proxy can't be trusted.
    2. Scene-relative depth cap. Reject depths > MAX_DEPTH_FRAC * scene
       diagonal (filters far-away floater Gaussians).
    3. AABB containment. The back-projected 3D point must land inside
       the scene's 5-95 percentile bounding box.

  Surviving 3D points per object are clustered with DBSCAN (min 3 views
  agreeing) — one cluster = one real instance of that object name in
  the room. Each cluster's centroid becomes a label anchor.

Output: semantics/<scene>/object_anchors.json
  [{label, anchor:[x,y,z], n_views, n_clusters, cluster_idx}, ...]

Usage:
  scripts/place_object_labels.py --scene <id> [--min-frames K]
        [--max-depth-frac 0.6] [--cluster-eps 0.4]
        [--show-raw] [--force]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData
from tqdm import tqdm


REPO = Path(__file__).resolve().parent.parent
QWEN_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# I/O helpers — copied/shared from lift_semantics_v2 so this script can run
# standalone without importing the v2 lift module.
# ---------------------------------------------------------------------------

def load_transforms(scene: str) -> dict:
    return json.loads((REPO / "data" / "scenes" / scene / "transforms.json").read_text())


def load_dataparser_transform(scene: str) -> tuple[np.ndarray, float]:
    runs = sorted((REPO / "outputs" / scene / "splatfacto_full" / "splatfacto").glob(
        "*/dataparser_transforms.json"))
    if not runs:
        raise FileNotFoundError(f"no dataparser_transforms.json for scene {scene}")
    d = json.loads(runs[-1].read_text())
    T = np.eye(4, dtype=np.float64)
    T[:3, :4] = np.asarray(d["transform"], dtype=np.float64)
    return T, float(d["scale"])


def load_splat_xyz(scene: str) -> np.ndarray:
    p = REPO / "outputs" / scene / "ply_full" / "splat.ply"
    ply = PlyData.read(str(p))
    el = ply["vertex"]
    return np.stack([el["x"], el["y"], el["z"]], axis=1).astype(np.float32)


def project_gaussians(xyz_h: np.ndarray, w2c: np.ndarray,
                      fx: float, fy: float, cx: float, cy: float,
                      W: int, H: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    p = (w2c @ xyz_h.T).T
    Xc, Yc, Zc = p[:, 0], -p[:, 1], -p[:, 2]
    in_front = Zc > 1e-3
    u = (fx * Xc / np.where(in_front, Zc, 1.0)) + cx
    v = (fy * Yc / np.where(in_front, Zc, 1.0)) + cy
    valid = in_front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    idx = np.where(valid)[0]
    return idx, u[idx].astype(np.int32), v[idx].astype(np.int32), Zc[idx].astype(np.float32)


def build_depth_and_density(ui: np.ndarray, vi: np.ndarray, z: np.ndarray,
                            H: int, W: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-pixel front-most z and per-pixel projected-Gaussian density.
    Density is the count of Gaussians whose centre projected to that pixel."""
    depth = np.full((H, W), np.inf, dtype=np.float32)
    np.minimum.at(depth, (vi, ui), z)
    density = np.zeros((H, W), dtype=np.int32)
    np.add.at(density, (vi, ui), 1)
    return depth, density


def back_project(c2w: np.ndarray, u_px: float, v_px: float, depth: float,
                 fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Pixel + camera-space depth -> world XYZ. Nerfstudio (OpenGL) convention."""
    Xc = (u_px - cx) * depth / fx
    Yc = (v_px - cy) * depth / fy
    Zc = depth
    # OpenCV -> OpenGL/nerfstudio: flip y, z.
    p_cam = np.array([Xc, -Yc, -Zc, 1.0], dtype=np.float64)
    return (c2w @ p_cam)[:3]


# ---------------------------------------------------------------------------
# Qwen bbox parsing
# ---------------------------------------------------------------------------

# Qwen2.5-VL emits boxes in a few formats; cover the common ones.
_BOX_PATTERNS = [
    # <|box_start|>(120,300),(450,600)<|box_end|>  or  <box>(...)</box>
    re.compile(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*[,;]\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)"),
    # Plain "120,300,450,600"
    re.compile(r"(?<!\d)(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)(?!\d)"),
    # "<box>120 300 450 600</box>"
    re.compile(r"(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"),
]


def parse_bbox(raw: str, W: int, H: int) -> tuple[int, int, int, int] | None:
    """Extract (x1, y1, x2, y2) pixel coords or None. Qwen2.5-VL outputs
    coords in pixel space already (not normalised) for this model."""
    s = raw.strip().lower()
    if "none" in s[:20] or "not visible" in s or "cannot" in s[:30]:
        return None
    for pat in _BOX_PATTERNS:
        m = pat.search(s)
        if not m:
            continue
        x1, y1, x2, y2 = (int(g) for g in m.groups())
        # Qwen 2.5-VL uses pixel coords. Some variants normalise to 0-1000;
        # if all four values <= 1000 and the image is bigger, that's the
        # normalised case — rescale.
        if max(x1, y1, x2, y2) <= 1001 and max(W, H) > 1000:
            x1 = int(round(x1 * W / 1000))
            x2 = int(round(x2 * W / 1000))
            y1 = int(round(y1 * H / 1000))
            y2 = int(round(y2 * H / 1000))
        x1, x2 = sorted((max(0, x1), min(W - 1, x2)))
        y1, y2 = sorted((max(0, y1), min(H - 1, y2)))
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None
        return (x1, y1, x2, y2)
    return None


def qwen_ground(model, proc, img: "Image.Image", obj: str) -> str:
    """Single Qwen call asking for the bbox of `obj` in `img`. Returns the
    raw text response so the caller can parse + log it."""
    import torch
    prompt = (
        f"Output the bounding box of the {obj} in the image. "
        f"If no {obj} is visible, output exactly: none"
    )
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": prompt},
    ]}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[img], padding=True, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        gen = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    out_ids = gen[:, inputs.input_ids.shape[1]:]
    return proc.batch_decode(out_ids, skip_special_tokens=True)[0]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def pick_keyframes(scene: str, n: int) -> list[dict]:
    t = load_transforms(scene)
    frames = t["frames"]
    idxs = np.linspace(0, len(frames) - 1, min(n, len(frames)), dtype=int)
    return [{"idx": int(i), **frames[int(i)]} for i in idxs]


def load_inventory(scene: str, min_frames: int) -> list[str]:
    p = REPO / "semantics" / scene / "scene_inventory.json"
    if not p.exists():
        log(f"ERROR: no scene_inventory.json — run `make scene-inventory SCENE={scene}` first")
        return []
    inv = json.loads(p.read_text())
    # Apply user threshold to the per-object counts, OR auto-default to ~1/3
    # of keyframes (matches the viewer's default).
    nk = int(inv.get("n_keyframes", 1))
    if min_frames <= 0:
        min_frames = max(2, nk // 3)
    keep = [name for name, c in inv["counts"] if int(c) >= min_frames]
    # Filter out non-physical "objects" that the VLM sometimes lists.
    SKIP = {"wall", "floor", "ceiling", "room", "scene", "background", "view"}
    keep = [k for k in keep if k not in SKIP]
    log(f"inventory: {len(keep)} strongly-seen objects (>= {min_frames} frames) "
        f"after dropping non-physical surfaces")
    return keep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--n-keyframes", type=int, default=15,
                    help="Same N used by scene_inventory for consistency.")
    ap.add_argument("--min-frames", type=int, default=0,
                    help="Inventory items below this frame count are skipped. "
                         "0 = auto (~1/3 of n_keyframes), matches viewer.")
    ap.add_argument("--patch", type=int, default=11,
                    help="Pixel patch size for the local Gaussian density check.")
    ap.add_argument("--min-density", type=int, default=10,
                    help="Min projected-Gaussian count inside patch to trust the depth.")
    ap.add_argument("--max-depth-frac", type=float, default=0.6,
                    help="Reject back-projections at depth > frac * scene diagonal.")
    ap.add_argument("--cluster-eps", type=float, default=0.4,
                    help="DBSCAN epsilon in splat units. Tune per scene scale.")
    ap.add_argument("--min-views-per-anchor", type=int, default=3,
                    help="DBSCAN min_samples — anchor needs this many views.")
    ap.add_argument("--show-raw", action="store_true",
                    help="Print Qwen's raw bbox response per (object, keyframe).")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = REPO / "semantics" / args.scene
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "object_anchors.json"
    if out_path.exists() and not args.force:
        log(f"cached: {out_path} (use --force to redo)")
        return 0

    objects = load_inventory(args.scene, args.min_frames)
    if not objects:
        return 2
    kfs = pick_keyframes(args.scene, args.n_keyframes)
    log(f"keyframes: {len(kfs)}, objects: {len(objects)}, "
        f"~{len(kfs) * len(objects)} Qwen calls upcoming")

    # ---- Camera + splat setup ---------------------------------------------
    xyz = load_splat_xyz(args.scene)
    n_gauss = xyz.shape[0]
    xyz_h = np.hstack([xyz, np.ones((n_gauss, 1), dtype=np.float32)])

    scene_lo = np.percentile(xyz, 5, axis=0)
    scene_hi = np.percentile(xyz, 95, axis=0)
    scene_diag = float(np.linalg.norm(scene_hi - scene_lo))
    max_depth = args.max_depth_frac * scene_diag
    log(f"scene 5-95 AABB: lo={scene_lo}, hi={scene_hi}, diag={scene_diag:.2f}, "
        f"max_depth={max_depth:.2f}")

    t = load_transforms(args.scene)
    fx, fy = float(t["fl_x"]), float(t["fl_y"])
    cx, cy = float(t["cx"]), float(t["cy"])
    W, H = int(t["w"]), int(t["h"])
    T_dp, s_dp = load_dataparser_transform(args.scene)

    # ---- Pre-compute per-keyframe depth + density maps --------------------
    log("pre-computing per-keyframe depth + density maps ...")
    kf_caches: list[dict] = []
    for k in tqdm(kfs, desc="proj", file=sys.stderr):
        c2w_ns = np.array(k["transform_matrix"], dtype=np.float64)
        c2w = T_dp @ c2w_ns
        c2w[:3, 3] *= s_dp
        w2c = np.linalg.inv(c2w)
        idx, ui, vi, z = project_gaussians(xyz_h, w2c, fx, fy, cx, cy, W, H)
        depth_map, density_map = build_depth_and_density(ui, vi, z, H, W)
        kf_caches.append({"kf": k, "c2w": c2w, "depth": depth_map, "density": density_map})

    # ---- Qwen --------------------------------------------------------------
    log("loading Qwen2.5-VL-3B ...")
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    proc = AutoProcessor.from_pretrained(QWEN_ID)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_ID, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    # detections[obj] -> list of (kf_idx, xyz_world, reject_reason or None)
    detections: dict[str, list[dict]] = defaultdict(list)
    n_calls = 0
    half = args.patch // 2

    for obj in tqdm(objects, desc="objects", file=sys.stderr):
        for cache in kf_caches:
            k = cache["kf"]
            img = Image.open(REPO / "data" / "scenes" / args.scene / k["file_path"]).convert("RGB")
            raw = qwen_ground(model, proc, img, obj)
            n_calls += 1
            box = parse_bbox(raw, W, H)
            entry = {"kf": k["idx"], "raw": raw.strip(), "bbox": box,
                     "reject": None, "xyz": None}
            if box is None:
                entry["reject"] = "no_box"
                detections[obj].append(entry)
                if args.show_raw:
                    log(f"[{obj}/kf{k['idx']:04d}] no box  raw={raw.strip()!r}")
                continue
            x1, y1, x2, y2 = box
            u_px = (x1 + x2) // 2
            v_px = (y1 + y2) // 2
            entry["center_px"] = [int(u_px), int(v_px)]

            # Safety 1: local density
            yy0 = max(0, v_px - half); yy1 = min(H, v_px + half + 1)
            xx0 = max(0, u_px - half); xx1 = min(W, u_px + half + 1)
            density_patch = cache["density"][yy0:yy1, xx0:xx1].sum()
            if density_patch < args.min_density:
                entry["reject"] = f"sparse_patch({density_patch})"
                detections[obj].append(entry)
                continue
            depth_patch = cache["depth"][yy0:yy1, xx0:xx1]
            valid_depths = depth_patch[np.isfinite(depth_patch)]
            if valid_depths.size == 0:
                entry["reject"] = "no_finite_depth"
                detections[obj].append(entry)
                continue
            d = float(np.median(valid_depths))   # median over patch is robust

            # Safety 2: scene-relative depth cap
            if d > max_depth:
                entry["reject"] = f"too_far({d:.2f}>{max_depth:.2f})"
                detections[obj].append(entry)
                continue

            # Back-project
            xyz_w = back_project(cache["c2w"], u_px, v_px, d, fx, fy, cx, cy)
            entry["xyz"] = xyz_w.tolist()
            entry["depth"] = d

            # Safety 3: must land inside the scene AABB
            if not (np.all(xyz_w >= scene_lo - 0.1 * scene_diag) and
                    np.all(xyz_w <= scene_hi + 0.1 * scene_diag)):
                entry["reject"] = "outside_aabb"
                detections[obj].append(entry)
                continue

            detections[obj].append(entry)
            if args.show_raw:
                log(f"[{obj}/kf{k['idx']:04d}] OK  px=({u_px},{v_px}) d={d:.2f} "
                    f"xyz=({xyz_w[0]:+.2f},{xyz_w[1]:+.2f},{xyz_w[2]:+.2f})")

    del model, proc
    torch.cuda.empty_cache()

    # ---- Cluster surviving detections per object --------------------------
    from sklearn.cluster import DBSCAN

    anchors_out: list[dict] = []
    summary: list[str] = []
    for obj, entries in detections.items():
        survivors = [e for e in entries if e["reject"] is None and e["xyz"] is not None]
        pts = np.array([e["xyz"] for e in survivors], dtype=np.float64)
        rejects = len(entries) - len(survivors)
        reject_reasons = defaultdict(int)
        for e in entries:
            if e["reject"]:
                reject_reasons[e["reject"].split("(")[0]] += 1
        if pts.shape[0] < args.min_views_per_anchor:
            summary.append(f"  {obj:<14}  views={pts.shape[0]:>2} rejects={rejects:>2} "
                           f"-> DROPPED (need >={args.min_views_per_anchor})")
            continue
        labels = DBSCAN(eps=args.cluster_eps,
                        min_samples=args.min_views_per_anchor).fit_predict(pts)
        clusters: dict[int, list[int]] = defaultdict(list)
        for i, lab in enumerate(labels):
            if lab >= 0:
                clusters[int(lab)].append(i)
        if not clusters:
            summary.append(f"  {obj:<14}  views={pts.shape[0]:>2} rejects={rejects:>2} "
                           f"-> no cluster (points too scattered)")
            continue
        for cidx, idxs in sorted(clusters.items()):
            cpts = pts[idxs]
            anchor = cpts.mean(axis=0)
            anchors_out.append({
                "label": obj,
                "anchor": [float(anchor[0]), float(anchor[1]), float(anchor[2])],
                "n_views": len(idxs),
                "n_clusters": len(clusters),
                "cluster_idx": cidx,
            })
        summary.append(f"  {obj:<14}  views={pts.shape[0]:>2} rejects={rejects:>2} "
                       f"clusters={len(clusters)} ({dict(reject_reasons)})")

    log(f"Qwen calls: {n_calls}; anchors produced: {len(anchors_out)}")
    log("per-object breakdown (rejects in parentheses):")
    for line in summary:
        log(line)

    out_path.write_text(json.dumps({
        "scene": args.scene,
        "n_keyframes": len(kfs),
        "min_frames": args.min_frames,
        "params": {
            "patch": args.patch,
            "min_density": args.min_density,
            "max_depth_frac": args.max_depth_frac,
            "cluster_eps": args.cluster_eps,
            "min_views_per_anchor": args.min_views_per_anchor,
        },
        "scene_aabb_lo": scene_lo.tolist(),
        "scene_aabb_hi": scene_hi.tolist(),
        "scene_diag": scene_diag,
        "anchors": anchors_out,
        "detections": {k: v for k, v in detections.items()},  # for debugging
    }, indent=2))
    log(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
