#!/usr/bin/env python3
"""
lift_semantics_v2.py — attach object instances + labels to Gaussians WITHOUT
overwriting the splat's colours.

Differences from v1:
  * Depth-aware voting. We build a per-keyframe sparse depth proxy from
    projected Gaussian z's, then only count a Gaussian for a mask if it is the
    front-most thing at its pixel (kills votes from Gaussians behind walls).
  * Cross-view mask association. Each mask's voters (visible Gaussians inside
    it) form a set; masks across keyframes whose voter sets overlap are linked.
    Union-find on the link graph -> stable instance IDs.
  * One VLM call per instance, not per mask per frame. The view with the most
    voters is used to label the whole instance.
  * Sidecar outputs only. splat.ply is never modified. Outputs:
      semantics/<scene>/gaussian_instances.npy        # (N,) int32 instance id, -1 = unassigned
      semantics/<scene>/instance_labels.json          # {id: {label, n_views, n_voters}}
      semantics/<scene>/instance_anchors.json         # {id: {anchor, top, n_gaussians, palette_rgb}}

Usage:
  scripts/lift_semantics_v2.py --scene <id>
        [--n-keyframes 30]
        [--max-masks-per-frame 25]
        [--min-mask-area-frac 0.005]
        [--depth-tol 0.10]
        [--assoc-jaccard 0.20]
        [--min-voters-per-instance 200]
        [--force]
        [--only stage]            # keyframes,masks,voters,associate,label,lift,export

The keyframes + masks stages are identical to v1 and will reuse v1's cache
(semantics/<scene>/keyframes.json, semantics/<scene>/masks/kf####.npz).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData
from tqdm import tqdm


REPO = Path(__file__).resolve().parent.parent
SAM_CKPT = REPO / "data" / "checkpoints" / "sam_vit_h_4b8939.pth"
QWEN_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# I/O helpers (shared with v1)
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


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Stage 1 — keyframes (compatible with v1 cache)
# ---------------------------------------------------------------------------

def stage_keyframes(scene: str, n_keyframes: int, force: bool) -> list[dict]:
    out = ensure_dir(REPO / "semantics" / scene) / "keyframes.json"
    if out.exists() and not force:
        kfs = json.loads(out.read_text())
        log(f"[1/7 keyframes] cached: {len(kfs)} keyframes")
        return kfs
    t = load_transforms(scene)
    frames = t["frames"]
    n = min(n_keyframes, len(frames))
    idxs = np.linspace(0, len(frames) - 1, n, dtype=int).tolist()
    kfs = [{"idx_in_transforms": int(i), **frames[i]} for i in idxs]
    out.write_text(json.dumps(kfs, indent=2))
    log(f"[1/7 keyframes] picked {len(kfs)} / {len(frames)}")
    return kfs


# ---------------------------------------------------------------------------
# Stage 2 — SAM auto-mask (identical to v1, reuses its cache)
# ---------------------------------------------------------------------------

def stage_masks(scene: str, kfs: list[dict], max_masks: int,
                min_area_frac: float, force: bool) -> None:
    out_dir = ensure_dir(REPO / "semantics" / scene / "masks")
    todo = [k for k in kfs if force or not (out_dir / f"kf{k['idx_in_transforms']:04d}.npz").exists()]
    if not todo:
        log(f"[2/7 masks] all {len(kfs)} cached")
        return

    log(f"[2/7 masks] loading SAM ViT-H ...")
    import torch
    import torchvision
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    _native_nms = torch.ops.torchvision.nms
    def _cpu_nms(boxes, scores, iou_threshold):
        return _native_nms(boxes.cpu(), scores.cpu(), iou_threshold).to(boxes.device)
    torchvision.ops.boxes.nms = lambda b, s, iou: _cpu_nms(b, s, iou)
    torchvision.ops.nms = torchvision.ops.boxes.nms

    sam = sam_model_registry["vit_h"](checkpoint=str(SAM_CKPT)).to("cuda").eval()
    gen = SamAutomaticMaskGenerator(
        sam, points_per_side=16, pred_iou_thresh=0.86,
        stability_score_thresh=0.92, crop_n_layers=0, min_mask_region_area=2000,
    )
    for k in tqdm(todo, desc="SAM", file=sys.stderr):
        img_path = REPO / "data" / "scenes" / scene / k["file_path"]
        img = np.array(Image.open(img_path).convert("RGB"))
        H, W = img.shape[:2]
        min_area = min_area_frac * H * W
        masks = gen.generate(img)
        masks = [m for m in masks if m["area"] >= min_area]
        masks.sort(key=lambda m: -m["area"])
        masks = masks[:max_masks]
        kf_idx = k["idx_in_transforms"]
        if not masks:
            np.savez_compressed(out_dir / f"kf{kf_idx:04d}.npz",
                                segs=np.zeros((0, H, W), dtype=bool),
                                bboxes=np.zeros((0, 4), dtype=np.int32),
                                areas=np.zeros((0,), dtype=np.int32))
            continue
        segs = np.stack([m["segmentation"] for m in masks])
        bboxes = np.array([m["bbox"] for m in masks], dtype=np.int32)
        areas = np.array([m["area"] for m in masks], dtype=np.int32)
        np.savez_compressed(out_dir / f"kf{kf_idx:04d}.npz",
                            segs=segs, bboxes=bboxes, areas=areas)
    del sam, gen
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Stage 3 — voters: for each (keyframe, mask), find the visible Gaussians
# inside the mask using a depth proxy built from front-most projected centres.
# ---------------------------------------------------------------------------

def project_gaussians(xyz_h: np.ndarray, w2c: np.ndarray,
                      fx: float, fy: float, cx: float, cy: float,
                      W: int, H: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (valid_idx, ui, vi, z) for Gaussians visible in this view.
    Convention: w2c is nerfstudio (OpenGL); we flip y,z to OpenCV for the K matrix."""
    p = (w2c @ xyz_h.T).T            # (N, 4)
    Xc, Yc, Zc = p[:, 0], -p[:, 1], -p[:, 2]
    in_front = Zc > 1e-3
    u = (fx * Xc / np.where(in_front, Zc, 1.0)) + cx
    v = (fy * Yc / np.where(in_front, Zc, 1.0)) + cy
    valid = in_front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
    idx = np.where(valid)[0]
    return idx, u[idx].astype(np.int32), v[idx].astype(np.int32), Zc[idx].astype(np.float32)


def build_depth_proxy(ui: np.ndarray, vi: np.ndarray, z: np.ndarray,
                      H: int, W: int) -> np.ndarray:
    """Per-pixel minimum z over projected Gaussian centres. Pixels with no hit
    are np.inf. We dilate with a 3x3 min filter to fill 1-pixel gaps so a
    Gaussian whose centre lands on a pixel with no neighbour doesn't get
    rejected by a hole in the proxy."""
    depth = np.full((H, W), np.inf, dtype=np.float32)
    # np.minimum.at handles duplicate indices correctly.
    np.minimum.at(depth, (vi, ui), z)
    # Cheap 3x3 min dilation via array slicing.
    d = depth
    out = d.copy()
    out[1:]   = np.minimum(out[1:],   d[:-1])
    out[:-1]  = np.minimum(out[:-1],  d[1:])
    out[:, 1:]  = np.minimum(out[:, 1:],  d[:, :-1])
    out[:, :-1] = np.minimum(out[:, :-1], d[:, 1:])
    return out


def stage_voters(scene: str, kfs: list[dict], depth_tol: float, force: bool) -> dict:
    """For each keyframe, for each mask, list the visible-Gaussian indices that
    project inside the mask. Cached as semantics/<scene>/voters.npz."""
    out_path = REPO / "semantics" / scene / "voters.npz"
    if out_path.exists() and not force:
        log(f"[3/7 voters] cached")
        data = np.load(out_path, allow_pickle=True)
        keys_arr = data["keys"]
        return {
            "keys": [(int(a), int(b)) for a, b in keys_arr],
            "voters": list(data["voters"]),
            "mask_areas": [int(x) for x in data["mask_areas"]],
        }

    xyz = load_splat_xyz(scene)
    n_gauss = xyz.shape[0]
    xyz_h = np.hstack([xyz, np.ones((n_gauss, 1), dtype=np.float32)])

    t = load_transforms(scene)
    fx, fy = float(t["fl_x"]), float(t["fl_y"])
    cx, cy = float(t["cx"]), float(t["cy"])
    W, H = int(t["w"]), int(t["h"])
    T_dp, s_dp = load_dataparser_transform(scene)

    masks_dir = REPO / "semantics" / scene / "masks"
    keys: list[tuple[int, int]] = []    # (kf_idx, mask_idx)
    voters: list[np.ndarray] = []       # list of int32 Gaussian indices
    mask_areas: list[int] = []

    for k in tqdm(kfs, desc="voters", file=sys.stderr):
        kf_idx = k["idx_in_transforms"]
        npz_path = masks_dir / f"kf{kf_idx:04d}.npz"
        if not npz_path.exists():
            continue
        d = np.load(npz_path)
        segs = d["segs"]
        if segs.shape[0] == 0:
            continue

        c2w_ns = np.array(k["transform_matrix"], dtype=np.float64)
        c2w = T_dp @ c2w_ns
        c2w[:3, 3] *= s_dp
        w2c = np.linalg.inv(c2w)

        idx, ui, vi, z = project_gaussians(xyz_h, w2c, fx, fy, cx, cy, W, H)
        if idx.size == 0:
            continue

        depth_proxy = build_depth_proxy(ui, vi, z, H, W)
        # A Gaussian is visible if its z is within (1+depth_tol) of the front-
        # most z at its pixel. Scaled tolerance handles distant geometry.
        front_z = depth_proxy[vi, ui]
        visible = z <= front_z * (1.0 + depth_tol) + 1e-3
        vis_idx = idx[visible]
        vis_ui = ui[visible]
        vis_vi = vi[visible]
        if vis_idx.size == 0:
            continue

        for mi, seg in enumerate(segs):
            hit = seg[vis_vi, vis_ui]
            if hit.sum() < 20:           # skip masks with too few voters
                continue
            voter_idx = vis_idx[hit].astype(np.int32)
            keys.append((int(kf_idx), int(mi)))
            voters.append(voter_idx)
            mask_areas.append(int(seg.sum()))

    log(f"[3/7 voters] {len(keys)} masks with voters across {len(kfs)} keyframes; "
        f"mean voters/mask = {np.mean([len(v) for v in voters]):.0f}")

    np.savez_compressed(
        out_path,
        keys=np.array(keys, dtype=np.int32),
        voters=np.array(voters, dtype=object),
        mask_areas=np.array(mask_areas, dtype=np.int32),
    )
    return {"keys": keys, "voters": voters, "mask_areas": mask_areas}


# ---------------------------------------------------------------------------
# Stage 4 — associate masks across views (union-find on voter Jaccard)
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self, n: int):
        self.p = list(range(n))
        self.r = [0] * n
    def find(self, a: int) -> int:
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a
    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return
        if self.r[ra] < self.r[rb]: ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]: self.r[ra] += 1


def stage_associate(scene: str, voter_data: dict, jaccard_thresh: float,
                    force: bool) -> dict:
    """Link masks whose voter sets overlap above jaccard_thresh. Connected
    components -> instance ids. Cached as instances_pre.json."""
    out_path = REPO / "semantics" / scene / "instances_pre.json"
    keys = voter_data["keys"]
    voters = voter_data["voters"]
    if out_path.exists() and not force:
        log(f"[4/7 associate] cached")
        return json.loads(out_path.read_text())

    # Build inverted index: Gaussian -> list of mask indices it voted in.
    inv: dict[int, list[int]] = defaultdict(list)
    for mi, vs in enumerate(voters):
        for g in vs.tolist():
            inv[g].append(mi)

    # Candidate mask pairs share at least one voter.
    pair_share: Counter = Counter()
    for ms in inv.values():
        if len(ms) < 2: continue
        # only emit each unordered pair once
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                a, b = ms[i], ms[j]
                if a > b: a, b = b, a
                pair_share[(a, b)] += 1

    n = len(keys)
    sizes = np.array([len(v) for v in voters], dtype=np.int64)
    uf = UnionFind(n)
    n_edges = 0
    for (a, b), shared in pair_share.items():
        jac = shared / (sizes[a] + sizes[b] - shared)
        if jac >= jaccard_thresh:
            uf.union(a, b)
            n_edges += 1

    # Collapse to instance ids
    roots = [uf.find(i) for i in range(n)]
    root_to_inst: dict[int, int] = {}
    inst_of_mask: list[int] = []
    for r in roots:
        if r not in root_to_inst:
            root_to_inst[r] = len(root_to_inst)
        inst_of_mask.append(root_to_inst[r])

    inst_to_masks: dict[int, list[int]] = defaultdict(list)
    for mi, ii in enumerate(inst_of_mask):
        inst_to_masks[ii].append(mi)

    log(f"[4/7 associate] {n} masks -> {len(inst_to_masks)} instances "
        f"({n_edges} links above jaccard={jaccard_thresh})")

    result = {
        "keys": [[int(a), int(b)] for a, b in keys],
        "inst_of_mask": [int(x) for x in inst_of_mask],
        "inst_to_masks": {str(int(k)): [int(m) for m in v]
                          for k, v in inst_to_masks.items()},
    }
    out_path.write_text(json.dumps(result))
    return result


# ---------------------------------------------------------------------------
# Stage 5 — label each instance once (best view), via Qwen2.5-VL
# ---------------------------------------------------------------------------

_LABEL_RE = re.compile(r"[a-zA-Z][a-zA-Z\- ]{0,20}")

def _clean_label(raw: str) -> str:
    raw = raw.strip().lower()
    m = _LABEL_RE.search(raw)
    if not m: return "unknown"
    label = m.group(0).strip()
    for pre in ("a ", "an ", "the "):
        if label.startswith(pre):
            label = label[len(pre):]
    return label.split()[0] if label else "unknown"


def _qwen_label_crop(model, proc, pil: "Image.Image", prompt: str) -> str:
    import torch
    messages = [{"role": "user", "content": [
        {"type": "image", "image": pil},
        {"type": "text", "text": prompt},
    ]}]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[pil], padding=True, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        gen_ids = model.generate(**inputs, max_new_tokens=8, do_sample=False)
    out_ids = gen_ids[:, inputs.input_ids.shape[1]:]
    return proc.batch_decode(out_ids, skip_special_tokens=True)[0]


def _build_label_crop(scene: str, kf: dict, seg: np.ndarray, bbox: np.ndarray,
                      pad_frac: float = 0.15, dim_factor: float = 0.5) -> "Image.Image":
    """Bigger context crop + softer dim so the VLM sees the object in situ."""
    img = np.array(Image.open(REPO / "data" / "scenes" / scene / kf["file_path"]).convert("RGB"))
    H, W = img.shape[:2]
    x, y, bw, bh = bbox
    pad = max(16, int(pad_frac * max(bw, bh)))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(W, x + bw + pad), min(H, y + bh + pad)
    crop = img[y0:y1, x0:x1].copy()
    seg_c = seg[y0:y1, x0:x1]
    crop[~seg_c] = (crop[~seg_c] * dim_factor).astype(np.uint8)
    return Image.fromarray(crop)


def stage_label(scene: str, kfs: list[dict], voter_data: dict, assoc: dict,
                inst_arr: np.ndarray, min_gaussians: int, label_views: int,
                force: bool) -> dict:
    """For each instance with enough lifted Gaussians, run Qwen on the top-K
    voter-count views and keep the majority label. Instances without a
    majority get label="unknown" and are filtered out by stage_export. Each
    labelled crop is dumped to semantics/<scene>/audit/ for spot-checks."""
    out_path = REPO / "semantics" / scene / "instance_labels.json"
    audit_dir = ensure_dir(REPO / "semantics" / scene / "audit")
    if out_path.exists() and not force:
        log(f"[6/7 label] cached")
        return json.loads(out_path.read_text())

    keys = voter_data["keys"]
    voters = voter_data["voters"]
    sizes = np.array([len(v) for v in voters], dtype=np.int64)
    inst_to_masks: dict[int, list[int]] = {
        int(k): v for k, v in assoc["inst_to_masks"].items()
    }

    # Pre-compute Gaussian count per instance from the lifted array.
    unique, counts = np.unique(inst_arr[inst_arr >= 0], return_counts=True)
    n_gauss_by_inst = dict(zip(unique.tolist(), counts.tolist()))

    keep: dict[int, dict] = {}
    skipped_small = 0
    for ii, mlist in inst_to_masks.items():
        n_g = int(n_gauss_by_inst.get(ii, 0))
        if n_g < min_gaussians:
            skipped_small += 1
            continue
        union = set()
        for mi in mlist:
            union.update(voters[mi].tolist())
        top_k = sorted(mlist, key=lambda mi: -sizes[mi])[:label_views]
        keep[ii] = {
            "top_k_mi": [int(mi) for mi in top_k],
            "n_views": len(mlist),
            "n_voters": len(union),
            "n_gaussians": n_g,
        }

    log(f"[6/7 label] {len(keep)} instances pass min-gaussians={min_gaussians} "
        f"(skipped {skipped_small} below threshold); K={label_views} views per instance")

    log(f"[6/7 label] loading Qwen2.5-VL-3B ...")
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    proc = AutoProcessor.from_pretrained(QWEN_ID)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_ID, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    prompt = (
        "This is a crop of an indoor scene with one object highlighted "
        "(surroundings darkened). What is the highlighted object? Reply with "
        "ONE common English noun in lowercase (e.g. chair, table, lamp). "
        "If you are unsure, reply exactly: unknown"
    )

    kfs_by_idx = {k["idx_in_transforms"]: k for k in kfs}
    masks_dir = REPO / "semantics" / scene / "masks"

    n_majority = 0
    n_unknown = 0
    for ii, info in tqdm(list(keep.items()), desc="Qwen", file=sys.stderr):
        votes: Counter = Counter()
        per_view: list[dict] = []
        best_voters = -1
        best_kf = best_mask = None
        for mi in info["top_k_mi"]:
            kf_idx, mask_idx = keys[mi]
            kf_idx = int(kf_idx); mask_idx = int(mask_idx)
            k = kfs_by_idx[kf_idx]
            d = np.load(masks_dir / f"kf{kf_idx:04d}.npz")
            seg = d["segs"][mask_idx]
            bbox = d["bboxes"][mask_idx]
            pil = _build_label_crop(scene, k, seg, bbox)
            raw = _qwen_label_crop(model, proc, pil, prompt)
            lab = _clean_label(raw)
            votes[lab] += 1
            per_view.append({"kf": kf_idx, "mask": mask_idx, "label": lab, "raw": raw.strip()})
            n_vot = int(sizes[mi])
            if n_vot > best_voters:
                best_voters, best_kf, best_mask = n_vot, kf_idx, mask_idx
                # Save the best-view crop for audit (one image per instance).
                pil.save(audit_dir / f"{ii:04d}_pending.jpg", quality=85)

        # Mode + majority test: require strictly more than half the views,
        # excluding "unknown" from the running.
        non_unk = Counter({k: v for k, v in votes.items() if k != "unknown"})
        if non_unk:
            top_label, top_count = non_unk.most_common(1)[0]
            majority = top_count > len(info["top_k_mi"]) / 2
        else:
            top_label, top_count, majority = "unknown", votes.get("unknown", 0), False

        info["label"] = top_label if majority else "unknown"
        info["confidence"] = top_count / max(1, len(info["top_k_mi"]))
        info["votes"] = dict(votes)
        info["per_view"] = per_view
        info["best_kf"] = best_kf
        info["best_mask"] = best_mask

        # Rename audit crop to include final label.
        pending = audit_dir / f"{ii:04d}_pending.jpg"
        if pending.exists():
            safe = re.sub(r"[^a-zA-Z0-9_-]", "_", info["label"])[:24]
            pending.rename(audit_dir / f"{ii:04d}_{safe}_c{info['confidence']:.2f}.jpg")
        if majority: n_majority += 1
        else:        n_unknown  += 1

    log(f"[6/7 label] majority: {n_majority}, unknown: {n_unknown}")
    summary = Counter(v["label"] for v in keep.values() if v["label"] != "unknown")
    log(f"[6/7 label] top labels: {summary.most_common(10)}")
    log(f"[6/7 label] audit crops -> {audit_dir}")

    out_path.write_text(json.dumps({str(k): v for k, v in keep.items()}, indent=2))
    del model, proc
    torch.cuda.empty_cache()
    return {str(k): v for k, v in keep.items()}


# ---------------------------------------------------------------------------
# Stage 5 — lift instance IDs to Gaussians.
#
# Each (mask, gaussian) pair contributes one vote to that mask's instance.
# This lets an instance that's seen in many keyframes outvote a single huge
# wall/floor mask, which was the failure mode of mask-area weighting (chairs
# kept losing their Gaussians to background masks).
# ---------------------------------------------------------------------------

def stage_lift(scene: str, voter_data: dict, assoc: dict) -> np.ndarray:
    cache_path = REPO / "semantics" / scene / "gaussian_instances_pre.npy"
    if cache_path.exists():
        log(f"[5/7 lift] cached")
        return np.load(cache_path)

    voters = voter_data["voters"]
    inst_of_mask: list[int] = assoc["inst_of_mask"]

    xyz = load_splat_xyz(scene)
    n_gauss = xyz.shape[0]
    log(f"[5/7 lift] voting over {n_gauss:,} Gaussians (vote-count weighting) ...")

    # gaussian_idx -> {instance_id: vote count across all masks of that instance}
    votes: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    for mi, vs in enumerate(voters):
        inst = inst_of_mask[mi]
        for g in vs.tolist():
            votes[g][inst] += 1.0

    inst_arr = np.full(n_gauss, -1, dtype=np.int32)
    for gi, inst_w in votes.items():
        best_inst, _ = max(inst_w.items(), key=lambda kv: kv[1])
        inst_arr[gi] = best_inst

    n_assigned = int((inst_arr >= 0).sum())
    log(f"[5/7 lift] {n_assigned:,} / {n_gauss:,} Gaussians assigned "
        f"({100*n_assigned/n_gauss:.1f}%)")
    np.save(cache_path, inst_arr)
    return inst_arr


# ---------------------------------------------------------------------------
# Stage 7 — write sidecars (instance ids + anchors + palette); never splat.ply
# ---------------------------------------------------------------------------

def palette_for(label: str, inst_id: int) -> tuple[int, int, int]:
    if not label or label == "unknown":
        return (128, 128, 128)
    h = hashlib.md5(f"{label}:{inst_id}".encode()).digest()
    return (int(h[0]), int(h[1]), int(h[2]))


def stage_export(scene: str, inst_arr: np.ndarray, labels: dict,
                 scatter_ratio: float, max_size_frac: float,
                 dedup_factor: float, up_axis: str) -> None:
    out_dir = ensure_dir(REPO / "semantics" / scene)
    xyz = load_splat_xyz(scene)

    np.save(out_dir / "gaussian_instances.npy", inst_arr)
    log(f"[7/7 export] wrote gaussian_instances.npy ({inst_arr.size} ints)")

    # Scene-relative size cap: drop instances whose inlier diag exceeds a
    # fraction of the scene's own diagonal. Catches floater clusters that span
    # the whole scene.
    scene_lo = np.percentile(xyz, 1, axis=0)
    scene_hi = np.percentile(xyz, 99, axis=0)
    scene_diag = float(np.linalg.norm(scene_hi - scene_lo))
    max_diag = max_size_frac * scene_diag
    log(f"[7/7 export] scene diag={scene_diag:.2f}, max instance diag={max_diag:.2f}")

    # Up-axis vector in splat coordinates. For nerfstudio splatfacto with
    # orientation_method=up the world gravity is aligned so +y is up; allow
    # an override flag for scenes oriented differently.
    up_vec = {"+y": np.array([0, 1, 0]), "-y": np.array([0, -1, 0]),
              "+z": np.array([0, 0, 1]), "-z": np.array([0, 0, -1])}[up_axis]
    up_idx = int(np.argmax(np.abs(up_vec)))
    up_sign = float(np.sign(up_vec[up_idx]))

    dropped_scatter = dropped_unknown = dropped_huge = 0
    raw_anchors: list[dict] = []
    for ii_str, info in labels.items():
        ii = int(ii_str)
        sel = (inst_arr == ii)
        n = int(sel.sum())
        if info.get("label", "unknown") in ("", "unknown"):
            dropped_unknown += 1
            continue
        if n == 0:
            continue
        pts = xyz[sel]
        # Robust centroid + spread; reject if very scattered (= merged with
        # background / split across the scene).
        centroid = np.median(pts, axis=0)
        d2c = np.linalg.norm(pts - centroid, axis=1)
        med = float(np.median(d2c)) + 1e-6
        p95 = float(np.percentile(d2c, 95))
        if p95 > scatter_ratio * med:
            dropped_scatter += 1
            continue
        # Diagonal of the inlier (5–95 pct) AABB — robust to a few outlier
        # Gaussians that snuck in.
        lo = np.percentile(pts, 5, axis=0)
        hi = np.percentile(pts, 95, axis=0)
        diag = float(np.linalg.norm(hi - lo))
        if diag > max_diag:
            dropped_huge += 1
            continue
        # Pick the "top" of the object along the up-axis (95th percentile if
        # up_sign>0, else 5th) and float the label just above it.
        if up_sign > 0:
            top_u = float(np.percentile(pts[:, up_idx], 95))
        else:
            top_u = float(np.percentile(pts[:, up_idx], 5))
        top_pos = centroid.copy()
        top_pos[up_idx] = top_u + up_sign * 0.10 * diag

        raw_anchors.append({
            "id": ii,
            "label": info["label"],
            "confidence": float(info.get("confidence", 1.0)),
            "n_gaussians": n,
            "n_views": int(info["n_views"]),
            "n_voters": int(info["n_voters"]),
            "anchor": [float(centroid[0]), float(centroid[1]), float(centroid[2])],
            "top": [float(top_pos[0]), float(top_pos[1]), float(top_pos[2])],
            "diag": diag,
            "palette_rgb": list(palette_for(info["label"], ii)),
        })

    log(f"[7/7 export] after filters: {len(raw_anchors)} instances "
        f"(dropped: unknown={dropped_unknown}, scattered={dropped_scatter}, "
        f"oversized={dropped_huge})")

    # De-dup: merge same-label instances whose anchors are close compared to
    # their own diagonals. Keep the entry with the most Gaussians per group.
    raw_anchors.sort(key=lambda a: -a["n_gaussians"])
    kept: list[dict] = []
    merged_into: dict[int, int] = {}
    for a in raw_anchors:
        absorbed = False
        for k in kept:
            if k["label"] != a["label"]:
                continue
            dist = float(np.linalg.norm(np.array(a["anchor"]) - np.array(k["anchor"])))
            # Use max(diag) so a big real instance absorbs nearby tiny splits.
            threshold = dedup_factor * max(a["diag"], k["diag"])
            if dist < threshold:
                merged_into[a["id"]] = k["id"]
                k["n_gaussians"] += a["n_gaussians"]
                absorbed = True
                break
        if not absorbed:
            kept.append(a)

    log(f"[7/7 export] after dedup: {len(kept)} instances "
        f"(merged {len(merged_into)} duplicates)")

    # Apply the merges to the per-Gaussian instance array so the viewer's
    # "tint by instance" is consistent with the kept anchors.
    if merged_into:
        for old, new in merged_into.items():
            inst_arr[inst_arr == old] = new
        np.save(out_dir / "gaussian_instances.npy", inst_arr)

    anchors_out: dict[str, dict] = {str(a["id"]): a for a in kept}
    (out_dir / "instance_anchors.json").write_text(json.dumps(anchors_out, indent=2))
    log(f"[7/7 export] wrote instance_anchors.json ({len(anchors_out)} instances)")

    summary = Counter(a["label"] for a in kept)
    log(f"[7/7 export] final labels: {summary.most_common(15)}")
    log(f"[7/7 export] DONE. splat.ply was NOT modified.")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

STAGES = ("keyframes", "masks", "voters", "associate", "lift", "label", "export")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--n-keyframes", type=int, default=30)
    ap.add_argument("--max-masks-per-frame", type=int, default=25)
    ap.add_argument("--min-mask-area-frac", type=float, default=0.005)
    ap.add_argument("--depth-tol", type=float, default=0.10,
                    help="A Gaussian is visible if z <= front_z * (1+tol).")
    ap.add_argument("--assoc-jaccard", type=float, default=0.20,
                    help="Min voter-set Jaccard to link two masks across views.")
    ap.add_argument("--label-views", type=int, default=3,
                    help="Run Qwen on the top-K voter-count views per instance and take the majority label.")
    ap.add_argument("--min-gaussians", type=int, default=20,
                    help="After lift, skip instances with fewer Gaussians than this (no label, no anchor).")
    ap.add_argument("--scatter-ratio", type=float, default=10.0,
                    help="Drop instances whose p95 distance-to-centroid > ratio * median (= scattered/merged).")
    ap.add_argument("--max-size-frac", type=float, default=0.25,
                    help="Drop instances whose inlier diag exceeds this fraction of the scene diag.")
    ap.add_argument("--dedup-factor", type=float, default=0.7,
                    help="Merge same-label instances whose anchors are within factor * max(diag).")
    ap.add_argument("--up-axis", choices=["+y", "-y", "+z", "-z"], default="+y",
                    help="Scene up direction in splat coordinates.")
    ap.add_argument("--only", choices=STAGES)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    log(f"scene={args.scene}")

    kfs = stage_keyframes(args.scene, args.n_keyframes, args.force)
    if args.only == "keyframes": return 0

    stage_masks(args.scene, kfs, args.max_masks_per_frame,
                args.min_mask_area_frac, args.force)
    if args.only == "masks": return 0

    voter_data = stage_voters(args.scene, kfs, args.depth_tol, args.force)
    if args.only == "voters": return 0

    assoc = stage_associate(args.scene, voter_data, args.assoc_jaccard, args.force)
    if args.only == "associate": return 0

    inst_arr = stage_lift(args.scene, voter_data, assoc)
    if args.only == "lift": return 0

    labels = stage_label(args.scene, kfs, voter_data, assoc, inst_arr,
                         args.min_gaussians, args.label_views, args.force)
    if args.only == "label": return 0

    stage_export(args.scene, inst_arr, labels,
                 scatter_ratio=args.scatter_ratio,
                 max_size_frac=args.max_size_frac,
                 dedup_factor=args.dedup_factor,
                 up_axis=args.up_axis)
    return 0


if __name__ == "__main__":
    sys.exit(main())
