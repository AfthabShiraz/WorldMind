#!/usr/bin/env python3
"""
lift_semantics.py — attach object labels to Gaussians in a trained splat.

Pipeline (each stage caches its output under semantics/<scene>/ and is
skipped if its outputs exist; use --force to redo):

  1. keyframes:    pick N evenly-spaced frames from transforms.json
  2. masks:        SAM auto-mask each keyframe -> masks.npz per frame
  3. labels:       Qwen2.5-VL-3B names each mask -> labels.json
  4. lift:         project Gaussian centers into each keyframe, accumulate
                   votes from masks they fall inside, write splat_semantic.ply
                   (colors replaced by class palette) + gaussian_labels.json

Usage:
  scripts/lift_semantics.py --scene <id>
        [--n-keyframes 30]
        [--max-masks-per-frame 25]
        [--min-mask-area-frac 0.005]
        [--force]
        [--only stage]            # one of: keyframes,masks,labels,lift
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement
from tqdm import tqdm


REPO = Path(__file__).resolve().parent.parent
SAM_CKPT = REPO / "data" / "checkpoints" / "sam_vit_h_4b8939.pth"
QWEN_ID = "Qwen/Qwen2.5-VL-3B-Instruct"


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def load_transforms(scene: str) -> dict:
    p = REPO / "data" / "scenes" / scene / "transforms.json"
    return json.loads(p.read_text())


def load_dataparser_transform(scene: str) -> tuple[np.ndarray, float]:
    """Return (T_4x4, scale) for the splatfacto run.

    The .ply lives in dataparser-transformed coordinates; cameras in
    transforms.json are pre-dataparser. We apply the same transform to the
    cameras so they line up with the splat.
    """
    runs = sorted((REPO / "outputs" / scene / "splatfacto_full" / "splatfacto").glob("*/dataparser_transforms.json"))
    if not runs:
        raise FileNotFoundError(f"no dataparser_transforms.json for scene {scene}")
    d = json.loads(runs[-1].read_text())
    T = np.eye(4, dtype=np.float64)
    T[:3, :4] = np.asarray(d["transform"], dtype=np.float64)
    return T, float(d["scale"])


def load_splat(scene: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (positions Nx3 float32, the raw structured array)."""
    p = REPO / "outputs" / scene / "ply_full" / "splat.ply"
    ply = PlyData.read(str(p))
    el = ply["vertex"]
    xyz = np.stack([el["x"], el["y"], el["z"]], axis=1).astype(np.float32)
    return xyz, el.data


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Stage 1 — keyframe selection
# ---------------------------------------------------------------------------

def stage_keyframes(scene: str, n_keyframes: int, force: bool) -> list[dict]:
    out_dir = ensure_dir(REPO / "semantics" / scene)
    kf_path = out_dir / "keyframes.json"
    if kf_path.exists() and not force:
        kfs = json.loads(kf_path.read_text())
        log(f"[1/4 keyframes] cached: {len(kfs)} keyframes")
        return kfs

    t = load_transforms(scene)
    frames = t["frames"]
    n = min(n_keyframes, len(frames))
    idxs = np.linspace(0, len(frames) - 1, n, dtype=int).tolist()
    kfs = [{"idx_in_transforms": int(i), **frames[i]} for i in idxs]
    kf_path.write_text(json.dumps(kfs, indent=2))
    log(f"[1/4 keyframes] picked {len(kfs)} / {len(frames)}, saved {kf_path}")
    return kfs


# ---------------------------------------------------------------------------
# Stage 2 — SAM auto-mask
# ---------------------------------------------------------------------------

def stage_masks(scene: str, kfs: list[dict], max_masks: int,
                min_area_frac: float, force: bool) -> None:
    out_dir = ensure_dir(REPO / "semantics" / scene / "masks")
    todo = [k for k in kfs if force or not (out_dir / f"kf{k['idx_in_transforms']:04d}.npz").exists()]
    if not todo:
        log(f"[2/4 masks] all {len(kfs)} cached")
        return

    log(f"[2/4 masks] loading SAM ViT-H ({SAM_CKPT.stat().st_size / 1e9:.1f} GB) ...")
    import torch
    import torchvision
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    # torchvision 0.24 ships NMS CUDA without sm_120 support on Blackwell GB10
    # and lacks a PTX fallback, so the call errors with cudaErrorNoKernelImage.
    # Route NMS through CPU — it's fast enough on the mask counts SAM produces.
    _native_nms = torch.ops.torchvision.nms
    def _cpu_nms(boxes, scores, iou_threshold):
        return _native_nms(boxes.cpu(), scores.cpu(), iou_threshold).to(boxes.device)
    torchvision.ops.boxes.nms = lambda b, s, iou: _cpu_nms(b, s, iou)
    torchvision.ops.nms = torchvision.ops.boxes.nms

    sam = sam_model_registry["vit_h"](checkpoint=str(SAM_CKPT))
    sam.to("cuda").eval()
    gen = SamAutomaticMaskGenerator(
        sam,
        points_per_side=16,                # fewer prompts -> fewer, bigger masks
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=0,
        min_mask_region_area=2000,
    )
    log(f"[2/4 masks] generating for {len(todo)} keyframes")

    for k in tqdm(todo, desc="SAM", file=sys.stderr):
        img_path = REPO / "data" / "scenes" / scene / k["file_path"]
        img = np.array(Image.open(img_path).convert("RGB"))
        H, W = img.shape[:2]
        min_area = min_area_frac * H * W
        masks = gen.generate(img)
        masks = [m for m in masks if m["area"] >= min_area]
        masks.sort(key=lambda m: -m["area"])
        masks = masks[:max_masks]

        if not masks:
            np.savez_compressed(out_dir / f"kf{k['idx_in_transforms']:04d}.npz",
                                segs=np.zeros((0, H, W), dtype=bool),
                                bboxes=np.zeros((0, 4), dtype=np.int32),
                                areas=np.zeros((0,), dtype=np.int32))
            continue
        segs = np.stack([m["segmentation"] for m in masks])
        bboxes = np.array([m["bbox"] for m in masks], dtype=np.int32)  # XYWH
        areas = np.array([m["area"] for m in masks], dtype=np.int32)
        np.savez_compressed(out_dir / f"kf{k['idx_in_transforms']:04d}.npz",
                            segs=segs, bboxes=bboxes, areas=areas)
    del sam, gen
    torch.cuda.empty_cache()
    log(f"[2/4 masks] done")


# ---------------------------------------------------------------------------
# Stage 3 — Qwen VLM labels
# ---------------------------------------------------------------------------

_LABEL_RE = re.compile(r"[a-zA-Z][a-zA-Z\- ]{0,20}")


def _clean_label(raw: str) -> str:
    raw = raw.strip().lower()
    # Take first plausible word-ish span; strip articles
    m = _LABEL_RE.search(raw)
    if not m:
        return "unknown"
    label = m.group(0).strip()
    for prefix in ("a ", "an ", "the "):
        if label.startswith(prefix):
            label = label[len(prefix):]
    return label.split()[0] if label else "unknown"


def stage_labels(scene: str, kfs: list[dict], force: bool) -> dict:
    out_path = REPO / "semantics" / scene / "labels.json"
    if out_path.exists() and not force:
        log(f"[3/4 labels] cached: {out_path}")
        return json.loads(out_path.read_text())

    log(f"[3/4 labels] loading Qwen2.5-VL-3B ...")
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    proc = AutoProcessor.from_pretrained(QWEN_ID)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_ID, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    prompt = ("What single object is shown in this image? "
              "Reply with ONE noun, lowercase, no article, no description.")
    masks_dir = REPO / "semantics" / scene / "masks"

    labels: dict[str, dict] = {}  # kf_idx -> {mask_idx: label}
    n_masks_total = 0
    for k in tqdm(kfs, desc="Qwen", file=sys.stderr):
        kf_idx = k["idx_in_transforms"]
        npz_path = masks_dir / f"kf{kf_idx:04d}.npz"
        if not npz_path.exists():
            continue
        data = np.load(npz_path)
        segs, bboxes = data["segs"], data["bboxes"]
        img_path = REPO / "data" / "scenes" / scene / k["file_path"]
        img = np.array(Image.open(img_path).convert("RGB"))
        H, W = img.shape[:2]

        kf_labels: dict[str, str] = {}
        for mi, (seg, bbox) in enumerate(zip(segs, bboxes)):
            x, y, bw, bh = bbox
            pad = max(8, int(0.05 * max(bw, bh)))
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(W, x + bw + pad), min(H, y + bh + pad)
            crop = img[y0:y1, x0:x1].copy()
            seg_crop = seg[y0:y1, x0:x1]
            # Darken outside the mask so the VLM focuses on the segment
            crop[~seg_crop] = (crop[~seg_crop] * 0.25).astype(np.uint8)
            pil = Image.fromarray(crop)

            messages = [{"role": "user", "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": prompt},
            ]}]
            text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = proc(text=[text], images=[pil], padding=True, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                gen_ids = model.generate(**inputs, max_new_tokens=8, do_sample=False)
            out_ids = gen_ids[:, inputs.input_ids.shape[1]:]
            raw = proc.batch_decode(out_ids, skip_special_tokens=True)[0]
            kf_labels[str(mi)] = _clean_label(raw)
            n_masks_total += 1
        labels[str(kf_idx)] = kf_labels

    out_path.write_text(json.dumps(labels, indent=2))
    summary = Counter()
    for kfl in labels.values():
        summary.update(kfl.values())
    log(f"[3/4 labels] labeled {n_masks_total} masks across {len(labels)} keyframes")
    log(f"[3/4 labels] top: {summary.most_common(10)}")
    del model, proc
    torch.cuda.empty_cache()
    return labels


# ---------------------------------------------------------------------------
# Stage 4 — Lift to Gaussians
# ---------------------------------------------------------------------------

def palette_for(label: str) -> tuple[int, int, int]:
    """Deterministic colour for each unique label string."""
    if label in ("unknown", ""):
        return (128, 128, 128)
    h = hashlib.md5(label.encode()).digest()
    return (h[0], h[1], h[2])


def stage_lift(scene: str, kfs: list[dict], labels: dict) -> None:
    out_dir = ensure_dir(REPO / "semantics" / scene)

    # ----- Load splat & cameras ---------------------------------------------
    xyz, vertex_data = load_splat(scene)
    n_gauss = xyz.shape[0]
    log(f"[4/4 lift] loaded {n_gauss:,} Gaussians from splat.ply")

    t = load_transforms(scene)
    fx, fy = float(t["fl_x"]), float(t["fl_y"])
    cx, cy = float(t["cx"]), float(t["cy"])
    W, H = int(t["w"]), int(t["h"])

    T_dp, s_dp = load_dataparser_transform(scene)
    log(f"[4/4 lift] dataparser scale={s_dp:.4f}")

    # ----- Voting -----------------------------------------------------------
    # gaussian_idx -> Counter({label: weight})
    votes: dict[int, Counter] = defaultdict(Counter)

    xyz_h = np.hstack([xyz, np.ones((n_gauss, 1), dtype=np.float32)])  # (N, 4)
    masks_dir = REPO / "semantics" / scene / "masks"

    for k in tqdm(kfs, desc="lift", file=sys.stderr):
        kf_idx = k["idx_in_transforms"]
        kfl = labels.get(str(kf_idx))
        if not kfl:
            continue
        npz_path = masks_dir / f"kf{kf_idx:04d}.npz"
        if not npz_path.exists():
            continue
        segs = np.load(npz_path)["segs"]
        if segs.shape[0] == 0:
            continue

        # Camera-to-world in nerfstudio frame, then apply the dataparser
        # transform so the camera lives in the splat's frame.
        c2w_ns = np.array(k["transform_matrix"], dtype=np.float64)
        c2w = T_dp @ c2w_ns
        c2w[:3, 3] *= s_dp
        w2c = np.linalg.inv(c2w)

        # Project Gaussians: w2c is in nerfstudio (OpenGL) convention.
        p_cam = (w2c @ xyz_h.T).T  # (N, 4)
        # Nerfstudio -> OpenCV: flip y, z so +Z is forward.
        Xc = p_cam[:, 0]
        Yc = -p_cam[:, 1]
        Zc = -p_cam[:, 2]
        in_front = Zc > 1e-3
        u = (fx * Xc / np.where(in_front, Zc, 1.0)) + cx
        v = (fy * Yc / np.where(in_front, Zc, 1.0)) + cy
        in_image = in_front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        valid = np.where(in_image)[0]
        if valid.size == 0:
            continue
        ui = u[valid].astype(np.int32)
        vi = v[valid].astype(np.int32)

        # For each mask in this keyframe, find which projected Gaussians
        # fall inside, vote with the mask area (bigger mask = more weight).
        for mi, seg in enumerate(segs):
            label = kfl.get(str(mi))
            if not label or label == "unknown":
                continue
            hit = seg[vi, ui]
            if not hit.any():
                continue
            area_w = float(seg.sum())  # mask area in pixels, as weight
            hit_idx = valid[hit]
            for gi in hit_idx:
                votes[int(gi)][label] += area_w

    log(f"[4/4 lift] {len(votes):,} / {n_gauss:,} Gaussians received votes "
        f"({100*len(votes)/n_gauss:.1f}%)")

    # ----- Resolve labels & write outputs -----------------------------------
    gauss_labels: list[str] = ["unknown"] * n_gauss
    for gi, ctr in votes.items():
        gauss_labels[gi] = ctr.most_common(1)[0][0]
    summary = Counter(gauss_labels)
    log(f"[4/4 lift] label distribution: {summary.most_common(15)}")

    (out_dir / "gaussian_labels.json").write_text(json.dumps({
        "n_gauss": int(n_gauss),
        "summary": summary.most_common(),
        "labels": gauss_labels,
    }))

    # Tinted PLY — replace the SH DC channels with a per-label palette colour.
    # splatfacto stores DC color as f_dc_0/1/2 (RGB at SH order 0). We override
    # those so SuperSplat renders the segmentation directly.
    palette = {lab: palette_for(lab) for lab in set(gauss_labels)}
    arr = vertex_data.copy()
    # Map SH DC to roughly [-1.7, 1.7] sigmoid-equivalent — splatfacto uses
    # 0.5 + SH * 0.28209479177387814 in linear, so to get color c in [0, 1]
    # we set f_dc = (c - 0.5) / 0.28209479177387814. Just clamp safely.
    SH_C0 = 0.28209479177387814
    for i in range(n_gauss):
        r, g, b = palette[gauss_labels[i]]
        for j, ch in enumerate((r, g, b)):
            arr[f"f_dc_{j}"][i] = (ch / 255.0 - 0.5) / SH_C0
    # Zero higher-order SH so view-dependent shading doesn't repaint.
    for field in arr.dtype.names:
        if field.startswith("f_rest_"):
            arr[field][:] = 0.0

    out_ply = REPO / "semantics" / scene / "splat_semantic.ply"
    PlyData([PlyElement.describe(arr, "vertex")], text=False).write(str(out_ply))
    log(f"[4/4 lift] wrote {out_ply}  ({out_ply.stat().st_size / 1e6:.1f} MB)")
    (out_dir / "palette.json").write_text(json.dumps(palette, indent=2))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--n-keyframes", type=int, default=30)
    ap.add_argument("--max-masks-per-frame", type=int, default=25)
    ap.add_argument("--min-mask-area-frac", type=float, default=0.005)
    ap.add_argument("--only", choices=["keyframes", "masks", "labels", "lift"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    log(f"scene={args.scene}  n_keyframes={args.n_keyframes}")
    kfs = stage_keyframes(args.scene, args.n_keyframes, args.force)
    if args.only == "keyframes":
        return 0
    stage_masks(args.scene, kfs, args.max_masks_per_frame,
                args.min_mask_area_frac, args.force)
    if args.only == "masks":
        return 0
    labels = stage_labels(args.scene, kfs, args.force)
    if args.only == "labels":
        return 0
    stage_lift(args.scene, kfs, labels)
    return 0


if __name__ == "__main__":
    sys.exit(main())
