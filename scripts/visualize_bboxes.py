#!/usr/bin/env python3
"""
visualize_bboxes.py — overlay Qwen's bounding boxes on the keyframes that
place_object_labels.py used, colour-coded by whether they were accepted
or which safety filter rejected them.

Reads:  semantics/<scene>/object_anchors.json  (has detections per object)
Writes: semantics/<scene>/bbox_audit/kfXXXX.jpg  (one per keyframe used)

Usage:
  scripts/visualize_bboxes.py --scene <id>
        [--accepted-only]            # hide filter-rejected boxes
        [--include-no-box]           # also note frames where Qwen said 'none'
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


REPO = Path(__file__).resolve().parent.parent


def load_dataparser_transform(scene: str) -> tuple[np.ndarray, float]:
    runs = sorted((REPO / "outputs" / scene / "splatfacto_full" / "splatfacto").glob(
        "*/dataparser_transforms.json"))
    if not runs:
        return np.eye(4, dtype=np.float64), 1.0
    d = json.loads(runs[-1].read_text())
    T = np.eye(4, dtype=np.float64)
    T[:3, :4] = np.asarray(d["transform"], dtype=np.float64)
    return T, float(d["scale"])


def project_world_to_pixel(xyz_world: np.ndarray, w2c: np.ndarray,
                           fx: float, fy: float, cx: float, cy: float,
                           W: int, H: int) -> tuple[float, float] | None:
    """Mirror of place_object_labels.project_gaussians for a single world
    point. Returns (u, v) in pixel coords or None if behind or off-frame."""
    p_h = np.array([xyz_world[0], xyz_world[1], xyz_world[2], 1.0], dtype=np.float64)
    p_cam = w2c @ p_h
    Xc, Yc, Zc = p_cam[0], -p_cam[1], -p_cam[2]
    if Zc <= 1e-3:
        return None
    u = fx * Xc / Zc + cx
    v = fy * Yc / Zc + cy
    if not (0 <= u < W and 0 <= v < H):
        return None
    return (float(u), float(v))


# Reject reason -> RGB. None = accepted -> green.
COLOR = {
    None:             (60, 220, 60),    # green: accepted, made it to anchor
    "sparse_patch":   (255, 200, 0),    # yellow: bbox over a void
    "too_far":        (255, 110, 0),    # orange: depth > scene cap
    "outside_aabb":   (200, 60, 200),   # purple: back-projection outside scene
    "no_finite_depth":(220, 60, 60),    # red: no depth at all
    "no_box":         (140, 140, 140),  # grey: Qwen said 'none' (no bbox to draw)
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def get_font(size: int) -> ImageFont.FreeTypeFont:
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
        "/Windows/Fonts/arial.ttf",
    ):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            pass
    return ImageFont.load_default()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--accepted-only", action="store_true",
                    help="Hide bboxes that were rejected by safety filters.")
    args = ap.parse_args()

    p = REPO / "semantics" / args.scene / "object_anchors.json"
    if not p.exists():
        log(f"ERROR: no {p} — run `make place-labels SCENE={args.scene}` first")
        return 2
    d = json.loads(p.read_text())

    # Group all (object, keyframe) detections by keyframe.
    by_kf: dict[int, list[dict]] = defaultdict(list)
    for obj, entries in d["detections"].items():
        for e in entries:
            if e.get("bbox") is None:
                continue
            by_kf[int(e["kf"])].append({"obj": obj, **e})

    if not by_kf:
        log("no parsed bounding boxes at all — Qwen returned 'none' for every "
            "(object, keyframe) pair. Check `--show-raw` output during "
            "place-labels.")
        return 1

    # Need source keyframe paths — read scene transforms.
    t = json.loads((REPO / "data" / "scenes" / args.scene / "transforms.json").read_text())
    file_by_idx = {i: f["file_path"] for i, f in enumerate(t["frames"])}
    fx, fy = float(t["fl_x"]), float(t["fl_y"])
    cx_i, cy_i = float(t["cx"]), float(t["cy"])
    W_i, H_i = int(t["w"]), int(t["h"])
    T_dp, s_dp = load_dataparser_transform(args.scene)

    # Pre-build w2c per keyframe so we can project 3D anchors back into pixels.
    w2c_by_idx: dict[int, np.ndarray] = {}
    for i, f in enumerate(t["frames"]):
        c2w_ns = np.array(f["transform_matrix"], dtype=np.float64)
        c2w = T_dp @ c2w_ns
        c2w[:3, 3] *= s_dp
        w2c_by_idx[i] = np.linalg.inv(c2w)

    # Collect every 3D anchor so we can mark "where this label LANDED in 3D"
    # by projecting it back into every keyframe.
    anchors = d.get("anchors", [])

    out_dir = REPO / "semantics" / args.scene / "bbox_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Clear any stale audit images from previous runs.
    for old in out_dir.glob("kf*.jpg"):
        old.unlink()

    big_font = get_font(22)
    small_font = get_font(16)

    log(f"writing audit JPEGs for {len(by_kf)} keyframes -> {out_dir}")

    for kf_idx in sorted(by_kf.keys()):
        dets = by_kf[kf_idx]
        if args.accepted_only:
            dets = [d_ for d_ in dets if d_["reject"] is None]
            if not dets:
                continue

        img_path = REPO / "data" / "scenes" / args.scene / file_by_idx[kf_idx]
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img, "RGBA")

        # Draw bboxes — sort so accepted (green) draws on top of rejected.
        dets.sort(key=lambda d_: 0 if d_["reject"] is None else 1, reverse=True)

        for det in dets:
            x1, y1, x2, y2 = det["bbox"]
            reason = det["reject"].split("(")[0] if det["reject"] else None
            col = COLOR.get(reason, (255, 255, 255))
            text = det["obj"]
            if reason:
                text = f"{det['obj']}  [{reason}]"
            # Box outline (alpha 255 = fully opaque).
            draw.rectangle([x1, y1, x2, y2], outline=col + (255,), width=4)
            # Label tab above the box, with a coloured backing for legibility.
            text_bb = draw.textbbox((x1, max(0, y1 - 26)), text, font=small_font)
            pad = 4
            draw.rectangle(
                [text_bb[0] - pad, text_bb[1] - pad, text_bb[2] + pad, text_bb[3] + pad],
                fill=col + (235,),
            )
            draw.text((x1, max(0, y1 - 26)), text, fill=(0, 0, 0), font=small_font)

        # --- Project each 3D anchor back into THIS keyframe -----------------
        # Each marker is a cyan crosshair + label at the pixel where the
        # anchor's 3D position lands. Compare to the bbox center: if the
        # cyan dot drifts off the object that owns that label, the 3D
        # anchor is misplaced even though the bbox was right.
        for a in anchors:
            uv = project_world_to_pixel(np.array(a["anchor"]),
                                        w2c_by_idx[kf_idx],
                                        fx, fy, cx_i, cy_i, W_i, H_i)
            if uv is None:
                continue
            u, v = uv
            cross = 14
            cyan = (0, 220, 255)
            draw.line([(u - cross, v), (u + cross, v)], fill=cyan + (255,), width=3)
            draw.line([(u, v - cross), (u, v + cross)], fill=cyan + (255,), width=3)
            draw.ellipse([u - 6, v - 6, u + 6, v + 6],
                         outline=cyan + (255,), width=3, fill=(0, 0, 0, 0))
            tag = a["label"]
            if a.get("n_clusters", 1) > 1:
                tag = f"{tag} {a['cluster_idx'] + 1}/{a['n_clusters']}"
            tag = f"◇ {tag}"
            tb = draw.textbbox((u + 10, v - 10), tag, font=small_font)
            draw.rectangle([tb[0] - 3, tb[1] - 3, tb[2] + 3, tb[3] + 3],
                           fill=(0, 0, 0, 200))
            draw.text((u + 10, v - 10), tag, fill=cyan, font=small_font)

        # Legend strip at top-left.
        legend_y = 10
        legend_items = [
            ("accepted (anchored a label)", COLOR[None]),
            ("rejected: bbox over void", COLOR["sparse_patch"]),
            ("rejected: too far away", COLOR["too_far"]),
            ("rejected: outside room AABB", COLOR["outside_aabb"]),
            ("◇  projected 3D anchor location", (0, 220, 255)),
        ]
        legend_w = 380
        legend_h = 24 * len(legend_items) + 10
        draw.rectangle([10, legend_y, 10 + legend_w, legend_y + legend_h],
                       fill=(0, 0, 0, 180))
        draw.text((20, legend_y + 4),
                  f"kf{kf_idx:04d}  ({len(dets)} bboxes)",
                  fill=(255, 255, 255), font=big_font)
        for i, (text, col) in enumerate(legend_items):
            yy = legend_y + 30 + i * 22
            draw.rectangle([20, yy, 36, yy + 14], fill=col + (255,))
            draw.text((44, yy - 2), text, fill=(240, 240, 240), font=small_font)

        out = out_dir / f"kf{kf_idx:04d}.jpg"
        img.save(out, quality=85)

    log(f"done. open the folder or one image at a time:")
    log(f"  {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
