#!/usr/bin/env python3
"""
view_semantics.py — viser viewer for splat + floating-label semantics overlay.

Loads:
  outputs/<scene>/ply_full/splat.ply               (unmodified)
  semantics/<scene>/gaussian_instances.npy         (per-Gaussian instance id)
  semantics/<scene>/instance_anchors.json          (label + 3D anchor + colour)

Renders Gaussians as a coloured point cloud (using each splat's SH DC
channel converted to RGB) and floats text labels above each instance's
top point. GUI checkboxes toggle labels, instance tinting, and top-N
distance-based culling.

Usage:
  scripts/view_semantics.py --scene <id> [--port 8080] [--max-points 250000]

splat.ply is never read for write. This viewer is purely visualisation.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from plyfile import PlyData


REPO = Path(__file__).resolve().parent.parent
SH_C0 = 0.28209479177387814


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def load_points_and_colors(ply_path: Path) -> tuple[np.ndarray, np.ndarray]:
    log(f"loading {ply_path} ...")
    ply = PlyData.read(str(ply_path))
    el = ply["vertex"]
    xyz = np.stack([el["x"], el["y"], el["z"]], axis=1).astype(np.float32)
    dc = np.stack([el["f_dc_0"], el["f_dc_1"], el["f_dc_2"]], axis=1).astype(np.float32)
    # splatfacto: rendered colour = 0.5 + SH_C0 * dc, then clamped to [0,1].
    rgb = np.clip(0.5 + SH_C0 * dc, 0.0, 1.0)
    rgb_u8 = (rgb * 255.0).astype(np.uint8)
    return xyz, rgb_u8


def subsample_points(xyz: np.ndarray, rgb: np.ndarray, inst: np.ndarray,
                     max_pts: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = xyz.shape[0]
    if n <= max_pts:
        return xyz, rgb, inst
    rng = np.random.default_rng(0)
    sel = rng.choice(n, size=max_pts, replace=False)
    sel.sort()
    return xyz[sel], rgb[sel], inst[sel]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--max-points", type=int, default=250_000,
                    help="Subsample Gaussian centres to at most N points for the viewer.")
    ap.add_argument("--point-size", type=float, default=0.005)
    ap.add_argument("--inventory-min-frames", type=int, default=0,
                    help="Items needing this many keyframe sightings to show "
                         "in the scene-inventory panel. 0 = auto (>=50%% of keyframes).")
    ap.add_argument("--share", action="store_true",
                    help="Request a public viser share URL so the viewer can be "
                         "opened from outside this machine (no SSH needed).")
    args = ap.parse_args()

    import viser

    ply_path = REPO / "outputs" / args.scene / "ply_full" / "splat.ply"
    if not ply_path.exists():
        log(f"ERROR: missing {ply_path} — run `make run VIDEO=...` first")
        return 2

    # Semantic sidecars are OPTIONAL. Without them the viewer just shows the
    # raw splat (no tinting, no inventory panel) — this is the default path
    # because the semantics pipeline is not part of the standard flow.
    inst_path = REPO / "semantics" / args.scene / "gaussian_instances.npy"
    anch_path = REPO / "semantics" / args.scene / "instance_anchors.json"
    has_instances = inst_path.exists() and anch_path.exists()

    xyz, rgb = load_points_and_colors(ply_path)
    if has_instances:
        inst_full = np.load(inst_path).astype(np.int32)
        anchors: dict[str, dict] = json.loads(anch_path.read_text())
        log(f"loaded splat: {xyz.shape[0]:,} Gaussians, "
            f"{(inst_full >= 0).sum():,} with instance ids, "
            f"{len(anchors)} labelled instances")
    else:
        inst_full = np.full(xyz.shape[0], -1, dtype=np.int32)
        anchors = {}
        log(f"loaded splat: {xyz.shape[0]:,} Gaussians (no semantic sidecars)")

    xyz_s, rgb_s, inst_s = subsample_points(xyz, rgb, inst_full, args.max_points)
    log(f"subsampled to {xyz_s.shape[0]:,} points for viewer")

    # Instance palette as a single lookup. Empty when no semantics are loaded.
    inst_id_to_color: dict[int, tuple[int, int, int]] = {}
    for ii_str, a in anchors.items():
        inst_id_to_color[int(ii_str)] = tuple(a["palette_rgb"])

    def colors_for_mode(tint: bool, tint_amount: float) -> np.ndarray:
        if not tint or not inst_id_to_color:
            return rgb_s
        out = rgb_s.astype(np.float32)
        # Build a per-point tint colour; -1 / unknown stays grey.
        tint_rgb = np.full_like(out, 128.0)
        for ii, col in inst_id_to_color.items():
            sel = (inst_s == ii)
            if sel.any():
                tint_rgb[sel] = col
        out = (1.0 - tint_amount) * out + tint_amount * tint_rgb
        return np.clip(out, 0, 255).astype(np.uint8)

    # --- Scene inventory (optional sidecar) ------------------------------
    inv_path = REPO / "semantics" / args.scene / "scene_inventory.json"
    inventory_items: list[tuple[str, int]] = []
    inv_n_keyframes = 0
    if inv_path.exists():
        inv = json.loads(inv_path.read_text())
        inv_n_keyframes = int(inv.get("n_keyframes", 0))
        threshold = (args.inventory_min_frames if args.inventory_min_frames > 0
                     else max(2, (inv_n_keyframes + 1) // 2))
        inventory_items = [(name, int(c)) for name, c in inv["counts"]
                           if int(c) >= threshold]
        log(f"scene inventory: {len(inventory_items)} items >= {threshold} frames "
            f"(of {inv_n_keyframes} keyframes)")
    else:
        log(f"no scene_inventory.json at {inv_path} "
            f"(run `make scene-inventory SCENE={args.scene}` to enable the panel)")

    server = viser.ViserServer(host="0.0.0.0", port=args.port)
    log(f"viser running at http://0.0.0.0:{args.port}")
    if args.share:
        try:
            url = server.request_share_url()
            log(f"PUBLIC SHARE URL: {url}")
            log("anyone with that link can view the scene from outside this machine")
        except Exception as e:
            log(f"could not get share URL ({e}); use --port + SSH tunnel instead")

    # --- GUI -------------------------------------------------------------
    if inventory_items:
        threshold = (args.inventory_min_frames if args.inventory_min_frames > 0
                     else max(2, (inv_n_keyframes + 1) // 2))
        with server.gui.add_folder("Scene inventory (VLM)"):
            md = [f"**{len(inventory_items)} items** seen in ≥ {threshold} "
                  f"of {inv_n_keyframes} keyframes:\n"]
            for name, c in inventory_items:
                md.append(f"- `{c:>2}`  {name}")
            server.gui.add_markdown("\n".join(md))
    elif inv_path.exists():
        with server.gui.add_folder("Scene inventory (VLM)"):
            server.gui.add_markdown(
                "_No items pass the frame threshold. Lower "
                "`--inventory-min-frames` to see fewer-confidence items._"
            )

    # Tint controls only make sense when semantic sidecars are loaded.
    if has_instances:
        tint_on = server.gui.add_checkbox("Tint by instance", initial_value=False)
        tint_amount = server.gui.add_slider("Tint amount", min=0.0, max=1.0,
                                            step=0.05, initial_value=0.5)
    else:
        tint_on = None
        tint_amount = None
    point_size = server.gui.add_slider("Point size", min=0.001, max=0.05,
                                       step=0.001, initial_value=args.point_size)

    # --- Point cloud ------------------------------------------------------
    pc = server.scene.add_point_cloud(
        name="/splat",
        points=xyz_s,
        colors=rgb_s,
        point_size=args.point_size,
        point_shape="circle",
    )

    # Floating per-instance 3D labels are disabled for now; the
    # instance_anchors.json sidecar is still on disk and the inventory panel
    # above shows the VLM's overall list. Re-enable here when needed.

    # --- Reactivity -------------------------------------------------------
    def refresh_colors():
        pc.colors = colors_for_mode(tint_on.value, float(tint_amount.value))

    if tint_on is not None:
        @tint_on.on_update
        def _(_):
            refresh_colors()

        @tint_amount.on_update
        def _(_):
            if tint_on.value:
                refresh_colors()

    @point_size.on_update
    def _(_):
        pc.point_size = float(point_size.value)

    log("viewer ready. Ctrl-C to quit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        log("shutting down")
    return 0


if __name__ == "__main__":
    sys.exit(main())
