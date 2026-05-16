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


def make_permutation(n: int, seed: int = 0) -> np.ndarray:
    """Stable random permutation; slicing perm[:k] gives a monotonic subset
    so the slider feels like 'add more detail' instead of reshuffling."""
    return np.random.default_rng(seed).permutation(n)


def subset(perm: np.ndarray, k: int) -> np.ndarray:
    """Return sorted indices for the first `k` elements of the permutation."""
    k = max(1, min(int(k), perm.size))
    sel = perm[:k].copy()
    sel.sort()
    return sel


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

    n_total = xyz.shape[0]
    perm = make_permutation(n_total)
    init_k = max(1, min(args.max_points, n_total))
    sel0 = subset(perm, init_k)
    # `view` holds the current visible subset and is mutated by the slider.
    view = {"xyz": xyz[sel0], "rgb": rgb[sel0], "inst": inst_full[sel0]}
    log(f"showing {view['xyz'].shape[0]:,} / {n_total:,} points "
        f"(slider can go up to all of them)")

    # Instance palette as a single lookup. Empty when no semantics are loaded.
    inst_id_to_color: dict[int, tuple[int, int, int]] = {}
    for ii_str, a in anchors.items():
        inst_id_to_color[int(ii_str)] = tuple(a["palette_rgb"])

    def colors_for_mode(tint: bool, tint_amount: float) -> np.ndarray:
        if not tint or not inst_id_to_color:
            return view["rgb"]
        out = view["rgb"].astype(np.float32)
        # Build a per-point tint colour; -1 / unknown stays grey.
        tint_rgb = np.full_like(out, 128.0)
        for ii, col in inst_id_to_color.items():
            sel = (view["inst"] == ii)
            if sel.any():
                tint_rgb[sel] = col
        out = (1.0 - tint_amount) * out + tint_amount * tint_rgb
        return np.clip(out, 0, 255).astype(np.uint8)

    # --- Object anchors (optional sidecar, from place_object_labels) ------
    anchors_path = REPO / "semantics" / args.scene / "object_anchors.json"
    object_anchors: list[dict] = []
    scene_diag = 1.0
    if anchors_path.exists():
        oa = json.loads(anchors_path.read_text())
        object_anchors = oa.get("anchors", [])
        scene_diag = float(oa.get("scene_diag", 1.0))
        log(f"object anchors: {len(object_anchors)} loaded "
            f"(scene diag {scene_diag:.2f})")

    # --- Scene inventory (optional sidecar) ------------------------------
    inv_path = REPO / "semantics" / args.scene / "scene_inventory.json"
    inventory_items: list[tuple[str, int]] = []
    inventory_relations: list[dict] = []
    inventory_description: str = ""
    inv_n_keyframes = 0
    if inv_path.exists():
        inv = json.loads(inv_path.read_text())
        inv_n_keyframes = int(inv.get("n_keyframes", 0))
        threshold = (args.inventory_min_frames if args.inventory_min_frames > 0
                     else max(2, inv_n_keyframes // 3))
        inventory_items = [(name, int(c)) for name, c in inv["counts"]
                           if int(c) >= threshold]
        inventory_relations = [r for r in inv.get("relations", [])
                               if int(r["frames"]) >= threshold]
        inventory_description = inv.get("description", "").strip()
        log(f"scene inventory: {len(inventory_items)} items + "
            f"{len(inventory_relations)} relations >= {threshold} frames "
            f"(of {inv_n_keyframes} keyframes)"
            + (f"; description loaded ({len(inventory_description)} chars)"
               if inventory_description else ""))
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
    if inventory_items or inventory_relations or inventory_description:
        threshold = (args.inventory_min_frames if args.inventory_min_frames > 0
                     else max(2, inv_n_keyframes // 3))
        with server.gui.add_folder("Scene inventory (VLM)"):
            md: list[str] = []
            if inventory_description:
                md.append("**Room description**\n")
                md.append(f"_{inventory_description}_")
            if inventory_items:
                if md:
                    md.append("")
                md.append(f"**Objects** ({len(inventory_items)}, seen in ≥ "
                          f"{threshold} of {inv_n_keyframes} keyframes):\n")
                for name, c in inventory_items:
                    md.append(f"- `{c:>2}`  {name}")
            if inventory_relations:
                if md:
                    md.append("")
                md.append(f"**Spatial relations** ({len(inventory_relations)}):\n")
                for r in inventory_relations:
                    md.append(f"- `{r['frames']:>2}`  "
                              f"{r['object']} *{r['relation']}* {r['anchor']}")
            server.gui.add_markdown("\n".join(md))
    elif inv_path.exists():
        with server.gui.add_folder("Scene inventory (VLM)"):
            server.gui.add_markdown(
                "_No items pass the frame threshold. Lower "
                "`--inventory-min-frames` to see fewer-confidence items._"
            )

    # Object labels (3D floating billboards above each anchor) ------------
    if object_anchors:
        show_obj_labels = server.gui.add_checkbox("Show object labels",
                                                  initial_value=True)
    else:
        show_obj_labels = None

    # Point count: slider lets the user trade density vs. structure clarity
    # live, between 1% of the splat and all of it.
    slider_min = max(1000, n_total // 100)
    slider_step = max(1000, (n_total - slider_min) // 200)
    point_count = server.gui.add_slider(
        "Points shown", min=slider_min, max=n_total,
        step=slider_step, initial_value=init_k,
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
    # viser 0.2.7's PointCloudHandle has no setters for points/colors/size,
    # so live updates require a remove + re-add cycle. The handle is held in
    # a one-element list so closures can mutate it.
    def _make_pc(points, colors, size):
        return server.scene.add_point_cloud(
            name="/splat",
            points=points, colors=colors,
            point_size=size, point_shape="circle",
        )

    pc_holder = [_make_pc(view["xyz"], view["rgb"], args.point_size)]

    # 3D floating labels for inventory objects, anchored just above their
    # back-projected position. Toggle via the "Show object labels" checkbox.
    label_handles: list = []
    if object_anchors:
        nudge_up = 0.04 * scene_diag   # float labels slightly above the surface
        # Group same-label clusters so we can suffix "(2)" etc. when the room
        # actually has multiple instances of the same object name.
        for a in object_anchors:
            text = a["label"]
            if a.get("n_clusters", 1) > 1:
                text = f"{text} ({a['cluster_idx'] + 1}/{a['n_clusters']})"
            ax, ay, az = a["anchor"]
            label_handles.append(server.scene.add_label(
                name=f"/obj_labels/{a['label']}_{a.get('cluster_idx', 0)}",
                text=text,
                position=(ax, ay + nudge_up, az),
            ))

    # --- Reactivity -------------------------------------------------------
    def remake_pc():
        tint_v = tint_on.value if tint_on is not None else False
        amt = float(tint_amount.value) if tint_amount is not None else 0.0
        colors_now = colors_for_mode(tint_v, amt)
        with server.atomic():
            pc_holder[0].remove()
            pc_holder[0] = _make_pc(view["xyz"], colors_now, float(point_size.value))

    def resample_to(k: int):
        sel = subset(perm, k)
        view["xyz"] = xyz[sel]
        view["rgb"] = rgb[sel]
        view["inst"] = inst_full[sel]
        remake_pc()

    @point_count.on_update
    def _(_):
        resample_to(int(point_count.value))

    if tint_on is not None:
        @tint_on.on_update
        def _(_):
            remake_pc()

        @tint_amount.on_update
        def _(_):
            if tint_on.value:
                remake_pc()

    @point_size.on_update
    def _(_):
        remake_pc()

    if show_obj_labels is not None:
        @show_obj_labels.on_update
        def _(_):
            for h in label_handles:
                h.visible = show_obj_labels.value

    log("viewer ready. Ctrl-C to quit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        log("shutting down")
    return 0


if __name__ == "__main__":
    sys.exit(main())
