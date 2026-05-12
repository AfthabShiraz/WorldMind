"""Render top-down + side views of a COLMAP sparse cloud + camera trajectory.

Emits two PNGs that you can Read into the chat. Works over SSH (no display needed).

Usage:
    python scripts/qc_render_sparse.py --scene room
"""
from __future__ import annotations

import argparse
import struct
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed; works over SSH
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _read_points3d(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (xyz N x 3, rgb N x 3) from a COLMAP points3D.bin."""
    xyzs: list[tuple[float, float, float]] = []
    rgbs: list[tuple[int, int, int]] = []
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            f.read(8)  # point id
            xyz = struct.unpack("<3d", f.read(24))
            rgb = struct.unpack("<3B", f.read(3))
            f.read(8)  # reproj error
            (track_len,) = struct.unpack("<Q", f.read(8))
            f.read(8 * track_len)
            xyzs.append(xyz)
            rgbs.append(rgb)
    return np.asarray(xyzs), np.asarray(rgbs)


def _read_images(path: Path) -> np.ndarray:
    """Return camera centers (N x 3) extracted from COLMAP images.bin.

    World position of a camera is `-R^T t`, where (qw, qx, qy, qz) is the
    rotation as a quaternion and t the translation in the camera's view of
    the world.
    """
    centers: list[np.ndarray] = []
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            f.read(4)  # image_id
            qw, qx, qy, qz = struct.unpack("<4d", f.read(32))
            tx, ty, tz = struct.unpack("<3d", f.read(24))
            f.read(4)  # camera_id
            while f.read(1) != b"\x00":
                pass  # name
            (n_pts,) = struct.unpack("<Q", f.read(8))
            f.read(24 * n_pts)
            # Quaternion (w,x,y,z) → rotation matrix
            r = np.array(
                [
                    [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                    [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                    [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
                ]
            )
            center = -r.T @ np.array([tx, ty, tz])
            centers.append(center)
    return np.asarray(centers)


def _render(scene_dir: Path, out_dir: Path) -> list[Path]:
    sparse = scene_dir / "colmap" / "sparse" / "0"
    pts_xyz, pts_rgb = _read_points3d(sparse / "points3D.bin")
    cams = _read_images(sparse / "images.bin")
    print(f"  {len(pts_xyz)} 3D points,  {len(cams)} camera centers")

    # Clip outlier points so the axes auto-scale to the bulk of the scene.
    keep = np.ones(len(pts_xyz), dtype=bool)
    for ax in range(3):
        lo, hi = np.percentile(pts_xyz[:, ax], [1, 99])
        keep &= (pts_xyz[:, ax] >= lo) & (pts_xyz[:, ax] <= hi)
    pts_in = pts_xyz[keep]
    rgb_in = pts_rgb[keep] / 255.0

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for view_name, (a, b, a_label, b_label) in {
        "top_down": (0, 2, "X (world)", "Z (world)"),
        "side": (0, 1, "X (world)", "Y (world)"),
    }.items():
        fig, ax = plt.subplots(figsize=(10, 10), dpi=120)
        ax.scatter(pts_in[:, a], pts_in[:, b], c=rgb_in, s=1.5, alpha=0.5, label="sparse 3D points")
        ax.scatter(cams[:, a], cams[:, b], c="red", s=12, marker="^", label=f"camera centers (n={len(cams)})")
        # Trajectory line
        order = np.argsort([i for i in range(len(cams))])  # already in image order
        ax.plot(cams[order, a], cams[order, b], "-", color="red", alpha=0.4, linewidth=0.8)
        ax.set_aspect("equal")
        ax.set_xlabel(a_label)
        ax.set_ylabel(b_label)
        ax.set_title(f"{scene_dir.name}: {view_name} view  ({len(pts_in)} pts, {len(cams)} cams)")
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.2)
        path = out_dir / f"{view_name}.png"
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        written.append(path)
        print(f"  wrote {path}")
    return written


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    args = p.parse_args()

    repo = Path(__file__).resolve().parent.parent
    scene_dir = repo / "data" / "scenes" / args.scene
    out_dir = repo / "outputs" / args.scene / "_qc"
    _render(scene_dir, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
