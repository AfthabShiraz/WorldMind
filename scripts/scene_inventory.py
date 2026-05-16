#!/usr/bin/env python3
"""
scene_inventory.py — "what does the VLM see in this room?"

No SAM, no lifting, no 3D. Just samples N keyframes from the scene and asks
Qwen2.5-VL to list significant objects/furniture/features in each. Aggregates
counts across frames and prints a sorted frequency list.

Usage:
  scripts/scene_inventory.py --scene <id> [--n-keyframes 15] [--show-raw]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm


REPO = Path(__file__).resolve().parent.parent
QWEN_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

PROMPT = (
    "Look at this photo of an indoor scene. List the significant objects, "
    "furniture, and architectural features you can clearly see in the image. "
    "Output ONLY a comma-separated list of single lowercase nouns. Examples: "
    "'chair, table, window, curtain, lamp, wardrobe, rug, bed'. Include "
    "furniture, appliances, doors, windows, curtains, rugs, plants, beds, "
    "shelves, mirrors, lamps. Do NOT include wall, floor, or ceiling unless "
    "they have a distinctive feature. Do NOT write sentences. Do NOT add "
    "explanations. Just the list."
)

STOP_PREFIXES = (
    "i ", "we ", "there ", "this ", "that ", "you ", "the room ", "the scene ",
    "objects ", "things ", "items ", "visible ", "in the ", "on the ", "at the ",
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def pick_keyframes(scene: str, n: int) -> list[tuple[int, str]]:
    p = REPO / "data" / "scenes" / scene / "transforms.json"
    t = json.loads(p.read_text())
    frames = t["frames"]
    idxs = np.linspace(0, len(frames) - 1, min(n, len(frames)), dtype=int)
    return [(int(i), frames[int(i)]["file_path"]) for i in idxs]


_WORD_RE = re.compile(r"^[a-z][a-z\- ]{0,30}$")

def clean_items(raw: str) -> list[str]:
    """Parse 'chair, table, window' (or messier variants) into a clean list."""
    s = raw.strip().lower()
    # If the model prefixed "objects visible:" or similar, drop everything
    # before the first colon (only if it's near the start).
    if ":" in s[:60]:
        s = s.split(":", 1)[1]
    parts = re.split(r"[,;\n]| and ", s)
    out: list[str] = []
    seen: set[str] = set()
    for raw_part in parts:
        it = raw_part.strip().rstrip(".")
        it = re.sub(r"^(a|an|the|some)\s+", "", it)
        if not _WORD_RE.match(it):
            continue
        if any(it.startswith(p) for p in STOP_PREFIXES):
            continue
        if len(it.split()) > 3:
            continue
        if it in seen:
            continue
        seen.add(it)
        out.append(it)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--n-keyframes", type=int, default=15)
    ap.add_argument("--show-raw", action="store_true",
                    help="Print Qwen's raw response for each keyframe.")
    ap.add_argument("--min-frames", type=int, default=2,
                    help="Items appearing in fewer than this many keyframes are "
                         "shown but marked as low-confidence in the summary.")
    args = ap.parse_args()

    kfs = pick_keyframes(args.scene, args.n_keyframes)
    log(f"scene={args.scene}  using {len(kfs)} keyframes")

    log("loading Qwen2.5-VL-3B ...")
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    proc = AutoProcessor.from_pretrained(QWEN_ID)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        QWEN_ID, torch_dtype=torch.bfloat16, device_map="cuda"
    ).eval()

    counts: Counter = Counter()
    per_frame: dict[str, dict] = {}

    for kf_idx, fp in tqdm(kfs, desc="Qwen", file=sys.stderr):
        img = Image.open(REPO / "data" / "scenes" / args.scene / fp).convert("RGB")
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": PROMPT},
        ]}]
        text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = proc(text=[text], images=[img], padding=True, return_tensors="pt").to("cuda")
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=150, do_sample=False)
        out_ids = gen[:, inputs.input_ids.shape[1]:]
        raw = proc.batch_decode(out_ids, skip_special_tokens=True)[0]
        items = clean_items(raw)
        per_frame[str(kf_idx)] = {"file": fp, "raw": raw.strip(), "items": items}
        for it in items:
            counts[it] += 1
        if args.show_raw:
            print(f"\n--- kf{kf_idx:04d} ({fp}) ---", file=sys.stderr)
            print(f"raw:    {raw.strip()}", file=sys.stderr)
            print(f"parsed: {items}", file=sys.stderr)

    # Save
    out_dir = REPO / "semantics" / args.scene
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scene_inventory.json"
    out_path.write_text(json.dumps({
        "scene": args.scene,
        "n_keyframes": len(kfs),
        "prompt": PROMPT,
        "counts": counts.most_common(),
        "per_frame": per_frame,
    }, indent=2))

    # Print summary
    print()
    print(f"=== Scene inventory: {args.scene}  ({len(kfs)} keyframes) ===")
    print(f"{'frames':>6}  item")
    print(f"{'-'*6}  {'-'*30}")
    confident, low_conf = [], []
    for item, c in counts.most_common():
        if c >= args.min_frames:
            confident.append((item, c))
        else:
            low_conf.append((item, c))
    for item, c in confident:
        print(f"{c:>6}  {item}")
    if low_conf:
        print(f"\n--- seen only in 1 frame (likely noise / one-off views) ---")
        for item, c in low_conf:
            print(f"{c:>6}  {item}")
    print(f"\nTotal unique items: {len(counts)}  "
          f"(>= {args.min_frames} frames: {len(confident)})")
    print(f"Saved to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
