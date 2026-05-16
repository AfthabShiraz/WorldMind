#!/usr/bin/env python3
"""
scene_inventory.py — "what does the VLM see in this room, and where is it?"

No SAM, no 3D. Samples N keyframes and asks Qwen2.5-VL to list significant
objects AND their spatial relations (e.g. 'book on desk', 'lamp on
nightstand'). Aggregates both object frequencies and relation triples
across frames.

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
    "Look at this photo of an indoor scene. List the significant visible "
    "objects and describe where each one is, using one short phrase per "
    "line. Format each line as:\n"
    "  <object> <relation> <other_object>\n"
    "Allowed relations: on, in, under, next to, near, against, behind, "
    "beside, above, below, on top of, in front of.\n"
    "If a thing is just standalone with no clear relation, write only its name.\n"
    "Examples (one per line):\n"
    "  book on desk\n"
    "  lamp on nightstand\n"
    "  rug on floor\n"
    "  curtain on window\n"
    "  chair next to desk\n"
    "  pillow on bed\n"
    "  bed against wall\n"
    "  plant\n"
    "Rules: one phrase per line; lowercase; single common English nouns; "
    "skip the wall/floor/ceiling themselves unless something rests on them; "
    "no sentences, no commentary, no bullet points."
)

# Multi-word relations listed first so they win longest-match.
RELATIONS = (
    "on top of", "in front of", "next to",
    "on", "in", "under", "above", "below",
    "behind", "beside", "near", "against", "by",
)

STOP_PREFIXES = (
    "i ", "we ", "there ", "this ", "that ", "you ", "the room ", "the scene ",
    "objects ", "things ", "items ", "visible ", "in the ", "on the ", "at the ",
    "here ", "this is ",
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def pick_keyframes(scene: str, n: int) -> list[tuple[int, str]]:
    p = REPO / "data" / "scenes" / scene / "transforms.json"
    t = json.loads(p.read_text())
    frames = t["frames"]
    idxs = np.linspace(0, len(frames) - 1, min(n, len(frames)), dtype=int)
    return [(int(i), frames[int(i)]["file_path"]) for i in idxs]


_NOUN_RE = re.compile(r"^[a-z][a-z\- ]{0,30}$")


def _strip_article(s: str) -> str:
    return re.sub(r"^(a|an|the|some)\s+", "", s.strip())


def _valid_noun(s: str) -> bool:
    s = s.strip()
    return bool(_NOUN_RE.match(s)) and len(s.split()) <= 3 \
        and not any(s.startswith(p) for p in STOP_PREFIXES)


def parse_line(raw: str) -> tuple[str, str | None, str | None] | None:
    """Parse 'book on desk' -> ('book', 'on', 'desk'); 'plant' -> ('plant', None, None).
    Returns None if the line is unparseable."""
    s = raw.strip().lower()
    s = re.sub(r"^[-*•\d\.\s]+", "", s)               # strip bullets / numbering
    s = s.rstrip(".,;:")
    if not s:
        return None
    # Drop "objects visible:" style prefixes (rare in the per-line format but possible).
    if ":" in s[:40]:
        s = s.split(":", 1)[1].strip()
    s = _strip_article(s)

    # Try each relation, longest first; require whole-word boundaries.
    for rel in RELATIONS:
        pat = rf"^(.+?)\s+{re.escape(rel)}\s+(.+)$"
        m = re.match(pat, s)
        if m:
            obj = _strip_article(m.group(1)).strip().rstrip(",")
            anc = _strip_article(m.group(2)).strip().rstrip(",")
            if _valid_noun(obj) and _valid_noun(anc):
                return (obj, rel, anc)
    # Standalone object: no relation.
    if _valid_noun(s):
        return (s, None, None)
    return None


def parse_response(raw: str) -> list[tuple[str, str | None, str | None]]:
    """Split Qwen's output on newlines (and commas as a fallback for cases
    where it ignored the per-line instruction) and parse each piece."""
    pieces = re.split(r"[\n;]", raw)
    # If we got just one big line and no newlines, fall back to comma-splitting.
    if len(pieces) == 1:
        pieces = re.split(r",", pieces[0])
    out: list[tuple[str, str | None, str | None]] = []
    seen: set[tuple] = set()
    for p in pieces:
        t = parse_line(p)
        if t is None:
            continue
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--n-keyframes", type=int, default=15)
    ap.add_argument("--show-raw", action="store_true",
                    help="Print Qwen's raw response for each keyframe.")
    ap.add_argument("--min-frames", type=int, default=2,
                    help="Items/relations appearing in fewer than this many "
                         "keyframes are still listed but marked low-confidence.")
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

    object_counts: Counter = Counter()       # noun -> frames mentioning it
    relation_counts: Counter = Counter()     # (obj, rel, anc) -> frames
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
            gen = model.generate(**inputs, max_new_tokens=220, do_sample=False)
        out_ids = gen[:, inputs.input_ids.shape[1]:]
        raw = proc.batch_decode(out_ids, skip_special_tokens=True)[0]

        triples = parse_response(raw)
        per_frame_objs = {t[0] for t in triples}
        for o in per_frame_objs:
            object_counts[o] += 1
        per_frame_rels = {(o, r, a) for (o, r, a) in triples if r is not None}
        for rel in per_frame_rels:
            relation_counts[rel] += 1

        per_frame[str(kf_idx)] = {
            "file": fp,
            "raw": raw.strip(),
            "triples": [list(t) for t in triples],
        }
        if args.show_raw:
            print(f"\n--- kf{kf_idx:04d} ({fp}) ---", file=sys.stderr)
            print(f"raw:    {raw.strip()}", file=sys.stderr)
            print(f"parsed: {triples}", file=sys.stderr)

    # Save
    out_dir = REPO / "semantics" / args.scene
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "scene_inventory.json"
    out_path.write_text(json.dumps({
        "scene": args.scene,
        "n_keyframes": len(kfs),
        "prompt": PROMPT,
        "counts": object_counts.most_common(),
        "relations": [
            {"object": o, "relation": r, "anchor": a, "frames": int(c)}
            for (o, r, a), c in relation_counts.most_common()
        ],
        "per_frame": per_frame,
    }, indent=2))

    # ---- Print summary -----------------------------------------------------
    print()
    print(f"=== Scene inventory: {args.scene}  ({len(kfs)} keyframes) ===")
    print()
    print("OBJECTS  (n frames mentioning each)")
    print(f"{'-'*6}  {'-'*30}")
    confident, low_conf = [], []
    for item, c in object_counts.most_common():
        (confident if c >= args.min_frames else low_conf).append((item, c))
    for item, c in confident:
        print(f"{c:>6}  {item}")
    if low_conf:
        print(f"\n  -- seen in <{args.min_frames} frames (low confidence) --")
        for item, c in low_conf[:20]:
            print(f"{c:>6}  {item}")
        if len(low_conf) > 20:
            print(f"        ... and {len(low_conf) - 20} more")

    print()
    print("RELATIONSHIPS  (n frames where this triple appeared)")
    print(f"{'-'*6}  {'-'*40}")
    rel_confident = [((o, r, a), c) for (o, r, a), c in relation_counts.most_common()
                     if c >= args.min_frames]
    rel_low = [((o, r, a), c) for (o, r, a), c in relation_counts.most_common()
               if c < args.min_frames]
    for (o, r, a), c in rel_confident:
        print(f"{c:>6}  {o} {r} {a}")
    if rel_low:
        print(f"\n  -- relations in <{args.min_frames} frames (low confidence) --")
        for (o, r, a), c in rel_low[:15]:
            print(f"{c:>6}  {o} {r} {a}")
        if len(rel_low) > 15:
            print(f"        ... and {len(rel_low) - 15} more")

    print()
    print(f"Unique objects:    {len(object_counts)} "
          f"(>= {args.min_frames} frames: {len(confident)})")
    print(f"Unique relations:  {len(relation_counts)} "
          f"(>= {args.min_frames} frames: {len(rel_confident)})")
    print(f"Saved to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
