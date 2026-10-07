#!/usr/bin/env python3
"""Convert SAM3 BDD-MOTS pickles -> RobMOTS tracker txt + HOTA.

RobMOTS line format (0-based time):
  <time> <track_id> <class_id> <conf> <height> <width> <rle_counts>

  python eval/tracking/score_bdd_mots.py --tracker sam3_1_driving
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_spec import CONCEPT_TO_CLASS_ROBMOTS  # noqa: E402
from score_kitti_mots import resolve_overlaps  # noqa: E402
from score_mots_challenge import _maybe_upsample  # noqa: E402

GT_ROOT = (
    ROOT
    / "Research_Data"
    / "RobMOTS"
    / "train_gt_unzipped"
    / "data"
    / "gt"
    / "rob_mots"
)
SEQMAP = GT_ROOT / "train" / "bdd_mots" / "seqmap.txt"
PRED_ROOT = ROOT / "outputs" / "SAM3.1" / "tracking" / "raw" / "bdd_mots" / "train"
TRACKERS = ROOT / "outputs" / "SAM3.1" / "tracking" / "predictions" / "bdd_mots" / "train"


def parse_seqmap(path: Path) -> dict[str, tuple[int, int, int]]:
    info: dict[str, tuple[int, int, int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        info[parts[0]] = (int(parts[1]), int(parts[2]), int(parts[3]))
    return info


def concept_to_class(concept: str) -> int | None:
    c = concept.lower().strip()
    if c in CONCEPT_TO_CLASS_ROBMOTS:
        return CONCEPT_TO_CLASS_ROBMOTS[c]
    for k, v in CONCEPT_TO_CLASS_ROBMOTS.items():
        if k in c:
            return v
    return None


def encode_mask(binary: np.ndarray) -> dict:
    return mask_utils.encode(np.asfortranarray(binary.astype(np.uint8)))


def convert_pkl(
    masks_pkl: Path,
    out_txt: Path,
    num_frames: int,
    target_hw: tuple[int, int],
) -> dict:
    with masks_pkl.open("rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "masks" in payload:
        mask_dict = payload["masks"]
        obj_to_concept = {
            int(k): str(v).lower() for k, v in payload.get("obj_to_concept", {}).items()
        }
    else:
        mask_dict, obj_to_concept = payload, {}

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    # global track ids across classes (RobMOTS uses free track ids + class column)
    next_tid = 1
    oid_to_tid: dict[int, int] = {}
    class_tracks: dict[int, set[int]] = {}
    n_lines = 0

    with out_txt.open("w", encoding="utf-8") as f:
        for t in range(num_frames):
            frame_masks = mask_dict.get(t) or mask_dict.get(str(t)) or {}
            if not frame_masks:
                continue
            typed: dict[int, tuple[int, np.ndarray]] = {}
            for oid, mask in frame_masks.items():
                oid = int(oid)
                cid = concept_to_class(obj_to_concept.get(oid, ""))
                if cid is None:
                    continue
                m = np.asarray(mask)
                if m.ndim == 3:
                    m = m[0]
                m = _maybe_upsample(m, target_hw)
                typed[oid] = (cid, m)
            if not typed:
                continue
            resolved = resolve_overlaps({oid: m for oid, (_c, m) in typed.items()})
            for oid, bin_m in resolved.items():
                cid = typed[oid][0]
                if oid not in oid_to_tid:
                    oid_to_tid[oid] = next_tid
                    next_tid += 1
                    class_tracks.setdefault(cid, set()).add(oid_to_tid[oid])
                tid = oid_to_tid[oid]
                rle = encode_mask(bin_m)
                counts = rle["counts"]
                if isinstance(counts, bytes):
                    counts = counts.decode("utf-8")
                h, w = rle["size"]
                # conf=1.0 for open-vocab tracks
                f.write(f"{t} {tid} {cid} 1.0 {h} {w} {counts}\n")
                n_lines += 1

    return {
        "n_lines": n_lines,
        "tracks": {str(c): len(ids) for c, ids in class_tracks.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker", default="sam3_1_driving")
    parser.add_argument("--pred-dir", type=Path, default=PRED_ROOT)
    parser.add_argument("--gt-folder", type=Path, default=GT_ROOT)
    parser.add_argument("--seqmap", type=Path, default=SEQMAP)
    parser.add_argument("--skip-hota", action="store_true")
    parser.add_argument(
        "--classes",
        default="person,car,truck,bus",
        help="Comma-separated RobMOTS class names for HOTA",
    )
    parser.add_argument("--seqs", default=None, help="Optional comma-separated subset")
    args = parser.parse_args()

    seq_info = parse_seqmap(args.seqmap)
    if args.seqs:
        wanted = {s.strip() for s in args.seqs.split(",") if s.strip()}
        seq_info = {k: v for k, v in seq_info.items() if k in wanted}

    tracker_data = TRACKERS / args.tracker / "data" / "bdd_mots"
    tracker_data.mkdir(parents=True, exist_ok=True)

    summary: dict = {}
    for seq, (n_frames, h, w) in seq_info.items():
        pkl = args.pred_dir / f"{seq}_driving_masks.pkl"
        if not pkl.is_file():
            print(f"[miss] {pkl}")
            continue
        # Only write frames we actually tracked (pkl may be shorter on smoke)
        with pkl.open("rb") as f:
            payload = pickle.load(f)
        masks = payload["masks"] if isinstance(payload, dict) and "masks" in payload else payload
        tracked = max((int(k) for k in masks.keys()), default=-1) + 1
        use_frames = min(n_frames, tracked) if tracked > 0 else n_frames
        out_txt = tracker_data / f"{seq}.txt"
        stats = convert_pkl(pkl, out_txt, use_frames, (h, w))
        summary[seq] = {**stats, "num_frames_written": use_frames}
        print(f"[ok] {seq}: {stats}")

    meta = tracker_data.parent.parent / "convert_summary.json"
    meta.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {meta}")

    if args.skip_hota or not summary:
        return

    # For partial smoke: write a temporary seqmap covering only converted seqs
    smoke_seqmap = TRACKERS / args.tracker / "seqmap_eval.txt"
    lines = []
    for seq, (n_frames, h, w) in seq_info.items():
        if seq not in summary:
            continue
        use = summary[seq].get("num_frames_written", n_frames)
        lines.append(f"{seq} {use} {h} {w}")
    smoke_seqmap.write_text("\n".join(lines) + "\n", encoding="utf-8")

    cmd = [
        sys.executable,
        str(ROOT / "eval" / "tracking" / "run_hota.py"),
        "--benchmark",
        "bdd_mots",
        "--split",
        "train",
        "--tracker",
        args.tracker,
        "--gt-folder",
        str(args.gt_folder),
        "--trackers-folder",
        str(TRACKERS.parent),  # .../preds/rob_mots  (split=train under it)
        "--classes",
        args.classes,
        "--seqmap-file",
        str(smoke_seqmap),
    ]
    # trackers folder for RobMOTS is parent of split: preds/rob_mots
    # Fix: TRACKERS = .../preds/rob_mots/train, so parent is .../preds/rob_mots
    cmd[cmd.index("--trackers-folder") + 1] = str(TRACKERS.parent)
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT))


if __name__ == "__main__":
    main()
