#!/usr/bin/env python3
"""Convert SAM 3.1 MOTS Challenge pickles -> TrackEval + HOTA.

MOTS Challenge time indices are 1-based (unlike KITTI MOTS 0-based).

  python eval/tracking/score_mots_challenge.py --tracker sam3_1_person
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
from metrics_spec import CONCEPT_TO_CLASS  # noqa: E402
from score_kitti_mots import resolve_overlaps  # noqa: E402

SEQMAP = Path(__file__).resolve().parent / "seqmaps" / "mots_challenge_train.seqmap"
PRED_ROOT = ROOT / "outputs" / "SAM3.1" / "tracking" / "raw" / "mots_challenge" / "train"
TRACKERS = ROOT / "outputs" / "SAM3.1" / "tracking" / "predictions" / "mots_challenge" / "train"
GT = ROOT / "outputs" / "benchmarks" / "ground_truth" / "mots_challenge"


def parse_seqmap(path: Path) -> dict[str, int]:
    info: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        seq = f"{int(parts[0]):04d}"
        start, end = int(parts[2]), int(parts[3])
        info[seq] = end - start + 1
    return info


def concept_to_class(concept: str) -> int | None:
    c = concept.lower()
    if c in CONCEPT_TO_CLASS:
        return CONCEPT_TO_CLASS[c]
    for k, v in CONCEPT_TO_CLASS.items():
        if k in c:
            return v
    return None


def encode_mask(binary: np.ndarray) -> dict:
    return mask_utils.encode(np.asfortranarray(binary.astype(np.uint8)))


def _maybe_upsample(mask: np.ndarray, target_hw: tuple[int, int] | None) -> np.ndarray:
    """Nearest upsample when inference ran at reduced resolution (e.g. --max-side)."""
    if target_hw is None:
        return mask
    th, tw = target_hw
    h, w = mask.shape[-2:]
    if (h, w) == (th, tw):
        return mask
    from PIL import Image

    im = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    im = im.resize((tw, th), Image.NEAREST)
    return (np.asarray(im) > 127).astype(np.uint8)


def convert_pkl(
    masks_pkl: Path,
    out_txt: Path,
    num_frames: int,
    *,
    frame_offset: int = 1,
    target_hw: tuple[int, int] | None = None,
) -> dict:
    """Write MOTS txt. frame_offset=1 maps SAM3 t=0 -> TrackEval time=1."""
    with masks_pkl.open("rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict) and "masks" in payload:
        model_version = payload.get("meta", {}).get("model_version")
        if model_version != "sam3.1":
            raise ValueError(
                f"{masks_pkl} is model_version={model_version!r}; expected 'sam3.1'"
            )
        mask_dict = payload["masks"]
        obj_to_concept = {int(k): str(v).lower() for k, v in payload.get("obj_to_concept", {}).items()}
    else:
        raise ValueError(f"{masks_pkl} is a legacy payload without SAM 3.1 metadata")

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    class_local: dict[int, dict[int, int]] = {2: {}}
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
                if cid != 2:
                    continue  # pedestrian only
                m = np.asarray(mask)
                if m.ndim == 3:
                    m = m[0]
                m = _maybe_upsample(m.astype(np.uint8), target_hw)
                typed[oid] = (cid, m)
            if not typed:
                continue
            resolved = resolve_overlaps({oid: m for oid, (_c, m) in typed.items()})
            time_frame = t + frame_offset
            for oid, bin_m in resolved.items():
                cid = typed[oid][0]
                local = class_local[cid]
                if oid not in local:
                    local[oid] = len(local) + 1
                mots_id = cid * 1000 + local[oid]
                rle = encode_mask(bin_m)
                counts = rle["counts"]
                if isinstance(counts, bytes):
                    counts = counts.decode("utf-8")
                h, w = rle["size"]
                f.write(f"{time_frame} {mots_id} {cid} {h} {w} {counts}\n")
                n_lines += 1
    return {"n_lines": n_lines, "tracks": {str(c): len(m) for c, m in class_local.items()}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker", default="sam3_1_person")
    parser.add_argument("--pred-dir", type=Path, default=PRED_ROOT)
    parser.add_argument("--skip-hota", action="store_true")
    args = parser.parse_args()

    seq_info = parse_seqmap(SEQMAP)
    tracker_data = TRACKERS / args.tracker / "data"
    tracker_data.mkdir(parents=True, exist_ok=True)

    summary = {}
    for seq, n_frames in seq_info.items():
        pkl = args.pred_dir / f"{seq}_person_masks.pkl"
        if not pkl.is_file():
            print(f"[miss] {pkl}")
            continue
        target_hw: tuple[int, int] | None = None
        gt_txt = GT / seq / "gt" / "gt.txt"
        if gt_txt.is_file():
            first = gt_txt.read_text(encoding="utf-8").splitlines()[0].split()
            # MOTS: time id class h w counts
            if len(first) >= 5:
                target_hw = (int(first[3]), int(first[4]))
        out_txt = tracker_data / f"{seq}.txt"
        stats = convert_pkl(pkl, out_txt, n_frames, frame_offset=1, target_hw=target_hw)
        summary[seq] = stats
        print(f"[ok] {seq}: {stats} target_hw={target_hw}")

    meta = tracker_data.parent / "convert_summary.json"
    meta.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {meta}")

    if args.skip_hota:
        return

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_hota.py"),
        "--benchmark",
        "mots_challenge",
        "--split",
        "train",
        "--tracker",
        args.tracker,
        "--gt-folder",
        str(GT),
        "--trackers-folder",
        str(TRACKERS),
        "--classes",
        "pedestrian",
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
