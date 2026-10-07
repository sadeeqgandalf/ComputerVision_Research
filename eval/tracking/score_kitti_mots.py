#!/usr/bin/env python3
"""Convert SAM 3.1 KITTI MOTS mask pickles -> TrackEval tracker txt + run HOTA.

  python eval/tracking/score_kitti_mots.py --split val --tracker sam3_1_person_car
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_spec import CONCEPT_TO_CLASS  # noqa: E402

SEQMAP_DIR = Path(__file__).resolve().parent / "seqmaps"
PRED_ROOT = ROOT / "outputs" / "SAM3.1" / "tracking" / "raw" / "kitti_mots"
TRACKERS = ROOT / "outputs" / "SAM3.1" / "tracking" / "predictions"
GT = ROOT / "outputs" / "benchmarks" / "ground_truth" / "kitti_mots"


def parse_seqmap(path: Path) -> dict[str, int]:
    info: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        info[f"{int(parts[0]):04d}"] = int(parts[3]) + 1
    return info


def encode_mask(binary: np.ndarray) -> dict:
    return mask_utils.encode(np.asfortranarray(binary.astype(np.uint8)))


def resolve_overlaps(masks: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
    if len(masks) <= 1:
        return {oid: (m > 0).astype(np.uint8) for oid, m in masks.items()}
    items = []
    for oid, m in masks.items():
        bin_m = (m > 0).astype(np.uint8)
        items.append((int(bin_m.sum()), oid, bin_m))
    items.sort(key=lambda x: x[0])
    h, w = items[0][2].shape
    claimed = np.zeros((h, w), dtype=np.uint8)
    out: dict[int, np.ndarray] = {}
    for _a, oid, bin_m in items:
        keep = bin_m & (1 - claimed)
        if keep.any():
            out[oid] = keep
            claimed |= keep
    return out


def concept_to_class(concept: str) -> int | None:
    c = concept.lower()
    if c in CONCEPT_TO_CLASS:
        return CONCEPT_TO_CLASS[c]
    for k, v in CONCEPT_TO_CLASS.items():
        if k in c:
            return v
    return None


def convert_pkl(masks_pkl: Path, out_txt: Path, num_frames: int) -> dict:
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
    class_local: dict[int, dict[int, int]] = {1: {}, 2: {}}
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
                typed[oid] = (cid, m)
            if not typed:
                continue
            resolved = resolve_overlaps({oid: m for oid, (_c, m) in typed.items()})
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
                f.write(f"{t} {mots_id} {cid} {h} {w} {counts}\n")
                n_lines += 1
    return {"n_lines": n_lines, "tracks": {str(c): len(m) for c, m in class_local.items()}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val")
    parser.add_argument("--tracker", default="sam3_1_person_car")
    parser.add_argument("--pred-dir", type=Path, default=None)
    parser.add_argument("--conf", type=float, default=None, help="Inference conf for scoreboard meta")
    parser.add_argument("--skip-hota", action="store_true")
    args = parser.parse_args()

    pred_dir = args.pred_dir or (PRED_ROOT / args.split)
    seq_info = parse_seqmap(SEQMAP_DIR / f"kitti_mots_{args.split}.seqmap")
    tracker_data = TRACKERS / "kitti_mots" / args.split / args.tracker / "data"
    tracker_data.mkdir(parents=True, exist_ok=True)

    summary = {}
    for seq, n_frames in seq_info.items():
        # Prefer conf-tagged names when present: 0014_person_car_c05_masks.pkl
        tagged = None
        if args.conf is not None:
            tag = f"c{str(args.conf).replace('.', '')}"
            cand = pred_dir / f"{seq}_person_car_{tag}_masks.pkl"
            if cand.is_file():
                tagged = cand
        pkl = tagged or (pred_dir / f"{seq}_person_car_masks.pkl")
        if not pkl.is_file():
            print(f"[miss] {pkl}")
            continue
        out_txt = tracker_data / f"{seq}.txt"
        stats = convert_pkl(pkl, out_txt, n_frames)
        summary[seq] = stats
        print(f"[ok] {seq}: {stats} <- {pkl.name}")

    meta = tracker_data.parent / "convert_summary.json"
    meta.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {meta}")

    if args.skip_hota:
        return

    # Delegate to run_hota.py
    import subprocess

    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent / "run_hota.py"),
        "--benchmark",
        "kitti_mots",
        "--split",
        args.split,
        "--tracker",
        args.tracker,
        "--gt-folder",
        str(GT),
        "--trackers-folder",
        str(TRACKERS / "kitti_mots" / args.split),
        "--model",
        "sam3.1",
        "--prompts",
        "person,car",
    ]
    if args.conf is not None:
        cmd.extend(["--conf", str(args.conf)])
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
