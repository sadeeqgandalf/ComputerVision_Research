#!/usr/bin/env python3
"""Prepare official TrackEval GT for KITTI MOTS from instance PNGs.

Converts Research_Data/instances/<seq>/*.png -> MOTS txt RLE.

  python eval/tracking/prepare_kitti_mots_gt.py --split val
  python eval/tracking/prepare_kitti_mots_gt.py --split train
  python eval/tracking/prepare_kitti_mots_gt.py --split all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parents[2]
INSTANCES = ROOT / "Research_Data" / "instances"
SEQMAP_DIR = Path(__file__).resolve().parent / "seqmaps"
OUT_GT = ROOT / "outputs" / "benchmarks" / "ground_truth" / "kitti_mots"


def parse_seqmap(path: Path) -> dict[str, int]:
    """Return {seq: num_frames} using official last-frame index (inclusive) + 1."""
    info: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        seq = f"{int(parts[0]):04d}"
        last = int(parts[3])
        info[seq] = last + 1
    return info


def encode_mask(binary: np.ndarray) -> dict:
    return mask_utils.encode(np.asfortranarray(binary.astype(np.uint8)))


def convert_seq(seq: str, num_frames: int, out_txt: Path) -> dict:
    src = INSTANCES / seq
    if not src.is_dir():
        raise FileNotFoundError(src)

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    n_lines = 0
    class_counts: dict[str, int] = {}

    with out_txt.open("w", encoding="utf-8") as f:
        for t in range(num_frames):
            png = src / f"{t:06d}.png"
            if not png.is_file():
                continue
            img = cv2.imread(str(png), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError(f"failed to read {png}")
            for oid in (int(x) for x in np.unique(img) if int(x) != 0):
                class_id = oid // 1000
                if class_id not in (1, 2, 10):
                    continue
                binary = img == oid
                if not binary.any():
                    continue
                rle = encode_mask(binary)
                counts = rle["counts"]
                if isinstance(counts, bytes):
                    counts = counts.decode("utf-8")
                h, w = rle["size"]
                f.write(f"{t} {oid} {class_id} {h} {w} {counts}\n")
                n_lines += 1
                key = str(class_id)
                class_counts[key] = class_counts.get(key, 0) + 1

    return {"seq": seq, "num_frames": num_frames, "n_lines": n_lines, "class_counts": class_counts}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val", "all"], default="val")
    args = parser.parse_args()

    splits = ["train", "val"] if args.split == "all" else [args.split]
    summary: dict = {"splits": {}}

    for split in splits:
        seqmap = SEQMAP_DIR / f"kitti_mots_{split}.seqmap"
        seq_info = parse_seqmap(seqmap)
        # TrackEval expects evaluate_mots.seqmap.<split> next to GT or via SEQMAP_FILE
        dest_seqmap = OUT_GT / f"evaluate_mots.seqmap.{split}"
        dest_seqmap.parent.mkdir(parents=True, exist_ok=True)
        dest_seqmap.write_text(seqmap.read_text(encoding="utf-8"), encoding="utf-8")

        split_stats = {}
        for seq, n_frames in seq_info.items():
            out_txt = OUT_GT / "instances_txt" / f"{seq}.txt"
            stats = convert_seq(seq, n_frames, out_txt)
            split_stats[seq] = stats
            print(f"[{split}] {seq}: frames={n_frames} lines={stats['n_lines']} classes={stats['class_counts']}")

        summary["splits"][split] = {
            "seqmap": str(dest_seqmap),
            "n_seqs": len(seq_info),
            "seqs": split_stats,
        }

    out_json = OUT_GT / "prepare_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nGT ready under {OUT_GT}")
    print(f"summary -> {out_json}")


if __name__ == "__main__":
    main()
