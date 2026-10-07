#!/usr/bin/env python3
"""Layout MOTS Challenge GT for TrackEval.

TrackEval expects (SKIP_SPLIT_FOL=True):
  outputs/benchmarks/ground_truth/mots_challenge/<seq>/gt/gt.txt

Source:
  Research_Data/MOTSChallenge/train/instances_txt/<seq>.txt
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEQMAP = Path(__file__).resolve().parent / "seqmaps" / "mots_challenge_train.seqmap"
SRC = ROOT / "Research_Data" / "MOTSChallenge" / "train" / "instances_txt"
OUT = ROOT / "outputs" / "benchmarks" / "ground_truth" / "mots_challenge"


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    seq_info = parse_seqmap(SEQMAP)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {}
    for seq, n_frames in seq_info.items():
        src = SRC / f"{seq}.txt"
        if not src.is_file():
            raise FileNotFoundError(src)
        dst_dir = args.out / seq / "gt"
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / "gt.txt"
        shutil.copy2(src, dst)
        n_lines = sum(1 for _ in dst.open(encoding="utf-8"))
        summary[seq] = {"n_frames": n_frames, "n_lines": n_lines, "path": str(dst)}
        print(f"[ok] {seq}: frames={n_frames} lines={n_lines} -> {dst}")

    meta = args.out / "prepare_summary.json"
    meta.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {meta}")


if __name__ == "__main__":
    main()
