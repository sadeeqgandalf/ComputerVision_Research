#!/usr/bin/env python3
"""Batch SAM 3.1 video tracking on MOTS Challenge train (pedestrian only).

  python scripts/run_sam3_mots_challenge_batch.py --skip-existing --skip-video
  python scripts/run_sam3_mots_challenge_batch.py --seqs 0002 --max-frames 50  # smoke
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEQMAP = ROOT / "eval" / "tracking" / "seqmaps" / "mots_challenge_train.seqmap"
FRAMES_ROOT = ROOT / "Research_Data" / "MOTSChallenge" / "train" / "images"
OUT_ROOT = ROOT / "outputs" / "SAM3.1" / "tracking" / "raw" / "mots_challenge" / "train"
VIDEO_SCRIPT = ROOT / "scripts" / "run_sam3_video_folder.py"


def parse_seqmap(path: Path) -> dict[str, int]:
    """MOTS Challenge seqmap: start/end are 1-based frame ids → length = end-start+1."""
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
    parser.add_argument("--prompts", default="person")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--max-objects", type=int, default=128)
    parser.add_argument("--seqs", default=None, help="Comma-separated seq ids")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--max-side",
        type=int,
        default=1024,
        help="Downscale longest side (default 1024) to avoid A100 OOM on MOTS 1080p crowds.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    seq_info = parse_seqmap(SEQMAP)
    if args.seqs:
        wanted = {s.strip().zfill(4) for s in args.seqs.split(",") if s.strip()}
        seq_info = {k: v for k, v in seq_info.items() if k in wanted}
        missing = wanted - set(seq_info)
        if missing:
            raise SystemExit(f"unknown seqs: {sorted(missing)}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    total = sum(min(n, args.max_frames) if args.max_frames else n for n in seq_info.values())
    print(f"MOTS Challenge train: {len(seq_info)} seqs, ~{total} frames × person")
    print(f"output -> {OUT_ROOT}")

    manifest: dict = {
        "model_version": "sam3.1",
        "prompts": args.prompts,
        "conf": args.conf,
        "max_objects": args.max_objects,
        "seqs": {},
    }
    for seq, n_frames in seq_info.items():
        use_frames = min(n_frames, args.max_frames) if args.max_frames else n_frames
        name = f"{seq}_person"
        frames = FRAMES_ROOT / seq
        masks_path = OUT_ROOT / f"{name}_masks.pkl"
        manifest["seqs"][seq] = {"num_frames": use_frames, "masks": str(masks_path), "status": "pending"}
        if not frames.is_dir():
            print(f"[miss frames] {frames}")
            manifest["seqs"][seq]["status"] = "missing_frames"
            continue
        if args.skip_existing and masks_path.is_file():
            print(f"[skip] {seq}")
            manifest["seqs"][seq]["status"] = "skipped_existing"
            continue

        cmd = [
            sys.executable,
            str(VIDEO_SCRIPT),
            "--frames",
            str(frames),
            "--prompts",
            args.prompts,
            "--conf",
            str(args.conf),
            "--max-objects",
            str(args.max_objects),
            "--name",
            name,
            "--out-dir",
            str(OUT_ROOT),
        ]
        if args.max_frames:
            cmd += ["--max-frames", str(args.max_frames)]
        if args.max_side:
            cmd += ["--max-side", str(args.max_side)]
        if args.skip_video:
            cmd.append("--skip-video")

        print(f"\n=== {seq} frames={use_frames} ===")
        print(" ".join(cmd))
        if args.dry_run:
            manifest["seqs"][seq]["status"] = "dry_run"
            continue
        subprocess.run(cmd, check=True, cwd=str(ROOT))
        manifest["seqs"][seq]["status"] = "done"

    (OUT_ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("batch complete")


if __name__ == "__main__":
    main()
