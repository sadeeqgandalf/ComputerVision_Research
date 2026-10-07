#!/usr/bin/env python3
"""Batch SAM3 video tracking on RobMOTS BDD-MOTS train (driving).

Eval classes (RobMOTS clsmap): person, bicycle, car, motorcycle, bus, truck.
Traffic signs / lights are NOT in BDD-MOTS GT — do not expect HOTA for them.

  python scripts/run_sam3_bdd_mots_batch.py --skip-existing --skip-video
  python scripts/run_sam3_bdd_mots_batch.py --seqs 0000f77c-6257be58 --max-frames 40  # smoke
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAMES_ROOT = ROOT / "Research_Data" / "RobMOTS" / "train" / "bdd_mots"
GT_SEQMAP = (
    ROOT
    / "Research_Data"
    / "RobMOTS"
    / "train_gt_unzipped"
    / "data"
    / "gt"
    / "rob_mots"
    / "train"
    / "bdd_mots"
    / "seqmap.txt"
)
OUT_ROOT = ROOT / "outputs" / "SAM3.1" / "tracking" / "raw" / "bdd_mots" / "train"
VIDEO_SCRIPT = ROOT / "scripts" / "run_sam3_video_folder.py"

# Default: dominant ADAS classes. Full clsmap = person,bicycle,car,motorcycle,bus,truck
DEFAULT_PROMPTS = "person,car,truck,bus"


def parse_seqmap(path: Path) -> dict[str, tuple[int, int, int]]:
    """seq -> (num_frames, height, width)."""
    info: dict[str, tuple[int, int, int]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        seq, n, h, w = parts[0], int(parts[1]), int(parts[2]), int(parts[3])
        info[seq] = (n, h, w)
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompts", default=DEFAULT_PROMPTS)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--seqs", default=None, help="Comma-separated sequence ids")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--max-side",
        type=int,
        default=1024,
        help="Downscale longest side (BDD is 1280x720; 1024 is safe on A100).",
    )
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--seqmap", type=Path, default=GT_SEQMAP)
    parser.add_argument("--frames-root", type=Path, default=FRAMES_ROOT)
    parser.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    args = parser.parse_args()

    if not args.seqmap.is_file():
        raise SystemExit(f"seqmap missing: {args.seqmap}")

    seq_info = parse_seqmap(args.seqmap)
    if args.seqs:
        wanted = {s.strip() for s in args.seqs.split(",") if s.strip()}
        seq_info = {k: v for k, v in seq_info.items() if k in wanted}
        missing = wanted - set(seq_info)
        if missing:
            raise SystemExit(f"unknown seqs: {sorted(missing)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    total = sum(min(n, args.max_frames) if args.max_frames else n for n, _h, _w in seq_info.values())
    n_prompts = len([p for p in args.prompts.split(",") if p.strip()])
    print(f"BDD-MOTS train: {len(seq_info)} seqs, ~{total} frames × {n_prompts} prompts")
    print(f"frames -> {args.frames_root}")
    print(f"output -> {args.out_dir}")

    manifest: dict = {"prompts": args.prompts, "conf": args.conf, "seqs": {}}
    for seq, (n_frames, h, w) in seq_info.items():
        use_frames = min(n_frames, args.max_frames) if args.max_frames else n_frames
        name = f"{seq}_driving"
        frames = args.frames_root / seq
        masks_path = args.out_dir / f"{name}_masks.pkl"
        manifest["seqs"][seq] = {
            "num_frames": use_frames,
            "hw": [h, w],
            "masks": str(masks_path),
            "status": "pending",
        }
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
            "--name",
            name,
            "--out-dir",
            str(args.out_dir),
            "--fps",
            str(args.fps),
        ]
        if args.max_frames:
            cmd += ["--max-frames", str(args.max_frames)]
        if args.max_side:
            cmd += ["--max-side", str(args.max_side)]
        if args.skip_video:
            cmd.append("--skip-video")

        print(f"\n=== {seq} frames={use_frames} hw={h}x{w} ===")
        print(" ".join(cmd))
        if args.dry_run:
            manifest["seqs"][seq]["status"] = "dry_run"
            continue
        subprocess.run(cmd, check=True, cwd=str(ROOT))
        manifest["seqs"][seq]["status"] = "done"

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("batch complete")


if __name__ == "__main__":
    main()
