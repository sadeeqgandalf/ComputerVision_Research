#!/usr/bin/env python3
"""Batch SAM 3.1 video tracking on KITTI MOTS official splits.

Runs person+car (open-vocab) on each sequence, saves masks.pkl for HOTA.

  # val (default) — official KITTI MOTS validation sequences
  python scripts/run_sam3_kitti_mots_batch.py --split val

  # train batches (non-eval sequences; see seqmaps/kitti_mots_train_b*.seqmap)
  python scripts/run_sam3_kitti_mots_batch.py --split train --batch b1
  python scripts/run_sam3_kitti_mots_batch.py --split train --batch b2
  python scripts/run_sam3_kitti_mots_batch.py --split train --batch b3

  # smoke: one short seq
  python scripts/run_sam3_kitti_mots_batch.py --split val --seqs 0014

  # resume (skip seqs that already have masks.pkl)
  python scripts/run_sam3_kitti_mots_batch.py --split val --skip-existing
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEQMAP_DIR = ROOT / "eval" / "tracking" / "seqmaps"
OUT_ROOT = ROOT / "outputs" / "SAM3.1" / "tracking" / "raw" / "kitti_mots"
VIDEO_SCRIPT = ROOT / "scripts" / "run_sam3_kitti_video.py"


def parse_seqmap(path: Path) -> dict[str, int]:
    info: dict[str, int] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 4:
            continue
        seq = f"{int(parts[0]):04d}"
        info[seq] = int(parts[3]) + 1
    return info


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument(
        "--batch",
        choices=["b1", "b2", "b3", "all"],
        default=None,
        help="Train-only chunk: b1 short, b2 medium, b3 long (seqmaps kitti_mots_train_b*.seqmap).",
    )
    parser.add_argument("--prompts", default="person,car")
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--max-objects", type=int, default=128)
    parser.add_argument("--seqs", default=None, help="Comma-separated seq ids (default: full split)")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Do not write overlay mp4 (faster / safer for HOTA batch).",
    )
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--video-width",
        type=int,
        default=1920,
        help="Overlay mp4 width (KITTI aspect preserved). Ignored with --skip-video.",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Cap frames per seq (smoke)")
    parser.add_argument(
        "--run-tag",
        default=None,
        help="Suffix for artifact names (default: c{conf} e.g. c05). Use '' for legacy untagged names.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.batch and args.split != "train":
        raise SystemExit("--batch only applies to --split train")
    if args.batch and args.seqs:
        raise SystemExit("use either --batch or --seqs, not both")

    conf_tag = args.run_tag
    if conf_tag is None:
        conf_tag = f"c{str(args.conf).replace('.', '')}"
    # Empty string => legacy untagged {seq}_person_car_masks.pkl

    if args.batch and args.batch != "all":
        seqmap_path = SEQMAP_DIR / f"kitti_mots_train_{args.batch}.seqmap"
    else:
        seqmap_path = SEQMAP_DIR / f"kitti_mots_{args.split}.seqmap"
    seq_info = parse_seqmap(seqmap_path)
    if args.seqs:
        wanted = {s.strip().zfill(4) for s in args.seqs.split(",") if s.strip()}
        seq_info = {k: v for k, v in seq_info.items() if k in wanted}
        missing = wanted - set(seq_info)
        if missing:
            raise SystemExit(f"unknown seqs for split {args.split}: {sorted(missing)}")

    # Isolate by conf so 0.3 and 0.5 runs do not overwrite each other.
    # Train batches share the same train_{tag} folder so HOTA can score the full set later.
    out_dir = OUT_ROOT / (f"{args.split}_{conf_tag}" if conf_tag else args.split)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_version": "sam3.1",
        "split": args.split,
        "batch": args.batch,
        "prompts": args.prompts,
        "conf": args.conf,
        "run_tag": conf_tag or None,
        "max_objects": args.max_objects,
        "seqs": {},
    }

    total_frames = sum(
        min(n, args.max_frames) if args.max_frames else n for n in seq_info.values()
    )
    n_prompts = len([p for p in args.prompts.split(",") if p.strip()])
    batch_label = f" batch={args.batch}" if args.batch else ""
    print(
        f"KITTI MOTS {args.split}{batch_label}: {len(seq_info)} seqs, ~{total_frames} frames × "
        f"{n_prompts} prompts = ~{total_frames * n_prompts} NP-passes"
    )
    print(f"conf={args.conf} tag={conf_tag or '(none)'} output -> {out_dir}")
    print(f"seqs: {', '.join(seq_info)}")

    for seq, n_frames in seq_info.items():
        use_frames = min(n_frames, args.max_frames) if args.max_frames else n_frames
        name = f"{seq}_person_car_{conf_tag}" if conf_tag else f"{seq}_person_car"
        masks_path = out_dir / f"{name}_masks.pkl"
        manifest["seqs"][seq] = {
            "num_frames": use_frames,
            "masks": str(masks_path),
            "status": "pending",
        }
        if args.skip_existing and masks_path.is_file():
            print(f"[skip] {seq} ({masks_path.name} exists)")
            manifest["seqs"][seq]["status"] = "skipped_existing"
            continue

        cmd = [
            sys.executable,
            "-u",
            str(VIDEO_SCRIPT),
            "--seq",
            seq,
            "--start",
            "0",
            "--num-frames",
            str(use_frames),
            "--prompts",
            args.prompts,
            "--conf",
            str(args.conf),
            "--max-objects",
            str(args.max_objects),
            "--name",
            name,
            "--out-dir",
            str(out_dir),
        ]
        if args.skip_video:
            cmd.append("--skip-video")
        else:
            cmd.extend(
                ["--fps", str(args.fps), "--video-width", str(args.video_width)]
            )
        print(f"\n=== {seq} frames={use_frames} conf={args.conf} ===")
        print(" ".join(cmd))
        if args.dry_run:
            manifest["seqs"][seq]["status"] = "dry_run"
            continue

        subprocess.run(cmd, check=True, cwd=str(ROOT))
        if not masks_path.is_file():
            raise FileNotFoundError(masks_path)
        manifest["seqs"][seq]["status"] = "done"
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"saved {masks_path}")

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nbatch complete")
    print(f"manifest -> {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
