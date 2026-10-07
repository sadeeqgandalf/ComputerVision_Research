#!/usr/bin/env python3
"""Run SAM 3.1 inference and HOTA evaluation on KITTI MOTS and MOTS Challenge."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], *, dry_run: bool) -> None:
    print("\n" + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True, cwd=ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=["both", "kitti_mots", "mots_challenge"],
        default="both",
    )
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--max-objects", type=int, default=128)
    parser.add_argument(
        "--mots-max-side",
        type=int,
        default=1024,
        help="MOTS Challenge inference resolution cap; use 0 for full resolution.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--write-videos", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    common_batch_args = [
        "--conf",
        str(args.conf),
        "--max-objects",
        str(args.max_objects),
    ]
    if args.skip_existing:
        common_batch_args.append("--skip-existing")
    if not args.write_videos:
        common_batch_args.append("--skip-video")

    if args.dataset in {"both", "kitti_mots"}:
        run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_sam3_kitti_mots_batch.py"),
                "--split",
                "val",
                *common_batch_args,
            ],
            dry_run=args.dry_run,
        )
        run(
            [
                sys.executable,
                str(ROOT / "eval" / "tracking" / "score_kitti_mots.py"),
                "--split",
                "val",
            ],
            dry_run=args.dry_run,
        )

    if args.dataset in {"both", "mots_challenge"}:
        mots_batch = [
            sys.executable,
            str(ROOT / "scripts" / "run_sam3_mots_challenge_batch.py"),
            *common_batch_args,
            "--max-side",
            str(args.mots_max_side),
        ]
        run(mots_batch, dry_run=args.dry_run)
        run(
            [
                sys.executable,
                str(ROOT / "eval" / "tracking" / "score_mots_challenge.py"),
            ],
            dry_run=args.dry_run,
        )


if __name__ == "__main__":
    main()
