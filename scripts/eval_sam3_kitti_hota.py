#!/usr/bin/env python3
"""Deprecated: use the unified SAM 3.1 tracking eval pipeline instead.

This script previously wrote a duplicate KITTI HOTA tree under
``outputs/kitti_hota/``. That path has been retired.

Canonical workflow:

  # 1) Inference (SAM 3.1)
  python scripts/run_sam3_kitti_mots_batch.py --split val --skip-video

  # 2) Convert + HOTA
  python eval/tracking/score_kitti_mots.py --split val --tracker sam3_1_person_car

Artifacts land under::

  outputs/SAM3.1/tracking/{raw,predictions,results,metrics}/
  outputs/benchmarks/ground_truth/kitti_mots/
"""

from __future__ import annotations

import sys


def main() -> None:
    print(
        "eval_sam3_kitti_hota.py is retired.\n"
        "Use:\n"
        "  python scripts/run_sam3_kitti_mots_batch.py --split val --skip-video\n"
        "  python eval/tracking/score_kitti_mots.py --split val\n"
        "Outputs: outputs/SAM3.1/tracking/ and outputs/benchmarks/ground_truth/",
        file=sys.stderr,
    )
    raise SystemExit(2)


if __name__ == "__main__":
    main()
