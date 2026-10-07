#!/usr/bin/env python3
"""Run SAM 3.1 video tracking on an arbitrary image folder (KITTI or MOTS Challenge).

SAM 3.1 needs a prompt — text NP, or later point/box. Frames alone do nothing.

  python scripts/run_sam3_video_folder.py \\
    --frames Research_Data/MOTSChallenge/train/images/0002 \\
    --prompts person --name mots_chal_0002_person --fps 30
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# reuse helpers from KITTI runner
from run_sam3_kitti_video import (  # noqa: E402
    OUT_ROOT,
    collect_propagation,
    merge_prompt_masks,
    parse_prompts,
    save_payload,
    summarize_concepts,
    write_video,
)


def prepare_work_dir(
    src: Path,
    work_dir: Path,
    max_frames: int | None,
    *,
    max_side: int | None = None,
) -> list[Path]:
    frames = sorted(src.glob("*.jpg")) + sorted(src.glob("*.png"))
    if not frames:
        raise FileNotFoundError(f"no images in {src}")
    if max_frames is not None:
        frames = frames[:max_frames]
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    out_paths: list[Path] = []
    from PIL import Image

    for i, fr in enumerate(frames):
        # SAM3 image-folder loader expects contiguous 00000.jpg-style names
        dst = work_dir / f"{i:05d}.jpg"
        im = Image.open(fr).convert("RGB")
        if max_side is not None:
            w, h = im.size
            scale = max_side / max(w, h)
            if scale < 1.0:
                im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
        im.save(dst, quality=95)
        out_paths.append(dst)
    print(f"prepared {len(out_paths)} frames -> {work_dir} (from {src}, max_side={max_side})")
    return out_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True, help="Folder of jpg/png frames")
    parser.add_argument("--prompts", default="person")
    parser.add_argument("--prompts-file", type=Path, default=None)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--name", required=True)
    parser.add_argument("--max-frames", type=int, default=None, help="Smoke cap")
    parser.add_argument(
        "--max-objects",
        type=int,
        default=128,
        help="Maximum SAM 3.1 tracks retained in a session.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=None,
        help="If set, downscale frames so longest side <= this (saves VRAM on MOTS 1080p).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Directory for masks.pkl / concepts.json / mp4 (default: SAM3.1 kitti_mots/dev)",
    )
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Do not write overlay mp4 (HOTA batch).",
    )
    parser.add_argument(
        "--from-masks",
        type=Path,
        default=None,
        help="Skip SAM3; write overlay mp4 from an existing *_masks.pkl",
    )
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    import torch

    if "SAM3_DEVICE" not in os.environ:
        os.environ["SAM3_DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"

    out_root = args.out_dir if args.out_dir is not None else OUT_ROOT
    out_root.mkdir(parents=True, exist_ok=True)
    work = out_root / f"frames_{args.name}"
    out_mp4 = out_root / f"{args.name}_track.mp4"
    masks_path = out_root / f"{args.name}_masks.pkl"
    meta_json = out_root / f"{args.name}_concepts.json"

    if args.from_masks is not None:
        from run_sam3_kitti_video import load_payload

        frame_paths = prepare_work_dir(args.frames, work, args.max_frames, max_side=args.max_side)
        mask_dict, obj_to_concept, _meta = load_payload(args.from_masks)
        write_video(frame_paths, mask_dict, out_mp4, fps=args.fps, obj_to_concept=obj_to_concept)
        print(f"wrote {out_mp4} from {args.from_masks}")
        return

    prompts = parse_prompts(args.prompts, args.prompts_file)
    if not prompts:
        raise SystemExit("need at least one text prompt (NP). Frames alone are not enough.")

    frame_paths = prepare_work_dir(args.frames, work, args.max_frames, max_side=args.max_side)

    from sam3.model.device_utils import install_cuda_compat_shim
    from sam3.model_builder import build_sam3_predictor

    device = install_cuda_compat_shim()
    use_fa3 = bool(
        str(device).startswith("cuda")
        and torch.cuda.is_available()
        and torch.cuda.get_device_capability()[0] >= 9
    )
    print(
        f"device={device} fa3={use_fa3} prompts={prompts} "
        f"frames={len(frame_paths)}"
    )

    predictor = build_sam3_predictor(
        version="sam3.1",
        compile=False,
        async_loading_frames=False,
        warm_up=False,
        max_num_objects=args.max_objects,
        use_fa3=use_fa3,
    )
    if hasattr(predictor.model, "fill_hole_area"):
        predictor.model.fill_hole_area = 0
    if hasattr(predictor.model, "tracker") and hasattr(predictor.model.tracker, "fill_hole_area"):
        predictor.model.tracker.fill_hole_area = 0

    resp = predictor.handle_request(
        {
            "type": "start_session",
            "resource_path": str(work),
            "offload_video_to_cpu": True,
        }
    )
    session_id = resp["session_id"]
    print(f"session={session_id}")

    combined: dict = {}
    obj_to_concept: dict[int, str] = {}
    per_prompt_stats: dict[str, int] = {}

    for pi, concept in enumerate(prompts):
        print(f"\n=== [{pi + 1}/{len(prompts)}] add_prompt text={concept!r} ===")
        predictor.handle_request(
            {
                "type": "add_prompt",
                "session_id": session_id,
                "frame_index": 0,
                "text": concept,
                "output_prob_thresh": args.conf,
            }
        )
        print("propagate_in_video...")
        prompt_masks = collect_propagation(
            predictor, session_id, output_prob_thresh=args.conf
        )
        n_objs = len({oid for masks in prompt_masks.values() for oid in masks})
        per_prompt_stats[concept] = n_objs
        print(f"  -> {n_objs} unique objects for {concept!r}")
        merge_prompt_masks(combined, prompt_masks, concept, obj_to_concept)
        meta = {
            "model_version": "sam3.1",
            "max_objects": args.max_objects,
            "use_fa3": use_fa3,
            "frames": str(args.frames),
            "num_frames": len(frame_paths),
            "conf": args.conf,
            "prompts": prompts,
            "completed_prompts": prompts[: pi + 1],
            "per_prompt_object_counts": per_prompt_stats,
            "merged_object_counts": summarize_concepts(combined, obj_to_concept),
        }
        save_payload(masks_path, combined, obj_to_concept, meta)
        meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if args.skip_video:
        print("skip-video: masks/json already checkpointed; not writing mp4")
    else:
        write_video(frame_paths, combined, out_mp4, fps=args.fps, obj_to_concept=obj_to_concept)
    try:
        predictor.handle_request({"type": "close_session", "session_id": session_id})
    except Exception as exc:  # noqa: BLE001
        print(f"close_session warning: {exc}")
    print("done")


if __name__ == "__main__":
    main()
