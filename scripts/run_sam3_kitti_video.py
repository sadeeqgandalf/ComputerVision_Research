#!/usr/bin/env python3
"""Run SAM3 video PCS + tracking on a KITTI sequence (single- or multi-NP).

SAM3 video holds one text concept at a time; multi-NP runs each prompt then
merges masklets (same pattern as SAM3's eval loop).

Mac-safe (CPU by default). Examples:

  # single concept
  python scripts/run_sam3_kitti_video.py --seq 0019 --prompts pedestrian \\
    --start 188 --num-frames 40

  # exploratory multi-NP (what the eye might miss)
  python scripts/run_sam3_kitti_video.py --seq 0019 \\
    --prompts "person,pedestrian,bicycle,umbrella,storefront,car,backpack,traffic light" \\
    --start 188 --num-frames 20 --conf 0.3
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
KITTI_IMG = ROOT / "Research_Data/data_tracking_image_2/training/image_02"
OUT_ROOT = ROOT / "outputs/kitti_sam3_video"

DEFAULT_PROMPTS = [
    "person",
    "pedestrian",
    "bicycle",
    "umbrella",
    "storefront",
    "car",
    "backpack",
    "traffic light",
]

MASK_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
    (0, 128, 255),
    (255, 64, 128),
    (64, 255, 128),
    (128, 255, 64),
]


def parse_prompts(raw: str | None, prompts_file: Path | None) -> list[str]:
    items: list[str] = []
    if prompts_file is not None:
        text = prompts_file.read_text(encoding="utf-8")
        for line in text.splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                items.extend(p.strip() for p in line.split(",") if p.strip())
    if raw:
        items.extend(p.strip() for p in raw.split(",") if p.strip())
    # de-dupe, preserve order
    seen: set[str] = set()
    out: list[str] = []
    for p in items:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def slugify_prompts(prompts: list[str], max_len: int = 60) -> str:
    parts = [re.sub(r"[^a-z0-9]+", "", p.lower()) or "np" for p in prompts]
    if len(prompts) == 1:
        return parts[0]
    joined = "+".join(parts)
    if len(joined) <= max_len:
        return joined
    return f"multi{len(prompts)}_{parts[0]}"


def prepare_frame_dir(seq: str, start: int, num_frames: int, work_dir: Path) -> Path:
    src = KITTI_IMG / seq
    if not src.is_dir():
        raise FileNotFoundError(src)
    frames = sorted(src.glob("*.png"))
    if not frames:
        raise RuntimeError(f"No frames in {src}")
    end = min(start + num_frames, len(frames))
    subset = frames[start:end]
    if not subset:
        raise RuntimeError(f"Empty subset start={start} num={num_frames} len={len(frames)}")

    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    for i, fr in enumerate(subset):
        dst = work_dir / f"{i:05d}.jpg"
        Image.open(fr).convert("RGB").save(dst, quality=95)
    print(f"prepared {len(subset)} frames -> {work_dir} (KITTI {seq} [{start}:{end}])")
    return work_dir


def overlay_masks(
    frame_rgb: np.ndarray,
    masks_by_obj: dict,
    obj_to_concept: dict[int, str] | None = None,
    alpha: float = 0.45,
) -> np.ndarray:
    out = frame_rgb.astype(np.float32).copy()
    for obj_id, mask in sorted(masks_by_obj.items()):
        color = MASK_COLORS[int(obj_id) % len(MASK_COLORS)]
        m = mask.astype(bool)
        if m.ndim == 3:
            m = m[0]
        if m.shape[:2] != out.shape[:2]:
            m = cv2.resize(
                m.astype(np.uint8),
                (out.shape[1], out.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        for c in range(3):
            out[:, :, c] = np.where(
                m, out[:, :, c] * (1 - alpha) + color[c] * alpha, out[:, :, c]
            )
        ys, xs = np.where(m)
        if len(xs):
            cx, cy = int(xs.mean()), int(ys.mean())
            concept = (obj_to_concept or {}).get(int(obj_id), "")
            short = (concept[:10] + "…") if len(concept) > 10 else concept
            label = f"{int(obj_id)}:{short}" if short else str(int(obj_id))
            cv2.putText(
                out,
                label,
                (cx, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
    return out.astype(np.uint8)


def collect_propagation(predictor, session_id: str) -> dict:
    mask_dict: dict = {}
    for response in predictor.handle_stream_request(
        {"type": "propagate_in_video", "session_id": session_id}
    ):
        frame_idx = response.get("frame_index")
        if frame_idx is None:
            continue
        outputs = response.get("outputs", {})
        obj_ids = outputs.get("out_obj_ids", [])
        binary_masks = outputs.get("out_binary_masks")
        if binary_masks is None:
            mask_dict[frame_idx] = {}
            continue
        if isinstance(obj_ids, torch.Tensor):
            obj_ids = obj_ids.cpu().numpy()
        if isinstance(binary_masks, torch.Tensor):
            binary_masks = binary_masks.cpu().numpy()
        masks = {}
        for i, oid in enumerate(obj_ids):
            m = binary_masks[i]
            if m.ndim == 3:
                m = m[0]
            masks[int(oid)] = m
        mask_dict[frame_idx] = masks
        if frame_idx % 5 == 0:
            print(f"  frame {frame_idx}: {len(masks)} objects")
    return mask_dict


def merge_prompt_masks(
    combined: dict,
    prompt_masks: dict,
    concept: str,
    obj_to_concept: dict[int, str],
) -> int:
    """Remap local obj ids into a global space; return next free id."""
    used = set(obj_to_concept.keys())
    for frame_masks in combined.values():
        used.update(frame_masks.keys())
    start_id = (max(used) + 1) if used else 0

    local_ids: set[int] = set()
    for masks in prompt_masks.values():
        local_ids.update(masks.keys())
    local_to_global = {lid: start_id + i for i, lid in enumerate(sorted(local_ids))}

    for frame_idx, masks in prompt_masks.items():
        bucket = combined.setdefault(int(frame_idx), {})
        for lid, mask in masks.items():
            gid = local_to_global[lid]
            bucket[gid] = mask
            obj_to_concept[gid] = concept
    return start_id + len(local_to_global)


def resolve_frame_paths(
    frame_dir: Path | None,
    seq: str,
    start: int,
    num_frames: int,
) -> list[Path]:
    if frame_dir is not None and frame_dir.is_dir():
        paths = sorted(frame_dir.glob("*.jpg")) + sorted(frame_dir.glob("*.png"))
        if paths:
            return paths
    src = KITTI_IMG / seq
    frames = sorted(src.glob("*.png"))
    end = min(start + num_frames, len(frames))
    subset = frames[start:end]
    if not subset:
        raise RuntimeError(f"no frames for {seq} [{start}:{end}]")
    return subset


def save_payload(path: Path, mask_dict: dict, obj_to_concept: dict[int, str], meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        pickle.dump(
            {"masks": mask_dict, "obj_to_concept": obj_to_concept, "meta": meta},
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"saved masks+meta -> {path}")


def load_payload(path: Path) -> tuple[dict, dict[int, str], dict]:
    with path.open("rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict) and "masks" in data:
        return data["masks"], {int(k): v for k, v in data.get("obj_to_concept", {}).items()}, data.get("meta", {})
    # legacy: bare mask dict
    return data, {}, {}


def write_video(
    frame_paths: list[Path],
    mask_dict: dict,
    out_mp4: Path,
    fps: int = 10,
    obj_to_concept: dict[int, str] | None = None,
) -> None:
    if not frame_paths:
        raise RuntimeError("no frames to write")
    first = cv2.imread(str(frame_paths[0]))
    if first is None:
        raise RuntimeError(f"failed to read {frame_paths[0]}")
    h, w = first.shape[:2]
    out_mp4.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        for i, p in enumerate(frame_paths):
            bgr = cv2.imread(str(p))
            if bgr is None:
                raise RuntimeError(f"failed to read {p}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            overlay = overlay_masks(rgb, mask_dict.get(i, {}), obj_to_concept)
            cv2.imwrite(str(tmp_dir / f"{i:05d}.jpg"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            cmd = [
                ffmpeg,
                "-y",
                "-framerate",
                str(fps),
                "-i",
                str(tmp_dir / "%05d.jpg"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(out_mp4),
            ]
            subprocess.run(cmd, check=True, capture_output=True)
        else:
            writer = cv2.VideoWriter(
                str(out_mp4),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (w, h),
            )
            for i in range(len(frame_paths)):
                writer.write(cv2.imread(str(tmp_dir / f"{i:05d}.jpg")))
            writer.release()

    preview = out_mp4.with_name(out_mp4.stem + "_preview.jpg")
    mid_i = len(frame_paths) // 2
    bgr = cv2.imread(str(frame_paths[mid_i]))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    overlay = overlay_masks(rgb, mask_dict.get(mid_i, {}), obj_to_concept)
    cv2.imwrite(str(preview), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"saved {out_mp4}")
    print(f"saved {preview}")


def summarize_concepts(mask_dict: dict, obj_to_concept: dict[int, str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for oid, concept in obj_to_concept.items():
        appeared = any(oid in masks for masks in mask_dict.values())
        if appeared:
            counts[concept] = counts.get(concept, 0) + 1
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", default="0019")
    parser.add_argument(
        "--prompts",
        default=None,
        help="Comma-separated NPs. Default: exploratory street set.",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Alias for a single NP (compat).",
    )
    parser.add_argument(
        "--prompts-file",
        type=Path,
        default=None,
        help="Text file with NPs (comma or newline separated).",
    )
    parser.add_argument("--start", type=int, default=188, help="start frame in KITTI seq")
    parser.add_argument("--num-frames", type=int, default=20)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument(
        "--from-masks",
        type=Path,
        default=None,
        help="Skip inference; write video from a saved masks pickle",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Output stem override (default derived from prompts).",
    )
    args = parser.parse_args()

    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    # Prefer CUDA when present; only fall back to CPU if nothing else is set.
    if "SAM3_DEVICE" not in os.environ:
        try:
            import torch

            os.environ["SAM3_DEVICE"] = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            os.environ["SAM3_DEVICE"] = "cpu"

    raw = args.prompts
    if args.prompt:
        raw = (raw + "," if raw else "") + args.prompt
    prompts = parse_prompts(raw, args.prompts_file)
    if not prompts:
        prompts = list(DEFAULT_PROMPTS)

    stem = args.name or f"{args.seq}_{slugify_prompts(prompts)}"
    out_mp4 = OUT_ROOT / f"{stem}_track.mp4"
    masks_path = OUT_ROOT / f"{stem}_masks.pkl"
    meta_json = OUT_ROOT / f"{stem}_concepts.json"
    work = OUT_ROOT / f"frames_{args.seq}_{args.start}_{args.num_frames}"

    if args.from_masks is not None:
        mask_dict, obj_to_concept, _meta = load_payload(args.from_masks)
        paths = resolve_frame_paths(work, args.seq, args.start, args.num_frames)
        print(f"write-only: {len(mask_dict)} mask frames, {len(paths)} images")
        write_video(paths, mask_dict, out_mp4, fps=args.fps, obj_to_concept=obj_to_concept)
        print("done")
        return

    from sam3.model.device_utils import install_cuda_compat_shim

    device = install_cuda_compat_shim()
    print(f"device={device} seq={args.seq} prompts={prompts} conf={args.conf}")
    print(
        f"note: SAM3 runs ONE NP at a time; {len(prompts)} prompts × "
        f"{args.num_frames} frames ≈ {len(prompts) * args.num_frames * 25 / 60:.0f} min on CPU"
    )

    frame_dir = prepare_frame_dir(args.seq, args.start, args.num_frames, work)

    from sam3.model_builder import build_sam3_predictor

    print("Loading SAM3 video predictor...")
    predictor = build_sam3_predictor(
        version="sam3",
        compile=False,
        async_loading_frames=False,
        warm_up=False,
    )
    if hasattr(predictor.model, "fill_hole_area"):
        predictor.model.fill_hole_area = 0
    if hasattr(predictor.model, "tracker") and hasattr(
        predictor.model.tracker, "fill_hole_area"
    ):
        predictor.model.tracker.fill_hole_area = 0

    resp = predictor.handle_request(
        {
            "type": "start_session",
            "resource_path": str(frame_dir),
            "offload_video_to_cpu": True,
            "offload_state_to_cpu": True,
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
        prompt_masks = collect_propagation(predictor, session_id)
        n_objs = len({oid for masks in prompt_masks.values() for oid in masks})
        per_prompt_stats[concept] = n_objs
        print(f"  -> {n_objs} unique objects for {concept!r}")
        merge_prompt_masks(combined, prompt_masks, concept, obj_to_concept)

        meta = {
            "seq": args.seq,
            "start": args.start,
            "num_frames": args.num_frames,
            "conf": args.conf,
            "prompts": prompts,
            "completed_prompts": prompts[: pi + 1],
            "per_prompt_object_counts": per_prompt_stats,
            "merged_object_counts": summarize_concepts(combined, obj_to_concept),
        }
        save_payload(masks_path, combined, obj_to_concept, meta)
        meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        # refresh video after each NP so partial results are viewable
        paths = resolve_frame_paths(frame_dir, args.seq, args.start, args.num_frames)
        write_video(paths, combined, out_mp4, fps=args.fps, obj_to_concept=obj_to_concept)

    print("\n=== concept hit summary (unique tracked objs) ===")
    for concept, n in per_prompt_stats.items():
        print(f"  {concept!r}: {n}")

    try:
        predictor.handle_request({"type": "close_session", "session_id": session_id})
    except Exception as exc:  # noqa: BLE001
        print(f"close_session warning (ignored): {exc}")
    print("done")


if __name__ == "__main__":
    main()
