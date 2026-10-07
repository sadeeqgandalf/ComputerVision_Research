"""Controlled occlusion synthesis for SAM3 stress-testing.

Design (intentional):
  - Do NOT relocate cars (flying-car paste looks fake).
  - Pick one primary pedestrian + one car (texture source).
  - Hide an exact height-fraction of the pedestrian (known ratio).
  - Fill hidden pixels with real car-mask pixels.
  - GT is exact by construction: visible / occluded partitions of target.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def tensor_masks_to_list(masks, height: int, width: int) -> list[np.ndarray]:
    if masks is None or len(masks) == 0:
        return []
    m = masks.detach().cpu().numpy()
    if m.ndim == 4:
        m = m[:, 0]
    out: list[np.ndarray] = []
    for i in range(m.shape[0]):
        mi = m[i] > 0.5
        if mi.shape != (height, width):
            mi = (
                np.array(
                    Image.fromarray((mi.astype(np.uint8) * 255)).resize(
                        (width, height), Image.NEAREST
                    )
                )
                > 127
            )
        if mi.any():
            out.append(mi.astype(bool))
    return out


def pick_largest(
    instances: list[np.ndarray],
    *,
    reject_if_centroid_below_frac: float | None = None,
) -> np.ndarray:
    if not instances:
        raise ValueError("no instances")
    h = instances[0].shape[0]
    scored: list[tuple[int, np.ndarray]] = []
    for inst in instances:
        ys, _xs = np.where(inst)
        if reject_if_centroid_below_frac is not None:
            if float(ys.mean()) > reject_if_centroid_below_frac * h:
                continue
        scored.append((int(inst.sum()), inst))
    if not scored:
        raise ValueError("no instances left after filters")
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[0][1]


def _bbox_hw(mask: np.ndarray) -> tuple[int, int]:
    ys, xs = np.where(mask)
    return int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)


def pick_primary_car(car_instances: list[np.ndarray]) -> np.ndarray:
    """Largest real vehicle; reject ego-hood strip and tall thin false positives."""
    if not car_instances:
        raise ValueError("no car instances")
    h = car_instances[0].shape[0]
    scored: list[tuple[int, np.ndarray]] = []
    for inst in car_instances:
        ys, xs = np.where(inst)
        cy = float(ys.mean())
        bh, bw = _bbox_hw(inst)
        aspect = bh / max(bw, 1)
        if cy > 0.82 * h:
            continue  # ego hood / dash
        if aspect > 1.2:
            continue  # likely a person mislabeled as car
        if bw < 40:
            continue
        scored.append((int(inst.sum()), inst))
    if not scored:
        return pick_largest(car_instances, reject_if_centroid_below_frac=0.82)
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[0][1]


def pick_primary_pedestrian(
    ped_instances: list[np.ndarray],
    car_instances: list[np.ndarray],
) -> np.ndarray:
    """Prefer a clear single person (tall aspect, moderate area), not a merged crowd blob."""
    if not ped_instances:
        raise ValueError("no pedestrian instances")
    car_union = (
        np.logical_or.reduce(car_instances)
        if car_instances
        else np.zeros_like(ped_instances[0])
    )
    scored: list[tuple[float, np.ndarray]] = []
    for inst in ped_instances:
        area = float(inst.sum())
        bh, bw = _bbox_hw(inst)
        aspect = bh / max(bw, 1)
        overlap = float((inst & car_union).sum()) / area
        if overlap > 0.25 or aspect < 1.3:
            continue
        # down-weight giant merged blobs
        size_term = area if area <= 12000 else 12000.0 * (12000.0 / area)
        scored.append((size_term * aspect, inst))
    if not scored:
        fallback = []
        for inst in ped_instances:
            overlap = float((inst & car_union).sum()) / float(inst.sum())
            fallback.append((overlap, -int(inst.sum()), inst))
        fallback.sort()
        return fallback[0][2]
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored[0][1]


def exact_ratio_cover(
    target: np.ndarray,
    ratio: float,
    mode: str = "bottom",
) -> np.ndarray:
    """Hide `ratio` of target pixels by vertical quantile (exact stress control)."""
    if not target.any():
        raise ValueError("empty target")
    ratio = float(np.clip(ratio, 0.05, 0.95))
    ys, xs = np.where(target)
    cover = np.zeros_like(target, dtype=bool)
    if mode == "bottom":
        y_cut = float(np.quantile(ys, 1.0 - ratio))
        sel = ys >= y_cut
    elif mode == "top":
        y_cut = float(np.quantile(ys, ratio))
        sel = ys <= y_cut
    else:
        raise ValueError(f"unknown mode={mode!r}")
    cover[ys[sel], xs[sel]] = True
    return cover


def fill_with_occluder_texture(
    rgb: np.ndarray,
    cover: np.ndarray,
    occluder: np.ndarray,
    seed: int = 0,  # noqa: ARG001
) -> np.ndarray:
    """Fill hidden ped pixels with smooth car-body color (readable, not speckled)."""
    hard = rgb.copy()
    if not cover.any() or not occluder.any():
        return hard
    # Use median car color — honest synthetic occluder, stable GT boundary.
    color = np.median(rgb[occluder], axis=0)
    hard[cover] = color
    return hard


def overlay_rgb(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int],
    alpha: float = 0.55,
) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    c = np.asarray(color, dtype=np.float32)
    out[mask] = out[mask] * (1.0 - alpha) + c * alpha
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_contour(
    rgb: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
) -> np.ndarray:
    out = rgb.copy()
    u8 = (mask.astype(np.uint8) * 255)
    cnts, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, color, thickness)
    return out


def synthesize_occlusion(
    rgb: np.ndarray,
    target: np.ndarray,
    occluder: np.ndarray,
    ratio: float = 0.5,
    mode: str = "bottom",
    seed: int = 0,
) -> dict[str, Any]:
    cover = exact_ratio_cover(target, ratio=ratio, mode=mode)
    visible = target & ~cover
    hard = fill_with_occluder_texture(rgb, cover, occluder, seed=seed)
    achieved = float(cover.sum() / max(int(target.sum()), 1))
    return {
        "hard_rgb": hard,
        "target_full": target.astype(bool),
        "target_visible": visible.astype(bool),
        "target_occluded": cover.astype(bool),
        "occluder_src": occluder.astype(bool),
        "ratio_requested": float(ratio),
        "ratio_achieved": achieved,
        "mode": mode,
    }


def run_from_processor(
    processor,
    image: Image.Image,
    image_np: np.ndarray,
    target_concept: str,
    occluder_concept: str,
    ratio: float = 0.5,
    conf_already_set: bool = True,  # noqa: ARG001
) -> dict[str, Any]:
    """Prompt SAM3, pick instances, synthesize. Returns occ dict (+ instance counts)."""
    h, w = image_np.shape[:2]
    state = processor.set_image(image)
    out_t = processor.set_text_prompt(state=state, prompt=target_concept)
    # Clone immediately — a later set_text_prompt can overwrite prior tensors.
    t_inst = tensor_masks_to_list(out_t["masks"].detach().clone(), h, w)

    out_o = processor.set_text_prompt(state=state, prompt=occluder_concept)
    o_inst = tensor_masks_to_list(out_o["masks"].detach().clone(), h, w)
    if not t_inst:
        raise RuntimeError(f"no masks for target={target_concept!r}")
    if not o_inst:
        raise RuntimeError(f"no masks for occluder={occluder_concept!r}")

    occluder = pick_primary_car(o_inst)
    target = pick_primary_pedestrian(t_inst, o_inst)

    occ = synthesize_occlusion(image_np, target, occluder, ratio=ratio, mode="bottom")
    occ["n_target_instances"] = len(t_inst)
    occ["n_occluder_instances"] = len(o_inst)
    occ["target_concept"] = target_concept
    occ["occluder_concept"] = occluder_concept
    occ["target_bbox_hw"] = _bbox_hw(target)
    occ["occluder_bbox_hw"] = _bbox_hw(occluder)
    return occ


def render_qc_figure(
    image_np: np.ndarray,
    occ: dict[str, Any],
    frame_name: str,
    out_path: Path | None = None,
) -> Path | None:
    target = occ["target_full"]
    visible = occ["target_visible"]
    cover = occ["target_occluded"]
    occluder = occ["occluder_src"]
    hard = occ["hard_rgb"]

    target_color = (0, 200, 90)
    occluder_color = (230, 90, 40)
    hidden_color = (80, 120, 255)

    h, w = image_np.shape[:2]
    ys, xs = np.where(target)
    pad = 40
    y0, y1 = max(0, int(ys.min()) - pad), min(h, int(ys.max()) + pad)
    x0, x1 = max(0, int(xs.min()) - pad), min(w, int(xs.max()) + pad)

    pre = overlay_rgb(image_np, target, target_color)
    pre = overlay_rgb(pre, occluder, occluder_color)

    plan = overlay_rgb(image_np, visible, target_color, 0.65)
    plan = overlay_rgb(plan, cover, hidden_color, 0.65)

    on_hard = draw_contour(hard, target, (255, 255, 255), 2)
    on_hard = overlay_rgb(on_hard, visible, target_color, 0.45)

    zoom = np.concatenate(
        [image_np[y0:y1, x0:x1], hard[y0:y1, x0:x1], plan[y0:y1, x0:x1]],
        axis=1,
    )

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        f"Exact-ratio occlusion | {frame_name} | "
        f"hide {occ['ratio_achieved']:.0%} ({occ['mode']}) of {occ.get('target_concept', 'target')}",
        fontsize=13,
    )
    panels = [
        (axes[0, 0], image_np, "Original"),
        (axes[0, 1], pre, "Target (green) + car texture src (orange)"),
        (axes[0, 2], hard, "Hard image"),
        (axes[1, 0], plan, "Plan on original: green stay / blue hide"),
        (axes[1, 1], on_hard, "Hard + full contour + visible"),
        (axes[1, 2], zoom, "Zoom: original | hard | plan"),
    ]
    for ax, img, title in panels:
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()

    saved = None
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        saved = out_path
    plt.show()
    plt.close(fig)
    return saved
