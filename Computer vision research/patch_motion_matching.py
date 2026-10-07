"""
Patch Motion Matching
---------------------
Match local image patches across consecutive frames to estimate short-range
motion (dx, dy). Pure NumPy + Pillow (no OpenCV required).

Typical use
  field = patch_motion_field(frame_t, frame_t1, PatchMatchConfig(...))
  fig = visualize_motion(frame_t, frame_t1, field, centers)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Config / results
# ---------------------------------------------------------------------------


@dataclass
class PatchMatchConfig:
    patch_size: int = 21          # odd preferred
    search_radius: int = 18       # pixels each side in t+1
    stride: int = 32              # grid spacing of patch centers
    metric: str = "ncc"           # "sad" | "ssd" | "ncc"
    search_step: int = 1          # subsample search (2 = 2x faster, coarser)
    max_patches: Optional[int] = None  # optional cap for demos


@dataclass
class MotionField:
    """Per-patch motion from frame t → t+1."""

    vectors: np.ndarray   # (N, 2) float — (dx, dy) in pixels
    scores: np.ndarray    # (N,) float — metric at best match
    centers: np.ndarray   # (N, 2) int — (y, x) of patch centers in frame t
    patch_size: int
    stride: int
    metric: str


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def load_gray(path: Path | str) -> np.ndarray:
    """Load image as float32 grayscale in [0, 1]."""
    img = Image.open(path).convert("L")
    return np.asarray(img, dtype=np.float32) / 255.0


def load_rgb(path: Path | str) -> np.ndarray:
    """Load image as uint8 RGB."""
    return np.asarray(Image.open(path).convert("RGB"))


def list_kitti_frames(seq_dir: Path | str) -> List[Path]:
    """Sorted PNG/JPG frames in a KITTI tracking sequence folder."""
    seq_dir = Path(seq_dir)
    frames = sorted(seq_dir.glob("*.png")) + sorted(seq_dir.glob("*.jpg"))
    return frames


def default_kitti_seq(
    repo_root: Optional[Path] = None,
    seq: str = "0019",
) -> Path:
    root = repo_root or Path(__file__).resolve().parents[1]
    return root / "Research_Data" / "data_tracking_image_2" / "training" / "image_02" / seq


# ---------------------------------------------------------------------------
# Core matching
# ---------------------------------------------------------------------------


def extract_patch(frame: np.ndarray, y: int, x: int, size: int) -> np.ndarray:
    """Extract a size×size patch with top-left chosen so center ≈ (y, x)."""
    r = size // 2
    return frame[y - r : y - r + size, x - r : x - r + size]


def _score_sad(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).sum())


def _score_ssd(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    return float((d * d).sum())


def _score_ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Negative NCC so that *lower* is always better (consistent with SAD/SSD)."""
    a0 = a - a.mean()
    b0 = b - b.mean()
    denom = float(np.linalg.norm(a0) * np.linalg.norm(b0)) + 1e-8
    return float(-(a0 * b0).sum() / denom)


_METRICS = {
    "sad": _score_sad,
    "ssd": _score_ssd,
    "ncc": _score_ncc,
}


def match_patch(
    patch: np.ndarray,
    search_region: np.ndarray,
    metric: str = "ncc",
    search_step: int = 1,
) -> Tuple[int, int, float]:
    """
    Find best location of `patch` inside `search_region`.

    Returns
    -------
    dy, dx, score
        Offset of the match relative to the top-left of search_region,
        plus the metric score (lower = better for all metrics here).
    """
    dy, dx, score, _cost = match_patch_with_costmap(
        patch, search_region, metric=metric, search_step=search_step
    )
    return dy, dx, score


def match_patch_with_costmap(
    patch: np.ndarray,
    search_region: np.ndarray,
    metric: str = "ncc",
    search_step: int = 1,
) -> Tuple[int, int, float, np.ndarray]:
    """
    Same as match_patch, but also returns the full matching-cost map.

    cost_map[y, x] = metric(patch, search_region[y:y+ph, x:x+pw])
    Unvisited (skipped by search_step) cells are NaN.
    """
    if metric not in _METRICS:
        raise ValueError(f"unknown metric {metric!r}; choose from {list(_METRICS)}")
    scorer = _METRICS[metric]
    ph, pw = patch.shape[:2]
    sh, sw = search_region.shape[:2]
    if sh < ph or sw < pw:
        raise ValueError("search_region smaller than patch")

    cost_h = sh - ph + 1
    cost_w = sw - pw + 1
    cost_map = np.full((cost_h, cost_w), np.nan, dtype=np.float32)

    best = (0, 0, float("inf"))
    step = max(1, int(search_step))
    for y in range(0, cost_h, step):
        for x in range(0, cost_w, step):
            cand = search_region[y : y + ph, x : x + pw]
            s = scorer(patch, cand)
            cost_map[y, x] = s
            if s < best[2]:
                best = (y, x, s)
    return best[0], best[1], best[2], cost_map


@dataclass
class SinglePatchMatch:
    """Everything needed to teach / debug one patch correspondence."""

    center_yx: Tuple[int, int]          # patch center in frame 1
    patch: np.ndarray                   # input patch (frame 1)
    search_window: np.ndarray           # cropped search region (frame 2)
    search_origin_yx: Tuple[int, int]   # top-left of search window in frame 2
    best_patch: np.ndarray              # best-matching patch (frame 2)
    best_center_yx: Tuple[int, int]     # center of best match in frame 2
    displacement_xy: Tuple[float, float]  # (dx, dy) = best - source
    cost: float
    cost_map: np.ndarray
    metric: str
    patch_size: int
    search_radius: int


def match_single_patch(
    frame1: np.ndarray,
    frame2: np.ndarray,
    center_yx: Tuple[int, int],
    *,
    patch_size: int = 11,
    search_radius: int = 16,
    metric: str = "ssd",
    search_step: int = 1,
) -> SinglePatchMatch:
    """
    MIT Vision Book Algorithm 46.1 — patch matching for ONE location:

        Descriptor C = 3×(2s+1)×(2s+1)  — RGB patch ℓ[n−s:n+s, m−s:m+s]
        Search only inside a local L×L neighborhood in frame 2
        Cost = Euclidean distance (SSD) between patches
        Motion = argmin over that neighborhood

    Book example: s=5 → C = 3×11×11; L=16 local search.
    Pass RGB HxWx3. `patch_size` = 2s+1; `search_radius` ≈ L (half-width of the
    candidate-center range we scan — same role as L in the book).
    """
    frame1 = np.asarray(frame1, dtype=np.float32)
    frame2 = np.asarray(frame2, dtype=np.float32)
    if frame1.max() > 1.5:
        frame1 = frame1 / 255.0
    if frame2.max() > 1.5:
        frame2 = frame2 / 255.0
    h, w = frame1.shape[:2]
    cy, cx = int(center_yx[0]), int(center_yx[1])
    r = patch_size // 2

    patch = extract_patch(frame1, cy, cx, patch_size)
    expected = (patch_size, patch_size) if frame1.ndim == 2 else (patch_size, patch_size, frame1.shape[2])
    if patch.shape != expected:
        raise ValueError(f"patch out of bounds at center={(cy, cx)} got {patch.shape}")

    y0 = max(0, cy - r - search_radius)
    x0 = max(0, cx - r - search_radius)
    y1 = min(h, cy - r + patch_size + search_radius)
    x1 = min(w, cx - r + patch_size + search_radius)
    window = frame2[y0:y1, x0:x1]

    dy_tl, dx_tl, cost, cost_map = match_patch_with_costmap(
        patch, window, metric=metric, search_step=search_step
    )
    best_y = y0 + dy_tl + r
    best_x = x0 + dx_tl + r
    best = extract_patch(frame2, best_y, best_x, patch_size)

    return SinglePatchMatch(
        center_yx=(cy, cx),
        patch=patch,
        search_window=window,
        search_origin_yx=(y0, x0),
        best_patch=best,
        best_center_yx=(best_y, best_x),
        displacement_xy=(float(best_x - cx), float(best_y - cy)),
        cost=float(cost),
        cost_map=cost_map,
        metric=metric,
        patch_size=patch_size,
        search_radius=search_radius,
    )


def _as_display(img: np.ndarray) -> np.ndarray:
    """Normalize float image for imshow (gray or RGB)."""
    x = np.asarray(img, dtype=np.float32)
    if x.max() > 1.5:
        x = x / 255.0
    return np.clip(x, 0.0, 1.0)


def _luma(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]


def visualize_single_patch_match(
    rgb1: np.ndarray,
    rgb2: np.ndarray,
    result: SinglePatchMatch,
):
    """
    Teaching figure aligned with MIT Vision Book Figure 46.4 / Algorithm 46.1:

      row0: frame1 + input patch | frame2 + search window + best match
      row1: input patch | best-match patch | matching cost (bright = low cost)
      row2: mid-row pixel chart | input↔match scatter
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    cy, cx = result.center_yx
    by, bx = result.best_center_yx
    y0, x0 = result.search_origin_yx
    r = result.patch_size // 2
    dx, dy = result.displacement_xy
    sh, sw = result.search_window.shape[:2]
    s = r  # book: spatial extent (2s+1); full descriptor C = 3×(2s+1)×(2s+1)
    L = result.search_radius  # book: local L×L search neighborhood
    ch = 3 if result.patch.ndim == 3 else 1

    fig = plt.figure(figsize=(12.5, 10.2))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.2, 0.95, 0.9], hspace=0.38, wspace=0.30)

    # --- frame 1 ---
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(rgb1)
    ax.add_patch(
        Rectangle(
            (cx - r, cy - r), result.patch_size, result.patch_size,
            fill=False, edgecolor="lime", linewidth=2,
        )
    )
    ax.set_title(
        f"Frame 1 — input patch  C={ch}×{result.patch_size}×{result.patch_size}  (s={s})"
    )
    ax.axis("off")

    # --- frame 2 ---
    ax = fig.add_subplot(gs[0, 1:])
    ax.imshow(rgb2)
    ax.add_patch(
        Rectangle(
            (x0, y0), sw, sh,
            fill=False, edgecolor="cyan", linewidth=2,
        )
    )
    ax.add_patch(
        Rectangle(
            (bx - r, by - r), result.patch_size, result.patch_size,
            fill=False, edgecolor="magenta", linewidth=2,
        )
    )
    ax.plot([cx, bx], [cy, by], "y-", lw=1.5)
    ax.plot(cx, cy, "g+", markersize=10)
    ax.plot(bx, by, "m+", markersize=10)
    ax.set_title(
        f"Frame 2 — L×L search (cyan) + best match (magenta)\n"
        f"L={L}  Δ=(dx,dy)=({dx:.0f},{dy:.0f})  {result.metric.upper()}={result.cost:.4f}"
    )
    ax.axis("off")

    # --- input patch ---
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(_as_display(result.patch), cmap="gray" if result.patch.ndim == 2 else None)
    ax.set_title("Input patch (frame 1)")
    ax.set_xticks([]); ax.set_yticks([])

    # --- best match patch (book shows this, not the whole search crop) ---
    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(_as_display(result.best_patch), cmap="gray" if result.best_patch.ndim == 2 else None)
    ax.set_title("Best match (frame 2)")
    ax.set_xticks([]); ax.set_yticks([])

    # --- matching cost: reversed gray like book (bright = small distance) ---
    ax = fig.add_subplot(gs[1, 2])
    cost = result.cost_map.copy()
    # invert so low cost is bright (book: "reversed grayscale")
    cmin = float(np.nanmin(cost))
    cmax = float(np.nanmax(cost))
    inv = cmax - cost
    im = ax.imshow(inv, cmap="gray")
    by_c, bx_c = np.unravel_index(np.nanargmin(cost), cost.shape)
    ax.plot(bx_c, by_c, "r*", markersize=14)
    ax.set_title(f"Matching cost  ({cost.shape[1]}×{cost.shape[0]}, bright=low)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # --- pixel charts ---
    mid = result.patch_size // 2
    p_row = _luma(_as_display(result.patch))[mid, :]
    b_row = _luma(_as_display(result.best_patch))[mid, :]
    ax = fig.add_subplot(gs[2, 0:2])
    ax.plot(p_row, "g-", lw=2, label="input mid-row")
    ax.plot(b_row, "m--", lw=2, label="best-match mid-row")
    ax.set_xlabel("column in patch")
    ax.set_ylabel("luma [0,1]")
    ax.set_title("Pixel chart — mid-row intensities")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, 2])
    src_f = _luma(_as_display(result.patch)).ravel()
    dst_f = _luma(_as_display(result.best_patch)).ravel()
    ax.scatter(src_f, dst_f, s=10, alpha=0.55, c="steelblue")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("input patch pixel")
    ax.set_ylabel("best-match pixel")
    ax.set_title("Pixel↔pixel (identity = perfect)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Algorithm 46.1 — patch matching (MIT Vision Book Fig. 46.4 layout)",
        fontsize=13,
        y=0.995,
    )
    return fig


def _grid_centers(h: int, w: int, size: int, stride: int) -> np.ndarray:
    r = size // 2
    ys = np.arange(r, h - r, stride, dtype=np.int32)
    xs = np.arange(r, w - r, stride, dtype=np.int32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([yy.ravel(), xx.ravel()], axis=1)


def patch_motion_field(
    frame_t: np.ndarray,
    frame_t1: np.ndarray,
    cfg: Optional[PatchMatchConfig] = None,
) -> MotionField:
    """
    Compute a coarse motion field by matching patches from frame_t in frame_t1.
    """
    cfg = cfg or PatchMatchConfig()
    if frame_t.ndim == 3:
        frame_t = frame_t.mean(axis=2)
    if frame_t1.ndim == 3:
        frame_t1 = frame_t1.mean(axis=2)
    frame_t = np.asarray(frame_t, dtype=np.float32)
    frame_t1 = np.asarray(frame_t1, dtype=np.float32)
    assert frame_t.shape == frame_t1.shape

    h, w = frame_t.shape
    size = cfg.patch_size
    radius = cfg.search_radius
    r = size // 2
    centers = _grid_centers(h, w, size, cfg.stride)
    if cfg.max_patches is not None and len(centers) > cfg.max_patches:
        # keep a spatial subsample for interactive demos
        idx = np.linspace(0, len(centers) - 1, cfg.max_patches).astype(int)
        centers = centers[idx]

    vectors = np.zeros((len(centers), 2), dtype=np.float32)
    scores = np.zeros(len(centers), dtype=np.float32)

    for i, (cy, cx) in enumerate(centers):
        patch = extract_patch(frame_t, int(cy), int(cx), size)
        if patch.shape != (size, size):
            vectors[i] = (0.0, 0.0)
            scores[i] = np.nan
            continue

        y0 = max(0, int(cy) - r - radius)
        x0 = max(0, int(cx) - r - radius)
        y1 = min(h, int(cy) - r + size + radius)
        x1 = min(w, int(cx) - r + size + radius)
        region = frame_t1[y0:y1, x0:x1]

        dy_tl, dx_tl, score = match_patch(
            patch, region, metric=cfg.metric, search_step=cfg.search_step
        )
        # top-left of match in full frame_t1
        match_y = y0 + dy_tl + r
        match_x = x0 + dx_tl + r
        vectors[i, 0] = float(match_x - cx)  # dx
        vectors[i, 1] = float(match_y - cy)  # dy
        scores[i] = score

    return MotionField(
        vectors=vectors,
        scores=scores,
        centers=centers,
        patch_size=size,
        stride=cfg.stride,
        metric=cfg.metric,
    )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def draw_patches_and_vectors(
    rgb_t: np.ndarray,
    field: MotionField,
    *,
    scale: float = 2.0,
    min_mag: float = 0.5,
    color_box=(0, 255, 80),
    color_arrow=(255, 40, 40),
) -> np.ndarray:
    """Overlay patch boxes + motion arrows on an RGB copy of frame t."""
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Rectangle

    h, w = rgb_t.shape[:2]
    fig = Figure(figsize=(w / 100, h / 100), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgb_t)
    ax.set_axis_off()
    r = field.patch_size // 2

    for (cy, cx), (dx, dy), sc in zip(field.centers, field.vectors, field.scores):
        if np.isnan(sc):
            continue
        mag = float(np.hypot(dx, dy))
        ax.add_patch(
            Rectangle(
                (cx - r, cy - r),
                field.patch_size,
                field.patch_size,
                fill=False,
                edgecolor=np.array(color_box) / 255.0,
                linewidth=0.8,
                alpha=0.7,
            )
        )
        if mag < min_mag:
            continue
        ax.annotate(
            "",
            xy=(cx + dx * scale, cy + dy * scale),
            xytext=(cx, cy),
            arrowprops=dict(
                arrowstyle="->",
                color=np.array(color_arrow) / 255.0,
                lw=1.2,
            ),
        )

    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return buf


def show_patch_gallery(
    frame_t: np.ndarray,
    frame_t1: np.ndarray,
    field: MotionField,
    indices: Sequence[int],
):
    """
    Side-by-side gallery: source patch (t) | matched patch (t+1) | residual.
    Returns a matplotlib Figure.
    """
    import matplotlib.pyplot as plt

    n = len(indices)
    fig, axes = plt.subplots(n, 3, figsize=(7, 2.2 * n))
    if n == 1:
        axes = np.array([axes])
    r = field.patch_size // 2
    gray_t = frame_t if frame_t.ndim == 2 else frame_t.mean(axis=2)
    gray_t1 = frame_t1 if frame_t1.ndim == 2 else frame_t1.mean(axis=2)

    for row, idx in enumerate(indices):
        cy, cx = field.centers[idx]
        dx, dy = field.vectors[idx]
        src = extract_patch(gray_t, int(cy), int(cx), field.patch_size)
        my, mx = int(cy + dy), int(cx + dx)
        dst = extract_patch(gray_t1, my, mx, field.patch_size)
        if dst.shape != src.shape:
            # clamp if match near border
            dst = np.zeros_like(src)
        resid = np.abs(src.astype(np.float32) - dst.astype(np.float32))

        axes[row, 0].imshow(src, cmap="gray", vmin=0, vmax=1)
        axes[row, 0].set_title(f"t  center=({cy},{cx})")
        axes[row, 1].imshow(dst, cmap="gray", vmin=0, vmax=1)
        axes[row, 1].set_title(f"t+1  Δ=({dx:.1f},{dy:.1f})")
        axes[row, 2].imshow(resid, cmap="magma", vmin=0, vmax=0.5)
        axes[row, 2].set_title(f"|resid|  score={field.scores[idx]:.3f}")
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Patch motion matching — source | match | residual", y=1.01)
    fig.tight_layout()
    return fig


def pick_interesting_indices(field: MotionField, k: int = 6) -> List[int]:
    """Pick patches with largest motion magnitude (skip NaN scores)."""
    mag = np.linalg.norm(field.vectors, axis=1)
    mag[np.isnan(field.scores)] = -1
    order = np.argsort(-mag)
    return [int(i) for i in order[:k] if mag[i] >= 0]


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------


def main() -> None:
    seq = default_kitti_seq(seq="0019")
    frames = list_kitti_frames(seq)
    if len(frames) < 2:
        raise SystemExit(f"need ≥2 frames in {seq}")

    # mid-sequence pair — ego motion is clear on KITTI 0019
    i0 = min(120, len(frames) - 2)
    path_t, path_t1 = frames[i0], frames[i0 + 1]
    print(f"pair: {path_t.name} → {path_t1.name}")

    gray_t = load_gray(path_t)
    gray_t1 = load_gray(path_t1)
    cfg = PatchMatchConfig(
        patch_size=21,
        search_radius=16,
        stride=40,
        metric="ncc",
        search_step=2,
    )
    field = patch_motion_field(gray_t, gray_t1, cfg)
    mag = np.linalg.norm(field.vectors, axis=1)
    print(
        f"patches={len(field.centers)}  "
        f"mean|v|={mag.mean():.2f}px  max|v|={mag.max():.2f}px"
    )


if __name__ == "__main__":
    main()
