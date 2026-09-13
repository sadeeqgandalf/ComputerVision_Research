#!/usr/bin/env python3
"""Goal 1 smoke demo: SAM3 text PCS on one BDD100K val image.

Prereqs:
  1. Request access: https://huggingface.co/facebook/sam3
  2. source .venv_sam3/bin/activate
  3. hf auth login
  4. python scripts/run_sam3_bdd_demo.py

Optional env:
  SAM3_PROMPT="bridge"             # open-vocab noun phrase
  SAM3_IMAGE=c0e631f4-cf85b543.jpg
  SAM3_DEVICE=cpu|mps|cuda
  SAM3_CONF=0.3
  # Geometric prompts (normalized cx,cy,w,h in [0,1]). Repeat with ';' for multiple.
  # label=True positive, label=False negative — refines text PCS.
  SAM3_POS_BOX=0.22,0.32,0.28,0.22
  SAM3_NEG_BOX=0.88,0.40,0.10,0.18
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
BDD_IMG_DIR = ROOT / "Research_Data/BDD100K/hf_dgural_bdd100k/data"
BDD_VAL_DIR = ROOT / "Research_Data/BDD100K/images/100k/val_data"
BDD_LABELS = ROOT / "Research_Data/BDD100K/labels/det_val_simplified.json"
OUT_DIR = ROOT / "outputs/bdd_sam3_demo"
BDD_DET_CATEGORIES = {
    "car",
    "traffic sign",
    "traffic light",
    "pedestrian",
    "truck",
    "bus",
    "bicycle",
    "rider",
    "motorcycle",
    "other vehicle",
    "train",
    "trailer",
    "other person",
}


def pick_device() -> str:
    override = os.environ.get("SAM3_DEVICE", "").strip().lower()
    if override in {"cpu", "mps", "cuda"}:
        return override
    # MPS can still hit Metal dtype asserts on some Macs; default CPU for correctness.
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def parse_boxes_env(name: str) -> list[list[float]]:
    """Parse 'cx,cy,w,h;cx,cy,w,h' normalized boxes from an env var."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return []
    boxes: list[list[float]] = []
    for part in raw.split(";"):
        part = part.strip()
        if not part:
            continue
        vals = [float(x) for x in part.split(",")]
        if len(vals) != 4:
            raise ValueError(f"{name} entries must be cx,cy,w,h got {part!r}")
        if not all(0.0 <= v <= 1.0 for v in vals):
            raise ValueError(f"{name} values must be in [0,1], got {vals}")
        boxes.append(vals)
    return boxes


def cxcywh_norm_to_xyxy_px(
    box: list[float], width: int, height: int
) -> tuple[float, float, float, float]:
    cx, cy, w, h = box
    bw, bh = w * width, h * height
    x0 = (cx * width) - bw / 2
    y0 = (cy * height) - bh / 2
    return x0, y0, x0 + bw, y0 + bh


def load_labels(labels_path: Path) -> list[dict]:
    with open(labels_path) as f:
        return json.load(f)


def resolve_image_path(name_or_path: str) -> Path:
    """Accept a filename or any path under hf/ or val_data/."""
    p = Path(name_or_path).expanduser()
    if p.is_file():
        return p.resolve()
    for base in (BDD_IMG_DIR, BDD_VAL_DIR):
        cand = base / Path(name_or_path).name
        if cand.is_file():
            return cand.resolve()
    raise FileNotFoundError(
        f"Image not found: {name_or_path}\n"
        f"Tried under {BDD_IMG_DIR} and {BDD_VAL_DIR}"
    )


def find_image_with_class(
    labels: list[dict],
    category: str,
    preferred_name: str | None = None,
    min_count: int = 6,
) -> tuple[dict, Path]:
    """Return (label_item_or_stub, image_path). Open-vocab prompts need SAM3_IMAGE."""
    if preferred_name:
        img_path = resolve_image_path(preferred_name)
        name = img_path.name
        for item in labels:
            if item["name"] == name:
                return item, img_path
        return {"name": name, "labels": []}, img_path

    best = None
    best_n = -1
    for item in labels:
        n = sum(1 for lab in item["labels"] if lab["category"] == category)
        if n < min_count:
            continue
        if not (BDD_IMG_DIR / item["name"]).exists():
            continue
        if n > best_n:
            best, best_n = item, n
    if best is not None:
        return best, BDD_IMG_DIR / best["name"]

    for item in labels:
        if any(lab["category"] == category for lab in item["labels"]):
            if (BDD_IMG_DIR / item["name"]).exists():
                return item, BDD_IMG_DIR / item["name"]

    if category not in BDD_DET_CATEGORIES:
        raise RuntimeError(
            f"Prompt {category!r} is not a BDD det class (open-vocab is fine), "
            f"but you must set SAM3_IMAGE to a frame that actually shows it.\n"
            f"Example:\n"
            f'  SAM3_IMAGE=some_daytime.jpg SAM3_PROMPT="gas station pump" '
            f"python scripts/run_sam3_bdd_demo.py"
        )
    raise RuntimeError(f"No BDD image found with category={category}")


def overlay_masks_and_boxes(
    image: Image.Image,
    masks: torch.Tensor,
    boxes: torch.Tensor | None = None,
    color=(0, 220, 80),
    alpha=0.55,
) -> Image.Image:
    """Bright mask fill + outline + predicted boxes so small objects stay visible."""
    arr = np.array(image.convert("RGB")).copy()
    if masks is not None and len(masks) > 0:
        m = masks.detach().cpu().numpy()
        if m.ndim == 4:
            m = m[:, 0]
        union = np.zeros(arr.shape[:2], dtype=bool)
        for i in range(m.shape[0]):
            mask = m[i] > 0.5
            if mask.shape[:2] != arr.shape[:2]:
                mask_img = Image.fromarray((mask.astype(np.uint8) * 255)).resize(
                    (arr.shape[1], arr.shape[0]), Image.NEAREST
                )
                mask = np.array(mask_img) > 127
            union |= mask
        for c in range(3):
            arr[union, c] = (
                arr[union, c] * (1 - alpha) + color[c] * alpha
            ).astype(np.uint8)

        from PIL import ImageFilter

        for i in range(m.shape[0]):
            mask = m[i] > 0.5
            if mask.shape[:2] != arr.shape[:2]:
                mask_img = Image.fromarray((mask.astype(np.uint8) * 255)).resize(
                    (arr.shape[1], arr.shape[0]), Image.NEAREST
                )
            else:
                mask_img = Image.fromarray((mask.astype(np.uint8) * 255))
            edge = mask_img.filter(ImageFilter.FIND_EDGES)
            edge_arr = np.array(edge) > 0
            arr[edge_arr] = np.array([0, 255, 120], dtype=np.uint8)

    out = Image.fromarray(arr)
    draw = ImageDraw.Draw(out)
    if boxes is not None and len(boxes) > 0:
        b = boxes.detach().cpu().numpy()
        for i in range(b.shape[0]):
            x0, y0, x1, y1 = [float(v) for v in b[i][:4]]
            draw.rectangle([x0, y0, x1, y1], outline=(0, 180, 255), width=2)
    return out


def draw_gt_boxes(image: Image.Image, item: dict, category: str) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for lab in item["labels"]:
        if lab["category"] != category:
            continue
        x, y, w, h = lab["bbox"]
        draw.rectangle([x, y, x + w, y + h], outline=(255, 40, 40), width=2)
    return img


def draw_prompt_boxes(
    image: Image.Image,
    pos_boxes: list[list[float]],
    neg_boxes: list[list[float]],
) -> Image.Image:
    """Draw geometric prompts: green=positive, red=negative."""
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for box in pos_boxes:
        xyxy = cxcywh_norm_to_xyxy_px(box, img.width, img.height)
        draw.rectangle(xyxy, outline=(0, 220, 80), width=3)
        draw.text((xyxy[0] + 4, xyxy[1] + 4), "+ box", fill=(0, 220, 80))
    for box in neg_boxes:
        xyxy = cxcywh_norm_to_xyxy_px(box, img.width, img.height)
        draw.rectangle(xyxy, outline=(255, 60, 60), width=3)
        draw.text((xyxy[0] + 4, xyxy[1] + 4), "- box", fill=(255, 60, 60))
    return img


def label_panel(image: Image.Image, text: str) -> Image.Image:
    img = image.copy()
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 28], fill=(0, 0, 0))
    draw.text((8, 6), text, fill=(255, 255, 255))
    return img


def main():
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    category = os.environ.get("SAM3_PROMPT", "traffic light")
    preferred = os.environ.get("SAM3_IMAGE", "").strip() or None
    conf = float(os.environ.get("SAM3_CONF", "0.4"))
    pos_boxes = parse_boxes_env("SAM3_POS_BOX")
    neg_boxes = parse_boxes_env("SAM3_NEG_BOX")
    device = pick_device()
    print(f"device={device} prompt={category!r} conf={conf}")
    if pos_boxes or neg_boxes:
        print(f"geo prompts: +{len(pos_boxes)} box(es), -{len(neg_boxes)} box(es)")

    labels = load_labels(BDD_LABELS)
    item, img_path = find_image_with_class(labels, category, preferred_name=preferred)
    n_gt = sum(1 for lab in item["labels"] if lab["category"] == category)
    open_vocab = category not in BDD_DET_CATEGORIES
    print(
        f"image={img_path}  gt_match_count={n_gt}"
        + ("  (open-vocab: no BDD GT for this prompt)" if open_vocab else "")
    )
    image = Image.open(img_path).convert("RGB")

    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    print("Loading SAM3 (downloads facebook/sam3 if needed)...")
    model = build_sam3_image_model(device=device, load_from_HF=True)
    processor = Sam3Processor(model, device=device, confidence_threshold=conf)

    state = processor.set_image(image)
    output = processor.set_text_prompt(state=state, prompt=category)
    for box in pos_boxes:
        output = processor.add_geometric_prompt(box=box, label=True, state=state)
    for box in neg_boxes:
        output = processor.add_geometric_prompt(box=box, label=False, state=state)

    masks = output["masks"]
    boxes = output["boxes"]
    scores = output["scores"]
    print(
        f"preds: masks={len(masks)} boxes={len(boxes)} "
        f"scores={scores[:8].tolist() if hasattr(scores, 'tolist') else scores}"
    )
    if len(masks) > 0:
        pix = int(masks.detach().float().sum().item())
        print(f"mask_positive_pixels={pix}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if pos_boxes or neg_boxes:
        left = draw_prompt_boxes(image, pos_boxes, neg_boxes)
        left_title = f"prompts +{len(pos_boxes)} / -{len(neg_boxes)} (+ green, - red)"
    elif open_vocab:
        left = draw_gt_boxes(image, item, category)
        left_title = "BDD GT (n/a for open-vocab)"
    else:
        left = draw_gt_boxes(image, item, category)
        left_title = f"BDD GT boxes ({n_gt})"

    gt_vis = label_panel(left, left_title)
    pred_vis = label_panel(
        overlay_masks_and_boxes(image, masks, boxes),
        f"SAM3 '{category}' ({len(masks)})",
    )
    side = Image.new("RGB", (image.width * 2, image.height))
    side.paste(gt_vis, (0, 0))
    side.paste(pred_vis, (image.width, 0))
    suffix = "_geo" if (pos_boxes or neg_boxes) else ""
    out_path = OUT_DIR / f"{img_path.stem}_{category.replace(' ', '_')}{suffix}.png"
    side.save(out_path)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
