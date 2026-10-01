"""Model-free checks of the KITTI SAM3 script helpers."""

import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")
pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_sam3_kitti_video as k  # noqa: E402


def test_overlay_with_label_returns_uint8():
    # Regression: putText on a float32 image raises on OpenCV >= 5.
    img = np.zeros((20, 40, 3), dtype=np.uint8)
    mask = np.zeros((20, 40), dtype=bool)
    mask[5:15, 5:30] = True
    out = k.overlay_masks(img, {0: mask}, {0: "car"})
    assert out.dtype == np.uint8 and out.shape == img.shape
    # colour 0 is red (RGB): red channel tinted, blue untouched outside text
    assert out[6, 6, 0] > 0 and out[6, 6, 2] == 0


def test_overlay_empty():
    img = np.full((4, 6, 3), 7, dtype=np.uint8)
    np.testing.assert_array_equal(k.overlay_masks(img, {}), img)


def test_merge_prompt_masks_gives_disjoint_ids():
    combined, o2c = {}, {}
    m = np.ones((2, 2), dtype=bool)
    k.merge_prompt_masks(combined, {0: {0: m, 1: m}, 1: {1: m}}, "car", o2c)
    k.merge_prompt_masks(combined, {0: {0: m}, 1: {}, 2: {5: m}}, "person", o2c)
    assert o2c == {0: "car", 1: "car", 2: "person", 3: "person"}
    assert {f: sorted(v) for f, v in combined.items()} == {0: [0, 1, 2], 1: [1], 2: [3]}
    assert k.summarize_concepts(combined, o2c) == {"car": 2, "person": 2}


def test_parse_prompts_dedupes_and_strips_comments(tmp_path):
    f = tmp_path / "p.txt"
    f.write_text("# header\nperson\ncar, bicycle # inline\n", encoding="utf-8")
    assert k.parse_prompts("Car,,pole", f) == ["person", "car", "bicycle", "pole"]
