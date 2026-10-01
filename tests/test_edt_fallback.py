"""CPU EDT fallback must match the Triton kernel's semantics:
distance from each True pixel to the nearest False pixel (scipy convention)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("cv2")
ndimage = pytest.importorskip("scipy.ndimage")

EDT_PATH = Path(__file__).resolve().parents[1] / "sam3/sam3/model/edt.py"


def _load_edt():
    spec = importlib.util.spec_from_file_location("sam3_edt_under_test", EDT_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_single_pixel():
    edt = _load_edt()
    d = np.zeros((1, 5, 5), dtype=bool)
    d[0, 2, 2] = True
    out = edt._edt_cv2(torch.from_numpy(d)).numpy()[0]
    expected = np.zeros((5, 5), dtype=np.float32)
    expected[2, 2] = 1.0
    np.testing.assert_allclose(out, expected, atol=1e-5)


def test_matches_scipy_on_random_masks():
    edt = _load_edt()
    rng = np.random.default_rng(0)
    for _ in range(10):
        d = rng.random((3, 37, 53)) > 0.3
        d[:, 0, 0] = False  # ensure at least one zero per image
        out = edt._edt_cv2(torch.from_numpy(d)).numpy()
        ref = np.stack([ndimage.distance_transform_edt(x) for x in d])
        np.testing.assert_allclose(out, ref, atol=1e-4)
