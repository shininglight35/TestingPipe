import os
import cv2
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from brighten import load_images, apply_clahe, save_image

INPUT_DIR = "input"


@pytest.fixture
def input_images():
    files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    assert files, "No images found in input/"
    return files


@pytest.fixture
def output_dir():
    out = Path("output")
    out.mkdir(exist_ok=True)
    return out


def test_load_images(input_images):
    images = list(load_images(INPUT_DIR, input_images))
    assert images

    for name, img in images:
        assert img is not None
        assert img.size > 0


def test_apply_clahe_basic(input_images):
    for name, img in load_images(INPUT_DIR, input_images):
        enhanced = apply_clahe(img)

        assert enhanced is not None
        assert enhanced.shape == img.shape
        assert enhanced.dtype == img.dtype

        gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_enh = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

        assert not np.array_equal(gray_orig, gray_enh), \
            f"CLAHE did not modify {name}"


def test_save_image(output_dir, input_images):
    for name, img in load_images(INPUT_DIR, input_images):
        enhanced = apply_clahe(img)
        path = save_image(str(output_dir), name, enhanced)

        assert os.path.exists(path)

        saved = cv2.imread(path)
        assert saved is not None
        assert not np.array_equal(img, saved)


def test_apply_clahe_invalid_inputs():
    with pytest.raises(ValueError):
        apply_clahe(None)

    empty = np.zeros((0, 0, 3), dtype=np.uint8)
    with pytest.raises(cv2.error):
        apply_clahe(empty)


def test_load_images_edge_cases():
    assert list(load_images(INPUT_DIR, [])) == []
    assert list(load_images(INPUT_DIR, ["missing.jpg"])) == []


def test_clahe_deterministic(input_images):
    for _, img in load_images(INPUT_DIR, input_images):
        out1 = apply_clahe(img)
        out2 = apply_clahe(img)
        assert np.array_equal(out1, out2)


def test_clahe_synthetic_image():
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[25:75, 25:75] = 50

    enhanced = apply_clahe(img)

    gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_enh = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)

    assert np.mean(gray_enh) > np.mean(gray_orig)
