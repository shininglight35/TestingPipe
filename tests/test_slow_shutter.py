import os
import cv2
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from slow_shutter import load_images, apply_slow_shutter, save_image

INPUT_DIR = "input"


@pytest.fixture
def images():
    files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    assert files, "No input images found"
    return list(load_images(INPUT_DIR, files))


@pytest.fixture
def output_dir():
    out = Path("output")
    out.mkdir(exist_ok=True)
    return out


def test_images_load(images):
    for name, img in images:
        assert img is not None
        assert img.size > 0


def test_slow_shutter_modifies_image(images):
    for name, img in images:
        result = apply_slow_shutter(img)
        assert not np.array_equal(img, result)


def test_slow_shutter_preserves_shape_and_dtype(images):
    for _, img in images:
        result = apply_slow_shutter(img)
        assert result.shape == img.shape
        assert result.dtype == img.dtype


def test_slow_shutter_parameters_change_output(images):
    for _, img in images:
        left = apply_slow_shutter(img, direction=-1)
        right = apply_slow_shutter(img, direction=1)
        short = apply_slow_shutter(img, trail_length=30)

        assert not np.array_equal(left, right)
        assert not np.array_equal(left, short)


def test_blend_original_effect(images):
    for _, img in images:
        full = apply_slow_shutter(img, blend_original=1.0)
        none = apply_slow_shutter(img, blend_original=0.0)

        diff_full = cv2.absdiff(img, full).sum()
        diff_none = cv2.absdiff(img, none).sum()

        assert diff_full < diff_none


def test_save_image(images, output_dir):
    for name, img in images:
        result = apply_slow_shutter(img)
        path = save_image(str(output_dir), name, result)

        assert os.path.exists(path)
        saved = cv2.imread(path)
        assert saved is not None
        assert not np.array_equal(img, saved)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        apply_slow_shutter(None)


def test_load_images_edge_cases():
    assert list(load_images(INPUT_DIR, [])) == []
    assert list(load_images(INPUT_DIR, ["missing.jpg"])) == []
