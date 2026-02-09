import os
import cv2
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from beautify import load_images, smooth_skin, save_image

INPUT_DIR = "input"
OUTPUT_DIR = "output"


@pytest.fixture
def images():
    assert os.path.exists(INPUT_DIR)
    files = [f for f in os.listdir(INPUT_DIR)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    assert files
    return list(load_images(INPUT_DIR, files))


@pytest.fixture
def output_dir():
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    return OUTPUT_DIR


def test_images_load(images):
    for name, img in images:
        assert img is not None
        assert img.size > 0


def test_smooth_skin_basic(images):
    for name, img in images:
        result = smooth_skin(img)

        assert result is not None
        assert result.shape == img.shape
        assert result.dtype == img.dtype
        assert not np.array_equal(img, result)


def test_smooth_skin_invalid_input():
    with pytest.raises(ValueError):
        smooth_skin(None)


def test_save_image(images, output_dir):
    for name, img in images:
        result = smooth_skin(img)
        path = save_image(output_dir, name, result)

        assert os.path.exists(path)

        saved = cv2.imread(path)
        assert saved is not None
        assert saved.shape == result.shape
        assert cv2.absdiff(img, saved).sum() > 0


def test_pixel_value_range(images):
    for _, img in images:
        result = smooth_skin(img)
        assert result.min() >= 0
        assert result.max() <= 255
