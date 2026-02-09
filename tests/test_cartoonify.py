import os
import cv2
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from cartoonify import load_images, cartoonify, save_image

INPUT_DIR = "input"
OUTPUT_DIR = "output"


@pytest.fixture(scope="session")
def images():
    assert os.path.exists(INPUT_DIR)
    files = [f for f in os.listdir(INPUT_DIR)
             if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    assert files
    return list(load_images(INPUT_DIR, files))


@pytest.fixture(scope="session")
def output_dir():
    Path(OUTPUT_DIR).mkdir(exist_ok=True)
    return OUTPUT_DIR


def test_images_load(images):
    for name, img in images:
        assert img is not None
        assert img.size > 0


def test_cartoonify_basic(images):
    for name, img in images:
        result = cartoonify(img)

        assert result is not None
        assert result.shape == img.shape
        assert result.dtype == img.dtype
        assert not np.array_equal(img, result)


def test_cartoonify_edges_exist(images):
    for _, img in images:
        result = cartoonify(img)
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        assert edges.sum() > 0


def test_color_reduction(images):
    for _, img in images:
        result = cartoonify(img)

        orig_colors = len(np.unique(img.reshape(-1, 3), axis=0))
        new_colors = len(np.unique(result.reshape(-1, 3), axis=0))

        if orig_colors > 20:
            assert new_colors < orig_colors


def test_image_saved(images, output_dir):
    for name, img in images:
        result = cartoonify(img)
        path = save_image(output_dir, name, result)

        assert os.path.exists(path)
        saved = cv2.imread(path)
        assert saved is not None
        assert saved.shape == result.shape


def test_invalid_input():
    with pytest.raises(ValueError):
        cartoonify(None)
