import os
import cv2
import numpy as np
import pytest
import sys
from pathlib import Path

# Add src/ to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from nostalgic import load_images, apply_nostalgic_effect, save_image

INPUT_DIR = "input"


@pytest.fixture
def images():
    files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    assert files, "No images found in input/"
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


def test_nostalgic_effect_basic(images):
    for name, img in images:
        result = apply_nostalgic_effect(img)

        assert result.shape == img.shape
        assert result.dtype == np.uint8
        assert not np.array_equal(img, result)


def test_nostalgic_effect_parameters(images):
    for _, img in images:
        warm = apply_nostalgic_effect(img, warmth=1.3)
        cool = apply_nostalgic_effect(img, warmth=0.8)
        grain = apply_nostalgic_effect(img, grain_strength=12)
        blur = apply_nostalgic_effect(img, blur_ksize=9)

        assert not np.array_equal(warm, cool)
        assert not np.array_equal(grain, blur)


def test_pixel_value_clipping(images):
    for _, img in images:
        result = apply_nostalgic_effect(img)
        assert result.min() >= 0
        assert result.max() <= 255


def test_save_image(images, output_dir):
    for name, img in images:
        result = apply_nostalgic_effect(img)
        path = save_image(str(output_dir), name, result)

        assert os.path.exists(path)
        saved = cv2.imread(path)
        assert saved is not None
        assert not np.array_equal(img, saved)


def test_invalid_inputs():
    with pytest.raises(ValueError):
        apply_nostalgic_effect(None)

    assert list(load_images(INPUT_DIR, [])) == []
