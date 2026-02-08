import os
import cv2
import numpy as np
import pytest
import sys
from pathlib import Path

# Add src/ to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from brighten import (
    load_images,
    apply_clahe,
    save_image
)

INPUT_DIR = "input"


@pytest.fixture
def output_dir():
    """
    Always use 'output/' folder for all runs.
    """
    out = Path("output")
    out.mkdir(parents=True, exist_ok=True)
    return out


@pytest.fixture
def dark_image():
    """
    Load the specific dark.jpg image for testing.
    """
    dark_path = os.path.join(INPUT_DIR, "dark.jpg")
    assert os.path.exists(dark_path), "dark.jpg not found in input folder"
    
    img = cv2.imread(dark_path)
    assert img is not None, "Failed to load dark.jpg"
    return img


# --------------------
# Tests - Specific to dark.jpg
# --------------------

def test_dark_image_loads_correctly(dark_image):
    """
    Ensure dark.jpg loads successfully.
    """
    assert dark_image is not None, "Failed to load dark.jpg"
    assert dark_image.size > 0, "dark.jpg is empty"
    assert dark_image.shape[2] == 3, "dark.jpg should have 3 channels (BGR)"


def test_apply_clahe_on_dark_image(dark_image):
    """
    Ensure CLAHE enhancement runs on dark.jpg and returns valid output.
    """
    enhanced_img = apply_clahe(dark_image)

    assert enhanced_img is not None, "CLAHE failed for dark.jpg"
    assert enhanced_img.shape == dark_image.shape, "Shape mismatch for dark.jpg"
    assert enhanced_img.dtype == dark_image.dtype, "Dtype mismatch for dark.jpg"


def test_apply_clahe_changes_dark_image_luminance(dark_image):
    """
    Verify CLAHE actually modifies dark.jpg luminance.
    """
    enhanced_img = apply_clahe(dark_image)

    # Convert to grayscale for luminance comparison
    gray_original = cv2.cvtColor(dark_image, cv2.COLOR_BGR2GRAY)
    gray_enhanced = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)

    # CLAHE should modify the image (not necessarily every pixel)
    assert not np.array_equal(gray_original, gray_enhanced), \
        "CLAHE did not modify dark.jpg"


def test_enhanced_dark_image_is_saved(output_dir, dark_image):
    """
    Ensure enhanced dark.jpg is written to disk.
    """
    enhanced_img = apply_clahe(dark_image)

    saved_path = save_image(str(output_dir), "dark.jpg", enhanced_img)
    assert os.path.exists(saved_path), "Enhanced dark.jpg not saved"


def test_enhanced_dark_image_is_modified(output_dir, dark_image):
    """
    Ensure enhanced dark.jpg is not identical to the original.
    """
    enhanced_img = apply_clahe(dark_image)

    saved_path = save_image(str(output_dir), "dark.jpg", enhanced_img)
    saved_img = cv2.imread(saved_path)
    assert saved_img is not None, "Failed to reload saved dark.jpg"

    diff = cv2.absdiff(dark_image, saved_img)
    assert diff.sum() > 0, "Enhanced dark.jpg was not modified"


def test_apply_clahe_invalid_input():
    """
    Test CLAHE with invalid inputs.
    """
    # Test with None input - should raise ValueError based on your function
    with pytest.raises(ValueError):
        apply_clahe(None)

    # Test with empty array - OpenCV raises cv2.error, not ValueError
    empty_img = np.array([], dtype=np.uint8).reshape(0, 0, 3)
    with pytest.raises(cv2.error):
        apply_clahe(empty_img)


def test_load_images_edge_cases():
    """
    Test load_images function with edge cases.
    """
    # Test with empty file list
    result = list(load_images(INPUT_DIR, []))
    assert len(result) == 0

    # Test with non-existent file
    result = list(load_images(INPUT_DIR, ["non_existent.jpg"]))
    assert len(result) == 0


def test_save_image_with_prefix(output_dir, dark_image):
    """
    Test save_image with custom prefix using dark.jpg.
    """
    enhanced_img = apply_clahe(dark_image)
    
    saved_path = save_image(
        str(output_dir),
        "dark.jpg",
        enhanced_img,
        prefix="clahe_"
    )

    assert os.path.exists(saved_path)
    assert os.path.basename(saved_path) == "clahe_dark.jpg"

    saved_img = cv2.imread(saved_path)
    assert saved_img is not None
    assert saved_img.shape == enhanced_img.shape


def test_clahe_brightens_dark_image(dark_image):
    """
    Test CLAHE specifically on dark.jpg to ensure it brightens it.
    """
    enhanced_img = apply_clahe(dark_image)
    
    # Convert to grayscale for brightness comparison
    gray_original = cv2.cvtColor(dark_image, cv2.COLOR_BGR2GRAY)
    gray_enhanced = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
    
    # Get brightness metrics
    avg_original = np.mean(gray_original)
    avg_enhanced = np.mean(gray_enhanced)
    
    print(f"dark.jpg - Original average brightness: {avg_original:.2f}")
    print(f"dark.jpg - Enhanced average brightness: {avg_enhanced:.2f}")
    
    # CLAHE should modify the image
    assert not np.array_equal(gray_original, gray_enhanced), \
        "CLAHE did not modify dark.jpg"
    
    # For a truly dark image, CLAHE should increase brightness
    # We'll check if it's actually a dark image first
    if avg_original < 100:  # Arbitrary threshold for "dark" image
        assert avg_enhanced > avg_original, \
            f"CLAHE did not increase brightness for dark.jpg"
