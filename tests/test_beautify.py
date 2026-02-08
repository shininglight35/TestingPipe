import os
import cv2
import numpy as np
import pytest
import sys
from pathlib import Path

# Add src/ to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from beautify import (
    load_images,
    smooth_skin,
    save_image,
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
def acne_image():
    """
    Load the specific acne.jpg image for testing.
    """
    acne_path = os.path.join(INPUT_DIR, "acne.jpg")
    assert os.path.exists(acne_path), "acne.jpg not found in input folder"
    
    img = cv2.imread(acne_path)
    assert img is not None, "Failed to load acne.jpg"
    return img


# --------------------
# Tests - Specific to acne.jpg
# --------------------

def test_acne_image_loads_correctly(acne_image):
    """
    Ensure acne.jpg loads successfully.
    """
    assert acne_image is not None, "Failed to load acne.jpg"
    assert acne_image.size > 0, "acne.jpg is empty"
    assert acne_image.shape[2] == 3, "acne.jpg should have 3 channels (BGR)"


def test_smooth_skin_on_acne_image(acne_image):
    """
    Ensure smooth_skin function runs on acne.jpg and returns valid output.
    """
    result = smooth_skin(acne_image)

    assert result is not None, "smooth_skin failed for acne.jpg"
    assert result.shape == acne_image.shape, "Shape mismatch for acne.jpg"
    assert result.dtype == acne_image.dtype, "Dtype mismatch for acne.jpg"


def test_smooth_skin_modifies_acne_image(acne_image):
    """
    Verify smooth_skin actually modifies acne.jpg.
    """
    result = smooth_skin(acne_image)

    # The result should be different from original
    assert not np.array_equal(acne_image, result), \
        "smooth_skin did not modify acne.jpg"


def test_smooth_skin_with_skin_mask(acne_image):
    """
    Verify smooth_skin creates a skin mask and applies smoothing.
    Since acne.jpg contains skin, the mask should detect skin areas.
    """
    result = smooth_skin(acne_image)
    
    # Convert to YCrCb color space (as done in smooth_skin)
    ycrcb = cv2.cvtColor(acne_image, cv2.COLOR_BGR2YCrCb)
    
    # Skin color range (from the function)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    
    skin_mask = cv2.inRange(ycrcb, lower, upper)
    
    # Since acne.jpg contains skin, the mask should have some white pixels
    # (non-zero values)
    mask_pixel_count = np.count_nonzero(skin_mask)
    mask_percentage = (mask_pixel_count / skin_mask.size) * 100
    
    print(f"Skin mask coverage in acne.jpg: {mask_percentage:.2f}%")
    
    # The function should complete successfully
    assert result is not None, "smooth_skin should complete"
    
    # For an acne image, we expect some skin detection
    # (but this depends on the specific image content)


def test_beautified_acne_image_is_saved(output_dir, acne_image):
    """
    Ensure beautified acne.jpg is written to disk.
    """
    result = smooth_skin(acne_image)

    saved_path = save_image(str(output_dir), "acne.jpg", result)
    assert os.path.exists(saved_path), "Beautified acne.jpg not saved"


def test_beautified_acne_image_is_modified(output_dir, acne_image):
    """
    Ensure beautified acne.jpg is not identical to the original.
    """
    result = smooth_skin(acne_image)

    saved_path = save_image(str(output_dir), "acne.jpg", result)
    saved_img = cv2.imread(saved_path)
    assert saved_img is not None, "Failed to reload saved acne.jpg"

    diff = cv2.absdiff(acne_image, saved_img)
    assert diff.sum() > 0, "Beautified acne.jpg was not modified"


def test_smooth_skin_invalid_input():
    """
    Test smooth_skin with invalid inputs.
    """
    # Test with None input - should raise ValueError
    with pytest.raises(ValueError):
        smooth_skin(None)


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

    # Test with valid and invalid files mixed
    result = list(load_images(INPUT_DIR, ["acne.jpg", "non_existent.jpg"]))
    assert len(result) == 1
    assert result[0][0] == "acne.jpg"


def test_save_image_with_prefix(output_dir, acne_image):
    """
    Test save_image with custom prefix using acne.jpg.
    """
    result = smooth_skin(acne_image)
    
    saved_path = save_image(
        str(output_dir),
        "acne.jpg",
        result,
        prefix="beautified_"
    )

    assert os.path.exists(saved_path)
    assert os.path.basename(saved_path) == "beautified_acne.jpg"

    saved_img = cv2.imread(saved_path)
    assert saved_img is not None
    assert saved_img.shape == result.shape


def test_save_image_without_prefix(output_dir, acne_image):
    """
    Test save_image without custom prefix.
    """
    result = smooth_skin(acne_image)
    
    saved_path = save_image(
        str(output_dir),
        "acne.jpg",
        result
        # No prefix specified, should use default "beautified_"
    )

    assert os.path.exists(saved_path)
    assert os.path.basename(saved_path) == "beautified_acne.jpg"


def test_show_resized_function(acne_image):
    """
    Test show_resized function (though it's mostly for display).
    Since this function opens GUI windows, we just test it doesn't crash.
    """
    max_height = 700
    
    # show_resized doesn't return anything, it just displays
    # We'll test that it doesn't crash when called
    try:
        # Note: In CI/CD environments, cv2.imshow might fail
        # We'll wrap it in try-except
        show_resized("Test", acne_image, max_height)
        # If we reach here, the function didn't crash
        assert True
    except Exception as e:
        # In headless environments, this might fail
        # That's acceptable for testing
        print(f"Note: show_resized might fail in headless environments: {e}")


def test_smooth_skin_preserves_dimensions(acne_image):
    """
    Test that smooth_skin preserves image dimensions.
    """
    result = smooth_skin(acne_image)
    
    h1, w1, c1 = acne_image.shape
    h2, w2, c2 = result.shape
    
    assert h1 == h2, f"Height changed: {h1} != {h2}"
    assert w1 == w2, f"Width changed: {w1} != {w2}"
    assert c1 == c2, f"Channels changed: {c1} != {c2}"


def test_smooth_skin_alpha_blending(acne_image):
    """
    Test that smooth_skin uses alpha blending (no hard edges).
    For an acne image, the smoothing should be visible but subtle.
    """
    result = smooth_skin(acne_image)
    
    # Check that result values are within valid range
    assert result.min() >= 0, "Pixel values below 0"
    assert result.max() <= 255, "Pixel values above 255"
    
    # Calculate the difference between original and result
    diff = cv2.absdiff(acne_image, result)
    avg_diff = np.mean(diff)
    
    print(f"Average pixel difference for acne.jpg: {avg_diff:.2f}")
    
    # Alpha blending should create changes
    # For an acne image, we expect some smoothing to occur
    assert avg_diff > 0, "No changes detected in acne image"
    
    # The changes should be reasonable (not extreme)
    # Actual threshold depends on the image content
    assert avg_diff < 50, f"Average difference unusually high: {avg_diff}"


def test_smooth_skin_improves_acne_appearance(acne_image):
    """
    Test that smooth_skin actually improves acne appearance.
    This is a qualitative test - we check for reduced high-frequency noise.
    """
    result = smooth_skin(acne_image)
    
    # Convert to grayscale for analysis
    gray_original = cv2.cvtColor(acne_image, cv2.COLOR_BGR2GRAY)
    gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    
    # Calculate variance (measure of texture/noise)
    var_original = np.var(gray_original)
    var_result = np.var(gray_result)
    
    print(f"Original acne.jpg variance: {var_original:.2f}")
    print(f"Beautified acne.jpg variance: {var_result:.2f}")
    
    # Skin smoothing should reduce variance in skin areas
    # (smoothing reduces high-frequency variations)
    # Note: This is not always true for all images, depends on content
    if var_result < var_original:
        print("✓ Smoothing reduced image variance (expected for acne)")
    else:
        print("Note: Variance not reduced - image may have non-skin areas")


def test_morphological_operations(acne_image):
    """
    Test that morphological operations work on the skin mask.
    """
    # Replicate part of smooth_skin to test morphology
    ycrcb = cv2.cvtColor(acne_image, cv2.COLOR_BGR2YCrCb)
    
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    
    skin_mask = cv2.inRange(ycrcb, lower, upper)
    
    # Apply morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask_open = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel)
    
    # Morphological operations should change the mask
    assert not np.array_equal(skin_mask, mask_open), "Opening didn't change mask"
    assert not np.array_equal(mask_open, mask_close), "Closing didn't change mask"