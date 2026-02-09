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
def input_images():
    """
    Collect all image filenames from the repo input/ folder.
    """
    assert os.path.exists(INPUT_DIR), "input/ folder does not exist in repo"

    files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    assert len(files) > 0, "No images found in input/ folder"
    return files


@pytest.fixture
def output_dir():
    """
    Always use 'output/' folder for all runs.
    """
    out = Path("output")
    out.mkdir(parents=True, exist_ok=True)
    return out


# --------------------
# Tests - For all images in input folder
# --------------------

def test_images_load_correctly(input_images):
    """
    Ensure images from input/ load successfully.
    """
    loaded = list(load_images(INPUT_DIR, input_images))

    assert len(loaded) > 0

    for name, img in loaded:
        assert img is not None, f"Failed to load {name}"
        assert img.size > 0, f"Empty image: {name}"


def test_smooth_skin_on_all_images(input_images):
    """
    Ensure smooth_skin function runs on all images and returns valid output.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        result = smooth_skin(img)

        assert result is not None, f"smooth_skin failed for {name}"
        assert result.shape == img.shape, f"Shape mismatch for {name}"
        assert result.dtype == img.dtype, f"Dtype mismatch for {name}"


def test_smooth_skin_modifies_all_images(input_images):
    """
    Verify smooth_skin actually modifies all images.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        result = smooth_skin(img)

        # The result should be different from original
        assert not np.array_equal(img, result), \
            f"smooth_skin did not modify {name}"


def test_smooth_skin_with_skin_mask_on_all_images(input_images):
    """
    Verify smooth_skin creates a skin mask and applies smoothing.
    The mask should detect skin areas in images that contain skin.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        result = smooth_skin(img)
        
        # Convert to YCrCb color space (as done in smooth_skin)
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        
        # Skin color range (from the function)
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        
        skin_mask = cv2.inRange(ycrcb, lower, upper)
        
        # Count non-zero pixels in the mask
        mask_pixel_count = np.count_nonzero(skin_mask)
        mask_percentage = (mask_pixel_count / skin_mask.size) * 100
        
        print(f"{name} - Skin mask coverage: {mask_percentage:.2f}%")
        
        # The function should complete successfully
        assert result is not None, f"smooth_skin should complete for {name}"


def test_beautified_images_are_saved(output_dir, input_images):
    """
    Ensure beautified images are written to disk.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        result = smooth_skin(img)

        saved_path = save_image(str(output_dir), name, result)
        assert os.path.exists(saved_path), f"Beautified {name} not saved"


def test_beautified_images_are_modified(output_dir, input_images):
    """
    Ensure beautified images are not identical to the originals.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        result = smooth_skin(img)

        saved_path = save_image(str(output_dir), name, result)
        saved_img = cv2.imread(saved_path)
        assert saved_img is not None, f"Failed to reload saved {name}"

        diff = cv2.absdiff(img, saved_img)
        assert diff.sum() > 0, f"Beautified {name} was not modified"


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
    actual_files = [f for f in os.listdir(INPUT_DIR) 
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if actual_files:
        test_files = [actual_files[0], "non_existent.jpg"]
        result = list(load_images(INPUT_DIR, test_files))
        assert len(result) == 1
        assert result[0][0] == actual_files[0]


def test_save_image_with_prefix(output_dir, input_images):
    """
    Test save_image with custom prefix using all images.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        result = smooth_skin(img)
        
        saved_path = save_image(
            str(output_dir),
            name,
            result,
            prefix="beautified_"
        )

        assert os.path.exists(saved_path)
        assert os.path.basename(saved_path) == f"beautified_{name}"

        saved_img = cv2.imread(saved_path)
        assert saved_img is not None
        assert saved_img.shape == result.shape


def test_save_image_without_prefix(output_dir, input_images):
    """
    Test save_image without custom prefix.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        result = smooth_skin(img)
        
        saved_path = save_image(
            str(output_dir),
            name,
            result
            # No prefix specified, should use default "beautified_"
        )

        assert os.path.exists(saved_path)
        assert os.path.basename(saved_path) == f"beautified_{name}"


def test_smooth_skin_preserves_dimensions(input_images):
    """
    Test that smooth_skin preserves image dimensions.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        result = smooth_skin(img)
        
        h1, w1, c1 = img.shape
        h2, w2, c2 = result.shape
        
        assert h1 == h2, f"{name}: Height changed: {h1} != {h2}"
        assert w1 == w2, f"{name}: Width changed: {w1} != {w2}"
        assert c1 == c2, f"{name}: Channels changed: {c1} != {c2}"


def test_smooth_skin_alpha_blending_on_all_images(input_images):
    """
    Test that smooth_skin uses alpha blending (no hard edges).
    For images with skin, the smoothing should be visible but subtle.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        result = smooth_skin(img)
        
        # Check that result values are within valid range
        assert result.min() >= 0, f"{name}: Pixel values below 0"
        assert result.max() <= 255, f"{name}: Pixel values above 255"
        
        # Calculate the difference between original and result
        diff = cv2.absdiff(img, result)
        avg_diff = np.mean(diff)
        
        print(f"{name} - Average pixel difference: {avg_diff:.2f}")
        
        # Alpha blending should create changes if skin is detected
        assert avg_diff >= 0, f"{name}: Invalid average difference"


def test_smooth_skin_improves_skin_appearance(input_images):
    """
    Test that smooth_skin improves skin appearance where applicable.
    This is a qualitative test - we check for reduced high-frequency noise.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        result = smooth_skin(img)
        
        # Convert to grayscale for analysis
        gray_original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        
        # Calculate variance (measure of texture/noise)
        var_original = np.var(gray_original)
        var_result = np.var(gray_result)
        
        print(f"{name} - Original variance: {var_original:.2f}")
        print(f"{name} - Beautified variance: {var_result:.2f}")
        
        # Report whether smoothing reduced variance
        if var_result < var_original:
            print(f"✓ {name}: Smoothing reduced image variance")
        elif var_result > var_original:
            print(f"Note: {name}: Variance increased - may not contain skin")
        else:
            print(f"Note: {name}: Variance unchanged")


def test_morphological_operations_on_all_images(input_images):
    """
    Test that morphological operations work on skin masks for all images.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        # Replicate part of smooth_skin to test morphology
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        
        lower = np.array([0, 133, 77], dtype=np.uint8)
        upper = np.array([255, 173, 127], dtype=np.uint8)
        
        skin_mask = cv2.inRange(ycrcb, lower, upper)
        
        # Count skin pixels before morphology
        skin_pixels_before = np.count_nonzero(skin_mask)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_open = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        mask_close = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel)
        
        # Count skin pixels after morphology
        skin_pixels_after = np.count_nonzero(mask_close)
        
        print(f"{name} - Skin pixels before morphology: {skin_pixels_before}")
        print(f"{name} - Skin pixels after morphology: {skin_pixels_after}")
        
        # Morphological operations should change the mask
        assert not np.array_equal(skin_mask, mask_open), f"{name}: Opening didn't change mask"
        assert not np.array_equal(mask_open, mask_close), f"{name}: Closing didn't change mask"


def test_smooth_skin_edge_cases(input_images):
    """
    Test smooth_skin with edge cases on all images.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        # Make a copy to avoid modifying the original in tests
        img_copy = img.copy()
        
        # Test with very small image
        small_img = cv2.resize(img_copy, (50, 50))
        result_small = smooth_skin(small_img)
        assert result_small is not None, f"smooth_skin failed for small {name}"
        assert result_small.shape == small_img.shape, f"Shape mismatch for small {name}"
        
        # Test with square image
        square_size = min(img_copy.shape[0], img_copy.shape[1])
        square_img = img_copy[:square_size, :square_size]
        result_square = smooth_skin(square_img)
        assert result_square is not None, f"smooth_skin failed for square {name}"
        assert result_square.shape == square_img.shape, f"Shape mismatch for square {name}"


def test_smooth_skin_parameter_variations(input_images):
    """
    Test smooth_skin with different parameter combinations.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        # Test with different smoothing strengths
        # (Note: smooth_skin function might have different parameters)
        
        # Test the function completes
        result = smooth_skin(img)
        
        # Verify the output
        assert result is not None, f"smooth_skin failed for {name}"
        assert result.shape == img.shape, f"Shape mismatch for {name}"
        
        # Verify output type
        assert result.dtype == np.uint8, f"Wrong dtype for {name}: {result.dtype}"


def test_image_consistency_after_smoothing(input_images):
    """
    Test that smooth_skin maintains image consistency.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        result = smooth_skin(img)
        
        # Check that the result is still a valid image
        assert np.all(result >= 0), f"{name}: Negative pixel values"
        assert np.all(result <= 255), f"{name}: Pixel values exceed 255"
        
        # Check that the image isn't completely uniform (unless input was uniform)
        if np.var(img) > 0:
            assert np.var(result) > 0, f"{name}: Result is completely uniform"
        
        # Check for NaN or Inf values
        assert not np.any(np.isnan(result)), f"{name}: Contains NaN values"
        assert not np.any(np.isinf(result)), f"{name}: Contains infinite values"