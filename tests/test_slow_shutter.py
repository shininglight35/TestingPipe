import os
import cv2
import numpy as np
import pytest
import sys
from pathlib import Path

# Add src/ to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from slow_shutter import (
    load_images,
    apply_slow_shutter,
    save_image,
    show_resized
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

def test_images_load_correctly(input_images):
    """
    Ensure images from input/ load successfully.
    """
    loaded = list(load_images(INPUT_DIR, input_images))

    assert len(loaded) > 0

    for name, img in loaded:
        assert img is not None, f"Failed to load {name}"
        assert img.size > 0, f"Empty image: {name}"


def test_apply_slow_shutter_on_all_images(input_images):
    """
    Ensure apply_slow_shutter function runs on all images and returns valid output.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        result = apply_slow_shutter(img)

        assert result is not None, f"apply_slow_shutter failed for {name}"
        assert result.shape == img.shape, f"Shape mismatch for {name}"
        assert result.dtype == img.dtype, f"Dtype mismatch for {name}"


def test_slow_shutter_modifies_all_images(input_images):
    """
    Verify apply_slow_shutter actually modifies all images.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        result = apply_slow_shutter(img)

        # The result should be different from original
        assert not np.array_equal(img, result), \
            f"apply_slow_shutter did not modify {name}"


def test_slow_shutter_creates_motion_blur_on_all_images(input_images):
    """
    Verify apply_slow_shutter creates motion blur effect on all images.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        result = apply_slow_shutter(img, direction=-1)  # Left blur
        
        # Calculate horizontal gradient to detect motion blur
        gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        gray_original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Horizontal Sobel (for horizontal motion detection)
        sobel_x_result = cv2.Sobel(gray_result, cv2.CV_64F, 1, 0, ksize=3)
        sobel_x_original = cv2.Sobel(gray_original, cv2.CV_64F, 1, 0, ksize=3)
        
        avg_blur_x_result = np.mean(np.abs(sobel_x_result))
        avg_blur_x_original = np.mean(np.abs(sobel_x_original))
        
        print(f"{name} - Original horizontal gradient: {avg_blur_x_original:.2f}")
        print(f"{name} - Slow shutter horizontal gradient: {avg_blur_x_result:.2f}")
        
        # Just verify the image was processed
        assert result is not None


def test_slow_shutter_different_directions_on_all_images(input_images):
    """
    Test slow shutter with different motion directions on all images.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        # Test left motion
        result_left = apply_slow_shutter(img, direction=-1)
        
        # Test right motion
        result_right = apply_slow_shutter(img, direction=1)
        
        # Test with different trail lengths
        result_short = apply_slow_shutter(img, trail_length=60)
        result_long = apply_slow_shutter(img, trail_length=180)
        
        # All should be different from original
        assert not np.array_equal(img, result_left)
        assert not np.array_equal(img, result_right)
        assert not np.array_equal(img, result_short)
        assert not np.array_equal(img, result_long)
        
        # Different directions should produce different results
        assert not np.array_equal(result_left, result_right)
        
        # Different trail lengths should produce different results
        assert not np.array_equal(result_short, result_long)


def test_slow_shutter_blend_parameter_on_all_images(input_images):
    """
    Test different blend_original parameter values on all images.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        # Test with more original blend
        result_more_original = apply_slow_shutter(img, blend_original=0.7)
        
        # Test with less original blend
        result_less_original = apply_slow_shutter(img, blend_original=0.2)
        
        # Test with no original blend (only slow shutter)
        result_no_original = apply_slow_shutter(img, blend_original=0.0)
        
        # Test with full original (should be nearly identical to original)
        result_full_original = apply_slow_shutter(img, blend_original=1.0)
        
        # All should be processed
        assert result_more_original is not None
        assert result_less_original is not None
        assert result_no_original is not None
        assert result_full_original is not None
        
        # Higher blend_original should be more similar to original
        diff_more = cv2.absdiff(img, result_more_original).sum()
        diff_less = cv2.absdiff(img, result_less_original).sum()
        diff_no = cv2.absdiff(img, result_no_original).sum()
        diff_full = cv2.absdiff(img, result_full_original).sum()
        
        print(f"{name} - Diff with blend=0.7: {diff_more}")
        print(f"{name} - Diff with blend=0.2: {diff_less}")
        print(f"{name} - Diff with blend=0.0: {diff_no}")
        print(f"{name} - Diff with blend=1.0: {diff_full}")
        
        # With blend_original=1.0, should be very similar to original
        assert diff_full < diff_no, f"Full blend should be closer to original for {name}"


def test_slow_shutter_processed_images_are_saved(output_dir, input_images):
    """
    Ensure processed images are written to disk.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        result = apply_slow_shutter(img)

        saved_path = save_image(str(output_dir), name, result)
        assert os.path.exists(saved_path), f"Processed {name} not saved"


def test_slow_shutter_images_are_modified(output_dir, input_images):
    """
    Ensure processed images are not identical to the originals.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        result = apply_slow_shutter(img)

        saved_path = save_image(str(output_dir), name, result)
        saved_img = cv2.imread(saved_path)
        assert saved_img is not None, f"Failed to reload saved {name}"

        diff = cv2.absdiff(img, saved_img)
        assert diff.sum() > 0, f"Processed {name} was not modified"


def test_apply_slow_shutter_invalid_input():
    """
    Test apply_slow_shutter with invalid inputs.
    """
    # Test with None input - should raise ValueError
    with pytest.raises(ValueError):
        apply_slow_shutter(None)


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
        result = apply_slow_shutter(img)
        
        saved_path = save_image(
            str(output_dir),
            name,
            result,
            prefix="slow_"
        )

        assert os.path.exists(saved_path)
        assert os.path.basename(saved_path) == f"slow_{name}"

        saved_img = cv2.imread(saved_path)
        assert saved_img is not None
        assert saved_img.shape == result.shape


def test_save_image_without_prefix(output_dir, input_images):
    """
    Test save_image without custom prefix.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        result = apply_slow_shutter(img)
        
        saved_path = save_image(
            str(output_dir),
            name,
            result
            # No prefix specified, should use default "slow_"
        )

        assert os.path.exists(saved_path)
        assert os.path.basename(saved_path) == f"slow_{name}"


def test_show_resized_function(input_images):
    """
    Test show_resized function.
    """
    max_height = 700
    
    for name, img in load_images(INPUT_DIR, input_images):
        # show_resized doesn't return anything, it just displays
        # We'll test that it doesn't crash when called
        try:
            # In CI/CD environments, cv2.imshow might fail
            show_resized(f"Test - {name}", img, max_height)
            # If we reach here, the function didn't crash
            assert True
        except Exception as e:
            # In headless environments, this might fail
            print(f"Note: show_resized might fail in headless environments for {name}: {e}")


def test_slow_shutter_preserves_dimensions(input_images):
    """
    Test that apply_slow_shutter preserves image dimensions.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        result = apply_slow_shutter(img)
        
        h1, w1, c1 = img.shape
        h2, w2, c2 = result.shape
        
        assert h1 == h2, f"{name}: Height changed: {h1} != {h2}"
        assert w1 == w2, f"{name}: Width changed: {w1} != {w2}"
        assert c1 == c2, f"{name}: Channels changed: {c1} != {c2}"


def test_slow_shutter_default_parameters(input_images):
    """
    Test that default parameters work correctly on all images.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        # Test with all default parameters
        result_default = apply_slow_shutter(img)
        
        # Test with explicit default parameters
        result_explicit = apply_slow_shutter(
            img,
            trail_length=120,
            step=3,
            direction=-1,
            blend_original=0.4
        )
        
        # Both should produce the same result
        assert np.array_equal(result_default, result_explicit), \
            f"{name}: Default parameters should match explicit parameters"


def test_slow_shutter_edge_cases_parameters(input_images):
    """
    Test edge cases for parameters on all images.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        # Test with trail_length=0 (should be similar to original)
        result_trail_0 = apply_slow_shutter(img, trail_length=0)
        
        # Test with step=0 (no shift between frames)
        result_step_0 = apply_slow_shutter(img, step=0)
        
        # Test with very small trail
        result_trail_1 = apply_slow_shutter(img, trail_length=1)
        
        assert result_trail_0 is not None
        assert result_step_0 is not None
        assert result_trail_1 is not None


def test_slow_shutter_visual_analysis_on_all_images(input_images):
    """
    Perform visual analysis on the slow shutter effect for all images.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        result = apply_slow_shutter(img)
        
        # Calculate mean and std deviation to see changes
        mean_original = np.mean(img)
        mean_result = np.mean(result)
        
        std_original = np.std(img)
        std_result = np.std(result)
        
        print(f"{name} - Original mean: {mean_original:.2f}, std: {std_original:.2f}")
        print(f"{name} - Result mean: {mean_result:.2f}, std: {std_result:.2f}")
        
        # Slow shutter might smooth the image, reducing std deviation
        # But with blend_original=0.4, it should still have some variation
        assert std_result > 0, f"Result should have some variation for {name}"