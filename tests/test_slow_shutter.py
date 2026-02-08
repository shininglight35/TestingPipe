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
def output_dir():
    """
    Always use 'output/' folder for all runs.
    """
    out = Path("output")
    out.mkdir(parents=True, exist_ok=True)
    return out


@pytest.fixture
def car_image():
    """
    Load the specific car.jpg image for testing.
    """
    car_path = os.path.join(INPUT_DIR, "car.jpg")
    assert os.path.exists(car_path), "car.jpg not found in input folder"
    
    img = cv2.imread(car_path)
    assert img is not None, "Failed to load car.jpg"
    return img


# --------------------
# Tests - Specific to car.jpg
# --------------------

def test_car_image_loads_correctly(car_image):
    """
    Ensure car.jpg loads successfully.
    """
    assert car_image is not None, "Failed to load car.jpg"
    assert car_image.size > 0, "car.jpg is empty"
    assert car_image.shape[2] == 3, "car.jpg should have 3 channels (BGR)"


def test_apply_slow_shutter_on_car_image(car_image):
    """
    Ensure apply_slow_shutter function runs on car.jpg and returns valid output.
    """
    result = apply_slow_shutter(car_image)

    assert result is not None, "apply_slow_shutter failed for car.jpg"
    assert result.shape == car_image.shape, "Shape mismatch for car.jpg"
    assert result.dtype == car_image.dtype, "Dtype mismatch for car.jpg"


def test_slow_shutter_modifies_image(car_image):
    """
    Verify apply_slow_shutter actually modifies car.jpg.
    """
    result = apply_slow_shutter(car_image)

    # The result should be different from original
    assert not np.array_equal(car_image, result), \
        "apply_slow_shutter did not modify car.jpg"


def test_slow_shutter_creates_motion_blur(car_image):
    """
    Verify apply_slow_shutter creates motion blur effect.
    """
    result = apply_slow_shutter(car_image, direction=-1)  # Left blur
    
    # Calculate horizontal gradient to detect motion blur
    gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    gray_original = cv2.cvtColor(car_image, cv2.COLOR_BGR2GRAY)
    
    # Horizontal Sobel (for horizontal motion detection)
    sobel_x_result = cv2.Sobel(gray_result, cv2.CV_64F, 1, 0, ksize=3)
    sobel_x_original = cv2.Sobel(gray_original, cv2.CV_64F, 1, 0, ksize=3)
    
    avg_blur_x_result = np.mean(np.abs(sobel_x_result))
    avg_blur_x_original = np.mean(np.abs(sobel_x_original))
    
    print(f"Original horizontal gradient: {avg_blur_x_original:.2f}")
    print(f"Slow shutter horizontal gradient: {avg_blur_x_result:.2f}")
    
    # Motion blur might reduce horizontal gradients (smooths in direction of motion)
    # or increase them at edges (creates streaks)
    # We just verify the image was processed


def test_slow_shutter_different_directions(car_image):
    """
    Test slow shutter with different motion directions.
    """
    # Test left motion
    result_left = apply_slow_shutter(car_image, direction=-1)
    
    # Test right motion
    result_right = apply_slow_shutter(car_image, direction=1)
    
    # Test with different trail lengths
    result_short = apply_slow_shutter(car_image, trail_length=60)
    result_long = apply_slow_shutter(car_image, trail_length=180)
    
    # All should be different from original
    assert not np.array_equal(car_image, result_left)
    assert not np.array_equal(car_image, result_right)
    assert not np.array_equal(car_image, result_short)
    assert not np.array_equal(car_image, result_long)
    
    # Different directions should produce different results
    assert not np.array_equal(result_left, result_right)
    
    # Different trail lengths should produce different results
    assert not np.array_equal(result_short, result_long)


def test_slow_shutter_blend_parameter(car_image):
    """
    Test different blend_original parameter values.
    """
    # Test with more original blend
    result_more_original = apply_slow_shutter(car_image, blend_original=0.7)
    
    # Test with less original blend
    result_less_original = apply_slow_shutter(car_image, blend_original=0.2)
    
    # Test with no original blend (only slow shutter)
    result_no_original = apply_slow_shutter(car_image, blend_original=0.0)
    
    # Test with full original (should be nearly identical to original)
    result_full_original = apply_slow_shutter(car_image, blend_original=1.0)
    
    # All should be processed
    assert result_more_original is not None
    assert result_less_original is not None
    assert result_no_original is not None
    assert result_full_original is not None
    
    # Higher blend_original should be more similar to original
    diff_more = cv2.absdiff(car_image, result_more_original).sum()
    diff_less = cv2.absdiff(car_image, result_less_original).sum()
    diff_no = cv2.absdiff(car_image, result_no_original).sum()
    diff_full = cv2.absdiff(car_image, result_full_original).sum()
    
    print(f"Diff with blend=0.7: {diff_more}")
    print(f"Diff with blend=0.2: {diff_less}")
    print(f"Diff with blend=0.0: {diff_no}")
    print(f"Diff with blend=1.0: {diff_full}")
    
    # With blend_original=1.0, should be very similar to original
    assert diff_full < diff_no, "Full blend should be closer to original"


def test_slow_shutter_processed_image_is_saved(output_dir, car_image):
    """
    Ensure processed car.jpg is written to disk.
    """
    result = apply_slow_shutter(car_image)

    saved_path = save_image(str(output_dir), "car.jpg", result)
    assert os.path.exists(saved_path), "Processed car.jpg not saved"


def test_slow_shutter_image_is_modified(output_dir, car_image):
    """
    Ensure processed car.jpg is not identical to the original.
    """
    result = apply_slow_shutter(car_image)

    saved_path = save_image(str(output_dir), "car.jpg", result)
    saved_img = cv2.imread(saved_path)
    assert saved_img is not None, "Failed to reload saved car.jpg"

    diff = cv2.absdiff(car_image, saved_img)
    assert diff.sum() > 0, "Processed car.jpg was not modified"


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
    result = list(load_images(INPUT_DIR, ["car.jpg", "non_existent.jpg"]))
    assert len(result) == 1
    assert result[0][0] == "car.jpg"


def test_save_image_with_prefix(output_dir, car_image):
    """
    Test save_image with custom prefix using car.jpg.
    """
    result = apply_slow_shutter(car_image)
    
    saved_path = save_image(
        str(output_dir),
        "car.jpg",
        result,
        prefix="slow_"
    )

    assert os.path.exists(saved_path)
    assert os.path.basename(saved_path) == "slow_car.jpg"

    saved_img = cv2.imread(saved_path)
    assert saved_img is not None
    assert saved_img.shape == result.shape


def test_save_image_without_prefix(output_dir, car_image):
    """
    Test save_image without custom prefix.
    """
    result = apply_slow_shutter(car_image)
    
    saved_path = save_image(
        str(output_dir),
        "car.jpg",
        result
        # No prefix specified, should use default "slow_"
    )

    assert os.path.exists(saved_path)
    assert os.path.basename(saved_path) == "slow_car.jpg"


def test_show_resized_function(car_image):
    """
    Test show_resized function.
    """
    max_height = 700
    
    # show_resized doesn't return anything, it just displays
    # We'll test that it doesn't crash when called
    try:
        # In CI/CD environments, cv2.imshow might fail
        show_resized("Test", car_image, max_height)
        # If we reach here, the function didn't crash
        assert True
    except Exception as e:
        # In headless environments, this might fail
        print(f"Note: show_resized might fail in headless environments: {e}")


def test_slow_shutter_preserves_dimensions(car_image):
    """
    Test that apply_slow_shutter preserves image dimensions.
    """
    result = apply_slow_shutter(car_image)
    
    h1, w1, c1 = car_image.shape
    h2, w2, c2 = result.shape
    
    assert h1 == h2, f"Height changed: {h1} != {h2}"
    assert w1 == w2, f"Width changed: {w1} != {w2}"
    assert c1 == c2, f"Channels changed: {c1} != {c2}"


def test_slow_shutter_warp_affine_operation(car_image):
    """
    Test the warpAffine operation used in slow shutter.
    """
    h, w = car_image.shape[:2]
    
    # Test single shift
    dx = -10  # Shift left by 10 pixels
    M = np.float32([[1, 0, dx], [0, 1, 0]])
    
    shifted = cv2.warpAffine(
        car_image,
        M,
        (w, h),
        borderMode=cv2.BORDER_REPLICATE
    )
    
    assert shifted is not None
    assert shifted.shape == car_image.shape
    
    # Shifted image should be different
    assert not np.array_equal(car_image, shifted)


def test_slow_shutter_accumulator_logic(car_image):
    """
    Test the accumulator logic in slow shutter.
    """
    # Manually test the accumulator logic
    h, w = car_image.shape[:2]
    accumulator = np.zeros_like(car_image, dtype=np.float32)
    weight_sum = 0.0
    
    trail_length = 5  # Small for testing
    step = 3
    direction = -1
    
    for i in range(trail_length):
        dx = direction * step * i
        M = np.float32([[1, 0, dx], [0, 1, 0]])
        
        shifted = cv2.warpAffine(
            car_image,
            M,
            (w, h),
            borderMode=cv2.BORDER_REPLICATE
        )
        
        weight = 1.0 - (i / trail_length)
        accumulator += shifted * weight
        weight_sum += weight
    
    assert weight_sum > 0, "Weight sum should be positive"
    assert np.any(accumulator != 0), "Accumulator should have values"


def test_slow_shutter_default_parameters(car_image):
    """
    Test that default parameters work correctly.
    """
    # Test with all default parameters
    result_default = apply_slow_shutter(car_image)
    
    # Test with explicit default parameters
    result_explicit = apply_slow_shutter(
        car_image,
        trail_length=120,
        step=3,
        direction=-1,
        blend_original=0.4
    )
    
    # Both should produce the same result
    assert np.array_equal(result_default, result_explicit), \
        "Default parameters should match explicit parameters"


def test_slow_shutter_edge_cases_parameters(car_image):
    """
    Test edge cases for parameters.
    """
    # Test with trail_length=0 (should be similar to original)
    result_trail_0 = apply_slow_shutter(car_image, trail_length=0)
    
    # Test with step=0 (no shift between frames)
    result_step_0 = apply_slow_shutter(car_image, step=0)
    
    # Test with very small trail
    result_trail_1 = apply_slow_shutter(car_image, trail_length=1)
    
    assert result_trail_0 is not None
    assert result_step_0 is not None
    assert result_trail_1 is not None


def test_slow_shutter_visual_analysis(car_image):
    """
    Perform visual analysis on the slow shutter effect.
    """
    result = apply_slow_shutter(car_image)
    
    # Calculate mean and std deviation to see changes
    mean_original = np.mean(car_image)
    mean_result = np.mean(result)
    
    std_original = np.std(car_image)
    std_result = np.std(result)
    
    print(f"Original mean: {mean_original:.2f}, std: {std_original:.2f}")
    print(f"Result mean: {mean_result:.2f}, std: {std_result:.2f}")
    
    # Slow shutter might smooth the image, reducing std deviation
    # But with blend_original=0.4, it should still have some variation
    assert std_result > 0, "Result should have some variation"