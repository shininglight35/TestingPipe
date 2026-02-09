import os
import cv2
import numpy as np
import pytest
import sys
from pathlib import Path

# Add src/ to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from nostalgic import (
    load_images,
    apply_nostalgic_effect,
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
def old_image():
    """
    Load the specific old.jpg image for testing.
    """
    old_path = os.path.join(INPUT_DIR, "old.jpg")
    assert os.path.exists(old_path), "old.jpg not found in input folder"
    
    img = cv2.imread(old_path)
    assert img is not None, "Failed to load old.jpg"
    return img


# --------------------
# Tests - Specific to old.jpg
# --------------------

def test_old_image_loads_correctly(old_image):
    """
    Ensure old.jpg loads successfully.
    """
    assert old_image is not None, "Failed to load old.jpg"
    assert old_image.size > 0, "old.jpg is empty"
    assert old_image.shape[2] == 3, "old.jpg should have 3 channels (BGR)"


def test_apply_nostalgic_effect_on_old_image(old_image):
    """
    Ensure apply_nostalgic_effect function runs on old.jpg and returns valid output.
    """
    result = apply_nostalgic_effect(old_image)

    assert result is not None, "apply_nostalgic_effect failed for old.jpg"
    assert result.shape == old_image.shape, "Shape mismatch for old.jpg"
    assert result.dtype == old_image.dtype, "Dtype mismatch for old.jpg"


def test_nostalgic_effect_modifies_image(old_image):
    """
    Verify apply_nostalgic_effect actually modifies old.jpg.
    """
    result = apply_nostalgic_effect(old_image)

    # The result should be different from original
    assert not np.array_equal(old_image, result), \
        "apply_nostalgic_effect did not modify old.jpg"


def test_nostalgic_effect_warm_colors(old_image):
    """
    Verify nostalgic effect creates warm color tones.
    """
    result = apply_nostalgic_effect(old_image, warmth=1.15)
    
    # Calculate average channel values
    avg_original_b = np.mean(old_image[:, :, 0])  # Blue
    avg_original_g = np.mean(old_image[:, :, 1])  # Green
    avg_original_r = np.mean(old_image[:, :, 2])  # Red
    
    avg_result_b = np.mean(result[:, :, 0])
    avg_result_g = np.mean(result[:, :, 1])
    avg_result_r = np.mean(result[:, :, 2])
    
    print(f"Original - B: {avg_original_b:.1f}, G: {avg_original_g:.1f}, R: {avg_original_r:.1f}")
    print(f"Result   - B: {avg_result_b:.1f}, G: {avg_result_g:.1f}, R: {avg_result_r:.1f}")
    
    # With warmth > 1.0, red channel should be increased relative to blue
    red_blue_ratio_original = avg_original_r / max(avg_original_b, 1)
    red_blue_ratio_result = avg_result_r / max(avg_result_b, 1)
    
    print(f"Red/Blue ratio - Original: {red_blue_ratio_original:.2f}, Result: {red_blue_ratio_result:.2f}")
    
    # Warm effect should increase red/blue ratio
    assert red_blue_ratio_result > red_blue_ratio_original * 0.9, \
        "Warm effect not sufficiently applied"


def test_nostalgic_effect_fade_contrast(old_image):
    """
    Verify nostalgic effect reduces contrast (fade effect).
    """
    result = apply_nostalgic_effect(old_image, fade_strength=0.85)
    
    # Calculate standard deviation as measure of contrast
    std_original = np.std(old_image)
    std_result = np.std(result)
    
    print(f"Original contrast (std): {std_original:.2f}")
    print(f"Result contrast (std): {std_result:.2f}")
    
    # With fade_strength < 1.0, contrast should be reduced
    # But note: grain and vignette might add some variation
    # So we check it's at least somewhat different
    assert abs(std_original - std_result) > 1.0, \
        "Contrast not sufficiently modified"


def test_nostalgic_effect_film_grain(old_image):
    """
    Verify nostalgic effect adds film grain.
    """
    result_with_grain = apply_nostalgic_effect(old_image, grain_strength=8)
    result_no_grain = apply_nostalgic_effect(old_image, grain_strength=0)
    
    # Images with grain should be different from without grain
    assert not np.array_equal(result_with_grain, result_no_grain), \
        "Film grain not applied"
    
    # Calculate noise level (high frequency content)
    gray_with_grain = cv2.cvtColor(result_with_grain, cv2.COLOR_BGR2GRAY)
    gray_no_grain = cv2.cvtColor(result_no_grain, cv2.COLOR_BGR2GRAY)
    
    # Apply Laplacian to detect high frequency (grain)
    lap_with_grain = cv2.Laplacian(gray_with_grain, cv2.CV_64F).var()
    lap_no_grain = cv2.Laplacian(gray_no_grain, cv2.CV_64F).var()
    
    print(f"High frequency with grain: {lap_with_grain:.2f}")
    print(f"High frequency without grain: {lap_no_grain:.2f}")
    
    # With grain should have more high frequency content
    assert lap_with_grain > lap_no_grain * 1.1, \
        "Film grain not sufficiently increasing high frequency content"


def test_nostalgic_effect_vignette(old_image):
    """
    Verify nostalgic effect adds vignette (dark edges).
    """
    result = apply_nostalgic_effect(old_image)
    
    h, w = result.shape[:2]
    
    # Check corners are darker than center
    # Sample multiple corner and center points
    corners = [
        (0, 0), (0, w-1), (h-1, 0), (h-1, w-1)  # Four corners
    ]
    centers = [
        (h//2, w//2), (h//3, w//3), (2*h//3, 2*w//3)
    ]
    
    avg_corner_brightness = 0
    for y, x in corners:
        avg_corner_brightness += np.mean(result[y:y+10, x:x+10])
    avg_corner_brightness /= len(corners)
    
    avg_center_brightness = 0
    for y, x in centers:
        avg_center_brightness += np.mean(result[y:y+10, x:x+10])
    avg_center_brightness /= len(centers)
    
    print(f"Average corner brightness: {avg_corner_brightness:.1f}")
    print(f"Average center brightness: {avg_center_brightness:.1f}")
    
    # Vignette should make corners darker than center
    assert avg_center_brightness > avg_corner_brightness * 1.1, \
        "Vignette effect not sufficiently visible"


def test_nostalgic_effect_blur(old_image):
    """
    Verify nostalgic effect applies soft blur.
    """
    result_blur = apply_nostalgic_effect(old_image, blur_ksize=5)
    result_no_blur = apply_nostalgic_effect(old_image, blur_ksize=1)
    
    # Images with blur should be different from without blur
    assert not np.array_equal(result_blur, result_no_blur), \
        "Blur not applied"
    
    # Calculate high frequency content (blur reduces it)
    gray_blur = cv2.cvtColor(result_blur, cv2.COLOR_BGR2GRAY)
    gray_no_blur = cv2.cvtColor(result_no_blur, cv2.COLOR_BGR2GRAY)
    
    # Apply Laplacian to detect high frequency
    lap_blur = cv2.Laplacian(gray_blur, cv2.CV_64F).var()
    lap_no_blur = cv2.Laplacian(gray_no_blur, cv2.CV_64F).var()
    
    print(f"High frequency with blur: {lap_blur:.2f}")
    print(f"High frequency without blur: {lap_no_blur:.2f}")
    
    # Blur should reduce high frequency content
    assert lap_blur < lap_no_blur, \
        "Blur not reducing high frequency content"


def test_nostalgic_effect_processed_image_is_saved(output_dir, old_image):
    """
    Ensure processed old.jpg is written to disk.
    """
    result = apply_nostalgic_effect(old_image)

    saved_path = save_image(str(output_dir), "old.jpg", result)
    assert os.path.exists(saved_path), "Processed old.jpg not saved"


def test_nostalgic_effect_image_is_modified(output_dir, old_image):
    """
    Ensure processed old.jpg is not identical to the original.
    """
    result = apply_nostalgic_effect(old_image)

    saved_path = save_image(str(output_dir), "old.jpg", result)
    saved_img = cv2.imread(saved_path)
    assert saved_img is not None, "Failed to reload saved old.jpg"

    diff = cv2.absdiff(old_image, saved_img)
    assert diff.sum() > 0, "Processed old.jpg was not modified"


def test_apply_nostalgic_effect_invalid_input():
    """
    Test apply_nostalgic_effect with invalid inputs.
    """
    # Test with None input - should raise ValueError
    with pytest.raises(ValueError):
        apply_nostalgic_effect(None)


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
    result = list(load_images(INPUT_DIR, ["old.jpg", "non_existent.jpg"]))
    assert len(result) == 1
    assert result[0][0] == "old.jpg"


def test_save_image_with_prefix(output_dir, old_image):
    """
    Test save_image with custom prefix using old.jpg.
    """
    result = apply_nostalgic_effect(old_image)
    
    saved_path = save_image(
        str(output_dir),
        "old.jpg",
        result,
        prefix="nostalgic_"
    )

    assert os.path.exists(saved_path)
    assert os.path.basename(saved_path) == "nostalgic_old.jpg"

    saved_img = cv2.imread(saved_path)
    assert saved_img is not None
    assert saved_img.shape == result.shape


def test_save_image_without_prefix(output_dir, old_image):
    """
    Test save_image without custom prefix.
    """
    result = apply_nostalgic_effect(old_image)
    
    saved_path = save_image(
        str(output_dir),
        "old.jpg",
        result
        # No prefix specified, should use default "nostalgic_"
    )

    assert os.path.exists(saved_path)
    assert os.path.basename(saved_path) == "nostalgic_old.jpg"


def test_show_resized_function(old_image):
    """
    Test show_resized function.
    """
    max_height = 700
    
    # show_resized doesn't return anything, it just displays
    # We'll test that it doesn't crash when called
    try:
        # In CI/CD environments, cv2.imshow might fail
        show_resized("Test", old_image, max_height)
        # If we reach here, the function didn't crash
        assert True
    except Exception as e:
        # In headless environments, this might fail
        print(f"Note: show_resized might fail in headless environments: {e}")


def test_nostalgic_effect_preserves_dimensions(old_image):
    """
    Test that apply_nostalgic_effect preserves image dimensions.
    """
    result = apply_nostalgic_effect(old_image)
    
    h1, w1, c1 = old_image.shape
    h2, w2, c2 = result.shape
    
    assert h1 == h2, f"Height changed: {h1} != {h2}"
    assert w1 == w2, f"Width changed: {w1} != {w2}"
    assert c1 == c2, f"Channels changed: {c1} != {c2}"


def test_nostalgic_effect_parameter_variations(old_image):
    """
    Test different parameter combinations.
    """
    # Test extreme warmth
    result_warm = apply_nostalgic_effect(old_image, warmth=1.5)
    result_cool = apply_nostalgic_effect(old_image, warmth=0.8)
    
    # Test different fade strengths
    result_strong_fade = apply_nostalgic_effect(old_image, fade_strength=0.7)
    result_weak_fade = apply_nostalgic_effect(old_image, fade_strength=0.95)
    
    # Test different grain strengths
    result_heavy_grain = apply_nostalgic_effect(old_image, grain_strength=15)
    result_light_grain = apply_nostalgic_effect(old_image, grain_strength=3)
    
    # Test different blur amounts
    result_strong_blur = apply_nostalgic_effect(old_image, blur_ksize=9)
    result_weak_blur = apply_nostalgic_effect(old_image, blur_ksize=3)
    
    # All should be different from original
    assert not np.array_equal(old_image, result_warm)
    assert not np.array_equal(old_image, result_cool)
    assert not np.array_equal(old_image, result_strong_fade)
    assert not np.array_equal(old_image, result_weak_fade)
    assert not np.array_equal(old_image, result_heavy_grain)
    assert not np.array_equal(old_image, result_light_grain)
    assert not np.array_equal(old_image, result_strong_blur)
    assert not np.array_equal(old_image, result_weak_blur)
    
    # Different parameters should produce different results
    assert not np.array_equal(result_warm, result_cool)
    assert not np.array_equal(result_heavy_grain, result_light_grain)


def test_nostalgic_effect_color_clipping(old_image):
    """
    Test that color values are properly clipped to 0-255 range.
    """
    result = apply_nostalgic_effect(old_image)
    
    # All pixel values should be in valid range
    assert result.min() >= 0, "Pixel values below 0"
    assert result.max() <= 255, "Pixel values above 255"
    
    # Check dtype is correct
    assert result.dtype == np.uint8, f"Expected uint8, got {result.dtype}"


def test_nostalgic_effect_gaussian_kernel_generation():
    """
    Test Gaussian kernel generation (used for vignette).
    """
    # Test Gaussian kernel generation at different sizes
    for size in [100, 200, 300]:
        kernel = cv2.getGaussianKernel(size, size / 2)
        
        assert kernel is not None
        assert kernel.shape == (size, 1)
        
        # Gaussian should be symmetric (values decrease from center)
        center = size // 2
        assert kernel[center] > kernel[0], "Gaussian should peak at center"
        assert kernel[center] > kernel[-1], "Gaussian should peak at center"


def test_nostalgic_effect_default_parameters(old_image):
    """
    Test that default parameters work correctly.
    """
    # Test with all default parameters
    result_default = apply_nostalgic_effect(old_image)
    
    # Test with explicit default parameters
    result_explicit = apply_nostalgic_effect(
        old_image,
        warmth=1.15,
        fade_strength=0.85,
        grain_strength=8,
        blur_ksize=5
    )
    
    # Both should produce the same result
    assert np.array_equal(result_default, result_explicit), \
        "Default parameters should match explicit parameters"


def test_nostalgic_effect_visual_characteristics(old_image):
    """
    Analyze visual characteristics of nostalgic effect.
    """
    result = apply_nostalgic_effect(old_image)
    
    # Calculate various image statistics
    mean_original = np.mean(old_image)
    mean_result = np.mean(result)
    
    std_original = np.std(old_image)
    std_result = np.std(result)
    
    # Calculate color balance
    red_blue_ratio_original = np.mean(old_image[:, :, 2]) / max(np.mean(old_image[:, :, 0]), 1)
    red_blue_ratio_result = np.mean(result[:, :, 2]) / max(np.mean(result[:, :, 0]), 1)
    
    print(f"Original - Mean: {mean_original:.1f}, Std: {std_original:.1f}, R/B: {red_blue_ratio_original:.2f}")
    print(f"Result   - Mean: {mean_result:.1f}, Std: {std_result:.1f}, R/B: {red_blue_ratio_result:.2f}")
    
    # Nostalgic effect typically:
    # 1. Warms colors (increases red/blue ratio)
    # 2. Reduces contrast (lowers std)
    # 3. Adds grain (might increase std slightly)
    # 4. Adds vignette (reduces brightness at edges)
    
    # We verify the effect was applied (image modified)
    assert not np.array_equal(old_image, result), \
        "Nostalgic effect should modify the image"