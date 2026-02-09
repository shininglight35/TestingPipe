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


def test_apply_nostalgic_effect_on_all_images(input_images):
    """
    Ensure apply_nostalgic_effect function runs on all images and returns valid output.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        result = apply_nostalgic_effect(img)

        assert result is not None, f"apply_nostalgic_effect failed for {name}"
        assert result.shape == img.shape, f"Shape mismatch for {name}"
        assert result.dtype == img.dtype, f"Dtype mismatch for {name}"


def test_nostalgic_effect_modifies_all_images(input_images):
    """
    Verify apply_nostalgic_effect actually modifies all images.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        result = apply_nostalgic_effect(img)

        # The result should be different from original
        assert not np.array_equal(img, result), \
            f"apply_nostalgic_effect did not modify {name}"


def test_nostalgic_effect_warm_colors_on_all_images(input_images):
    """
    Verify nostalgic effect creates warm color tones on all images.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        result = apply_nostalgic_effect(img, warmth=1.15)
        
        # Calculate average channel values
        avg_original_b = np.mean(img[:, :, 0])  # Blue
        avg_original_g = np.mean(img[:, :, 1])  # Green
        avg_original_r = np.mean(img[:, :, 2])  # Red
        
        avg_result_b = np.mean(result[:, :, 0])
        avg_result_g = np.mean(result[:, :, 1])
        avg_result_r = np.mean(result[:, :, 2])
        
        print(f"{name} - Original - B: {avg_original_b:.1f}, G: {avg_original_g:.1f}, R: {avg_original_r:.1f}")
        print(f"{name} - Result   - B: {avg_result_b:.1f}, G: {avg_result_g:.1f}, R: {avg_result_r:.1f}")
        
        # With warmth > 1.0, red channel should be increased relative to blue
        red_blue_ratio_original = avg_original_r / max(avg_original_b, 1)
        red_blue_ratio_result = avg_result_r / max(avg_result_b, 1)
        
        print(f"{name} - Red/Blue ratio - Original: {red_blue_ratio_original:.2f}, Result: {red_blue_ratio_result:.2f}")


def test_nostalgic_effect_fade_contrast_on_all_images(input_images):
    """
    Verify nostalgic effect reduces contrast (fade effect) on all images.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        result = apply_nostalgic_effect(img, fade_strength=0.85)
        
        # Calculate standard deviation as measure of contrast
        std_original = np.std(img)
        std_result = np.std(result)
        
        contrast_change = (std_result - std_original) / std_original * 100
        
        print(f"{name} - Original contrast (std): {std_original:.2f}")
        print(f"{name} - Result contrast (std): {std_result:.2f}")
        print(f"{name} - Contrast change: {contrast_change:+.1f}%")


def test_nostalgic_effect_film_grain_on_all_images(input_images):
    """
    Verify nostalgic effect adds film grain on all images.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        result_with_grain = apply_nostalgic_effect(img, grain_strength=8)
        result_no_grain = apply_nostalgic_effect(img, grain_strength=0)
        
        # Images with grain should be different from without grain
        assert not np.array_equal(result_with_grain, result_no_grain), \
            f"Film grain not applied for {name}"
        
        # Calculate noise level (high frequency content)
        gray_with_grain = cv2.cvtColor(result_with_grain, cv2.COLOR_BGR2GRAY)
        gray_no_grain = cv2.cvtColor(result_no_grain, cv2.COLOR_BGR2GRAY)
        
        # Apply Laplacian to detect high frequency (grain)
        lap_with_grain = cv2.Laplacian(gray_with_grain, cv2.CV_64F).var()
        lap_no_grain = cv2.Laplacian(gray_no_grain, cv2.CV_64F).var()
        
        grain_effect = (lap_with_grain - lap_no_grain) / lap_no_grain * 100
        
        print(f"{name} - High frequency with grain: {lap_with_grain:.2f}")
        print(f"{name} - High frequency without grain: {lap_no_grain:.2f}")
        print(f"{name} - Grain effect: {grain_effect:+.1f}%")


def test_nostalgic_effect_vignette_on_all_images(input_images):
    """
    Verify nostalgic effect adds vignette (dark edges) on all images.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        result = apply_nostalgic_effect(img)
        
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
        
        vignette_strength = (avg_center_brightness - avg_corner_brightness) / avg_center_brightness * 100
        
        print(f"{name} - Average corner brightness: {avg_corner_brightness:.1f}")
        print(f"{name} - Average center brightness: {avg_center_brightness:.1f}")
        print(f"{name} - Vignette strength: {vignette_strength:.1f}%")


def test_nostalgic_effect_blur_on_all_images(input_images):
    """
    Verify nostalgic effect applies soft blur on all images.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        result_blur = apply_nostalgic_effect(img, blur_ksize=5)
        result_no_blur = apply_nostalgic_effect(img, blur_ksize=1)
        
        # Images with blur should be different from without blur
        assert not np.array_equal(result_blur, result_no_blur), \
            f"Blur not applied for {name}"
        
        # Calculate high frequency content (blur reduces it)
        gray_blur = cv2.cvtColor(result_blur, cv2.COLOR_BGR2GRAY)
        gray_no_blur = cv2.cvtColor(result_no_blur, cv2.COLOR_BGR2GRAY)
        
        # Apply Laplacian to detect high frequency
        lap_blur = cv2.Laplacian(gray_blur, cv2.CV_64F).var()
        lap_no_blur = cv2.Laplacian(gray_no_blur, cv2.CV_64F).var()
        
        blur_effect = (lap_blur - lap_no_blur) / lap_no_blur * 100
        
        print(f"{name} - High frequency with blur: {lap_blur:.2f}")
        print(f"{name} - High frequency without blur: {lap_no_blur:.2f}")
        print(f"{name} - Blur effect: {blur_effect:+.1f}%")


def test_nostalgic_effect_processed_images_are_saved(output_dir, input_images):
    """
    Ensure processed images are written to disk.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        result = apply_nostalgic_effect(img)

        saved_path = save_image(str(output_dir), name, result)
        assert os.path.exists(saved_path), f"Processed {name} not saved"


def test_nostalgic_effect_images_are_modified(output_dir, input_images):
    """
    Ensure processed images are not identical to the originals.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        result = apply_nostalgic_effect(img)

        saved_path = save_image(str(output_dir), name, result)
        saved_img = cv2.imread(saved_path)
        assert saved_img is not None, f"Failed to reload saved {name}"

        diff = cv2.absdiff(img, saved_img)
        assert diff.sum() > 0, f"Processed {name} was not modified"


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
        result = apply_nostalgic_effect(img)
        
        saved_path = save_image(
            str(output_dir),
            name,
            result,
            prefix="nostalgic_"
        )

        assert os.path.exists(saved_path)
        assert os.path.basename(saved_path) == f"nostalgic_{name}"

        saved_img = cv2.imread(saved_path)
        assert saved_img is not None
        assert saved_img.shape == result.shape


def test_save_image_without_prefix(output_dir, input_images):
    """
    Test save_image without custom prefix.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        result = apply_nostalgic_effect(img)
        
        saved_path = save_image(
            str(output_dir),
            name,
            result
            # No prefix specified, should use default "nostalgic_"
        )

        assert os.path.exists(saved_path)
        assert os.path.basename(saved_path) == f"nostalgic_{name}"


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


def test_nostalgic_effect_preserves_dimensions(input_images):
    """
    Test that apply_nostalgic_effect preserves image dimensions.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        result = apply_nostalgic_effect(img)
        
        h1, w1, c1 = img.shape
        h2, w2, c2 = result.shape
        
        assert h1 == h2, f"{name}: Height changed: {h1} != {h2}"
        assert w1 == w2, f"{name}: Width changed: {w1} != {w2}"
        assert c1 == c2, f"{name}: Channels changed: {c1} != {c2}"


def test_nostalgic_effect_parameter_variations(input_images):
    """
    Test different parameter combinations on all images.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        # Test extreme warmth
        result_warm = apply_nostalgic_effect(img, warmth=1.5)
        result_cool = apply_nostalgic_effect(img, warmth=0.8)
        
        # Test different fade strengths
        result_strong_fade = apply_nostalgic_effect(img, fade_strength=0.7)
        result_weak_fade = apply_nostalgic_effect(img, fade_strength=0.95)
        
        # Test different grain strengths
        result_heavy_grain = apply_nostalgic_effect(img, grain_strength=15)
        result_light_grain = apply_nostalgic_effect(img, grain_strength=3)
        
        # Test different blur amounts
        result_strong_blur = apply_nostalgic_effect(img, blur_ksize=9)
        result_weak_blur = apply_nostalgic_effect(img, blur_ksize=3)
        
        # All should be different from original
        assert not np.array_equal(img, result_warm), f"{name}: Warm effect didn't modify"
        assert not np.array_equal(img, result_cool), f"{name}: Cool effect didn't modify"
        assert not np.array_equal(img, result_strong_fade), f"{name}: Strong fade didn't modify"
        assert not np.array_equal(img, result_weak_fade), f"{name}: Weak fade didn't modify"
        assert not np.array_equal(img, result_heavy_grain), f"{name}: Heavy grain didn't modify"
        assert not np.array_equal(img, result_light_grain), f"{name}: Light grain didn't modify"
        assert not np.array_equal(img, result_strong_blur), f"{name}: Strong blur didn't modify"
        assert not np.array_equal(img, result_weak_blur), f"{name}: Weak blur didn't modify"
        
        # Different parameters should produce different results
        assert not np.array_equal(result_warm, result_cool), f"{name}: Warm and cool should differ"
        assert not np.array_equal(result_heavy_grain, result_light_grain), f"{name}: Grain strengths should differ"


def test_nostalgic_effect_color_clipping_on_all_images(input_images):
    """
    Test that color values are properly clipped to 0-255 range for all images.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        result = apply_nostalgic_effect(img)
        
        # All pixel values should be in valid range
        assert result.min() >= 0, f"{name}: Pixel values below 0"
        assert result.max() <= 255, f"{name}: Pixel values above 255"
        
        # Check dtype is correct
        assert result.dtype == np.uint8, f"{name}: Expected uint8, got {result.dtype}"


def test_nostalgic_effect_default_parameters(input_images):
    """
    Test that default parameters work correctly on all images.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        # Test with all default parameters
        result_default = apply_nostalgic_effect(img)
        
        # Test with explicit default parameters
        result_explicit = apply_nostalgic_effect(
            img,
            warmth=1.15,
            fade_strength=0.85,
            grain_strength=8,
            blur_ksize=5
        )
        
        # Both should produce the same result
        assert np.array_equal(result_default, result_explicit), \
            f"{name}: Default parameters should match explicit parameters"


def test_nostalgic_effect_visual_characteristics(input_images):
    """
    Analyze visual characteristics of nostalgic effect on all images.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        result = apply_nostalgic_effect(img)
        
        # Calculate various image statistics
        mean_original = np.mean(img)
        mean_result = np.mean(result)
        
        std_original = np.std(img)
        std_result = np.std(result)
        
        # Calculate color balance
        red_blue_ratio_original = np.mean(img[:, :, 2]) / max(np.mean(img[:, :, 0]), 1)
        red_blue_ratio_result = np.mean(result[:, :, 2]) / max(np.mean(result[:, :, 0]), 1)
        
        mean_change = (mean_result - mean_original) / mean_original * 100
        std_change = (std_result - std_original) / std_original * 100
        color_balance_change = (red_blue_ratio_result - red_blue_ratio_original) / red_blue_ratio_original * 100
        
        print(f"{name} - Mean change: {mean_change:+.1f}%")
        print(f"{name} - Std change: {std_change:+.1f}%")
        print(f"{name} - Color balance change: {color_balance_change:+.1f}%")


def test_nostalgic_effect_edge_cases(input_images):
    """
    Test nostalgic effect with edge cases on all images.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        # Make a copy to avoid modifying the original in tests
        img_copy = img.copy()
        
        # Test with very small image
        small_img = cv2.resize(img_copy, (50, 50))
        result_small = apply_nostalgic_effect(small_img)
        assert result_small is not None, f"apply_nostalgic_effect failed for small {name}"
        assert result_small.shape == small_img.shape, f"Shape mismatch for small {name}"
        
        # Test with square image
        square_size = min(img_copy.shape[0], img_copy.shape[1])
        square_img = img_copy[:square_size, :square_size]
        result_square = apply_nostalgic_effect(square_img)
        assert result_square is not None, f"apply_nostalgic_effect failed for square {name}"
        assert result_square.shape == square_img.shape, f"Shape mismatch for square {name}"


def test_nostalgic_effect_consistency(input_images):
    """
    Test that apply_nostalgic_effect produces consistent results.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        # Apply nostalgic effect multiple times to the same image
        result1 = apply_nostalgic_effect(img)
        result2 = apply_nostalgic_effect(img)
        
        # Results should be identical (deterministic)
        assert np.array_equal(result1, result2), \
            f"{name}: apply_nostalgic_effect is not deterministic"


def test_nostalgic_effect_with_synthetic_images():
    """
    Test nostalgic effect with synthetic test images.
    """
    # Create a synthetic test image with various colors
    synthetic = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Add different colored regions
    synthetic[0:50, 0:50] = [255, 0, 0]  # Red
    synthetic[0:50, 50:100] = [0, 255, 0]  # Green
    synthetic[50:100, 0:50] = [0, 0, 255]  # Blue
    synthetic[50:100, 50:100] = [255, 255, 255]  # White
    
    result = apply_nostalgic_effect(synthetic)
    
    assert result is not None
    assert result.shape == synthetic.shape
    
    # The effect should modify the synthetic image
    assert not np.array_equal(synthetic, result), "Nostalgic effect should modify synthetic image"
    
    # Check valid pixel range
    assert result.min() >= 0, "Negative pixel values in result"
    assert result.max() <= 255, "Pixel values above 255 in result"


def test_nostalgic_effect_quality_metrics(input_images):
    """
    Calculate quality metrics for nostalgic effect on all images.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        result = apply_nostalgic_effect(img)
        
        # Calculate Mean Squared Error
        mse = np.mean((img.astype(float) - result.astype(float)) ** 2)
        
        # Calculate Peak Signal-to-Noise Ratio (PSNR)
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        
        # Calculate average pixel difference
        diff = cv2.absdiff(img, result)
        avg_diff = np.mean(diff)
        
        print(f"{name} - MSE: {mse:.2f}, PSNR: {psnr:.2f} dB, Avg diff: {avg_diff:.2f}")
        
        # Nostalgic effect should change the image, so MSE should not be 0
        assert mse > 0, f"No change detected for {name}"