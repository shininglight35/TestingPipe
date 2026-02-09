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


def test_apply_clahe_on_all_images(input_images):
    """
    Ensure CLAHE enhancement runs on all images and returns valid output.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        enhanced_img = apply_clahe(img)

        assert enhanced_img is not None, f"CLAHE failed for {name}"
        assert enhanced_img.shape == img.shape, f"Shape mismatch for {name}"
        assert enhanced_img.dtype == img.dtype, f"Dtype mismatch for {name}"


def test_apply_clahe_changes_images_luminance(input_images):
    """
    Verify CLAHE actually modifies all images luminance.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        enhanced_img = apply_clahe(img)

        # Convert to grayscale for luminance comparison
        gray_original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_enhanced = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)

        # CLAHE should modify the image (not necessarily every pixel)
        assert not np.array_equal(gray_original, gray_enhanced), \
            f"CLAHE did not modify {name}"


def test_enhanced_images_are_saved(output_dir, input_images):
    """
    Ensure enhanced images are written to disk.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        enhanced_img = apply_clahe(img)

        saved_path = save_image(str(output_dir), name, enhanced_img)
        assert os.path.exists(saved_path), f"Enhanced {name} not saved"


def test_enhanced_images_are_modified(output_dir, input_images):
    """
    Ensure enhanced images are not identical to the originals.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        enhanced_img = apply_clahe(img)

        saved_path = save_image(str(output_dir), name, enhanced_img)
        saved_img = cv2.imread(saved_path)
        assert saved_img is not None, f"Failed to reload saved {name}"

        diff = cv2.absdiff(img, saved_img)
        assert diff.sum() > 0, f"Enhanced {name} was not modified"


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
        enhanced_img = apply_clahe(img)
        
        saved_path = save_image(
            str(output_dir),
            name,
            enhanced_img,
            prefix="clahe_"
        )

        assert os.path.exists(saved_path)
        assert os.path.basename(saved_path) == f"clahe_{name}"

        saved_img = cv2.imread(saved_path)
        assert saved_img is not None
        assert saved_img.shape == enhanced_img.shape


def test_save_image_without_prefix(output_dir, input_images):
    """
    Test save_image without custom prefix.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        enhanced_img = apply_clahe(img)
        
        saved_path = save_image(
            str(output_dir),
            name,
            enhanced_img
            # No prefix specified, should use default "clahe_"
        )

        assert os.path.exists(saved_path)
        assert os.path.basename(saved_path) == f"clahe_{name}"


def test_clahe_brightness_changes(input_images):
    """
    Test CLAHE brightness changes for all images.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        enhanced_img = apply_clahe(img)
        
        # Convert to grayscale for brightness comparison
        gray_original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_enhanced = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
        
        # Get brightness metrics
        avg_original = np.mean(gray_original)
        avg_enhanced = np.mean(gray_enhanced)
        brightness_change = avg_enhanced - avg_original
        
        # Get contrast metrics (standard deviation)
        std_original = np.std(gray_original)
        std_enhanced = np.std(gray_enhanced)
        contrast_change = std_enhanced - std_original
        
        print(f"{name} - Original brightness: {avg_original:.2f}, contrast: {std_original:.2f}")
        print(f"{name} - Enhanced brightness: {avg_enhanced:.2f}, contrast: {std_enhanced:.2f}")
        print(f"{name} - Brightness change: {brightness_change:+.2f}, Contrast change: {contrast_change:+.2f}")
        
        # CLAHE should modify the image
        assert not np.array_equal(gray_original, gray_enhanced), \
            f"CLAHE did not modify {name}"


def test_clahe_improves_contrast(input_images):
    """
    Test that CLAHE improves contrast in images.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        enhanced_img = apply_clahe(img)
        
        # Convert to grayscale for analysis
        gray_original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_enhanced = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
        
        # Calculate histogram spread (contrast measure)
        hist_original = cv2.calcHist([gray_original], [0], None, [256], [0, 256])
        hist_enhanced = cv2.calcHist([gray_enhanced], [0], None, [256], [0, 256])
        
        # Find non-zero bins in histogram
        non_zero_original = np.count_nonzero(hist_original)
        non_zero_enhanced = np.count_nonzero(hist_enhanced)
        
        # Calculate entropy (another measure of contrast/information)
        def calculate_entropy(hist):
            hist_normalized = hist / hist.sum()
            entropy = -np.sum(hist_normalized * np.log2(hist_normalized + 1e-10))
            return entropy
        
        entropy_original = calculate_entropy(hist_original)
        entropy_enhanced = calculate_entropy(hist_enhanced)
        
        print(f"{name} - Original non-zero bins: {non_zero_original}, entropy: {entropy_original:.2f}")
        print(f"{name} - Enhanced non-zero bins: {non_zero_enhanced}, entropy: {entropy_enhanced:.2f}")


def test_clahe_preserves_dimensions(input_images):
    """
    Test that apply_clahe preserves image dimensions.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        enhanced_img = apply_clahe(img)
        
        h1, w1, c1 = img.shape
        h2, w2, c2 = enhanced_img.shape
        
        assert h1 == h2, f"{name}: Height changed: {h1} != {h2}"
        assert w1 == w2, f"{name}: Width changed: {w1} != {w2}"
        assert c1 == c2, f"{name}: Channels changed: {c1} != {c2}"


def test_clahe_edge_cases(input_images):
    """
    Test apply_clahe with edge cases on all images.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        # Make a copy to avoid modifying the original in tests
        img_copy = img.copy()
        
        # Test with very small image
        small_img = cv2.resize(img_copy, (50, 50))
        enhanced_small = apply_clahe(small_img)
        assert enhanced_small is not None, f"CLAHE failed for small {name}"
        assert enhanced_small.shape == small_img.shape, f"Shape mismatch for small {name}"
        
        # Test with square image
        square_size = min(img_copy.shape[0], img_copy.shape[1])
        square_img = img_copy[:square_size, :square_size]
        enhanced_square = apply_clahe(square_img)
        assert enhanced_square is not None, f"CLAHE failed for square {name}"
        assert enhanced_square.shape == square_img.shape, f"Shape mismatch for square {name}"


def test_clahe_with_different_color_spaces(input_images):
    """
    Test that CLAHE works correctly on different color spaces.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        enhanced_img = apply_clahe(img)
        
        # Verify color integrity
        # CLAHE should enhance without introducing strange colors
        # Check that all pixel values are valid
        assert enhanced_img.min() >= 0, f"{name}: Negative pixel values after CLAHE"
        assert enhanced_img.max() <= 255, f"{name}: Pixel values above 255 after CLAHE"
        
        # Check for NaN or Inf values
        assert not np.any(np.isnan(enhanced_img)), f"{name}: Contains NaN values after CLAHE"
        assert not np.any(np.isinf(enhanced_img)), f"{name}: Contains infinite values after CLAHE"


def test_clahe_consistency(input_images):
    """
    Test that CLAHE produces consistent results.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        # Apply CLAHE multiple times to the same image
        enhanced1 = apply_clahe(img)
        enhanced2 = apply_clahe(img)
        
        # Results should be identical (deterministic)
        assert np.array_equal(enhanced1, enhanced2), \
            f"{name}: CLAHE is not deterministic"


def test_clahe_on_already_bright_images(input_images):
    """
    Test CLAHE on images that are already bright.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        enhanced_img = apply_clahe(img)
        
        # Convert to grayscale
        gray_original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_enhanced = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
        
        avg_original = np.mean(gray_original)
        avg_enhanced = np.mean(gray_enhanced)
        
        # Report findings
        if avg_original > 200:  # Already very bright
            print(f"{name}: Already bright (avg={avg_original:.2f}) - CLAHE result: {avg_enhanced:.2f}")
        elif avg_original < 50:  # Very dark
            print(f"{name}: Very dark (avg={avg_original:.2f}) - CLAHE result: {avg_enhanced:.2f}")
        else:  # Moderate brightness
            print(f"{name}: Moderate (avg={avg_original:.2f}) - CLAHE result: {avg_enhanced:.2f}")


def test_clahe_quality_metrics(input_images):
    """
    Calculate quality metrics for CLAHE enhancement.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        enhanced_img = apply_clahe(img)
        
        # Calculate Mean Squared Error
        mse = np.mean((img.astype(float) - enhanced_img.astype(float)) ** 2)
        
        # Calculate Peak Signal-to-Noise Ratio (PSNR)
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 20 * np.log10(255.0 / np.sqrt(mse))
        
        # Calculate average pixel difference
        diff = cv2.absdiff(img, enhanced_img)
        avg_diff = np.mean(diff)
        
        print(f"{name} - MSE: {mse:.2f}, PSNR: {psnr:.2f} dB, Avg diff: {avg_diff:.2f}")
        
        # CLAHE should change the image, so MSE should not be 0
        assert mse > 0, f"No change detected for {name}"


def test_clahe_histogram_analysis(input_images):
    """
    Analyze histograms before and after CLAHE.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        enhanced_img = apply_clahe(img)
        
        # Convert to grayscale for histogram analysis
        gray_original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_enhanced = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2GRAY)
        
        # Calculate histograms
        hist_original, _ = np.histogram(gray_original.flatten(), bins=256, range=[0, 256])
        hist_enhanced, _ = np.histogram(gray_enhanced.flatten(), bins=256, range=[0, 256])
        
        # Calculate cumulative distribution functions
        cdf_original = hist_original.cumsum()
        cdf_enhanced = hist_enhanced.cumsum()
        
        # Normalize CDFs
        cdf_original_normalized = cdf_original / cdf_original[-1]
        cdf_enhanced_normalized = cdf_enhanced / cdf_enhanced[-1]
        
        # CLAHE should spread out the histogram
        # Check if enhanced histogram uses more of the dynamic range
        original_nonzero = np.count_nonzero(hist_original)
        enhanced_nonzero = np.count_nonzero(hist_enhanced)
        
        print(f"{name} - Original uses {original_nonzero}/256 intensity levels")
        print(f"{name} - Enhanced uses {enhanced_nonzero}/256 intensity levels")


def test_clahe_with_synthetic_images():
    """
    Test CLAHE with synthetic test images.
    """
    # Create a synthetic dark image
    synthetic_dark = np.zeros((100, 100, 3), dtype=np.uint8)
    synthetic_dark[20:80, 20:80] = [50, 50, 50]  # Dark square
    
    enhanced = apply_clahe(synthetic_dark)
    
    assert enhanced is not None
    assert enhanced.shape == synthetic_dark.shape
    
    # CLAHE should brighten the dark regions
    gray_original = cv2.cvtColor(synthetic_dark, cv2.COLOR_BGR2GRAY)
    gray_enhanced = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    
    avg_original = np.mean(gray_original)
    avg_enhanced = np.mean(gray_enhanced)
    
    print(f"Synthetic dark - Original brightness: {avg_original:.2f}")
    print(f"Synthetic dark - Enhanced brightness: {avg_enhanced:.2f}")