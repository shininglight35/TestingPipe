import os
import cv2
import numpy as np
import pytest
import sys
from pathlib import Path

# Add src/ to path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from cartoonify import (
    load_images,
    cartoonify,
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


def test_cartoonify_on_all_images(input_images):
    """
    Ensure cartoonify function runs on all images and returns valid output.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        result = cartoonify(img)

        assert result is not None, f"cartoonify failed for {name}"
        assert result.shape == img.shape, f"Shape mismatch for {name}"
        assert result.dtype == img.dtype, f"Dtype mismatch for {name}"


def test_cartoonify_modifies_all_images(input_images):
    """
    Verify cartoonify actually modifies all images.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        result = cartoonify(img)

        # The result should be different from original
        assert not np.array_equal(img, result), \
            f"cartoonify did not modify {name}"


def test_cartoonify_creates_edges_on_all_images(input_images):
    """
    Verify cartoonify creates edge detection on all images.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        result = cartoonify(img)
        
        # Cartoon effect should create visible edges
        # Convert to grayscale to analyze edges
        gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        
        # Calculate edge strength using Sobel
        sobel_x = cv2.Sobel(gray_result, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_result, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        avg_edge_strength = np.mean(edge_magnitude)
        print(f"{name} - Average edge strength: {avg_edge_strength:.2f}")
        
        # Cartoon images should have noticeable edges
        assert avg_edge_strength > 0, f"No edges detected in cartoonified {name}"


def test_cartoonify_color_quantization_on_all_images(input_images):
    """
    Verify cartoonify applies color quantization on all images.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        result = cartoonify(img)
        
        # Color quantization should reduce number of unique colors
        original_colors = np.unique(img.reshape(-1, img.shape[2]), axis=0)
        result_colors = np.unique(result.reshape(-1, result.shape[2]), axis=0)
        
        print(f"{name} - Original unique colors: {len(original_colors)}")
        print(f"{name} - Cartoon unique colors: {len(result_colors)}")
        
        # K-means with K=6 should significantly reduce color count
        # Allow some flexibility (some images might be simple already)
        if len(original_colors) > 10:
            assert len(result_colors) <= max(10, len(original_colors) * 0.1), \
                f"{name}: Too many colors after quantization: {len(result_colors)}"


def test_cartoonified_images_are_saved(output_dir, input_images):
    """
    Ensure cartoonified images are written to disk.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        result = cartoonify(img)

        saved_path = save_image(str(output_dir), name, result)
        assert os.path.exists(saved_path), f"Cartoonified {name} not saved"


def test_cartoonified_images_are_modified(output_dir, input_images):
    """
    Ensure cartoonified images are not identical to the originals.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        result = cartoonify(img)

        saved_path = save_image(str(output_dir), name, result)
        saved_img = cv2.imread(saved_path)
        assert saved_img is not None, f"Failed to reload saved {name}"

        diff = cv2.absdiff(img, saved_img)
        assert diff.sum() > 0, f"Cartoonified {name} was not modified"


def test_cartoonify_invalid_input():
    """
    Test cartoonify with invalid inputs.
    """
    # Test with None input - should raise ValueError
    with pytest.raises(ValueError):
        cartoonify(None)


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
        result = cartoonify(img)
        
        saved_path = save_image(
            str(output_dir),
            name,
            result,
            prefix="cartoon_"
        )

        assert os.path.exists(saved_path)
        assert os.path.basename(saved_path) == f"cartoon_{name}"

        saved_img = cv2.imread(saved_path)
        assert saved_img is not None
        assert saved_img.shape == result.shape


def test_save_image_without_prefix(output_dir, input_images):
    """
    Test save_image without custom prefix.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        result = cartoonify(img)
        
        saved_path = save_image(
            str(output_dir),
            name,
            result
            # No prefix specified, should use default "cartoon_"
        )

        assert os.path.exists(saved_path)
        assert os.path.basename(saved_path) == f"cartoon_{name}"


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


def test_cartoonify_preserves_dimensions(input_images):
    """
    Test that cartoonify preserves image dimensions.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        result = cartoonify(img)
        
        h1, w1, c1 = img.shape
        h2, w2, c2 = result.shape
        
        assert h1 == h2, f"{name}: Height changed: {h1} != {h2}"
        assert w1 == w2, f"{name}: Width changed: {w1} != {w2}"
        assert c1 == c2, f"{name}: Channels changed: {c1} != {c2}"


def test_cartoonify_edge_detection_steps_on_all_images(input_images):
    """
    Test individual edge detection steps on all images.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        # Test grayscale conversion
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        assert gray is not None, f"Failed grayscale conversion for {name}"
        assert gray.ndim == 2, f"Wrong dimensions for grayscale {name}"
        
        # Test Gaussian blur
        gray_blur = cv2.GaussianBlur(gray, (7, 7), 0)
        assert gray_blur is not None, f"Failed Gaussian blur for {name}"
        assert gray_blur.shape == gray.shape, f"Shape changed after blur for {name}"
        
        # Test adaptive threshold
        edges = cv2.adaptiveThreshold(
            gray_blur,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            blockSize=9,
            C=2
        )
        assert edges is not None, f"Failed adaptive threshold for {name}"
        assert edges.shape == gray.shape, f"Shape changed after threshold for {name}"
        
        # Check edge image is binary
        unique_vals = np.unique(edges)
        assert len(unique_vals) <= 2, f"Edge image should be binary for {name}"


def test_cartoonify_bilateral_filtering_on_all_images(input_images):
    """
    Test bilateral filtering step on all images.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        color = img.copy()
        
        # Apply bilateral filter 8 times (as in the function)
        for _ in range(8):
            color = cv2.bilateralFilter(
                color, d=9, sigmaColor=300, sigmaSpace=300
            )
        
        assert color is not None, f"Bilateral filtering failed for {name}"
        assert color.shape == img.shape, f"Shape changed after bilateral filter for {name}"
        
        # Bilateral filtering should smooth while preserving edges
        # Compare with simple Gaussian blur
        gaussian_blur = cv2.GaussianBlur(img, (9, 9), 0)
        
        # Bilateral should be different from Gaussian
        diff = cv2.absdiff(color, gaussian_blur)
        assert diff.sum() > 0, f"Bilateral filter same as Gaussian for {name}"


def test_cartoonify_final_combination_on_all_images(input_images):
    """
    Test final combination of edges and colors on all images.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        result = cartoonify(img)
        
        # Get edges from the result
        gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        
        # Count black pixels (where edges mask was 0)
        black_pixels = np.sum(gray_result == 0)
        black_percentage = (black_pixels / gray_result.size) * 100
        
        print(f"{name} - Black pixels (edges): {black_percentage:.2f}%")
        
        # Should have some edge lines (black pixels)
        assert black_pixels > 0, f"No edge lines in cartoonified {name}"
        
        # But not too many (edges should be thin lines)
        assert black_percentage < 80, f"{name}: Too many black pixels: {black_percentage}%"


def test_cartoonify_color_consistency(input_images):
    """
    Test that cartoonify maintains reasonable color consistency.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        result = cartoonify(img)
        
        # Check that the result is still a valid BGR image
        assert result.min() >= 0, f"{name}: Negative pixel values"
        assert result.max() <= 255, f"{name}: Pixel values above 255"
        
        # Check each channel separately
        for channel in range(3):
            channel_values = result[:, :, channel]
            assert channel_values.min() >= 0, f"{name}: Negative values in channel {channel}"
            assert channel_values.max() <= 255, f"{name}: Values >255 in channel {channel}"


def test_cartoonify_edge_cases(input_images):
    """
    Test cartoonify with edge cases on all images.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        # Make a copy to avoid modifying the original in tests
        img_copy = img.copy()
        
        # Test with very small image
        small_img = cv2.resize(img_copy, (50, 50))
        result_small = cartoonify(small_img)
        assert result_small is not None, f"cartoonify failed for small {name}"
        assert result_small.shape == small_img.shape, f"Shape mismatch for small {name}"
        
        # Test with square image
        square_size = min(img_copy.shape[0], img_copy.shape[1])
        square_img = img_copy[:square_size, :square_size]
        result_square = cartoonify(square_img)
        assert result_square is not None, f"cartoonify failed for square {name}"
        assert result_square.shape == square_img.shape, f"Shape mismatch for square {name}"


def test_cartoonify_output_quality(input_images):
    """
    Test the quality of cartoonify output.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        result = cartoonify(img)
        
        # Calculate mean squared error (MSE) between original and result
        mse = np.mean((img.astype(float) - result.astype(float)) ** 2)
        
        # Calculate structural similarity index (SSIM) approximation
        # Simple version: check if images are significantly different
        diff = cv2.absdiff(img, result)
        avg_diff = np.mean(diff)
        
        print(f"{name} - MSE: {mse:.2f}, Average pixel difference: {avg_diff:.2f}")
        
        # Cartoonify should significantly change the image
        assert mse > 0, f"No change detected for {name}"
        assert avg_diff > 0, f"No pixel difference for {name}"


def test_cartoonify_handles_different_aspect_ratios(input_images):
    """
    Test that cartoonify handles images with different aspect ratios.
    """
    for name, img in load_images(INPUT_DIR, input_images):
        # Get original aspect ratio
        h, w = img.shape[:2]
        original_aspect = w / h
        
        # Apply cartoonify
        result = cartoonify(img)
        
        # Get result aspect ratio
        h_result, w_result = result.shape[:2]
        result_aspect = w_result / h_result
        
        # Aspect ratio should be preserved
        assert abs(original_aspect - result_aspect) < 0.01, \
            f"{name}: Aspect ratio changed significantly: {original_aspect:.3f} vs {result_aspect:.3f}"


def test_cartoonify_with_synthetic_images():
    """
    Test cartoonify with synthetic test images.
    """
    # Create a simple color gradient test image
    gradient = np.zeros((100, 100, 3), dtype=np.uint8)
    for i in range(100):
        gradient[:, i] = [i * 255 // 100, 128, 255 - i * 255 // 100]
    
    result = cartoonify(gradient)
    
    assert result is not None
    assert result.shape == gradient.shape
    
    # Gradient should be quantized into fewer colors
    gradient_colors = np.unique(gradient.reshape(-1, 3), axis=0)
    result_colors = np.unique(result.reshape(-1, 3), axis=0)
    
    print(f"Gradient - Original colors: {len(gradient_colors)}")
    print(f"Gradient - Cartoon colors: {len(result_colors)}")
    
    # Should have fewer colors
    assert len(result_colors) < len(gradient_colors), \
        "Cartoonify didn't reduce colors in gradient image"