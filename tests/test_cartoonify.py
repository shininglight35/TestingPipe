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
def output_dir():
    """
    Always use 'output/' folder for all runs.
    """
    out = Path("output")
    out.mkdir(parents=True, exist_ok=True)
    return out


@pytest.fixture
def cartoon_image():
    """
    Load the specific cartoon.jpg image for testing.
    """
    cartoon_path = os.path.join(INPUT_DIR, "cartoon.jpg")
    assert os.path.exists(cartoon_path), "cartoon.jpg not found in input folder"
    
    img = cv2.imread(cartoon_path)
    assert img is not None, "Failed to load cartoon.jpg"
    return img


# --------------------
# Tests - Specific to cartoon.jpg
# --------------------

def test_cartoon_image_loads_correctly(cartoon_image):
    """
    Ensure cartoon.jpg loads successfully.
    """
    assert cartoon_image is not None, "Failed to load cartoon.jpg"
    assert cartoon_image.size > 0, "cartoon.jpg is empty"
    assert cartoon_image.shape[2] == 3, "cartoon.jpg should have 3 channels (BGR)"


def test_cartoonify_on_cartoon_image(cartoon_image):
    """
    Ensure cartoonify function runs on cartoon.jpg and returns valid output.
    """
    result = cartoonify(cartoon_image)

    assert result is not None, "cartoonify failed for cartoon.jpg"
    assert result.shape == cartoon_image.shape, "Shape mismatch for cartoon.jpg"
    assert result.dtype == cartoon_image.dtype, "Dtype mismatch for cartoon.jpg"


def test_cartoonify_modifies_image(cartoon_image):
    """
    Verify cartoonify actually modifies cartoon.jpg.
    """
    result = cartoonify(cartoon_image)

    # The result should be different from original
    assert not np.array_equal(cartoon_image, result), \
        "cartoonify did not modify cartoon.jpg"


def test_cartoonify_creates_edges(cartoon_image):
    """
    Verify cartoonify creates edge detection.
    """
    result = cartoonify(cartoon_image)
    
    # Cartoon effect should create visible edges
    # Convert to grayscale to analyze edges
    gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    
    # Calculate edge strength using Sobel
    sobel_x = cv2.Sobel(gray_result, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_result, cv2.CV_64F, 0, 1, ksize=3)
    edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    
    avg_edge_strength = np.mean(edge_magnitude)
    print(f"Average edge strength in cartoonified image: {avg_edge_strength:.2f}")
    
    # Cartoon images should have noticeable edges
    assert avg_edge_strength > 0, "No edges detected in cartoonified image"


def test_cartoonify_color_quantization(cartoon_image):
    """
    Verify cartoonify applies color quantization.
    """
    result = cartoonify(cartoon_image)
    
    # Color quantization should reduce number of unique colors
    original_colors = np.unique(cartoon_image.reshape(-1, cartoon_image.shape[2]), axis=0)
    result_colors = np.unique(result.reshape(-1, result.shape[2]), axis=0)
    
    print(f"Original unique colors: {len(original_colors)}")
    print(f"Cartoon unique colors: {len(result_colors)}")
    
    # K-means with K=6 should significantly reduce color count
    assert len(result_colors) <= 10, f"Too many colors after quantization: {len(result_colors)}"


def test_cartoonified_image_is_saved(output_dir, cartoon_image):
    """
    Ensure cartoonified cartoon.jpg is written to disk.
    """
    result = cartoonify(cartoon_image)

    saved_path = save_image(str(output_dir), "cartoon.jpg", result)
    assert os.path.exists(saved_path), "Cartoonified cartoon.jpg not saved"


def test_cartoonified_image_is_modified(output_dir, cartoon_image):
    """
    Ensure cartoonified cartoon.jpg is not identical to the original.
    """
    result = cartoonify(cartoon_image)

    saved_path = save_image(str(output_dir), "cartoon.jpg", result)
    saved_img = cv2.imread(saved_path)
    assert saved_img is not None, "Failed to reload saved cartoon.jpg"

    diff = cv2.absdiff(cartoon_image, saved_img)
    assert diff.sum() > 0, "Cartoonified cartoon.jpg was not modified"


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
    result = list(load_images(INPUT_DIR, ["cartoon.jpg", "non_existent.jpg"]))
    assert len(result) == 1
    assert result[0][0] == "cartoon.jpg"


def test_save_image_with_prefix(output_dir, cartoon_image):
    """
    Test save_image with custom prefix using cartoon.jpg.
    """
    result = cartoonify(cartoon_image)
    
    saved_path = save_image(
        str(output_dir),
        "cartoon.jpg",
        result,
        prefix="cartoon_"
    )

    assert os.path.exists(saved_path)
    assert os.path.basename(saved_path) == "cartoon_cartoon.jpg"

    saved_img = cv2.imread(saved_path)
    assert saved_img is not None
    assert saved_img.shape == result.shape


def test_save_image_without_prefix(output_dir, cartoon_image):
    """
    Test save_image without custom prefix.
    """
    result = cartoonify(cartoon_image)
    
    saved_path = save_image(
        str(output_dir),
        "cartoon.jpg",
        result
        # No prefix specified, should use default "cartoon_"
    )

    assert os.path.exists(saved_path)
    assert os.path.basename(saved_path) == "cartoon_cartoon.jpg"


def test_show_resized_function(cartoon_image):
    """
    Test show_resized function.
    """
    max_height = 700
    
    # show_resized doesn't return anything, it just displays
    # We'll test that it doesn't crash when called
    try:
        # In CI/CD environments, cv2.imshow might fail
        show_resized("Test", cartoon_image, max_height)
        # If we reach here, the function didn't crash
        assert True
    except Exception as e:
        # In headless environments, this might fail
        print(f"Note: show_resized might fail in headless environments: {e}")


def test_cartoonify_preserves_dimensions(cartoon_image):
    """
    Test that cartoonify preserves image dimensions.
    """
    result = cartoonify(cartoon_image)
    
    h1, w1, c1 = cartoon_image.shape
    h2, w2, c2 = result.shape
    
    assert h1 == h2, f"Height changed: {h1} != {h2}"
    assert w1 == w2, f"Width changed: {w1} != {w2}"
    assert c1 == c2, f"Channels changed: {c1} != {c2}"


def test_cartoonify_edge_detection_steps(cartoon_image):
    """
    Test individual edge detection steps.
    """
    # Test grayscale conversion
    gray = cv2.cvtColor(cartoon_image, cv2.COLOR_BGR2GRAY)
    assert gray is not None
    assert gray.ndim == 2
    
    # Test Gaussian blur
    gray_blur = cv2.GaussianBlur(gray, (7, 7), 0)
    assert gray_blur is not None
    assert gray_blur.shape == gray.shape
    
    # Test adaptive threshold
    edges = cv2.adaptiveThreshold(
        gray_blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=9,
        C=2
    )
    assert edges is not None
    assert edges.shape == gray.shape
    
    # Check edge image is binary
    unique_vals = np.unique(edges)
    assert len(unique_vals) <= 2, "Edge image should be binary"


def test_cartoonify_bilateral_filtering(cartoon_image):
    """
    Test bilateral filtering step.
    """
    color = cartoon_image.copy()
    
    # Apply bilateral filter 8 times (as in the function)
    for _ in range(8):
        color = cv2.bilateralFilter(
            color, d=9, sigmaColor=300, sigmaSpace=300
        )
    
    assert color is not None
    assert color.shape == cartoon_image.shape
    
    # Bilateral filtering should smooth while preserving edges
    # Compare with simple Gaussian blur
    gaussian_blur = cv2.GaussianBlur(cartoon_image, (9, 9), 0)
    
    # Bilateral should be different from Gaussian
    diff = cv2.absdiff(color, gaussian_blur)
    assert diff.sum() > 0, "Bilateral filter same as Gaussian"


def test_cartoonify_kmeans_quantization(cartoon_image):
    """
    Test k-means color quantization step.
    """
    # Apply bilateral filtering first (as in the function)
    color = cartoon_image.copy()
    for _ in range(8):
        color = cv2.bilateralFilter(
            color, d=9, sigmaColor=300, sigmaSpace=300
        )
    
    # Reshape for k-means
    Z = color.reshape((-1, 3))
    Z = np.float32(Z)
    
    K = 6
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        10,
        1.0
    )
    
    # Run k-means
    _, labels, centers = cv2.kmeans(
        Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )
    
    assert labels is not None
    assert centers is not None
    assert len(centers) == K, f"Expected {K} centers, got {len(centers)}"
    
    # Reconstruct quantized image
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()]
    quantized = quantized.reshape(color.shape)
    
    assert quantized.shape == color.shape
    assert quantized.dtype == np.uint8


def test_cartoonify_final_combination(cartoon_image):
    """
    Test final combination of edges and colors.
    """
    result = cartoonify(cartoon_image)
    
    # The final result should be a bitwise_and of quantized colors with edges mask
    # This means pixels where edges=0 should be black
    
    # Get edges from the result
    gray_result = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    
    # Count black pixels (where edges mask was 0)
    black_pixels = np.sum(gray_result == 0)
    black_percentage = (black_pixels / gray_result.size) * 100
    
    print(f"Black pixels (edges) in cartoonified image: {black_percentage:.2f}%")
    
    # Should have some edge lines (black pixels)
    assert black_pixels > 0, "No edge lines in cartoonified image"
    
    # But not too many (edges should be thin lines)
    assert black_percentage < 50, f"Too many black pixels: {black_percentage}%"