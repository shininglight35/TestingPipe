import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from brighten import (
    apply_clahe,
    load_images,
    save_image,
)

# ---------- Fixtures ----------

@pytest.fixture
def dummy_image():
    """Create a simple dummy BGR image."""
    return np.full((100, 100, 3), 128, dtype=np.uint8)


@pytest.fixture
def temp_input_dir(tmp_path, dummy_image):
    """Create temp input dir with one valid and one invalid image."""
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    valid_path = input_dir / "test.jpg"
    cv2.imwrite(str(valid_path), dummy_image)

    return input_dir


@pytest.fixture
def temp_output_dir(tmp_path):
    return tmp_path / "output"


# ---------- Tests ----------

def test_apply_clahe_basic(dummy_image):
    result = apply_clahe(dummy_image)

    assert result is not None
    assert result.shape == dummy_image.shape
    assert result.dtype == dummy_image.dtype


def test_apply_clahe_none_input():
    with pytest.raises(ValueError):
        apply_clahe(None)


def test_load_images_valid_and_invalid(temp_input_dir):
    filenames = ["test.jpg", "missing.jpg"]

    results = list(load_images(str(temp_input_dir), filenames))

    assert len(results) == 1
    name, img = results[0]

    assert name == "test.jpg"
    assert img is not None
    assert img.shape[2] == 3


def test_save_image(temp_output_dir, dummy_image):
    output_path = save_image(
        str(temp_output_dir),
        "image.jpg",
        dummy_image,
        prefix="clahe_"
    )

    assert os.path.exists(output_path)

    saved = cv2.imread(output_path)
    assert saved is not None
    assert saved.shape == dummy_image.shape


def test_clahe_changes_luminance(dummy_image):
    """CLAHE should modify luminance (not necessarily every pixel)."""
    processed = apply_clahe(dummy_image)

    # Convert to grayscale for comparison
    gray_orig = cv2.cvtColor(dummy_image, cv2.COLOR_BGR2GRAY)
    gray_proc = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

    assert not np.array_equal(gray_orig, gray_proc)
