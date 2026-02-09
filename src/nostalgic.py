import cv2
import os
import numpy as np

def show_resized(win_name, img, max_height=700):
    h, w = img.shape[:2]
    if h > max_height:
        scale = max_height / h
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    cv2.imshow(win_name, img)

def apply_nostalgic_effect(
    img,
    warmth=1.15,
    fade_strength=0.85,
    grain_strength=8,
    blur_ksize=5
):
    if img is None:
        raise ValueError("Input image is None")

    # ---------- Warm color tone ----------
    nostalgic = img.astype(np.float32)
    nostalgic[:, :, 0] *= 0.9           # Blue down
    nostalgic[:, :, 1] *= 1.05          # Green slight up
    nostalgic[:, :, 2] *= warmth        # Red up
    nostalgic = np.clip(nostalgic, 0, 255)

    # ---------- Fade contrast ----------
    nostalgic = nostalgic * fade_strength + 30
    nostalgic = np.clip(nostalgic, 0, 255).astype(np.uint8)

    # ---------- Soft blur ----------
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    nostalgic = cv2.GaussianBlur(nostalgic, (blur_ksize, blur_ksize), 0)

    # ---------- Film grain ----------
    grain = np.random.normal(
        0, grain_strength, nostalgic.shape
    ).astype(np.int16)
    nostalgic = nostalgic.astype(np.int16) + grain
    nostalgic = np.clip(nostalgic, 0, 255).astype(np.uint8)

    # ---------- Vignette ----------
    h, w = nostalgic.shape[:2]
    kernel_x = cv2.getGaussianKernel(w, w / 2)
    kernel_y = cv2.getGaussianKernel(h, h / 2)
    mask = kernel_y * kernel_x.T
    mask = mask / mask.max()

    for i in range(3):
        nostalgic[:, :, i] = nostalgic[:, :, i] * mask

    return nostalgic.astype(np.uint8)

def load_images(input_dir, filenames):
    for name in filenames:
        name = name.strip()
        path = os.path.join(input_dir, name)

        img = cv2.imread(path)
        if img is None:
            print(f"Skipped: {name}")
            continue

        yield name, img

def save_image(output_dir, filename, img, prefix="nostalgic_"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{prefix}{filename}")
    cv2.imwrite(output_path, img)
    return output_path

def main():
    input_dir = "input"
    output_dir = "output"

    selected_files = input(
        "Enter image filenames (comma separated): "
    ).split(",")

    for filename, img in load_images(input_dir, selected_files):
        processed = apply_nostalgic_effect(img)

        show_resized("Original", img)
        show_resized("Nostalgic", processed)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        path = save_image(output_dir, filename, processed)
        print(f"Saved: {path}")

if __name__ == "__main__":
    main()
