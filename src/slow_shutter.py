import cv2
import os
import numpy as np

def show_resized(win_name, img, max_height=700):
    h, w = img.shape[:2]
    if h > max_height:
        scale = max_height / h
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    cv2.imshow(win_name, img)

def apply_slow_shutter(
    img,
    trail_length=120,
    step=3,
    direction=-1,
    blend_original=0.4
):
    if img is None:
        raise ValueError("Input image is None")

    h, w = img.shape[:2]

    accumulator = np.zeros_like(img, dtype=np.float32)
    weight_sum = 0.0

    for i in range(trail_length):
        dx = direction * step * i
        M = np.float32([[1, 0, dx], [0, 1, 0]])

        shifted = cv2.warpAffine(
            img,
            M,
            (w, h),
            borderMode=cv2.BORDER_REPLICATE
        )

        weight = 1.0 - (i / trail_length)
        accumulator += shifted * weight
        weight_sum += weight

    slow_shutter = (accumulator / weight_sum).astype(np.uint8)

    slow_shutter = cv2.addWeighted(
        img,
        blend_original,
        slow_shutter,
        1.0 - blend_original,
        0
    )

    return slow_shutter

def load_images(input_dir, filenames):
    for name in filenames:
        name = name.strip()
        path = os.path.join(input_dir, name)

        img = cv2.imread(path)
        if img is None:
            print(f"Skipped: {name}")
            continue

        yield name, img

def save_image(output_dir, filename, img, prefix="slow_"):
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
        processed = apply_slow_shutter(img)

        show_resized("Original", img)
        show_resized("Slow Shutter", processed)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        path = save_image(output_dir, filename, processed)
        print(f"Saved: {path}")

if __name__ == "__main__":
    main()
