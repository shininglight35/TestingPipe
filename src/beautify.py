import cv2
import numpy as np
import os

def show_resized(win_name, img, max_height=700):
    h, w = img.shape[:2]
    if h > max_height:
        scale = max_height / h
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    cv2.imshow(win_name, img)

def smooth_skin(img):
    if img is None:
        raise ValueError("Input image is None")

    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

    # Skin color range
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)

    skin_mask = cv2.inRange(ycrcb, lower, upper)

    # Refine mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
    skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)

    # Feather the mask (KEY FIX)
    skin_mask = cv2.GaussianBlur(skin_mask, (21, 21), 0)

    # Normalize mask to 0–1
    alpha = skin_mask.astype(np.float32) / 255.0
    alpha = cv2.merge([alpha, alpha, alpha])

    # Smooth image
    smooth = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

    # Alpha blend (NO hard edges)
    result = (smooth * alpha + img * (1 - alpha)).astype(np.uint8)

    return result

def load_images(input_dir, filenames):
    for name in filenames:
        name = name.strip()
        path = os.path.join(input_dir, name)

        img = cv2.imread(path)
        if img is None:
            print(f"Skipped: {name}")
            continue

        yield name, img

def save_image(output_dir, filename, img, prefix="beautified_"):
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
        result = smooth_skin(img)

        show_resized("Original", img)
        show_resized("Beautified", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        path = save_image(output_dir, filename, result)
        print(f"Saved: {path}")

if __name__ == "__main__":
    main()
