import cv2
import numpy as np
import os

def show_resized(win_name, img, max_height=700):
    h, w = img.shape[:2]
    if h > max_height:
        scale = max_height / h
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    cv2.imshow(win_name, img)

def cartoonify(img):

    if img is None:
        raise ValueError("Input image is None")

    # --- Edge detection ---
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.GaussianBlur(gray, (7, 7), 0)

    edges = cv2.adaptiveThreshold(
        gray_blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=9,
        C=2
    )

    edges = cv2.medianBlur(edges, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)

    edges = cv2.bitwise_not(edges)
    edges = cv2.medianBlur(edges, 3)
    edges = cv2.bitwise_not(edges)

    # --- Color smoothing ---
    color = img.copy()
    for _ in range(8):
        color = cv2.bilateralFilter(
            color, d=9, sigmaColor=300, sigmaSpace=300
        )

    # --- Color quantization ---
    Z = color.reshape((-1, 3))
    Z = np.float32(Z)

    K = 6
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        10,
        1.0
    )
    _, labels, centers = cv2.kmeans(
        Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    centers = np.uint8(centers)
    quantized = centers[labels.flatten()]
    quantized = quantized.reshape(color.shape)

    # --- Combine edges and colors ---
    return cv2.bitwise_and(quantized, quantized, mask=edges)

def load_images(input_dir, filenames):

    for name in filenames:
        name = name.strip()
        path = os.path.join(input_dir, name)

        img = cv2.imread(path)
        if img is None:
            print(f"Skipped: {name}")
            continue

        yield name, img

def save_image(output_dir, filename, img, prefix="cartoon_"):
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
        cartoon = cartoonify(img)

        show_resized("Original", img)
        show_resized("Cartoon", cartoon)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        path = save_image(output_dir, filename, cartoon)
        print(f"Saved: {path}")

if __name__ == "__main__":
    main()
