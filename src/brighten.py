import cv2
import os

def show_resized(win_name, img, max_height=700):
    h, w = img.shape[:2]
    if h > max_height:
        scale = max_height / h
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    cv2.imshow(win_name, img)

def apply_clahe(img, clip_limit=3.0, tile_grid_size=(8, 8)):

    if img is None:
        raise ValueError("Input image is None")

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )
    l = clahe.apply(l)

    merged = cv2.merge((l, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

def load_images(input_dir, filenames):

    for name in filenames:
        name = name.strip()
        path = os.path.join(input_dir, name)

        img = cv2.imread(path)
        if img is None:
            print(f"Skipped: {name}")
            continue

        yield name, img

def save_image(output_dir, filename, img, prefix="clahe_"):
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
        processed = apply_clahe(img)

        show_resized("Original", img)
        show_resized("CLAHE", processed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        path = save_image(output_dir, filename, processed)
        print(f"Saved: {path}")

if __name__ == "__main__":
    main()
