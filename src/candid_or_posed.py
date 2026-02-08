import cv2
import os
import numpy as np

def show_resized(win_name, img, max_height=700):
    h, w = img.shape[:2]
    if h > max_height:
        scale = max_height / h
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    cv2.imshow(win_name, img)

def load_images(input_dir, filenames):
    for name in filenames:
        name = name.strip()
        path = os.path.join(input_dir, name)

        img = cv2.imread(path)
        if img is None:
            print(f"Skipped: {name}")
            continue

        yield name, img

def save_image(output_dir, filename, img, prefix="classified_"):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{prefix}{filename}")
    cv2.imwrite(output_path, img)
    return output_path

def classify_candid_or_posed(img):
    import cv2
    import numpy as np

    # ---- Load cascades locally ----
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_eye.xml"
    )

    # ---- Blur score (local helper) ----
    def blur_score(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    # ---- Detection ----
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5
    )

    h, w = img.shape[:2]
    center_x = w // 2

    looking_at_camera = False
    off_center = False

    for (x, y, fw, fh) in faces:
        face_roi_gray = gray[y:y+fh, x:x+fw]
        eyes = eye_cascade.detectMultiScale(face_roi_gray)

        if len(eyes) >= 2:
            looking_at_camera = True

        face_center_x = x + fw // 2
        if abs(face_center_x - center_x) > w * 0.15:
            off_center = True

        # Draw detections
        cv2.rectangle(img, (x, y), (x+fw, y+fh), (0, 255, 0), 2)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(
                img[y:y+fh, x:x+fw],
                (ex, ey),
                (ex+ew, ey+eh),
                (255, 0, 0),
                1
            )

    # ---- Scoring ----
    blur_var = blur_score(img)
    is_blurry = blur_var < 100

    score = 0
    if len(faces) == 0:
        score += 3
    if not looking_at_camera:
        score += 2
    if is_blurry:
        score += 1
    if off_center:
        score += 1
    if len(faces) > 1:
        score += 1

    label = "CANDID" if score >= 3 else "POSED"

    return label, score, blur_var

def main():
    input_dir = "input"
    output_dir = "output"

    selected_files = input(
        "Enter image filenames (comma separated): "
    ).split(",")

    for filename, img in load_images(input_dir, selected_files):
        label, score, blur_var = classify_candid_or_posed(img)

        cv2.putText(
            img,
            label,
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (0, 0, 255) if label == "CANDID" else (0, 255, 0),
            3
        )

        show_resized("Candid or Posed", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        path = save_image(output_dir, filename, img)
        print(f"Saved: {path}")

if __name__ == "__main__":
    main()
