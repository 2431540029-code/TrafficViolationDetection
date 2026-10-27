
import cv2
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2] / "dataset_v02_yolo"
IMAGE_DIR = BASE_DIR / "images" / "train"
LABEL_DIR = BASE_DIR / "labels" / "train"
OUTPUT_DIR = BASE_DIR / "preview_trippling"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = {
    0: "Bike",
    1: "Helmet_On",
    2: "No_helmet",
    3: "Rider",
    4: "Trippling",
}

shown = 0
max_images = 10

for label_path in sorted(LABEL_DIR.glob("*.txt")):
    lines = label_path.read_text(encoding="utf-8").splitlines()

    # Chi xem nhung anh co nhan Trippling (ID 4)
    if not any(line.split()[0] == "4" for line in lines if line.strip()):
        continue

    image_path = IMAGE_DIR / f"{label_path.stem}.jpg"

    # Neu anh khong phai JPG, tim theo ten goc
    if not image_path.exists():
        candidates = [
            p for p in IMAGE_DIR.glob(f"{label_path.stem}.*")
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
        ]
        if not candidates:
            continue
        image_path = candidates[0]

    image = cv2.imread(str(image_path))
    if image is None:
        continue

    height, width = image.shape[:2]

    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue

        class_id = int(parts[0])
        x, y, w, h = map(float, parts[1:])

        x1 = int((x - w / 2) * width)
        y1 = int((y - h / 2) * height)
        x2 = int((x + w / 2) * width)
        y2 = int((y + h / 2) * height)

        color = (0, 0, 255) if class_id == 4 else (0, 255, 0)

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            CLASS_NAMES.get(class_id, str(class_id)),
            (max(0, x1), max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    output_path = OUTPUT_DIR / image_path.name
    cv2.imwrite(str(output_path), image)
    shown += 1

    if shown >= max_images:
        break

print(f"Da tao {shown} anh xem nhan tai:")
print(OUTPUT_DIR)

if shown == 0:
    print("Khong tim thay anh co nhan Trippling.")
