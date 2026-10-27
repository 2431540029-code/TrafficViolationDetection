
import json
import shutil
from pathlib import Path
from collections import Counter


SOURCE_DIR = Path(
    r"D:\Traffic Rule Violation Detection Dataset in Dhaka\traffic_violation\augmented_dataset"
)

# Tao dataset YOLO rieng, khong sua du lieu goc
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "dataset_v02_yolo"

# Bo qua lop 0 vi khong co annotation
# Remap ID COCO sang ID YOLO lien tuc tu 0 den 4
CLASS_MAP = {
    1: 0,  # Bike
    2: 1,  # Helmet_On
    3: 2,  # No_helmet
    4: 3,  # Rider
    5: 4,  # Trippling
}

SPLITS = ["train", "valid", "test"]


def convert_split(split):
    source_split = SOURCE_DIR / split
    json_path = source_split / "_annotations.coco.json"

    if not json_path.exists():
        raise FileNotFoundError(
            f"Khong tim thay file nhan: {json_path}"
        )

    with open(json_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {
        image["id"]: image
        for image in coco["images"]
    }

    output_images = OUTPUT_DIR / "images" / split
    output_labels = OUTPUT_DIR / "labels" / split

    output_images.mkdir(parents=True, exist_ok=True)
    output_labels.mkdir(parents=True, exist_ok=True)

    annotations_by_image = {}
    for ann in coco.get("annotations", []):
        annotations_by_image.setdefault(
            ann["image_id"], []
        ).append(ann)

    missing_images = 0
    total_labels = Counter()

    for image_id, image_info in images.items():
        filename = Path(image_info["file_name"]).name
        source_image = source_split / filename

        if not source_image.exists():
            missing_images += 1
            print(f"Thieu anh: {source_image}")
            continue

        # Copy anh sang bo du lieu YOLO
        shutil.copy2(
            source_image,
            output_images / filename
        )

        width = image_info["width"]
        height = image_info["height"]

        label_path = (
            output_labels / f"{Path(filename).stem}.txt"
        )

        lines = []

        for ann in annotations_by_image.get(image_id, []):
            category_id = ann["category_id"]

            if category_id not in CLASS_MAP:
                continue

            x, y, w, h = ann["bbox"]

            if width <= 0 or height <= 0 or w <= 0 or h <= 0:
                continue

            # COCO: x, y, w, h (goc tren trai)
            # YOLO: class, tam_x, tam_y, rong, cao
            center_x = (x + w / 2) / width
            center_y = (y + h / 2) / height
            norm_w = w / width
            norm_h = h / height

            # Gioi han toa do trong khoang [0, 1]
            center_x = min(max(center_x, 0), 1)
            center_y = min(max(center_y, 0), 1)
            norm_w = min(max(norm_w, 0), 1)
            norm_h = min(max(norm_h, 0), 1)

            if norm_w <= 0 or norm_h <= 0:
                continue

            yolo_id = CLASS_MAP[category_id]

            lines.append(
                f"{yolo_id} {center_x:.6f} "
                f"{center_y:.6f} {norm_w:.6f} {norm_h:.6f}"
            )

            total_labels[yolo_id] += 1

        label_path.write_text(
            "\n".join(lines),
            encoding="utf-8"
        )

    print(f"\n{split.upper()}:")
    print("Anh trong JSON:", len(images))
    print("Anh khong tim thay:", missing_images)
    print("So nhan da chuyen:")
    for class_id, count in sorted(total_labels.items()):
        print(f"  ID {class_id}: {count}")


def main():
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(
            f"Duong dan dataset khong ton tai: {SOURCE_DIR}"
        )

    for split in SPLITS:
        convert_split(split)

    # Tao file data.yaml cho Ultralytics YOLO
    yaml_path = OUTPUT_DIR / "data.yaml"

    yaml_content = f"""path: {OUTPUT_DIR.as_posix()}
train: images/train
val: images/valid
test: images/test

names:
  0: Bike
  1: Helmet_On
  2: No_helmet
  3: Rider
  4: Trippling
"""

    yaml_path.write_text(yaml_content, encoding="utf-8")

    print("\nDa chuyen doi xong.")
    print("Dataset YOLO:", OUTPUT_DIR)
    print("File cau hinh:", yaml_path)


if __name__ == "__main__":
    main()
