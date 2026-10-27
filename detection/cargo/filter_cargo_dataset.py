from pathlib import Path
import shutil

# Dataset gốc
SOURCE = Path("dataset_cargo/Anomaly detection2.v1i.yolov11")

# Dataset mới chỉ dành cho V03
DEST = Path("dataset_cargo_filtered")

SPLITS = ["train", "valid", "test"]

# Các đuôi ảnh có thể gặp
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def find_image(image_dir, image_name):
    """
    Tìm ảnh tương ứng với file label.
    """
    for ext in IMAGE_EXTENSIONS:
        image_path = image_dir / (image_name + ext)

        if image_path.exists():
            return image_path

    return None


total_images = 0
total_annotations = 0

for split in SPLITS:

    source_images = SOURCE / split / "images"
    source_labels = SOURCE / split / "labels"

    dest_images = DEST / split / "images"
    dest_labels = DEST / split / "labels"

    dest_images.mkdir(parents=True, exist_ok=True)
    dest_labels.mkdir(parents=True, exist_ok=True)

    image_count = 0
    annotation_count = 0

    print(f"\n===== {split.upper()} =====")

    for label_file in source_labels.glob("*.txt"):

        with open(label_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Chỉ giữ class 0 = Bulky Goods Transport
        cargo_lines = []

        for line in lines:

            parts = line.strip().split()

            if len(parts) >= 5:
                class_id = int(parts[0])

                if class_id == 0:
                    cargo_lines.append(line)

        # Nếu ảnh không có hàng cồng kềnh thì bỏ qua
        if not cargo_lines:
            continue

        image_name = label_file.stem

        image_path = find_image(source_images, image_name)

        if image_path is None:
            print(f"Khong tim thay anh: {image_name}")
            continue

        # Copy ảnh
        shutil.copy2(
            image_path,
            dest_images / image_path.name
        )

        # Ghi label mới
        new_label = dest_labels / label_file.name

        with open(new_label, "w", encoding="utf-8") as f:

            for line in cargo_lines:
                # Class 0 vẫn giữ nguyên
                f.write(line)

        image_count += 1
        annotation_count += len(cargo_lines)

    print(f"So anh: {image_count}")
    print(f"So annotation cargo: {annotation_count}")

    total_images += image_count
    total_annotations += annotation_count


# Tạo data.yaml
yaml_content = """train: train/images
val: valid/images
test: test/images

nc: 1
names: ['bulky_cargo']
"""

with open(DEST / "data.yaml", "w", encoding="utf-8") as f:
    f.write(yaml_content)


print("\n==============================")
print("HOAN THANH LOC DATASET V03")
print("==============================")
print(f"Tong so anh: {total_images}")
print(f"Tong annotation: {total_annotations}")
print(f"Dataset moi: {DEST}")