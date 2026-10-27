import argparse
from pathlib import Path

import cv2


DEFAULT_DATASET_DIR = Path("dataset_cargo") / "roboflow-v3"

SPLITS = ["train", "valid", "test"]
CLASS_NAMES = {0: "bulky_cargo"}


def check_labels(dataset_dir=DEFAULT_DATASET_DIR):
    total_images = 0
    total_boxes = 0
    missing_labels = 0
    invalid_labels = 0
    orphan_labels = 0
    output_dir = dataset_dir / "label_preview_checked"

    output_dir.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        image_dir = dataset_dir / split / "images"
        label_dir = dataset_dir / split / "labels"
        output_split = output_dir / split
        output_split.mkdir(parents=True, exist_ok=True)

        if not image_dir.exists():
            print(f"[WARNING] Khong tim thay: {image_dir}")
            continue

        images = [
            p for p in image_dir.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
        ]
        image_stems = {image_path.stem for image_path in images}
        for label_path in label_dir.glob("*.txt"):
            if label_path.stem not in image_stems:
                orphan_labels += 1
                print(f"[NHAN MO COI] Khong co anh: {label_path.name}")

        print(f"\n===== {split.upper()} =====")

        for image_path in images:
            total_images += 1

            label_path = label_dir / f"{image_path.stem}.txt"
            image = cv2.imread(str(image_path))

            if image is None:
                print(f"[LOI] Khong doc duoc anh: {image_path.name}")
                continue

            height, width = image.shape[:2]

            if not label_path.exists():
                missing_labels += 1
                print(f"[THIEU NHAN] {image_path.name}")
                continue

            with open(label_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]

            for line_number, line in enumerate(lines, start=1):
                parts = line.split()

                if len(parts) != 5:
                    invalid_labels += 1
                    print(
                        f"[NHAN SAI] {label_path.name}, "
                        f"dong {line_number}: can 5 gia tri"
                    )
                    continue

                try:
                    class_id = int(parts[0])
                    x_center, y_center, box_width, box_height = map(
                        float, parts[1:]
                    )
                except ValueError:
                    invalid_labels += 1
                    print(f"[NHAN SAI] Khong doc duoc: {label_path.name}")
                    continue

                if (
                    class_id not in CLASS_NAMES
                    or not all(
                        0 <= value <= 1
                        for value in [
                            x_center, y_center, box_width, box_height
                        ]
                    )
                    or box_width <= 0
                    or box_height <= 0
                    or x_center - box_width / 2 < 0
                    or y_center - box_height / 2 < 0
                    or x_center + box_width / 2 > 1
                    or y_center + box_height / 2 > 1
                ):
                    invalid_labels += 1
                    print(
                        f"[NHAN SAI] Toa do khong hop le: "
                        f"{label_path.name}, dong {line_number}"
                    )
                    continue

                # Chuyen toa do YOLO sang pixel
                x1 = int((x_center - box_width / 2) * width)
                y1 = int((y_center - box_height / 2) * height)
                x2 = int((x_center + box_width / 2) * width)
                y2 = int((y_center + box_height / 2) * height)

                # Ve khung va ten lop
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    image,
                    CLASS_NAMES[class_id],
                    (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                total_boxes += 1

            # Luu anh xem truoc
            output_path = output_split / image_path.name
            if not cv2.imwrite(str(output_path), image):
                raise OSError(f"Khong luu duoc anh xem nhan: {output_path}")
            print(f"[OK] Da tao anh xem nhan: {output_path}")

    print("\n========== TONG KET ==========")
    print(f"Tong so anh: {total_images}")
    print(f"Tong so khung nhan hop le: {total_boxes}")
    print(f"So anh thieu file nhan: {missing_labels}")
    print(f"So dong nhan khong hop le: {invalid_labels}")
    print(f"So file nhan khong co anh: {orphan_labels}")
    print(f"Anh ket qua duoc luu tai: {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Kiem tra nhan cargo va tao anh xem truoc."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Thu muc dataset co cac split train/valid/test.",
    )
    check_labels(parser.parse_args().dataset)
