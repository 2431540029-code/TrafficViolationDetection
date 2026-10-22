from pathlib import Path
from collections import Counter

DATASET_DIR = Path("dataset_cargo/Anomaly detection2.v1i.yolov11")

splits = ["train", "valid", "test"]

class_names = [
    "Bulky Goods Transport",
    "Illegal Parking",
    "Pedestrian-on-Street",
    "wrong direction"
]

for split in splits:
    label_dir = DATASET_DIR / split / "labels"

    counter = Counter()
    total_labels = 0

    if not label_dir.exists():
        print(f"\n{split}: KHONG TIM THAY thu muc labels")
        continue

    for label_file in label_dir.glob("*.txt"):
        with open(label_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                if len(parts) >= 5:
                    class_id = int(parts[0])
                    counter[class_id] += 1
                    total_labels += 1

    print(f"\n===== {split.upper()} =====")
    print(f"Tong so annotation: {total_labels}")

    for class_id, class_name in enumerate(class_names):
        print(
            f"{class_id}: {class_name} -> "
            f"{counter[class_id]} annotations"
        )