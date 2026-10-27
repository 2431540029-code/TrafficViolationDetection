from pathlib import Path
from collections import Counter
import yaml

DATASET_DIR = Path("dataset_cargo/Anomaly detection2.v1i.yolov11")

splits = ["train", "valid", "test"]

class_names = [
    "Bulky Goods Transport",
    "Illegal Parking",
    "Pedestrian-on-Street",
    "wrong direction"
]


def summarize_dataset(data_yaml):
    data_yaml = Path(data_yaml).resolve()
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    dataset_root = Path(config.get("path", data_yaml.parent))
    if not dataset_root.is_absolute():
        dataset_root = data_yaml.parent / dataset_root

    counts = {}
    for split_name, key in (("train", "train"), ("val", "val"), ("test", "test")):
        split_path = Path(config.get(key, ""))
        if not split_path.is_absolute():
            split_path = dataset_root / split_path
        label_dir = split_path.parent / "labels"
        count = 0
        if label_dir.is_dir():
            for label_path in label_dir.glob("*.txt"):
                for line in label_path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines():
                    parts = line.split()
                    if len(parts) >= 5 and int(parts[0]) == 0:
                        count += 1
        counts[split_name] = count
    return counts


def main():
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


if __name__ == "__main__":
    main()