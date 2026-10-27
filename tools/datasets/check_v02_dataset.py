
import json
from pathlib import Path
from collections import Counter

DATASET_DIR = Path(
    r"D:\Traffic Rule Violation Detection Dataset in Dhaka\traffic_violation\augmented_dataset"
)

splits = ["train", "valid", "test"]

for split in splits:
    annotation_file = DATASET_DIR / split / "_annotations.coco.json"

    print(f"\n===== {split.upper()} =====")

    if not annotation_file.exists():
        print(f"KHONG TIM THAY: {annotation_file}")
        continue

    with open(annotation_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    categories = {
        category["id"]: category["name"]
        for category in data.get("categories", [])
    }

    counts = Counter(
        annotation["category_id"]
        for annotation in data.get("annotations", [])
    )

    print("So anh:", len(data.get("images", [])))
    print("So annotation:", len(data.get("annotations", [])))
    print("So luong tung lop:")

    for category_id, category_name in categories.items():
        print(
            f"  ID {category_id}: {category_name} "
            f"= {counts.get(category_id, 0)}"
        )
