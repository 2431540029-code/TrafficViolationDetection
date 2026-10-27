import argparse
import csv
import json
from pathlib import Path

import yaml
from ultralytics import YOLO

from detection.cargo.classes import normalize_cargo_classes
from detection.cargo.paths import latest_cargo_checkpoint


BASE_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = latest_cargo_checkpoint(BASE_DIR)
DEFAULT_DATA = BASE_DIR / "dataset_cargo" / "roboflow-v3" / "data.yaml"


def resolve_split_directories(data_yaml, split):
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    dataset_root = Path(config.get("path", data_yaml.parent))
    if not dataset_root.is_absolute():
        dataset_root = data_yaml.parent / dataset_root
    split_entry = config.get(split)
    if not split_entry:
        raise ValueError(
            f"Dataset YAML khong khai bao {split}: {data_yaml}"
        )
    image_dir = Path(split_entry)
    if not image_dir.is_absolute():
        image_dir = dataset_root / image_dir
    label_dir = image_dir.parent / "labels"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Khong tim thay anh test: {image_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Khong tim thay nhan test: {label_dir}")
    return image_dir, label_dir


def read_ground_truth(label_path, image_width, image_height):
    boxes = []
    if not label_path.is_file():
        return boxes
    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"Nhãn YOLO không hợp lệ: {label_path}:{line_number}")
        class_id, center_x, center_y, width, height = map(float, parts)
        if int(class_id) != 0:
            continue
        boxes.append(
            (
                (center_x - width / 2) * image_width,
                (center_y - height / 2) * image_height,
                (center_x + width / 2) * image_width,
                (center_y + height / 2) * image_height,
            )
        )
    return boxes


def calculate_iou(first, second):
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, first[2] - first[0]) * max(
        0.0, first[3] - first[1]
    )
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0, second[3] - second[1]
    )
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def match_detections(predictions, ground_truth, iou_threshold):
    candidates = sorted(
        (
            (calculate_iou(prediction["box"], truth), pred_index, truth_index)
            for pred_index, prediction in enumerate(predictions)
            for truth_index, truth in enumerate(ground_truth)
        ),
        reverse=True,
    )
    matched_predictions = set()
    matched_truth = set()
    for overlap, pred_index, truth_index in candidates:
        if overlap < iou_threshold:
            break
        if pred_index in matched_predictions or truth_index in matched_truth:
            continue
        matched_predictions.add(pred_index)
        matched_truth.add(truth_index)
    false_positives = [
        prediction for index, prediction in enumerate(predictions)
        if index not in matched_predictions
    ]
    false_negatives = [
        truth for index, truth in enumerate(ground_truth)
        if index not in matched_truth
    ]
    return len(matched_predictions), false_positives, false_negatives


def main():
    parser = argparse.ArgumentParser(
        description="Danh gia cargo tren test set va liet ke FP/FN."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="test",
        help="Tập đánh giá (mặc định: test).",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Tên thư mục kết quả trong runs/detect.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Confidence dùng để liệt kê FP/FN (mặc định: 0.25).",
    )
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument(
        "--device",
        default="0",
        help="GPU index hoac 'cpu'.",
    )
    args = parser.parse_args()

    if not args.model.is_file():
        raise FileNotFoundError(f"Khong tim thay model cargo: {args.model}")
    if not args.data.is_file():
        raise FileNotFoundError(f"Khong tim thay dataset YAML: {args.data}")
    if not 0 < args.confidence <= 1 or not 0 < args.iou <= 1:
        parser.error("--confidence va --iou phai nam trong khoang (0, 1].")

    model = YOLO(str(args.model))
    cargo_class_ids, _ = normalize_cargo_classes(model.names)
    ultralytics_confidence = 0.001
    evaluation_name = args.name or (
        "cargo_protrusion_test_evaluation"
        if args.split == "test"
        else "cargo_protrusion_val_evaluation"
    )

    metrics = model.val(
        data=str(args.data.resolve()),
        split=args.split,
        conf=ultralytics_confidence,
        device=args.device,
        classes=cargo_class_ids,
        plots=True,
        project=str(BASE_DIR / "runs" / "detect"),
        name=evaluation_name,
    )
    image_dir, label_dir = resolve_split_directories(
        args.data.resolve(), args.split
    )
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(
        path for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in image_extensions
    )
    if not images:
        raise ValueError(f"Khong co anh {args.split} trong {image_dir}")
    image_stems = {image_path.stem for image_path in images}
    orphan_labels = sorted(
        label_path.name
        for label_path in label_dir.rglob("*.txt")
        if label_path.stem not in image_stems
    )
    if orphan_labels:
        print(
            f"CANH BAO: {len(orphan_labels)} file nhan test khong co anh "
            f"tuong ung; vi du: {orphan_labels[:3]}"
        )

    true_positives = 0
    false_positives = []
    false_negatives = []
    for image_path in images:
        result = model.predict(
            source=str(image_path),
            conf=args.confidence,
            classes=cargo_class_ids,
            device=args.device,
            verbose=False,
        )[0]
        height, width = result.orig_shape
        label_path = label_dir / f"{image_path.stem}.txt"
        ground_truth = read_ground_truth(label_path, width, height)
        predictions = []
        if result.boxes is not None:
            predictions = [
                {
                    "box": tuple(float(value) for value in box.xyxy[0].tolist()),
                    "confidence": float(box.conf[0].item()),
                }
                for box in result.boxes
                if int(box.cls[0].item()) == 0
            ]
        matched, frame_fp, frame_fn = match_detections(
            predictions, ground_truth, args.iou
        )
        true_positives += matched
        false_positives.extend(
            {"image": image_path.name, **item} for item in frame_fp
        )
        false_negatives.extend(
            {"image": image_path.name, "box": box} for box in frame_fn
        )

    precision = (
        true_positives / (true_positives + len(false_positives))
        if true_positives + len(false_positives)
        else 0.0
    )
    recall = (
        true_positives / (true_positives + len(false_negatives))
        if true_positives + len(false_negatives)
        else 0.0
    )
    output_dir = Path(metrics.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    error_csv = output_dir / "cargo_false_positives_and_misses.csv"
    with error_csv.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["type", "image", "confidence", "x1", "y1", "x2", "y2"],
        )
        writer.writeheader()
        for item in false_positives:
            writer.writerow(
                {
                    "type": "false_positive",
                    "image": item["image"],
                    "confidence": item["confidence"],
                    **dict(zip(("x1", "y1", "x2", "y2"), item["box"])),
                }
            )
        for item in false_negatives:
            writer.writerow(
                {
                    "type": "missed_ground_truth",
                    "image": item["image"],
                    **dict(zip(("x1", "y1", "x2", "y2"), item["box"])),
                }
            )

    report = {
        f"{args.split}_images": len(images),
        "orphan_label_files": len(orphan_labels),
        "model_classes": model.names,
        "iou_threshold_for_error_listing": args.iou,
        "confidence_threshold": args.confidence,
        "ultralytics_confidence_threshold": ultralytics_confidence,
        "true_positives": true_positives,
        "false_positives": len(false_positives),
        "missed_ground_truth": len(false_negatives),
        "precision_from_error_listing": precision,
        "recall_from_error_listing": recall,
        "ultralytics_precision": float(metrics.box.mp),
        "ultralytics_recall": float(metrics.box.mr),
        "ultralytics_map50": float(metrics.box.map50),
        "ultralytics_map50_95": float(metrics.box.map),
    }
    (output_dir / f"cargo_{args.split}_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"Chi tiet FP/FN: {error_csv}")


if __name__ == "__main__":
    main()
