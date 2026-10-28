import csv
import json
import math
import sqlite3
import tempfile
import time
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

import cv2
import torch
import yaml
from ultralytics import YOLO

from detection.cargo.classes import normalize_cargo_classes


BASE_DIR = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MODEL_ARCHITECTURES = {
    "yolov8s": {
        "label": "YOLOv8s",
        "model_family": "Ultralytics YOLOv8s",
        "pretrained_weights": Path("yolov8s.pt"),
    },
    "yolo11n": {
        "label": "YOLO11n",
        "model_family": "Ultralytics YOLO11n",
        "pretrained_weights": Path("yolo11n.pt"),
    },
    "yolo26n": {
        "label": "YOLO26n",
        "model_family": "Ultralytics YOLO26n",
        "pretrained_weights": Path("weights/yolo26n.pt"),
    },
}

TASKS = {
    "helmet": {
        "label": "Không đội mũ bảo hiểm",
        "dataset_id": "helmet_dataset",
        "classes": ["helmet", "no_helmet"],
        "used_classes": ["helmet", "no_helmet"],
        "confidence": 0.35,
        "models": {
            "helmet_yolo11n": {
                "architecture": "yolo11n",
                "weight_config": "HELMET_MODEL",
                "weight_file": "runs/detect/helmet_model-2/weights/best.pt",
                "run_dir": "runs/detect/helmet_model-2",
                "version": "YOLO11n · checkpoint hiện tại",
            },
            "helmet_yolov8s": {
                "architecture": "yolov8s",
                "weight_config": "HELMET_YOLOV8S_MODEL",
                "weight_file": "runs/detect/helmet_yolov8s/weights/best.pt",
                "run_dir": "runs/detect/helmet_yolov8s",
                "ready_marker": "comparison-ready.txt",
                "version": "YOLOv8s · fine-tune trên bộ mũ bảo hiểm",
            },
            "helmet_yolo26n": {
                "architecture": "yolo26n",
                "weight_config": "HELMET_YOLO26N_MODEL",
                "weight_file": "runs/detect/helmet_yolo26n/weights/best.pt",
                "run_dir": "runs/detect/helmet_yolo26n",
                "ready_marker": "comparison-ready.txt",
                "version": "YOLO26n · fine-tune trên bộ mũ bảo hiểm",
            },
        },
    },
    "passenger": {
        "label": "Xe máy chở quá số người",
        "dataset_id": "passenger_dataset",
        "classes": ["Bike", "Helmet_On", "No_helmet", "Rider", "Trippling"],
        "used_classes": ["Bike", "Rider", "Trippling"],
        "confidence": 0.3,
        "models": {
            "passenger_yolo11n": {
                "architecture": "yolo11n",
                "weight_config": "PASSENGER_MODEL",
                "weight_file": "runs/detect/v02_trippling/weights/best.pt",
                "run_dir": "runs/detect/v02_trippling",
                "version": "YOLO11n · checkpoint hiện tại",
            },
            "passenger_yolov8s": {
                "architecture": "yolov8s",
                "weight_config": "PASSENGER_YOLOV8S_MODEL",
                "weight_file": "runs/detect/passenger_yolov8s/weights/best.pt",
                "run_dir": "runs/detect/passenger_yolov8s",
                "ready_marker": "comparison-ready.txt",
                "version": "YOLOv8s · fine-tune trên bộ V02",
            },
            "passenger_yolo26n": {
                "architecture": "yolo26n",
                "weight_config": "PASSENGER_YOLO26N_MODEL",
                "weight_file": "runs/detect/passenger_yolo26n/weights/best.pt",
                "run_dir": "runs/detect/passenger_yolo26n",
                "ready_marker": "comparison-ready.txt",
                "version": "YOLO26n · fine-tune trên bộ V02",
            },
        },
    },
    "cargo": {
        "label": "Hàng hóa cồng kềnh",
        "dataset_id": "cargo_dataset",
        "classes": ["bulky_cargo"],
        "used_classes": ["bulky_cargo"],
        "confidence": 0.35,
        "unavailable_splits": {
            "test": (
                "Tập test cargo đã được dùng để đánh giá checkpoint "
                "cargo_protrusion_relabel_v1; không còn là holdout độc lập."
            ),
        },
        "models": {
            "cargo_yolo11n": {
                "architecture": "yolo11n",
                "weight_config": "CARGO_MODEL",
                "weight_file": "runs/detect/cargo_protrusion_relabel_v1/weights/best.pt",
                "run_dir": "runs/detect/cargo_protrusion_relabel_v1",
                "version": "YOLO11n · fine-tune Roboflow v2 đã gắn nhãn lại",
            },
            "cargo_yolov8s": {
                "architecture": "yolov8s",
                "weight_config": "CARGO_YOLOV8S_MODEL",
                "weight_file": "runs/detect/cargo_yolov8s/weights/best.pt",
                "run_dir": "runs/detect/cargo_yolov8s",
                "ready_marker": "comparison-ready.txt",
                "version": "YOLOv8s · fine-tune trên bộ Cargo",
            },
            "cargo_yolo26n": {
                "architecture": "yolo26n",
                "weight_config": "CARGO_YOLO26N_MODEL",
                "weight_file": "runs/detect/cargo_yolo26n/weights/best.pt",
                "run_dir": "runs/detect/cargo_yolo26n",
                "ready_marker": "comparison-ready.txt",
                "version": "YOLO26n · fine-tune trên bộ Cargo",
            },
        },
    },
}

DATASETS = {
    "helmet_dataset": {
        "label": "Bộ dữ liệu mũ bảo hiểm",
        "yaml": Path("dataset/data.yaml"),
        "root": Path("dataset"),
    },
    "passenger_dataset": {
        "label": "Bộ dữ liệu V02",
        "yaml": Path("dataset_v02_yolo/data.yaml"),
        "root": Path("dataset_v02_yolo"),
    },
    "cargo_dataset": {
        "label": "Roboflow Cargo v2 · gắn nhãn lại, chia theo cảnh",
        "yaml": Path("dataset_cargo/roboflow-v5-prepared/data.yaml"),
        "root": Path("dataset_cargo/roboflow-v5-prepared"),
    },
}


def _read_dataset_definition(task_id):
    task = TASKS[task_id]
    dataset = DATASETS[task["dataset_id"]]
    yaml_path = BASE_DIR / dataset["yaml"]
    if not yaml_path.is_file():
        raise FileNotFoundError("Không tìm thấy cấu hình bộ dữ liệu.")
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("Cấu hình bộ dữ liệu không hợp lệ.")
    configured_names = config.get("names", {})
    if isinstance(configured_names, dict):
        configured_classes = {
            str(name).strip().lower() for name in configured_names.values()
        }
    elif isinstance(configured_names, list):
        configured_classes = {
            str(name).strip().lower() for name in configured_names
        }
    else:
        raise ValueError("Tên lớp trong bộ dữ liệu không hợp lệ.")
    if task_id == "cargo":
        _, normalized_names = normalize_cargo_classes(configured_names)
        configured_classes = {
            str(name).strip().lower()
            for name in normalized_names.values()
        }
    expected_classes = {
        name.lower() for name in TASKS[task_id]["classes"]
    }
    if configured_classes != expected_classes:
        raise ValueError("Bộ dữ liệu không tương thích với bài toán đã chọn.")
    return dataset, yaml_path, config


def _dataset_root(dataset):
    root = (BASE_DIR / dataset["root"]).resolve()
    if not root.is_dir():
        raise FileNotFoundError("Không tìm thấy thư mục bộ dữ liệu.")
    return root


def _display_checkpoint_path(weight_path):
    try:
        return weight_path.relative_to(BASE_DIR).as_posix()
    except ValueError:
        return "Checkpoint được cấu hình cho bài toán"


def _is_within(path, root):
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _split_paths(task_id, split):
    if split not in {"val", "test"}:
        raise ValueError("Chỉ hỗ trợ tập validation hoặc test.")
    dataset, yaml_path, config = _read_dataset_definition(task_id)
    split_value = config.get(split)
    if not isinstance(split_value, str) or not split_value.strip():
        raise ValueError("Bộ dữ liệu không khai báo tập đánh giá này.")

    root = _dataset_root(dataset)
    relative_image_dir = Path(split_value)
    if relative_image_dir.is_absolute():
        raise ValueError("Đường dẫn bộ dữ liệu không hợp lệ.")
    image_dir = (root / relative_image_dir).resolve()
    if not _is_within(image_dir, root) or not image_dir.is_dir():
        raise FileNotFoundError("Không tìm thấy thư mục ảnh của tập đánh giá.")

    parts = list(relative_image_dir.parts)
    if "images" not in parts:
        raise ValueError("Cấu hình tập ảnh không có thư mục images.")
    parts[parts.index("images")] = "labels"
    label_dir = (root / Path(*parts)).resolve()
    if not _is_within(label_dir, root) or not label_dir.is_dir():
        raise FileNotFoundError("Không tìm thấy nhãn của tập đánh giá.")
    return dataset, yaml_path, config, root, image_dir, label_dir


def _split_images_and_counts(task_id, split):
    dataset, yaml_path, config, root, image_dir, label_dir = _split_paths(
        task_id, split
    )
    images = sorted(
        path
        for path in image_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise ValueError("Tập đánh giá không có ảnh hợp lệ.")

    object_count = 0
    expected_labels = set()
    class_names = config.get("names", {})
    if isinstance(class_names, dict):
        valid_class_ids = {int(class_id) for class_id in class_names}
    elif isinstance(class_names, list):
        valid_class_ids = set(range(len(class_names)))
    else:
        raise ValueError("Tên lớp trong bộ dữ liệu không hợp lệ.")
    for image_path in images:
        relative = image_path.relative_to(root)
        parts = list(relative.parts)
        parts[parts.index("images")] = "labels"
        label_path = root.joinpath(*parts).with_suffix(".txt")
        if not label_path.is_file():
            raise ValueError("Tập đánh giá thiếu file nhãn cho một số ảnh.")
        expected_labels.add(label_path.resolve())
        for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(
                    f"Nhãn không hợp lệ trong tập {split} ở dòng {line_number}."
                )
            try:
                class_id = int(parts[0])
                coordinates = [float(value) for value in parts[1:]]
            except ValueError as error:
                raise ValueError(
                    f"Nhãn không hợp lệ trong tập {split} ở dòng {line_number}."
                ) from error
            if (
                class_id not in valid_class_ids
                or not all(math.isfinite(value) and 0 <= value <= 1 for value in coordinates)
                or coordinates[2] == 0
                or coordinates[3] == 0
            ):
                raise ValueError(
                    f"Nhãn ngoài phạm vi hoặc sai class trong tập {split} "
                    f"ở dòng {line_number}."
                )
            object_count += 1
    label_paths = list(label_dir.rglob("*.txt"))
    orphan_label_count = sum(
        1
        for path in label_paths
        if not path.is_file() or path.resolve() not in expected_labels
    )

    source_group_count = None
    manifest_path = root / "split_manifest.csv"
    if manifest_path.is_file():
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as stream:
            manifest_rows = csv.DictReader(stream)
            if {"source_group", "split", "image"}.issubset(
                manifest_rows.fieldnames or ()
            ):
                manifest_split = "valid" if split == "val" else split
                image_names = {path.name for path in images}
                source_group_count = len({
                    row["source_group"]
                    for row in manifest_rows
                    if row["split"] == manifest_split
                    and row["image"] in image_names
                    and row["source_group"]
                })

    return {
        "dataset_id": TASKS[task_id]["dataset_id"],
        "dataset_label": dataset["label"],
        "split": split,
        "images": images,
        "image_count": len(images),
        "object_count": object_count,
        "source_group_count": source_group_count,
        "orphan_label_count": orphan_label_count,
        "image_dir": image_dir,
        "label_dir": label_dir,
        "root": root,
        "yaml_path": yaml_path,
        "yaml_config": config,
    }


def _read_training_metrics(run_dir, validation_report=None):
    metrics_path = run_dir / "results.csv"
    if not metrics_path.is_file():
        return None
    with metrics_path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    metric_names = {
        "precision": "metrics/precision(B)",
        "recall": "metrics/recall(B)",
        "map50": "metrics/mAP50(B)",
        "map50_95": "metrics/mAP50-95(B)",
    }
    valid_rows = [
        row for row in rows
        if all(row.get(column) not in (None, "") for column in metric_names.values())
    ]
    if not valid_rows:
        return None
    best = max(valid_rows, key=lambda row: float(row["metrics/mAP50-95(B)"]))
    result = {
        name: float(best[column])
        for name, column in metric_names.items()
    }
    result.update({
        "epoch": int(float(best["epoch"])),
        "source": "validation",
        "status": "validation",
    })
    if validation_report is not None and validation_report.is_file():
        report = json.loads(validation_report.read_text(encoding="utf-8"))
        validation = report["metrics"]["validation"]
        report_metrics = {
            name: float(validation[name])
            for name in metric_names
        }
        if not all(
            math.isfinite(value) and 0 <= value <= 1
            for value in report_metrics.values()
        ):
            raise ValueError("Metric validation trong báo cáo không hợp lệ.")
        result.update(report_metrics)
        result["epoch"] = int(report["training"]["best_epoch"])
        result["source"] = "post_training_validation"
    return result


def model_weight_path(config, task_id, model_id):
    task = TASKS[task_id]
    model = task["models"].get(model_id)
    if model is None:
        raise ValueError("Mô hình không tương thích với bài toán đã chọn.")
    configured_path = config.get(model["weight_config"])
    if configured_path is not None:
        return Path(configured_path).resolve()
    return (BASE_DIR / model["weight_file"]).resolve()


def model_is_ready(config, task_id, model_id):
    task = TASKS[task_id]
    model = task["models"].get(model_id)
    if model is None:
        raise ValueError("Mô hình không tương thích với bài toán đã chọn.")
    weight_path = model_weight_path(config, task_id, model_id)
    if not weight_path.is_file():
        return False
    ready_marker = model.get("ready_marker")
    if ready_marker is None:
        return True
    return (weight_path.parents[1] / ready_marker).is_file()


def get_model_catalog(config):
    catalog = []
    for task_id, definition in TASKS.items():
        split_info = {}
        for split in ("val", "test"):
            unavailable_reason = definition.get(
                "unavailable_splits", {}
            ).get(split)
            if unavailable_reason:
                split_info[split] = {
                    "available": False,
                    "error": unavailable_reason,
                }
                continue
            try:
                dataset_info = _split_images_and_counts(task_id, split)
            except (FileNotFoundError, ValueError) as error:
                split_info[split] = {
                    "available": False,
                    "error": str(error),
                }
            else:
                split_info[split] = {
                    "available": True,
                    "image_count": dataset_info["image_count"],
                    "object_count": dataset_info["object_count"],
                    "source_group_count": dataset_info["source_group_count"],
                    "orphan_label_count": dataset_info["orphan_label_count"],
                    "dataset_label": dataset_info["dataset_label"],
                }
        for model_id, model_definition in definition["models"].items():
            weight_path = model_weight_path(config, task_id, model_id)
            run_dir = BASE_DIR / model_definition["run_dir"]
            validation_report = model_definition.get("validation_report")
            metrics = _read_training_metrics(
                run_dir,
                BASE_DIR / validation_report if validation_report else None,
            )
            checkpoint_exists = weight_path.is_file()
            ready = model_is_ready(config, task_id, model_id)
            training_incomplete = (
                not ready
                and (run_dir / "weights" / "last.pt").is_file()
            )
            model_size = (
                weight_path.stat().st_size if checkpoint_exists else None
            )
            architecture = MODEL_ARCHITECTURES[
                model_definition["architecture"]
            ]
            catalog.append({
                "id": model_id,
                "task": task_id,
                "task_label": definition["label"],
                "name": model_definition["version"],
                "architecture": architecture["label"],
                "model_family": architecture["model_family"],
                "classes": definition["classes"],
                "used_classes": definition["used_classes"],
                "weight_file": _display_checkpoint_path(weight_path),
                "available": ready,
                "training_incomplete": training_incomplete,
                "model_size_bytes": model_size,
                "model_size_mb": (
                    model_size / (1024 * 1024) if model_size is not None else None
                ),
                "image_size": 640,
                "system_confidence": definition["confidence"],
                "training_metrics": metrics,
                "dataset_label": DATASETS[definition["dataset_id"]]["label"],
                "splits": split_info,
            })
    return catalog


def build_evaluation_yaml(dataset_info, split):
    config = dict(dataset_info["yaml_config"])
    config["path"] = str(dataset_info["root"])
    config[split] = str(dataset_info["image_dir"])
    config["names"] = config.get("names", {})
    temporary_directory = tempfile.TemporaryDirectory(prefix="traffic-ai-eval-")
    yaml_path = Path(temporary_directory.name) / "dataset.yaml"
    yaml_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return temporary_directory, yaml_path


def _device_name():
    if torch.cuda.is_available():
        return "0", torch.cuda.get_device_name(0)
    return "cpu", "CPU"


def _synchronize(device):
    if device != "cpu":
        torch.cuda.synchronize()


def evaluate_model(
    model_path, task_id, model_id, split, confidence, repeats, job_id
):
    if task_id not in TASKS:
        raise ValueError("Bài toán đánh giá không được hỗ trợ.")
    model_definition = TASKS[task_id]["models"].get(model_id)
    if model_definition is None:
        raise ValueError("Mô hình không tương thích với bài toán đã chọn.")
    if not model_path.is_file():
        raise FileNotFoundError("File trọng số không tồn tại.")

    dataset_info = _split_images_and_counts(task_id, split)
    temporary_directory, eval_yaml = build_evaluation_yaml(
        dataset_info, split
    )
    device, device_label = _device_name()
    try:
        cache_path = dataset_info["label_dir"].with_suffix(".cache")
        if cache_path.is_file():
            # Rebuild Ultralytics' derived index against the current files.
            cache_path.unlink()
        model = YOLO(str(model_path))
        model_names = model.names
        if isinstance(model_names, dict):
            model_classes = {
                str(name).strip().lower() for name in model_names.values()
            }
        else:
            model_classes = {
                str(name).strip().lower() for name in model_names
            }
        if task_id == "cargo":
            _, normalized_names = normalize_cargo_classes(model_names)
            model_classes = {
                str(name).strip().lower()
                for name in normalized_names.values()
            }
        expected_classes = {
            name.lower() for name in TASKS[task_id]["classes"]
        }
        if model.task != "detect" or model_classes != expected_classes:
            raise ValueError("Checkpoint không tương thích với bài toán đã chọn.")
        metrics = model.val(
            data=str(eval_yaml),
            split=split,
            conf=confidence,
            imgsz=640,
            batch=4,
            workers=0,
            device=device,
            project=str(BASE_DIR / "runs" / "detect" / "ai_comparison"),
            name=f"{job_id}_{model_id}",
            exist_ok=False,
            plots=False,
            verbose=False,
        )
        box_metrics = metrics.box
        image_paths = dataset_info["images"]
        first_image = cv2.imread(str(image_paths[0]))
        if first_image is None:
            raise ValueError("Không đọc được ảnh trong tập đo tốc độ.")

        model.predict(
            first_image,
            conf=confidence,
            imgsz=640,
            device=device,
            verbose=False,
        )
        measured_times = []
        for _ in range(repeats):
            for image_path in image_paths:
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError("Không đọc được ảnh trong tập đo tốc độ.")
                _synchronize(device)
                started = time.perf_counter()
                model.predict(
                    image,
                    conf=confidence,
                    imgsz=640,
                    device=device,
                    verbose=False,
                )
                _synchronize(device)
                measured_times.append(
                    (time.perf_counter() - started) * 1000
                )

        return {
            "model_id": model_id,
            "task": task_id,
            "task_label": TASKS[task_id]["label"],
            "model_name": model_definition["version"],
            "weight_file": _display_checkpoint_path(model_path.resolve()),
            "dataset": dataset_info["dataset_label"],
            "split": split,
            "precision": float(box_metrics.mp),
            "recall": float(box_metrics.mr),
            "map50": float(box_metrics.map50),
            "map50_95": float(box_metrics.map),
            "inference_ms": sum(measured_times) / len(measured_times),
            "inference_min_ms": min(measured_times),
            "inference_max_ms": max(measured_times),
            "model_size_mb": model_path.stat().st_size / (1024 * 1024),
            "image_count": dataset_info["image_count"],
            "object_count": dataset_info["object_count"],
            "orphan_label_count": dataset_info["orphan_label_count"],
            "confidence": confidence,
            "repeats": repeats,
            "image_size": 640,
            "device": device_label,
            "timing_scope": (
                "Bao gồm tiền xử lý, suy luận và hậu xử lý Ultralytics; "
                "không gồm đọc ảnh từ đĩa."
            ),
            "evaluated_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds"
            ),
            "status": "Đã đánh giá",
        }
    finally:
        temporary_directory.cleanup()


def initialize_evaluation_history(database_path):
    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS ai_model_evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    task TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    weight_file TEXT NOT NULL,
                    dataset TEXT NOT NULL,
                    split TEXT NOT NULL,
                    evaluated_at TEXT NOT NULL,
                    precision REAL NOT NULL,
                    recall REAL NOT NULL,
                    map50 REAL NOT NULL,
                    map50_95 REAL NOT NULL,
                    inference_ms REAL NOT NULL,
                    inference_min_ms REAL NOT NULL,
                    inference_max_ms REAL NOT NULL,
                    model_size_mb REAL NOT NULL,
                    image_count INTEGER NOT NULL,
                    object_count INTEGER NOT NULL,
                    orphan_label_count INTEGER NOT NULL DEFAULT 0,
                    confidence REAL NOT NULL,
                    repeats INTEGER NOT NULL,
                    image_size INTEGER NOT NULL,
                    device TEXT NOT NULL,
                    timing_scope TEXT NOT NULL
                )
                """
            )
            columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(ai_model_evaluations)"
                ).fetchall()
            }
            if "orphan_label_count" not in columns:
                connection.execute(
                    """
                    ALTER TABLE ai_model_evaluations
                    ADD COLUMN orphan_label_count INTEGER NOT NULL DEFAULT 0
                    """
                )


def save_evaluation_results(database_path, job_id, results):
    with closing(sqlite3.connect(database_path)) as connection:
        with connection:
            connection.executemany(
                """
                INSERT INTO ai_model_evaluations (
                    job_id, model_id, task, model_name, weight_file, dataset,
                    split, evaluated_at, precision, recall, map50, map50_95,
                    inference_ms, inference_min_ms, inference_max_ms,
                    model_size_mb, image_count, object_count,
                    orphan_label_count, confidence, repeats, image_size,
                    device, timing_scope
                ) VALUES (
                    :job_id, :model_id, :task, :model_name, :weight_file,
                    :dataset, :split, :evaluated_at, :precision, :recall,
                    :map50, :map50_95, :inference_ms, :inference_min_ms,
                    :inference_max_ms, :model_size_mb, :image_count,
                    :object_count, :orphan_label_count, :confidence,
                    :repeats, :image_size, :device, :timing_scope
                )
                """,
                [{"job_id": job_id, **result} for result in results],
            )


def load_evaluation_history(database_path, task=None, limit=30):
    query = "SELECT * FROM ai_model_evaluations"
    parameters = []
    if task:
        query += " WHERE task = ?"
        parameters.append(task)
    query += " ORDER BY evaluated_at DESC, id DESC LIMIT ?"
    parameters.append(limit)
    with closing(sqlite3.connect(database_path)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(query, parameters).fetchall()
    return [dict(row) for row in rows]
