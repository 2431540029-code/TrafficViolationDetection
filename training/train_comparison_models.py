import argparse

from ultralytics import YOLO

from web.model_comparison import (
    BASE_DIR,
    DATASETS,
    MODEL_ARCHITECTURES,
    TASKS,
)


EPOCHS_BY_TASK = {
    "helmet": 50,
    "passenger": 30,
    "cargo": 20,
}
COMPARISON_ARCHITECTURES = ("yolov8s", "yolo26n")


def train_comparison_models(tasks, architectures, batch, device):
    for task_id in tasks:
        task = TASKS[task_id]
        dataset_yaml = BASE_DIR / DATASETS[task["dataset_id"]]["yaml"]
        if not dataset_yaml.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy cấu hình dữ liệu: {dataset_yaml}"
            )

        for architecture_id in architectures:
            architecture = MODEL_ARCHITECTURES[architecture_id]
            pretrained_weights = BASE_DIR / architecture["pretrained_weights"]
            if not pretrained_weights.is_file():
                raise FileNotFoundError(
                    f"Không tìm thấy trọng số khởi tạo: {pretrained_weights}"
                )

            model_id = f"{task_id}_{architecture_id}"
            run_dir = BASE_DIR / "runs" / "detect" / model_id
            ready_marker = run_dir / "comparison-ready.txt"
            if ready_marker.is_file():
                raise FileExistsError(
                    f"Mô hình đã huấn luyện hoàn tất: {run_dir}"
                )

            print(
                f"Fine-tune {architecture['label']} cho {task['label']} "
                f"({EPOCHS_BY_TASK[task_id]} epoch tối đa, batch={batch}, "
                f"imgsz=640, device={device})"
            )
            last_checkpoint = run_dir / "weights" / "last.pt"
            resume = last_checkpoint.is_file()
            if run_dir.exists() and not resume:
                raise FileExistsError(
                    f"Run đã tồn tại nhưng không có checkpoint để tiếp tục: "
                    f"{run_dir}"
                )
            model = YOLO(
                str(last_checkpoint if resume else pretrained_weights)
            )
            model.train(
                data=str(dataset_yaml),
                epochs=EPOCHS_BY_TASK[task_id],
                imgsz=640,
                batch=batch,
                device=device,
                workers=0,
                project=str(BASE_DIR / "runs" / "detect"),
                name=model_id,
                exist_ok=False,
                patience=10,
                pretrained=True,
                seed=0,
                deterministic=True,
                amp=True,
                plots=True,
                save=True,
                resume=resume,
            )

            best_checkpoint = run_dir / "weights" / "best.pt"
            if not best_checkpoint.is_file():
                raise FileNotFoundError(
                    f"Huấn luyện kết thúc nhưng không có checkpoint: "
                    f"{best_checkpoint}"
                )
            ready_marker.write_text(
                "Training completed successfully.\n",
                encoding="utf-8",
            )
            print(f"Checkpoint sẵn sàng: {best_checkpoint}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune YOLOv8s và YOLO26n trên cùng dataset của từng tác vụ "
            "để so sánh với YOLO11n."
        )
    )
    parser.add_argument(
        "--task",
        choices=(*TASKS.keys(), "all"),
        default="all",
        help="Tác vụ cần train; mặc định train cả ba tác vụ.",
    )
    parser.add_argument(
        "--architecture",
        choices=(*COMPARISON_ARCHITECTURES, "all"),
        default="all",
        help="Kiến trúc cần train; mặc định train cả YOLOv8s và YOLO26n.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=2,
        help="Batch size dùng trong tất cả các lần train (mặc định: 2).",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Thiết bị Ultralytics, ví dụ 0 hoặc cpu (mặc định: 0).",
    )
    args = parser.parse_args()
    if args.batch < 1:
        parser.error("--batch phải lớn hơn 0.")

    tasks = list(TASKS) if args.task == "all" else [args.task]
    architectures = (
        list(COMPARISON_ARCHITECTURES)
        if args.architecture == "all"
        else [args.architecture]
    )
    train_comparison_models(tasks, architectures, args.batch, args.device)


if __name__ == "__main__":
    main()
