import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

from detection.cargo.classes import CARGO_CLASS_NAMES, normalize_cargo_classes
from detection.cargo.paths import DEFAULT_CARGO_RUN_NAME


BASE_DIR = Path(__file__).resolve().parents[2]

# Bộ dữ liệu Cargo v3 mới tải từ Roboflow
DEFAULT_DATA = BASE_DIR / "dataset_cargo" / "roboflow-v3" / "data.yaml"

# Mô hình YOLO11n pretrained
DEFAULT_WEIGHTS = BASE_DIR / "yolo11n.pt"


def main():
    parser = argparse.ArgumentParser(
        description="Train YOLO11n nhận diện xe máy chở hàng hóa cồng kềnh."
    )

    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Số worker đọc dữ liệu (mặc định: 0).",
    )
    parser.add_argument(
        "--name",
        default=DEFAULT_CARGO_RUN_NAME,
        help="Tên run trong runs/detect (mặc định là run cargo hiện tại).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Tiếp tục chính xác run đã dừng từ weights/last.pt.",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="Dung 'cpu' neu khong co GPU CUDA; hoac '0' neu co GPU CUDA."
    )

    args = parser.parse_args()

    if not args.data.is_file():
        raise FileNotFoundError(
            f"Khong tim thay file dataset YAML: {args.data}"
        )

    if not args.weights.is_file():
        raise FileNotFoundError(
            f"Khong tim thay model YOLO: {args.weights}"
        )

    if (
        args.epochs <= 0
        or args.imgsz <= 0
        or args.batch <= 0
        or args.workers < 0
    ):
        parser.error(
            "--epochs, --imgsz, --batch phai lon hon 0; --workers "
            "khong duoc am."
        )
    run_dir = BASE_DIR / "runs" / "detect" / args.name
    if args.resume:
        expected_last_checkpoint = run_dir / "weights" / "last.pt"
        if args.weights.resolve() != expected_last_checkpoint.resolve():
            parser.error(
                "--resume yêu cầu --weights trỏ đúng last.pt của run --name."
            )
        if not expected_last_checkpoint.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy checkpoint để tiếp tục: "
                f"{expected_last_checkpoint}"
            )
        saved_args = yaml.safe_load(
            (run_dir / "args.yaml").read_text(encoding="utf-8")
        )
        if (
            not isinstance(saved_args, dict)
            or not isinstance(saved_args.get("data"), str)
        ):
            raise ValueError(
                f"Không đọc được cấu hình dataset của run: {run_dir}"
            )
        if Path(saved_args["data"]).resolve() != args.data.resolve():
            parser.error(
                "--resume yêu cầu dataset YAML giống với run đã bắt đầu."
            )
    elif run_dir.exists():
        raise FileExistsError(
            f"Run đã tồn tại, không ghi đè: {run_dir}. "
            "Dùng --resume cùng last.pt nếu cần tiếp tục."
        )

    data_config = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    dataset_names = data_config.get("names", {})
    try:
        cargo_class_ids, normalized_names = normalize_cargo_classes(dataset_names)
    except ValueError as error:
        parser.error(str(error))
    if len(dataset_names) != 1 or len(cargo_class_ids) != 1:
        parser.error(
            "Dataset huan luyen cargo phai co dung mot class cargo "
            f"({', '.join(sorted(CARGO_CLASS_NAMES))})."
        )

    print("=" * 50)
    print("HUAN LUYEN NHAN DIEN HANG HOA CONG KENH")
    print("=" * 50)
    print(f"Dataset: {args.data}")
    print(f"Model: {args.weights}")
    print(f"Epochs toi da: {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {args.batch}")
    print(f"Workers: {args.workers}")
    print(f"Device: {args.device}")
    print(
        "Class dataset: "
        f"{dataset_names}; class hien thi trong ung dung: "
        f"{normalized_names[cargo_class_ids[0]]}"
    )
    print("=" * 50)

    model = YOLO(str(args.weights))
    if args.resume:
        model.train(resume=True)
    else:
        model.train(
            data=str(args.data.resolve()),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            project=str(BASE_DIR / "runs" / "detect"),
            name=args.name,
            patience=15,
            save=True,
            plots=True,
        )


if __name__ == "__main__":
    main()