import argparse
import csv
from pathlib import Path

import cv2

from core.violation_pipeline import ViolationPipeline
from detection.cargo.paths import latest_cargo_checkpoint


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_HELMET_MODEL = (
    BASE_DIR / "runs" / "detect" / "helmet_model-2" / "weights" / "best.pt"
)
DEFAULT_PASSENGER_MODEL = (
    BASE_DIR / "runs" / "detect" / "v02_trippling" / "weights" / "best.pt"
)
DEFAULT_CARGO_MODEL = latest_cargo_checkpoint(BASE_DIR)
DEFAULT_SOURCE = BASE_DIR / "videos" / "videotest.mp4"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Nhan dien mu bao hiem, cho qua nguoi va hang cong kenh."
    )
    parser.add_argument(
        "--source",
        default=str(DEFAULT_SOURCE),
        help="Duong dan video hoac so camera (mac dinh: videos/videotest.mp4).",
    )
    parser.add_argument(
        "--helmet-model",
        default=str(DEFAULT_HELMET_MODEL),
        help="Checkpoint YOLO cho helmet/no_helmet.",
    )
    parser.add_argument(
        "--passenger-model",
        default=str(DEFAULT_PASSENGER_MODEL),
        help="Checkpoint YOLO cho Bike/Rider/Trippling.",
    )
    parser.add_argument(
        "--cargo-model",
        help=(
            "Checkpoint YOLO class bulky_cargo/protrusion. Neu bo trong, "
            "dung checkpoint best.pt moi nhat trong "
            "runs/detect/cargo_protrusion_v3_fixed*/weights neu ton tai."
        ),
    )
    parser.add_argument(
        "--output",
        help="Luu video da danh dau vao duong dan nay.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Khong mo cua so video (phu hop chay kiem tra/khong co GUI).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Dung sau so frame nay; huu ich khi kiem tra nhanh.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.35,
        help="Nguong tin cay YOLO tu 0 den 1 (mac dinh: 0.35).",
    )
    return parser.parse_args()


def parse_source(source):
    return int(source) if source.isdecimal() else source


def main():
    args = parse_args()
    if not 0 < args.confidence <= 1:
        raise SystemExit("--confidence phai nam trong khoang (0, 1].")
    if args.max_frames is not None and args.max_frames <= 0:
        raise SystemExit("--max-frames phai lon hon 0.")

    cargo_model = (
        Path(args.cargo_model)
        if args.cargo_model
        else DEFAULT_CARGO_MODEL if DEFAULT_CARGO_MODEL.is_file() else None
    )
    if args.cargo_model and not cargo_model.is_file():
        raise FileNotFoundError(f"Khong tim thay model cargo: {cargo_model}")

    pipeline = ViolationPipeline(
        helmet_model_path=Path(args.helmet_model),
        passenger_model_path=Path(args.passenger_model),
        cargo_model_path=cargo_model,
        confidence=args.confidence,
    )

    source = parse_source(args.source)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"Khong mo duoc nguon video: {args.source}")

    writer = None
    output_path = Path(args.output) if args.output else None
    frame_index = 0
    log_file = None
    log_writer = None

    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25.0

        while args.max_frames is None or frame_index < args.max_frames:
            success, frame = capture.read()
            if not success:
                break

            annotated, events = pipeline.process(frame)
            frame_index += 1

            if output_path is not None:
                if writer is None:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    height, width = annotated.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(
                            f"Khong tao duoc video ket qua: {output_path}"
                        )
                writer.write(annotated)

            if log_writer is None:
                log_path = (
                    output_path.with_suffix(".csv")
                    if output_path is not None
                    else BASE_DIR / "violations_detected.csv"
                )
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_file = log_path.open("w", newline="", encoding="utf-8-sig")
                log_writer = csv.DictWriter(
                    log_file,
                    fieldnames=[
                        "frame",
                        "track_id",
                        "violation",
                        "passenger_count",
                        "confidence",
                    ],
                )
                log_writer.writeheader()

            for event in events:
                log_writer.writerow({"frame": frame_index, **event})

            if not args.no_display:
                cv2.imshow("Traffic Violation Detection", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        if frame_index == 0:
            raise RuntimeError(f"Nguon video khong co frame: {args.source}")
        print(f"Da xu ly {frame_index} frame.")
        if output_path is not None:
            print(f"Video ket qua: {output_path}")
        if log_file is not None:
            print(f"Nhật ký vi phạm: {log_file.name}")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if log_file is not None:
            log_file.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
