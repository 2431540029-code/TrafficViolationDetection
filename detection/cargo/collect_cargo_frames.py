import argparse
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Trich frame tu video de tao du lieu chua gan nhan cho cargo."
        )
    )
    parser.add_argument("--video", required=True, help="Video can lay frame.")
    parser.add_argument(
        "--output",
        default="dataset_cargo_staging/images",
        help="Thu muc luu anh de gan nhan.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        help="Khoang cach giua hai frame duoc lay, tinh bang giay.",
    )
    args = parser.parse_args()
    if args.interval <= 0:
        parser.error("--interval phai lon hon 0.")

    video_path = Path(args.video)
    if not video_path.is_file():
        raise FileNotFoundError(f"Khong tim thay video: {video_path}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Khong mo duoc video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        capture.release()
        raise RuntimeError(f"Khong doc duoc FPS cua video: {video_path}")

    frame_interval = max(1, round(fps * args.interval))
    frame_index = 0
    saved = 0
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame_index % frame_interval == 0:
                filename = (
                    f"{video_path.stem}_frame_{frame_index:08d}.jpg"
                )
                target = output_dir / filename
                if target.exists():
                    raise FileExistsError(
                        f"Anh da ton tai, dung de tranh ghi de: {target}"
                    )
                if not cv2.imwrite(str(target), frame):
                    raise OSError(f"Khong luu duoc anh: {target}")
                saved += 1
            frame_index += 1
    finally:
        capture.release()

    print(f"Da trich {saved} frame vao: {output_dir.resolve()}")
    print("Cac frame nay chua co nhan YOLO; hay gan nhan bulky_cargo truoc khi train.")


if __name__ == "__main__":
    main()
