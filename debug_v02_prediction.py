
from pathlib import Path

import cv2
from ultralytics import YOLO

# ==========================================
# 1. CAU HINH
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / "runs"
    / "detect"
    / "v02_trippling"
    / "weights"
    / "best.pt"
)

# Doi ten file nay cho khop voi anh thuc te cua ban.
IMAGE_PATH = (
    BASE_DIR
    / "dataset_v02"
    / "images"
    / "xe_cho_4_nguoi.jpeg"
)

RESULT_DIR = BASE_DIR / "dataset_v02" / "results" / "debug"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

CONFIDENCE_LEVELS = [0.15, 0.05, 0.01]

WINDOW_NAME = "Kiem tra nhan dien xe cho qua so nguoi"
MAX_WIDTH = 800
MAX_HEIGHT = 500


# ==========================================
# 2. KIEM TRA DUONG DAN
# ==========================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Khong tim thay model:\n{MODEL_PATH}"
    )

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Khong tim thay anh:\n{IMAGE_PATH}\n"
        "Hay kiem tra ten file va duoi anh .jpg, .jpeg hoac .webp."
    )


# ==========================================
# 3. TAI MODEL VA ANH
# ==========================================

model = YOLO(str(MODEL_PATH))

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise ValueError(f"OpenCV khong doc duoc anh: {IMAGE_PATH}")

print("Ten lop:", model.names)
print("Anh kiem tra:", IMAGE_PATH)
print("Kich thuoc anh goc:", image.shape[1], "x", image.shape[0])


# ==========================================
# 4. HAM HIEN THI ANH VUA MAN HINH
# ==========================================

def show_image_fit(window_name, image):
    height, width = image.shape[:2]

    scale = min(
        MAX_WIDTH / width,
        MAX_HEIGHT / height,
        1.0
    )

    display_width = max(1, int(width * scale))
    display_height = max(1, int(height * scale))

    if scale < 1.0:
        display_image = cv2.resize(
            image,
            (display_width, display_height),
            interpolation=cv2.INTER_AREA
        )
    else:
        display_image = image.copy()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        window_name,
        display_width,
        display_height
    )
    cv2.imshow(window_name, display_image)

    print("Nhan phim bat ky de xem muc tiep theo.")
    print("Nhan ESC de dung.")

    key = cv2.waitKey(0)
    cv2.destroyWindow(window_name)

    return key


# ==========================================
# 5. THU NHAN DIEN VOI CAC MUC CONFIDENCE
# ==========================================

for conf in CONFIDENCE_LEVELS:
    print(f"\n{'=' * 45}")
    print(f"CONFIDENCE = {conf}")
    print("=" * 45)

    result = model.predict(
        source=image,
        conf=conf,
        imgsz=640,
        verbose=False
    )[0]

    # Tao anh co khung nhan dien va nhan lop.
    output = result.plot()

    if result.boxes is None or len(result.boxes) == 0:
        print("Khong phat hien doi tuong nao.")
    else:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            coords = [
                round(value, 1)
                for value in box.xyxy[0].tolist()
            ]

            print(
                f"Lop: {model.names[class_id]} | "
                f"Confidence: {confidence:.4f} | "
                f"Box: {coords}"
            )

    # Luu anh ket qua voi ten rieng cho tung nguong.
    output_path = (
        RESULT_DIR
        / f"{IMAGE_PATH.stem}_conf_{conf:.2f}.jpg"
    )

    saved = cv2.imwrite(str(output_path), output)

    if saved:
        print("Da luu anh:", output_path)
    else:
        print("Loi khi luu anh:", output_path)

    # Hien thi anh da thu nho.
    key = show_image_fit(WINDOW_NAME, output)

    if key == 27:
        print("Da dung theo yeu cau.")
        break

print("\nDa ket thuc kiem tra.")