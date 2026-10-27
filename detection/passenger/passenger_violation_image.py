
from pathlib import Path

import cv2
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parents[2]

IMAGE_DIR = BASE_DIR / "dataset_v02" / "images"
RESULT_DIR = BASE_DIR / "dataset_v02" / "results"
MODEL_PATH = (
    BASE_DIR / "runs" / "detect" / "v02_trippling"
    / "weights" / "best.pt"
)

CONFIDENCE = 0.35
PASSENGER_LIMIT = 2
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Kich thuoc toi da cua cua so hien thi
MAX_WIDTH = 900
MAX_HEIGHT = 650

WINDOW_NAME = "Kiem thu nhan dien xe cho qua so nguoi"

RESULT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 1. KIEM TRA DUONG DAN
# =========================
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Khong tim thay model:\n{MODEL_PATH}"
    )

if not IMAGE_DIR.exists():
    raise FileNotFoundError(
        f"Khong tim thay thu muc anh:\n{IMAGE_DIR}"
    )

# =========================
# 2. TAI MODEL YOLO
# =========================
model = YOLO(str(MODEL_PATH))

names = model.names
if isinstance(names, list):
    names = dict(enumerate(names))

class_ids = {
    str(name).strip().lower(): int(class_id)
    for class_id, name in names.items()
}

print("Cac lop cua model:", names)

required = ["bike", "rider", "trippling"]
missing = [name for name in required if name not in class_ids]

if missing:
    raise ValueError(
        f"Model khong co cac lop {missing}.\n"
        f"Cac lop hien tai: {names}"
    )

BIKE_ID = class_ids["bike"]
RIDER_ID = class_ids["rider"]
TRIPPLING_ID = class_ids["trippling"]

# =========================
# 3. LAY DANH SACH ANH
# =========================
images = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.is_file()
    and p.suffix.lower() in IMAGE_EXTENSIONS
    and not p.stem.endswith("_result")
)

if not images:
    print(f"Khong tim thay anh trong: {IMAGE_DIR}")
    raise SystemExit

print(f"Tim thay {len(images)} anh.")
print("Nhan phim bat ky de xem anh tiep theo.")
print("Nhan ESC de thoat.")

# =========================
# 4. HAM THU NHO ANH HIEN THI
# =========================
def resize_for_display(image):
    height, width = image.shape[:2]

    scale = min(
        MAX_WIDTH / width,
        MAX_HEIGHT / height,
        1.0
    )

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    if scale < 1.0:
        image = cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA
        )

    return image


# =========================
# 5. XU LY TUNG ANH
# =========================
for image_path in images:

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Khong doc duoc anh: {image_path.name}")
        continue

    # Nhan dien tren anh goc
    result = model.predict(
        source=image,
        classes=[BIKE_ID, RIDER_ID, TRIPPLING_ID],
        conf=CONFIDENCE,
        imgsz=640,
        verbose=False
    )[0]

    bikes = []
    riders = []
    tripplings = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            coords = tuple(
                int(v) for v in box.xyxy[0].tolist()
            )

            item = (coords, confidence)

            if class_id == BIKE_ID:
                bikes.append(item)
            elif class_id == RIDER_ID:
                riders.append(item)
            elif class_id == TRIPPLING_ID:
                tripplings.append(item)

    output = image.copy()

    # -------------------------
    # 5.1 VE KHUNG XANH DUONG CHO RIDER
    # -------------------------
    for (x1, y1, x2, y2), conf in riders:
        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (255, 180, 0),
            2
        )

        cv2.putText(
            output,
            f"Rider {conf:.2f}",
            (x1, max(22, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 180, 0),
            2,
            cv2.LINE_AA
        )

    # -------------------------
    # 5.2 KHONG VE KHUNG DO RIENG CHO TRIPPLING
    # Trippling van duoc thong ke o phia tren.
    # Mau khung xe se dua vao so Rider dem duoc.
    # -------------------------

    total_violations = 0

    # -------------------------
    # 5.3 DEM RIDER TRONG VUNG CUA TUNG XE
    # -------------------------
    for bike, bike_conf in bikes:
        bx1, by1, bx2, by2 = bike

        bw = max(1, bx2 - bx1)
        bh = max(1, by2 - by1)

        # Mo rong vung dem len phia tren xe
        rx1 = bx1 - 0.15 * bw
        rx2 = bx2 + 0.15 * bw
        ry1 = by1 - 1.5 * bh
        ry2 = by2 + 0.05 * bh

        # Dem cac Rider co tam khung nam trong vung xe
        count = 0

        for (px1, py1, px2, py2), _ in riders:
            pcx = (px1 + px2) / 2
            pcy = (py1 + py2) / 2

            if (
                rx1 <= pcx <= rx2
                and ry1 <= pcy <= ry2
            ):
                count += 1

        # Chi ket luan vi pham dua tren so Rider
        violation = count > PASSENGER_LIMIT

        if violation:
            color = (0, 0, 255)  # Do
            label = f"VI PHAM: CHO QUA SO NGUOI ({count})"
            total_violations += 1
        else:
            color = (0, 200, 0)  # Xanh la
            label = None

        # Ve khung bao quanh xe
        cv2.rectangle(
            output,
            (bx1, by1),
            (bx2, by2),
            color,
            4
        )

        # Chi hien chu vi pham khi co tu 3 Rider tro len
        if label is not None:
            text_x = max(5, bx1)
            text_y = max(55, by1 - 10)

            cv2.putText(
                output,
                label,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
                cv2.LINE_AA
            )

    # -------------------------
    # 5.4 THANH THONG KE TONG
    # -------------------------
    summary = (
        f"Bikes: {len(bikes)} | "
        f"Riders: {len(riders)} | "
        f"Trippling: {len(tripplings)}"
    )

    cv2.rectangle(
        output,
        (0, 0),
        (output.shape[1], 40),
        (30, 30, 30),
        -1
    )

    cv2.putText(
        output,
        summary,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # -------------------------
    # 5.5 LUU ANH KET QUA
    # -------------------------
    save_path = RESULT_DIR / f"{image_path.stem}_result.jpg"

    if cv2.imwrite(str(save_path), output):
        print(f"\nAnh: {image_path.name}")
        print(f"So Bike: {len(bikes)}")
        print(f"So Rider: {len(riders)}")
        print(f"So Trippling: {len(tripplings)}")
        print(f"So xe bi danh dau vi pham: {total_violations}")
        print(f"Da luu ket qua: {save_path}")
    else:
        print(f"Khong luu duoc anh: {save_path}")

    # -------------------------
    # 5.6 HIEN THI ANH DA THU NHO
    # -------------------------
    display_image = resize_for_display(output)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(
        WINDOW_NAME,
        display_image.shape[1],
        display_image.shape[0]
    )
    cv2.imshow(WINDOW_NAME, display_image)

    key = cv2.waitKey(0)
    cv2.destroyAllWindows()

    if key == 27:
        print("Da dung chuong trinh.")
        break

print("\nDa ket thuc chuong trinh.")
