
from pathlib import Path

import cv2
from ultralytics import YOLO


# ==========================================
# 1. CAU HINH
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

IMAGE_DIR = BASE_DIR / "dataset_v02" / "images"
RESULT_DIR = BASE_DIR / "dataset_v02" / "results"

MODEL_PATH = "yolo11n.pt"
CONFIDENCE = 0.35
PASSENGER_LIMIT = 2

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. KHOI TAO MO HINH YOLO
# ==========================================

model = YOLO(MODEL_PATH)


# ==========================================
# 3. HAM UOC LUONG NGUOI THUOC XE NAO
# ==========================================

def person_bike_score(person, bike):
    """
    Uoc luong kha nang nguoi dang ngoi tren xe.

    Luu y:
    Day la heuristic dua tren vi tri khung hinh,
    khong phai mo hinh chuyen biet dem nguoi tren xe.
    """

    px1, py1, px2, py2 = person
    bx1, by1, bx2, by2 = bike

    pw = max(1, px2 - px1)
    ph = max(1, py2 - py1)
    bw = max(1, bx2 - bx1)
    bh = max(1, by2 - by1)

    person_cx = (px1 + px2) / 2
    person_bottom = py2

    bike_cx = (bx1 + bx2) / 2

    # Kiem tra nguoi co nam gan chieu ngang cua xe khong
    horizontal_ratio = (person_cx - bx1) / bw

    if horizontal_ratio < -0.10 or horizontal_ratio > 1.10:
        return 0.0

    # Vung uoc luong nguoi ngoi tren xe.
    # Vung nay mo rong len tren khung xe vi nguoi cao hon xe.
    seat_top = by1 - 0.85 * bh
    seat_bottom = by1 + 0.45 * bh

    if person_bottom < seat_top or py1 > seat_bottom:
        return 0.0

    # Diem gan tam xe theo chieu ngang
    horizontal_score = max(
        0.0,
        1.0 - abs(person_cx - bike_cx) / (bw * 0.65)
    )

    # Diem gan vung ngoi theo chieu doc
    seat_center_y = by1 - 0.15 * bh
    vertical_distance = abs(person_bottom - seat_center_y)

    vertical_score = max(
        0.0,
        1.0 - vertical_distance / (bh + ph * 0.25)
    )

    # Ket hop hai diem
    score = (
        0.55 * horizontal_score
        + 0.45 * vertical_score
    )

    return score


def assign_people_to_motorcycles(people, motorcycles):
    """
    Gan moi nguoi vao toi da mot xe may.

    Nguoi khong dat nguong se khong duoc gan.
    """

    assigned = [[] for _ in motorcycles]

    for person_index, person in enumerate(people):
        best_bike = -1
        best_score = 0.0

        for bike_index, bike in enumerate(motorcycles):
            score = person_bike_score(person, bike)

            if score > best_score:
                best_score = score
                best_bike = bike_index

        if best_bike >= 0 and best_score >= 0.35:
            assigned[best_bike].append(person_index)

    return assigned


# ==========================================
# 4. KIEM TRA THU MUC ANH
# ==========================================

if not IMAGE_DIR.exists():
    print(f"Khong tim thay thu muc anh: {IMAGE_DIR}")
    raise SystemExit

images = sorted(
    path
    for path in IMAGE_DIR.iterdir()
    if path.is_file()
    and path.suffix.lower() in IMAGE_EXTENSIONS
)

if not images:
    print(f"Khong tim thay anh trong: {IMAGE_DIR}")
    raise SystemExit

print(f"Tim thay {len(images)} anh.")
print("Bat dau nhan dien...\n")


# ==========================================
# 5. NHAN DIEN TUNG ANH
# ==========================================

for image_path in images:

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Khong doc duoc anh: {image_path.name}")
        continue

    # Chi nhan dien person va motorcycle
    result = model.predict(
        source=image,
        classes=[0, 3],
        conf=CONFIDENCE,
        verbose=False
    )[0]

    people = []
    motorcycles = []

    if result.boxes is not None:
        for box in result.boxes:

            class_id = int(box.cls[0].item())

            coords = tuple(
                int(value)
                for value in box.xyxy[0].tolist()
            )

            if class_id == 0:
                people.append(coords)

            elif class_id == 3:
                motorcycles.append(coords)

    # Tao anh de ve ket qua
    output = image.copy()

    # Ve khung nguoi mau xanh duong
    for person in people:
        x1, y1, x2, y2 = person

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

    # Gan nguoi vao xe may
    assigned_people = assign_people_to_motorcycles(
        people,
        motorcycles
    )

    # ======================================
    # 6. VE KHUNG XE VA HIEN THI SO NGUOI
    # ======================================

    for bike_index, bike in enumerate(motorcycles):

        x1, y1, x2, y2 = bike
        count = len(assigned_people[bike_index])

        if count > PASSENGER_LIMIT:
            color = (0, 0, 255)
            label = f"NGHI VI PHAM: {count} nguoi"

        else:
            color = (0, 200, 0)
            label = f"Uoc tinh: {count} nguoi"

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        # Nen cho chu de de doc
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.60
        thickness = 2

        (text_w, text_h), baseline = cv2.getTextSize(
            label,
            font,
            font_scale,
            thickness
        )

        label_y = max(text_h + 8, y1 - 8)

        cv2.rectangle(
            output,
            (x1, label_y - text_h - 6),
            (min(output.shape[1] - 1, x1 + text_w + 8), label_y + baseline),
            color,
            -1
        )

        cv2.putText(
            output,
            label,
            (x1 + 4, label_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness
        )

    # ======================================
    # 7. HIEN THI TONG SO DOI TUONG
    # ======================================

    summary = (
        f"Detected people: {len(people)} | "
        f"Motorcycles: {len(motorcycles)}"
    )

    cv2.rectangle(
        output,
        (0, 0),
        (output.shape[1], 38),
        (30, 30, 30),
        -1
    )

    cv2.putText(
        output,
        summary,
        (10, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    # ======================================
    # 8. LUU ANH KET QUA
    # ======================================

    save_path = RESULT_DIR / f"{image_path.stem}_result.jpg"

    saved = cv2.imwrite(str(save_path), output)

    # ======================================
    # 9. IN KET QUA RA TERMINAL
    # ======================================

    print(f"Anh: {image_path.name}")
    print(f"Tong so nguoi phat hien: {len(people)}")
    print(f"Tong so xe may phat hien: {len(motorcycles)}")

    for index, assigned in enumerate(assigned_people, start=1):
        print(
            f"Xe {index}: uoc tinh {len(assigned)} nguoi"
        )

    if saved:
        print(f"Da luu anh ket qua: {save_path}")
    else:
        print(f"Loi khi luu anh: {save_path}")

    print("-" * 45)

    # ======================================
    # 10. HIEN THI ANH BANG OPENCV
    # ======================================

    window_title = (
        f"Nhan dien xe cho qua so nguoi - "
        f"{image_path.name}"
    )

    cv2.imshow(window_title, output)

    # Nhan phim bat ky de xem anh tiep theo
    # Nhan ESC de dung chuong trinh
    key = cv2.waitKey(0)

    cv2.destroyAllWindows()

    if key == 27:
        print("Da dung theo yeu cau.")
        break


print("\nDa ket thuc chuong trinh.")
