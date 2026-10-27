from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import easyocr
import cv2
import re
from collections import defaultdict, Counter


# =====================================================
# 1. LOAD MODEL XE MÁY
# =====================================================

traffic_model = YOLO("yolo11n.pt")

print("✅ Đã load model xe máy!")


# =====================================================
# 2. LOAD MODEL BIỂN SỐ
# =====================================================

model_path = hf_hub_download(
    repo_id="Koushim/yolov8-license-plate-detection",
    filename="best.pt"
)

plate_model = YOLO(model_path)

print("✅ Đã load model biển số!")


# =====================================================
# 3. LOAD EASY OCR
# =====================================================

print("⏳ Đang khởi động EasyOCR...")

reader = easyocr.Reader(
    ['en'],
    gpu=False
)

print("✅ EasyOCR đã sẵn sàng!")


# =====================================================
# 4. MỞ VIDEO
# =====================================================

cap = cv2.VideoCapture(
    "videos/videotest.mp4"
)

if not cap.isOpened():

    print("❌ Không thể mở video!")

    exit()


# =====================================================
# 5. CÁC BIẾN
# =====================================================

frame_count = 0

# Cứ 5 frame OCR một lần
OCR_INTERVAL = 3

# Lưu các kết quả OCR của từng xe
plate_history = defaultdict(list)

# Lưu biển số ổn định cuối cùng
stable_plates = {}


# =====================================================
# 6. HÀM TÌM XE GẦN BIỂN SỐ
# =====================================================

def find_matching_vehicle(
    plate_center_x,
    plate_center_y,
    motorcycles
):

    best_vehicle = None

    best_distance = 999999


    for (
        track_id,
        x1,
        y1,
        x2,
        y2
    ) in motorcycles:

        vehicle_width = x2 - x1
        vehicle_height = y2 - y1


        # Mở rộng nhẹ vùng xe
        expanded_x1 = max(
            0,
            x1 - int(vehicle_width * 0.15)
        )

        expanded_x2 = min(
            99999,
            x2 + int(vehicle_width * 0.15)
        )

        expanded_y1 = max(
            0,
            y1 - int(vehicle_height * 0.10)
        )

        expanded_y2 = (
            y2 + int(vehicle_height * 0.10)
        )


        # Biển số phải nằm trong vùng xe
        if not (
            expanded_x1 <= plate_center_x <= expanded_x2
            and
            expanded_y1 <= plate_center_y <= expanded_y2
        ):

            continue


        # Biển số nên nằm ở phần dưới xe
        relative_y = (
            plate_center_y - y1
        ) / max(vehicle_height, 1)


        if relative_y < 0.55:

            continue


        # Khoảng cách tới tâm xe
        vehicle_center_x = (
            x1 + x2
        ) // 2

        vehicle_center_y = (
            y1 + y2
        ) // 2


        distance = (
            (plate_center_x - vehicle_center_x) ** 2
            +
            (plate_center_y - vehicle_center_y) ** 2
        )


        if distance < best_distance:

            best_distance = distance

            best_vehicle = track_id


    return best_vehicle


# =====================================================
# 7. HÀM OCR
# =====================================================

def read_plate(plate_crop):

    if plate_crop is None:
        return "", 0

    if plate_crop.size == 0:
        return "", 0

    # ==========================================
    # PHÓNG TO BIỂN SỐ
    # ==========================================

    plate_big = cv2.resize(
        plate_crop,
        None,
        fx=5,
        fy=5,
        interpolation=cv2.INTER_CUBIC
    )

    # ==========================================
    # TẠO 3 PHIÊN BẢN ẢNH
    # ==========================================

    gray = cv2.cvtColor(
        plate_big,
        cv2.COLOR_BGR2GRAY
    )

    # Cách 1: ảnh grayscale
    image1 = gray

    # Cách 2: tăng tương phản
    image2 = cv2.equalizeHist(gray)

    # Cách 3: threshold
    _, image3 = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    images = [
        image1,
        image2,
        image3
    ]

    best_text = ""
    best_confidence = 0

    # ==========================================
    # OCR 3 LẦN
    # ==========================================

    for image in images:

        results = reader.readtext(
            image,
            detail=1,
            paragraph=False,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )

        current_text = ""
        current_confidence = 0

        for detection in results:

            detected_text = detection[1]

            detected_confidence = float(
                detection[2]
            )

            current_text += detected_text

            current_confidence = max(
                current_confidence,
                detected_confidence
            )

        # ======================================
        # LÀM SẠCH
        # ======================================

        current_text = re.sub(
            r'[^A-Z0-9]',
            '',
            current_text.upper()
        )

        # ======================================
        # CHỌN KẾT QUẢ TỐT NHẤT
        # ======================================

        if (
            len(current_text) >= 4
            and current_confidence > best_confidence
        ):

            best_text = current_text

            best_confidence = (
                current_confidence
            )

    return best_text, best_confidence


    # ---------------------------------------------
    # Phóng to
    # ---------------------------------------------

    plate_big = cv2.resize(
        plate_crop,
        None,
        fx=4,
        fy=4,
        interpolation=cv2.INTER_CUBIC
    )


    # ---------------------------------------------
    # Grayscale
    # ---------------------------------------------

    gray = cv2.cvtColor(
        plate_big,
        cv2.COLOR_BGR2GRAY
    )


    # ---------------------------------------------
    # Làm mịn nhẹ
    # ---------------------------------------------

    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )


    # ---------------------------------------------
    # Tăng tương phản
    # ---------------------------------------------

    gray = cv2.equalizeHist(
        gray
    )


    # ---------------------------------------------
    # OCR
    # ---------------------------------------------

    results = reader.readtext(
        gray,
        detail=1,
        paragraph=False,
        allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    )


    text = ""

    best_confidence = 0


    for detection in results:

        detected_text = detection[1]

        detected_confidence = float(
            detection[2]
        )


        text += detected_text

        best_confidence = max(
            best_confidence,
            detected_confidence
        )


    # ---------------------------------------------
    # Làm sạch
    # ---------------------------------------------

    text = re.sub(
        r'[^A-Z0-9]',
        '',
        text.upper()
    )


    return text, best_confidence


# =====================================================
# 8. VÒNG LẶP VIDEO
# =====================================================

while True:

    ret, frame = cap.read()


    if not ret:

        print("\n✅ Video đã kết thúc!")

        break


    frame_count += 1


    # =================================================
    # 9. TRACK XE MÁY
    # =================================================

    traffic_results = traffic_model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[3],
        conf=0.4,
        imgsz=640,
        verbose=False
    )


    traffic_result = traffic_results[0]

    motorcycles = []


    if traffic_result.boxes.id is not None:

        track_ids = (
            traffic_result.boxes.id
            .int()
            .cpu()
            .tolist()
        )


        vehicle_boxes = (
            traffic_result.boxes.xyxy
            .int()
            .cpu()
            .tolist()
        )


        for track_id, box in zip(
            track_ids,
            vehicle_boxes
        ):

            x1, y1, x2, y2 = box


            motorcycles.append(
                (
                    track_id,
                    x1,
                    y1,
                    x2,
                    y2
                )
            )


            # Vẽ xe
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 0),
                2
            )


            cv2.putText(
                frame,
                f"XE ID: {track_id}",
                (
                    x1,
                    max(y1 - 10, 30)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )


    # =================================================
    # 10. DETECT BIỂN SỐ
    # =================================================

    plate_results = plate_model(
        frame,
        conf=0.35,
        iou=0.45,
        imgsz=640,
        verbose=False
    )


    plate_result = plate_results[0]


    # =================================================
    # 11. CHỈ OCR MỖI 5 FRAME
    # =================================================

    run_ocr = (
        frame_count % OCR_INTERVAL == 0
    )


    for box in plate_result.boxes:

        px1, py1, px2, py2 = map(
            int,
            box.xyxy[0]
        )


        detection_confidence = float(
            box.conf[0]
        )


        plate_width = px2 - px1

        plate_height = py2 - py1


        if (
            plate_width <= 0
            or plate_height <= 0
        ):

            continue


        # =================================================
        # 12. LỌC HÌNH DẠNG BIỂN SỐ
        # =================================================

        ratio = (
            plate_width /
            plate_height
        )


        if ratio < 1.2 or ratio > 6:

            continue


        # Không lấy vùng quá nhỏ
        if (
            plate_width < 20
            or plate_height < 8
        ):

            continue


        # =================================================
        # 13. TÂM BIỂN SỐ
        # =================================================

        plate_center_x = (
            px1 + px2
        ) // 2

        plate_center_y = (
            py1 + py2
        ) // 2


        # =================================================
        # 14. TÌM XE TƯƠNG ỨNG
        # =================================================

        matched_vehicle = find_matching_vehicle(
            plate_center_x,
            plate_center_y,
            motorcycles
        )


        if matched_vehicle is None:

            continue


        # =================================================
        # 15. OCR
        # =================================================

        if run_ocr:

            plate_crop = frame[
                py1:py2,
                px1:px2
            ]


            text, ocr_confidence = read_plate(
                plate_crop
            )


            # =============================================
            # 16. CHỈ GIỮ OCR ĐỦ TIN CẬY
            # =============================================

            # Các từ thường xuất hiện trên logo/thân xe
            invalid_words = {
                "HONDA",
                "YAMAHA",
                "SUZUKI",
                "PIAGGIO",
                "SYM",
                "VINFAST",
                "VESPA"
            }

            # Kiểm tra kết quả OCR
            if (
                    len(text) >= 4
                    and
                    sum(char.isdigit() for char in text) >= 2
                    and
                    ocr_confidence >= 0.15
                    and
                    text not in invalid_words
            ):
                plate_history[
                    matched_vehicle
                ].append(text)


                # Chỉ giữ 15 kết quả gần nhất
                plate_history[
                    matched_vehicle
                ] = plate_history[
                    matched_vehicle
                ][-15:]


                # =========================================
                # 17. TÌM KẾT QUẢ XUẤT HIỆN NHIỀU NHẤT
                # =========================================

                counter = Counter(
                    plate_history[
                        matched_vehicle
                    ]
                )


                most_common = counter.most_common(
                    1
                )


                if most_common:

                    stable_plate = (
                        most_common[0][0]
                    )

                    stable_count = (
                        most_common[0][1]
                    )


                    stable_plates[
                        matched_vehicle
                    ] = stable_plate


                    print(
                        f"XE ID {matched_vehicle} "
                        f"-> BIEN SO: "
                        f"{stable_plate} "
                        f"({stable_count} lan)"
                    )


        # =================================================
        # 18. VẼ KHUNG BIỂN SỐ
        # =================================================

        cv2.rectangle(
            frame,
            (px1, py1),
            (px2, py2),
            (0, 255, 0),
            2
        )


        # =================================================
        # 19. HIỂN THỊ BIỂN SỐ ỔN ĐỊNH
        # =================================================

        if matched_vehicle in stable_plates:

            plate_text = stable_plates[
                matched_vehicle
            ]


            cv2.putText(
                frame,
                f"BIEN SO: {plate_text}",
                (
                    px1,
                    max(py1 - 10, 30)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 255, 0),
                2
            )



        else:
            cv2.putText(
                frame,
                "DANG DOC BIEN SO...",
                (
                    px1,
                    max(py1 - 10, 30)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )


    # =================================================
    # 20. THÔNG TIN TRÊN MÀN HÌNH
    # =================================================

    cv2.putText(
        frame,
        f"FRAME: {frame_count}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"XE MAY: {len(motorcycles)}",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        "Q: Quit",
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # =================================================
    # 21. HIỂN THỊ
    # =================================================

    cv2.imshow(
        "Traffic - License Plate OCR",
        frame
    )


    key = cv2.waitKey(1) & 0xFF


    if key == ord("q") or key == ord("Q"):

        break


# =====================================================
# 22. KẾT THÚC
# =====================================================

cap.release()

cv2.destroyAllWindows()

print("\n======================================")
print("🛑 ĐÃ DỪNG CHƯƠNG TRÌNH")
print("======================================")

print("\nBIỂN SỐ ĐÃ NHẬN DIỆN:")

for vehicle_id, plate in stable_plates.items():

    print(
        f"XE ID {vehicle_id}: {plate}"
    )