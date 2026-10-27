from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import easyocr
import cv2
import re


# ==========================================
# 1. LOAD MODEL BIỂN SỐ
# ==========================================

model_path = hf_hub_download(
    repo_id="Koushim/yolov8-license-plate-detection",
    filename="best.pt"
)

plate_model = YOLO(model_path)

print("✅ Đã load model biển số!")


# ==========================================
# 2. LOAD EASY OCR
# ==========================================

print("⏳ Đang khởi động EasyOCR...")

reader = easyocr.Reader(
    ['en'],
    gpu=False
)

print("✅ EasyOCR đã sẵn sàng!")


# ==========================================
# 3. MỞ VIDEO
# ==========================================

cap = cv2.VideoCapture(
    "videos/videotest.mp4"
)

if not cap.isOpened():
    print("❌ Không thể mở video!")
    exit()


# ==========================================
# 4. BIẾN ĐIỀU KHIỂN
# ==========================================

frame_count = 0

OCR_INTERVAL = 5

last_plate_text = ""
last_ocr_confidence = 0


# ==========================================
# 5. VIDEO
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("✅ Video đã kết thúc!")
        break

    frame_count += 1


    # ======================================
    # 6. YOLO PHÁT HIỆN BIỂN SỐ
    # ======================================

    results = plate_model(
        frame,
        conf=0.35,
        iou=0.45,
        imgsz=640,
        verbose=False
    )

    result = results[0]


    # ======================================
    # 7. CHỈ OCR MỖI 5 FRAME
    # ======================================

    run_ocr = (
        frame_count % OCR_INTERVAL == 0
    )


    for box in result.boxes:

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0]
        )

        confidence = float(
            box.conf[0]
        )


        # ==================================
        # KIỂM TRA KÍCH THƯỚC
        # ==================================

        width = x2 - x1
        height = y2 - y1

        if width <= 0 or height <= 0:
            continue


        # ==================================
        # KIỂM TRA TỶ LỆ BIỂN SỐ
        # ==================================

        ratio = width / height

        if ratio < 1.2 or ratio > 6:
            continue


        # ==================================
        # CROP BIỂN SỐ
        # ==================================

        plate_crop = frame[
            y1:y2,
            x1:x2
        ]


        if plate_crop.size == 0:
            continue


        # ==================================
        # OCR
        # ==================================

        if run_ocr:

            # ------------------------------
            # PHÓNG TO
            # ------------------------------

            plate_big = cv2.resize(
                plate_crop,
                None,
                fx=4,
                fy=4,
                interpolation=cv2.INTER_CUBIC
            )


            # ------------------------------
            # GRAYSCALE
            # ------------------------------

            gray = cv2.cvtColor(
                plate_big,
                cv2.COLOR_BGR2GRAY
            )


            # ------------------------------
            # LÀM SẮC NÉT
            # ------------------------------

            sharpen_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (3, 3)
            )

            gray = cv2.morphologyEx(
                gray,
                cv2.MORPH_CLOSE,
                sharpen_kernel
            )


            # ------------------------------
            # TĂNG TƯƠNG PHẢN
            # ------------------------------

            gray = cv2.equalizeHist(
                gray
            )


            # ------------------------------
            # EASY OCR
            # ------------------------------

            ocr_results = reader.readtext(
                gray,
                detail=1,
                paragraph=False,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
            )


            # ------------------------------
            # GHÉP KẾT QUẢ
            # ------------------------------

            text = ""
            best_confidence = 0

            for detection in ocr_results:

                detected_text = detection[1]

                detected_confidence = float(
                    detection[2]
                )

                text += detected_text

                best_confidence = max(
                    best_confidence,
                    detected_confidence
                )


            # ------------------------------
            # LỌC KÝ TỰ
            # ------------------------------

            text = re.sub(
                r'[^A-Z0-9]',
                '',
                text.upper()
            )


            # ------------------------------
            # CHỈ NHẬN KẾT QUẢ CÓ Ý NGHĨA
            # ------------------------------

            if (
                    len(text) >= 4
                    and best_confidence >= 0.30
            ):
                last_plate_text = text

                last_ocr_confidence = (
                    best_confidence
                )


        # ==================================
        # VẼ KHUNG BIỂN SỐ
        # ==================================

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )


        # ==================================
        # HIỂN THỊ BIỂN SỐ
        # ==================================

        if last_plate_text:

            display_text = (
                f"BIEN SO: {last_plate_text}"
            )

            cv2.putText(
                frame,
                display_text,
                (x1, max(y1 - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"OCR: {last_ocr_confidence:.2f}",
                (x1, y2 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

        else:

            cv2.putText(
                frame,
                "BIEN SO: CHUA DOC DUOC",
                (x1, max(y1 - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )


    # ======================================
    # THÔNG TIN
    # ======================================

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
        "Q: Quit",
        (20, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )


    # ======================================
    # HIỂN THỊ
    # ======================================

    cv2.imshow(
        "License Plate OCR",
        frame
    )


    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == ord("Q"):
        break


# ==========================================
# KẾT THÚC
# ==========================================

cap.release()

cv2.destroyAllWindows()

print("🛑 Đã dừng chương trình.")