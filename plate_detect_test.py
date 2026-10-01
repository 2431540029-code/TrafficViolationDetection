from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import cv2


# ==========================================
# 1. LOAD MODEL XE MÁY
# ==========================================

traffic_model = YOLO("yolo11n.pt")


# ==========================================
# 2. LOAD MODEL MŨ BẢO HIỂM
# ==========================================

helmet_model = YOLO(
    "runs/detect/helmet_model-2/weights/best.pt"
)


# ==========================================
# 3. LOAD MODEL BIỂN SỐ
# ==========================================

model_path = hf_hub_download(
    repo_id="Koushim/yolov8-license-plate-detection",
    filename="best.pt"
)

plate_model = YOLO(model_path)


print("======================================")
print("✅ Đã load model xe máy!")
print("✅ Đã load model mũ bảo hiểm!")
print("✅ Đã load model biển số!")
print("======================================")


# ==========================================
# 4. MỞ VIDEO
# ==========================================

cap = cv2.VideoCapture("videos/videotest.mp4")

if not cap.isOpened():
    print("❌ Không thể mở video!")
    exit()


# ==========================================
# 5. VÒNG LẶP VIDEO
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("✅ Video đã kết thúc!")
        break


    # Giữ frame gốc
    original_frame = frame.copy()


    # ======================================
    # 6. PHÁT HIỆN XE MÁY + TRACKING
    # ======================================

    traffic_results = traffic_model.track(
        original_frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[3],
        conf=0.4,
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
                (track_id, x1, y1, x2, y2)
            )


            # -----------------------------
            # VẼ KHUNG XE
            # -----------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 0),
                2
            )


            # -----------------------------
            # HIỂN THỊ XE ID
            # -----------------------------

            cv2.putText(
                frame,
                f"XE ID: {track_id}",
                (x1, max(y1 - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )


    # ======================================
    # 7. NHẬN DIỆN MŨ BẢO HIỂM
    # ======================================

    helmet_results = helmet_model(
        original_frame,
        conf=0.65,
        imgsz=640,
        verbose=False
    )

    helmet_result = helmet_results[0]


    # Số người không đội mũ hợp lệ
    no_helmet_count = 0


    # Danh sách vùng có mũ
    helmet_boxes = []


    # ======================================
    # 7.1 LƯU CÁC VÙNG ĐƯỢC NHẬN LÀ HELMET
    # ======================================

    for box in helmet_result.boxes:

        hx1, hy1, hx2, hy2 = map(
            int,
            box.xyxy[0]
        )


        class_id = int(box.cls[0])

        class_name = helmet_model.names[class_id]

        confidence = float(box.conf[0])


        # Chỉ lưu HELMET có độ tin cậy >= 0.65
        if (
            class_name == "helmet"
            and confidence >= 0.65
        ):

            helmet_boxes.append(
                (
                    hx1,
                    hy1,
                    hx2,
                    hy2,
                    confidence
                )
            )


    # ======================================
    # 7.2 XỬ LÝ NO_HELMET
    # ======================================

    for box in helmet_result.boxes:

        hx1, hy1, hx2, hy2 = map(
            int,
            box.xyxy[0]
        )


        class_id = int(box.cls[0])

        class_name = helmet_model.names[class_id]

        confidence = float(box.conf[0])


        # ----------------------------------
        # CHỈ NHẬN DETECTION >= 0.65
        # ----------------------------------

        if confidence < 0.65:
            continue


        # ----------------------------------
        # CHỈ XỬ LÝ NO_HELMET
        # ----------------------------------

        if class_name != "no_helmet":
            continue


        # ==================================
        # TÂM VÙNG ĐẦU
        # ==================================

        center_x = (hx1 + hx2) // 2
        center_y = (hy1 + hy2) // 2


        # ==================================
        # KIỂM TRA CÓ HELMET TRÙNG VÙNG KHÔNG
        # ==================================

        has_helmet = False


        for (
            helmet_x1,
            helmet_y1,
            helmet_x2,
            helmet_y2,
            helmet_conf
        ) in helmet_boxes:


            # Nếu tâm vùng no_helmet nằm
            # bên trong vùng helmet
            if (
                helmet_x1 <= center_x <= helmet_x2
                and
                helmet_y1 <= center_y <= helmet_y2
            ):

                has_helmet = True
                break


        # ----------------------------------
        # NẾU CÓ MŨ → BỎ QUA
        # ----------------------------------

        if has_helmet:
            continue


        # ==================================
        # TÌM XE TƯƠNG ỨNG
        # ==================================

        matched_vehicle = None


        for (
            track_id,
            x1,
            y1,
            x2,
            y2
        ) in motorcycles:


            vehicle_width = x2 - x1

            vehicle_height = y2 - y1


            # --------------------------------
            # MỞ RỘNG VÙNG PHÍA TRÊN XE
            # --------------------------------

            expanded_y1 = max(
                0,
                y1 - int(vehicle_height * 0.45)
            )


            # --------------------------------
            # KIỂM TRA ĐẦU CÓ THUỘC XE KHÔNG
            # --------------------------------

            if (
                x1 <= center_x <= x2
                and
                expanded_y1 <= center_y <= y2
            ):

                matched_vehicle = track_id
                break


        # ==================================
        # NẾU KHÔNG GẮN ĐƯỢC VỚI XE
        # → KHÔNG BÁO VI PHẠM
        # ==================================

        if matched_vehicle is None:
            continue


        # ==================================
        # XÁC NHẬN VI PHẠM
        # ==================================

        no_helmet_count += 1


        # ==================================
        # VẼ KHUNG ĐỎ NO_HELMET
        # ==================================

        cv2.rectangle(
            frame,
            (hx1, hy1),
            (hx2, hy2),
            (0, 0, 255),
            3
        )


        # ==================================
        # HIỂN THỊ VI PHẠM
        # ==================================

        cv2.putText(
            frame,
            f"VI PHAM: KHONG DOI MU ({confidence:.2f})",
            (hx1, max(hy1 - 10, 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2
        )


        # ==================================
        # HIỂN THỊ XE ID
        # ==================================

        cv2.putText(
            frame,
            f"XE ID: {matched_vehicle}",
            (hx1, hy2 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 255),
            2
        )


    # ======================================
    # 8. PHÁT HIỆN BIỂN SỐ
    # ======================================

    plate_results = plate_model(
        original_frame,
        conf=0.35,
        iou=0.45,
        imgsz=1280,
        verbose=False
    )

    plate_result = plate_results[0]

    valid_plates = 0


    # ======================================
    # 9. XỬ LÝ BIỂN SỐ
    # ======================================

    for box in plate_result.boxes:

        px1, py1, px2, py2 = map(
            int,
            box.xyxy[0]
        )


        confidence = float(box.conf[0])


        # ==================================
        # KÍCH THƯỚC BIỂN SỐ
        # ==================================

        plate_width = px2 - px1

        plate_height = py2 - py1


        if (
            plate_width <= 0
            or plate_height <= 0
        ):
            continue


        # ==================================
        # TỶ LỆ BIỂN SỐ
        # ==================================

        ratio = (
            plate_width /
            plate_height
        )


        # ----------------------------------
        # LỌC HÌNH DẠNG
        # ----------------------------------

        if ratio < 1.2 or ratio > 6.0:
            continue


        # ----------------------------------
        # LỌC BIỂN SỐ QUÁ NHỎ
        # ----------------------------------

        if (
            plate_width < 20
            or plate_height < 8
        ):
            continue


        # ==================================
        # TÂM BIỂN SỐ
        # ==================================

        plate_center_x = (
            px1 + px2
        ) // 2

        plate_center_y = (
            py1 + py2
        ) // 2


        matched_vehicle = None


        # ==================================
        # TÌM XE CHỨA BIỂN SỐ
        # ==================================

        for (
            track_id,
            x1,
            y1,
            x2,
            y2
        ) in motorcycles:


            vehicle_width = x2 - x1

            vehicle_height = y2 - y1


            # --------------------------------
            # MỞ RỘNG NHẸ KHUNG XE
            # --------------------------------

            expanded_x1 = max(
                0,
                x1 - int(vehicle_width * 0.1)
            )


            expanded_x2 = min(
                frame.shape[1],
                x2 + int(vehicle_width * 0.1)
            )


            expanded_y1 = max(
                0,
                y1 - int(vehicle_height * 0.1)
            )


            expanded_y2 = min(
                frame.shape[0],
                y2 + int(vehicle_height * 0.1)
            )


            # ==================================
            # BIỂN SỐ PHẢI NẰM TRONG XE
            # ==================================

            if (
                expanded_x1
                <= plate_center_x
                <= expanded_x2
                and
                expanded_y1
                <= plate_center_y
                <= expanded_y2
            ):


                # ------------------------------
                # BIỂN SỐ PHẢI Ở PHÍA DƯỚI XE
                # ------------------------------

                relative_y = (
                    plate_center_y - y1
                ) / max(vehicle_height, 1)


                if relative_y >= 0.35:

                    matched_vehicle = track_id
                    break


        # ==================================
        # KHÔNG THUỘC XE NÀO
        # ==================================

        if matched_vehicle is None:
            continue


        # ==================================
        # BIỂN SỐ HỢP LỆ
        # ==================================

        valid_plates += 1


        # ==================================
        # VẼ KHUNG BIỂN SỐ
        # ==================================

        cv2.rectangle(
            frame,
            (px1, py1),
            (px2, py2),
            (0, 255, 0),
            3
        )


        # ==================================
        # HIỂN THỊ BIỂN SỐ
        # ==================================

        cv2.putText(
            frame,
            f"BIEN SO - XE ID {matched_vehicle} {confidence:.2f}",
            (px1, max(py1 - 10, 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )


    # ======================================
    # 10. HIỂN THỊ THÔNG TIN
    # ======================================

    cv2.putText(
        frame,
        f"XE MAY: {len(motorcycles)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    cv2.putText(
        frame,
        f"KHONG DOI MU: {no_helmet_count}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2
    )


    cv2.putText(
        frame,
        f"BIEN SO HOP LE: {valid_plates}",
        (20, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


    cv2.putText(
        frame,
        "Q: Quit",
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ======================================
    # 11. HIỂN THỊ VIDEO
    # ======================================

    cv2.imshow(
        "Traffic Violation + License Plate",
        frame
    )


    # ======================================
    # 12. NHẤN Q ĐỂ THOÁT
    # ======================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q") or key == ord("Q"):
        break


# ==========================================
# 13. KẾT THÚC
# ==========================================

cap.release()

cv2.destroyAllWindows()

print("🛑 Đã dừng chương trình.")