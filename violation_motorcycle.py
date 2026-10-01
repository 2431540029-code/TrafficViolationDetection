from ultralytics import YOLO
import cv2
import os
import csv


# =========================================================
# 1. LOAD MODEL
# =========================================================

# Model YOLO11 nhận diện phương tiện
traffic_model = YOLO("yolo11n.pt")

# Model YOLO đã train để nhận diện mũ bảo hiểm
helmet_model = YOLO(
    "runs/detect/helmet_model-2/weights/best.pt"
)


# =========================================================
# 2. MỞ VIDEO
# =========================================================

cap = cv2.VideoCapture("videos/videotest.mp4")

if not cap.isOpened():
    print("❌ Không thể mở video!")
    exit()


# =========================================================
# 3. TẠO THƯ MỤC LƯU ẢNH VI PHẠM
# =========================================================

os.makedirs("violations", exist_ok=True)


# =========================================================
# 4. DANH SÁCH VI PHẠM
# =========================================================

violations = {}


# =========================================================
# 5. TRẠNG THÁI PAUSE
# =========================================================

paused = False

# Frame hiện tại
current_frame = None


# =========================================================
# 6. HÀM GHI VI PHẠM
# =========================================================

def add_violation(track_id, violation_type, video_time):

    # Nếu ID xe này đã được ghi nhận vi phạm
    # thì không ghi lại nhiều lần
    if track_id in violations:
        return

    violations[track_id] = {
        "vehicle_id": track_id,
        "plate": "CHUA NHAN DIEN",
        "color": "CHUA NHAN DIEN",
        "violation": violation_type,
        "time": video_time
    }

    print("\n======================================")
    print("🚨 PHÁT HIỆN VI PHẠM")
    print("======================================")
    print("Xe ID       :", track_id)
    print("Biển số     :", "CHUA NHAN DIEN")
    print("Màu xe      :", "CHUA NHAN DIEN")
    print("Vi phạm     :", violation_type)
    print("Thời gian   :", video_time)
    print("======================================")


# =========================================================
# 7. HÀM LƯU ẢNH
# =========================================================

def save_violation_image(frame, track_id):

    filename = f"violations/violation_vehicle_{track_id}.jpg"

    cv2.imwrite(filename, frame)

    print("📸 Đã lưu ảnh:", filename)


# =========================================================
# 8. CHẠY VIDEO
# =========================================================

while True:

    # -----------------------------------------------------
    # Nếu video KHÔNG pause
    # thì đọc frame tiếp theo
    # -----------------------------------------------------

    if not paused:

        ret, frame = cap.read()

        if not ret:
            print("\n✅ Video đã kết thúc!")
            break

        current_frame = frame.copy()

    else:

        # Nếu pause thì giữ nguyên frame hiện tại
        frame = current_frame.copy()


    # =====================================================
    # 9. TRACKING XE MÁY
    # =====================================================

    traffic_results = traffic_model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[3],       # COCO class 3 = motorcycle
        conf=0.4,
        verbose=False
    )

    traffic_result = traffic_results[0]


    # Danh sách xe máy đang xuất hiện
    motorcycles = []


    # =====================================================
    # 10. LẤY ID CỦA TỪNG XE
    # =====================================================

    if traffic_result.boxes.id is not None:

        track_ids = (
            traffic_result.boxes.id
            .int()
            .cpu()
            .tolist()
        )

        boxes = (
            traffic_result.boxes.xyxy
            .int()
            .cpu()
            .tolist()
        )


        for track_id, box in zip(track_ids, boxes):

            x1, y1, x2, y2 = box

            motorcycles.append(
                (track_id, x1, y1, x2, y2)
            )


            # Vẽ khung xe
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 0),
                2
            )


            # Hiển thị ID xe
            cv2.putText(
                frame,
                f"XE ID: {track_id}",
                (x1, max(y1 - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )


    # =====================================================
    # 11. NHẬN DIỆN MŨ BẢO HIỂM
    # =====================================================

    helmet_results = helmet_model(
        frame,
        conf=0.5,
        verbose=False
    )

    helmet_result = helmet_results[0]


    # =====================================================
    # 12. KIỂM TRA CÁC BOX MŨ
    # =====================================================

    for box in helmet_result.boxes:

        hx1, hy1, hx2, hy2 = map(
            int,
            box.xyxy[0]
        )

        class_id = int(box.cls[0])

        class_name = helmet_model.names[class_id]

        confidence = float(box.conf[0])


        # -------------------------------------------------
        # KHÔNG ĐỘI MŨ
        # -------------------------------------------------

        if class_name == "no_helmet":

            # Vẽ màu đỏ
            cv2.rectangle(
                frame,
                (hx1, hy1),
                (hx2, hy2),
                (0, 0, 255),
                3
            )


            cv2.putText(
                frame,
                "VI PHAM: KHONG DOI MU BAO HIEM",
                (hx1, max(hy1 - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )


            # -------------------------------------------------
            # TÌM XE GẦN VỚI NGƯỜI KHÔNG ĐỘI MŨ
            # -------------------------------------------------

            helmet_center_x = (hx1 + hx2) // 2
            helmet_center_y = (hy1 + hy2) // 2


            matched_vehicle = None


            for track_id, x1, y1, x2, y2 in motorcycles:

                # Mở rộng vùng xe lên phía trên
                # để bao phủ cả người lái
                expanded_y1 = max(
                    0,
                    y1 - int((y2 - y1) * 1.5)
                )


                if (
                    x1 <= helmet_center_x <= x2
                    and
                    expanded_y1 <= helmet_center_y <= y2
                ):

                    matched_vehicle = track_id
                    break


            # -------------------------------------------------
            # NẾU TÌM ĐƯỢC XE
            # -------------------------------------------------

            if matched_vehicle is not None:

                # Thời gian hiện tại trong video
                milliseconds = cap.get(
                    cv2.CAP_PROP_POS_MSEC
                )

                seconds = int(
                    milliseconds / 1000
                )

                minutes = seconds // 60
                remaining_seconds = seconds % 60

                video_time = (
                    f"{minutes:02d}:{remaining_seconds:02d}"
                )


                # Ghi nhận vi phạm
                add_violation(
                    matched_vehicle,
                    "Khong doi mu bao hiem",
                    video_time
                )


    # =====================================================
    # 13. HIỂN THỊ TRẠNG THÁI PAUSE
    # =====================================================

    if paused:

        cv2.putText(
            frame,
            "PAUSED - PRESS P TO RESUME",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )

    else:

        cv2.putText(
            frame,
            "P: Pause | S: Save | Q: Quit",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )


    # =====================================================
    # 14. HIỂN THỊ VIDEO
    # =====================================================

    cv2.imshow(
        "Traffic Violation Detection",
        frame
    )


    # =====================================================
    # 15. XỬ LÝ PHÍM
    # =====================================================

    if paused:

        # Khi pause, chờ người dùng nhấn phím
        key = cv2.waitKey(0) & 0xFF

    else:

        # Khi chạy bình thường
        key = cv2.waitKey(1) & 0xFF


    # -----------------------------------------------------
    # P = PAUSE / RESUME
    # -----------------------------------------------------

    if key == ord("p") or key == ord("P"):

        paused = not paused

        if paused:
            print("\n⏸ VIDEO ĐÃ PAUSE")

        else:
            print("\n▶ VIDEO TIẾP TỤC")


    # -----------------------------------------------------
    # S = LƯU ẢNH FRAME HIỆN TẠI
    # -----------------------------------------------------

    elif key == ord("s") or key == ord("S"):

        if current_frame is not None:

            filename = (
                "violations/"
                "current_frame.jpg"
            )

            cv2.imwrite(
                filename,
                frame
            )

            print(
                "\n📸 Đã lưu frame:",
                filename
            )


    # -----------------------------------------------------
    # Q = THOÁT
    # -----------------------------------------------------

    elif key == ord("q") or key == ord("Q"):

        print("\n🛑 Đã dừng chương trình.")
        break


# =========================================================
# 16. GIẢI PHÓNG VIDEO
# =========================================================

cap.release()
cv2.destroyAllWindows()


# =========================================================
# 17. GHI FILE CSV
# =========================================================

csv_file = "violations.csv"


with open(
    csv_file,
    "w",
    newline="",
    encoding="utf-8-sig"
) as file:

    fieldnames = [
        "vehicle_id",
        "plate",
        "color",
        "violation",
        "time"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for violation in violations.values():

        writer.writerow(violation)


# =========================================================
# 18. IN DANH SÁCH VI PHẠM
# =========================================================

print("\n\n")
print("==============================================")
print("          DANH SÁCH VI PHẠM")
print("==============================================")


if len(violations) == 0:

    print("Không phát hiện vi phạm.")


else:

    for violation in violations.values():

        print(
            f"Xe ID: {violation['vehicle_id']} | "
            f"Biển số: {violation['plate']} | "
            f"Màu: {violation['color']} | "
            f"Vi phạm: {violation['violation']} | "
            f"Thời gian: {violation['time']}"
        )


print("==============================================")
print(f"📄 File nhật ký: {csv_file}")
print("==============================================")