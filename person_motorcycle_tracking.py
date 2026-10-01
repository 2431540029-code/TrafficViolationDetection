from ultralytics import YOLO
import cv2
import math

# ==========================================
# 1. LOAD MODEL
# ==========================================

model = YOLO("yolo11n.pt")

# ==========================================
# 2. OPEN VIDEO
# ==========================================

cap = cv2.VideoCapture("videos/videotest.mp4")

if not cap.isOpened():
    print("❌ Không thể mở video!")
    exit()

print("======================================")
print("   PHAT HIEN CHO QUA SO NGUOI")
print("======================================")
print("Q: Thoat")
print()

# ==========================================
# 3. MAIN LOOP
# ==========================================

while True:

    ret, frame = cap.read()

    if not ret:
        print("✅ Video đã kết thúc!")
        break

    # ==========================================
    # 4. YOLO + BYTETRACK
    # ==========================================

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0, 3],       # 0 = person, 3 = motorcycle
        conf=0.4,
        verbose=False
    )

    result = results[0]

    motorcycles = []
    persons = []

    # ==========================================
    # 5. LẤY PERSON + MOTORCYCLE
    # ==========================================

    if result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().tolist()
        classes = result.boxes.cls.int().cpu().tolist()
        track_ids = result.boxes.id.int().cpu().tolist()

        for box, class_id, track_id in zip(
            boxes,
            classes,
            track_ids
        ):

            x1, y1, x2, y2 = map(int, box)

            # -----------------------------
            # MOTORCYCLE
            # -----------------------------

            if class_id == 3:

                motorcycles.append({
                    "id": track_id,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                })

            # -----------------------------
            # PERSON
            # -----------------------------

            elif class_id == 0:

                persons.append({
                    "id": track_id,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2
                })

    # ==========================================
    # 6. KHỞI TẠO DANH SÁCH NGƯỜI TRÊN XE
    # ==========================================

    motorcycle_people = {}

    for motorcycle in motorcycles:

        motorcycle_people[motorcycle["id"]] = []

    # ==========================================
    # 7. GÁN TỪNG NGƯỜI VÀO XE GẦN NHẤT
    # ==========================================

    for person in persons:

        px1 = person["x1"]
        py1 = person["y1"]
        px2 = person["x2"]
        py2 = person["y2"]

        person_id = person["id"]

        # Tâm ngang của người
        person_center_x = (px1 + px2) / 2

        # Điểm chân / đáy người
        person_bottom_y = py2

        best_motorcycle = None
        best_distance = float("inf")

        # ------------------------------------------
        # TÌM XE PHÙ HỢP NHẤT
        # ------------------------------------------

        for motorcycle in motorcycles:

            mx1 = motorcycle["x1"]
            my1 = motorcycle["y1"]
            mx2 = motorcycle["x2"]
            my2 = motorcycle["y2"]

            motorcycle_id = motorcycle["id"]

            motorcycle_center_x = (mx1 + mx2) / 2
            motorcycle_width = mx2 - mx1
            motorcycle_height = my2 - my1

            # Tránh lỗi chia cho 0
            if motorcycle_width <= 0:
                continue

            # Khoảng cách ngang giữa người và xe
            horizontal_distance = abs(
                person_center_x - motorcycle_center_x
            )

            # ------------------------------------------
            # ĐIỀU KIỆN 1:
            # Người phải nằm tương đối gần tâm ngang của xe
            # ------------------------------------------

            if horizontal_distance > motorcycle_width * 0.65:
                continue

            # ------------------------------------------
            # ĐIỀU KIỆN 2:
            # Người không được nằm quá xa phía trên xe
            # ------------------------------------------

            top_limit = my1 - motorcycle_height * 1.2

            if person_bottom_y < top_limit:
                continue

            # ------------------------------------------
            # ĐIỀU KIỆN 3:
            # Chân người không được quá xa phía dưới xe
            # ------------------------------------------

            bottom_limit = my2 + motorcycle_height * 0.5

            if person_bottom_y > bottom_limit:
                continue

            # ------------------------------------------
            # TÍNH KHOẢNG CÁCH
            # ------------------------------------------

            normalized_distance = (
                horizontal_distance / motorcycle_width
            )

            if normalized_distance < best_distance:

                best_distance = normalized_distance
                best_motorcycle = motorcycle_id

        # ------------------------------------------
        # GÁN NGƯỜI VÀO XE
        # ------------------------------------------

        if best_motorcycle is not None:

            motorcycle_people[
                best_motorcycle
            ].append(person_id)

    # ==========================================
    # 8. HIỂN THỊ MOTORCYCLE
    # ==========================================

    for motorcycle in motorcycles:

        motorcycle_id = motorcycle["id"]

        x1 = motorcycle["x1"]
        y1 = motorcycle["y1"]
        x2 = motorcycle["x2"]
        y2 = motorcycle["y2"]

        people_count = len(
            motorcycle_people.get(
                motorcycle_id,
                []
            )
        )

        # ======================================
        # KIỂM TRA VI PHẠM
        # ======================================

        if people_count > 2:

            box_color = (0, 0, 255)

            violation_text = (
                f"VI PHAM: CHO {people_count} NGUOI"
            )

            cv2.putText(
                frame,
                violation_text,
                (
                    x1,
                    max(y1 - 35, 30)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 255),
                2
            )

            print(
                f"XE ID {motorcycle_id} "
                f"-> {people_count} nguoi "
                f"-> VI PHAM"
            )

        else:

            box_color = (255, 255, 0)

        # ======================================
        # VẼ BOX XE
        # ======================================

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            box_color,
            2
        )

        # ======================================
        # HIỂN THỊ ID + SỐ NGƯỜI
        # ======================================

        text = (
            f"XE ID: {motorcycle_id} "
            f"| NGUOI: {people_count}"
        )

        cv2.putText(
            frame,
            text,
            (
                x1,
                min(
                    y2 + 25,
                    frame.shape[0] - 10
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            box_color,
            2
        )

    # ==========================================
    # 9. VẼ PERSON
    # ==========================================

    for person in persons:

        px1 = person["x1"]
        py1 = person["y1"]
        px2 = person["x2"]
        py2 = person["y2"]

        person_id = person["id"]

        cv2.rectangle(
            frame,
            (px1, py1),
            (px2, py2),
            (255, 0, 0),
            2
        )

        cv2.putText(
            frame,
            f"Person {person_id}",
            (
                px1,
                max(py1 - 5, 20)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2
        )

    # ==========================================
    # 10. HƯỚNG DẪN
    # ==========================================

    cv2.putText(
        frame,
        "Q: Thoat",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # ==========================================
    # 11. SHOW
    # ==========================================

    cv2.imshow(
        "Phat Hien Cho Qua So Nguoi",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == ord("Q"):
        print("\n🛑 Đã dừng chương trình.")
        break


# ==========================================
# 12. CLEAN UP
# ==========================================

cap.release()
cv2.destroyAllWindows()

print()
print("======================================")
print("       CHUONG TRINH KET THUC")
print("======================================")