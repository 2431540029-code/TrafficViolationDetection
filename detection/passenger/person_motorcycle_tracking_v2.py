from ultralytics import YOLO
import cv2
from collections import defaultdict, deque, Counter


# =========================================================
# 1. CẤU HÌNH
# =========================================================

VIDEO_PATH = "videos/videotest.mp4"
MODEL_PATH = "yolo11n.pt"

# Độ tin cậy YOLO
CONFIDENCE = 0.40

# Số frame dùng để ổn định kết quả
HISTORY_LENGTH = 7

# Trong 7 frame, nếu số người nào xuất hiện
# ít nhất số lần này thì lấy làm kết quả ổn định
MIN_STABLE_COUNT = 4

# Người phải nằm trong vùng tương đối gần xe
HORIZONTAL_RATIO = 0.65

# Phạm vi phía trên xe có thể chứa người
TOP_RATIO = 1.2

# Phạm vi phía dưới xe
BOTTOM_RATIO = 0.5


# =========================================================
# 2. LOAD MODEL
# =========================================================

model = YOLO(MODEL_PATH)


# =========================================================
# 3. MỞ VIDEO
# =========================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Không thể mở video!")
    exit()


# =========================================================
# 4. LỊCH SỬ SỐ NGƯỜI CỦA TỪNG XE
# =========================================================

person_history = defaultdict(
    lambda: deque(maxlen=HISTORY_LENGTH)
)


# Xe đã được xác nhận vi phạm
reported_violations = set()


# =========================================================
# 5. HÀM TÌM XE PHÙ HỢP NHẤT CHO NGƯỜI
# =========================================================

def find_best_motorcycle(person, motorcycles):

    px1 = person["x1"]
    py1 = person["y1"]
    px2 = person["x2"]
    py2 = person["y2"]

    # Tâm ngang của người
    person_center_x = (px1 + px2) / 2

    # Điểm đáy của người
    person_bottom_y = py2

    best_motorcycle = None
    best_score = float("inf")

    for motorcycle in motorcycles:

        mx1 = motorcycle["x1"]
        my1 = motorcycle["y1"]
        mx2 = motorcycle["x2"]
        my2 = motorcycle["y2"]

        motorcycle_width = mx2 - mx1
        motorcycle_height = my2 - my1

        if motorcycle_width <= 0 or motorcycle_height <= 0:
            continue

        motorcycle_center_x = (mx1 + mx2) / 2

        # ---------------------------------------------
        # 1. Khoảng cách ngang
        # ---------------------------------------------

        horizontal_distance = abs(
            person_center_x - motorcycle_center_x
        )

        max_horizontal_distance = (
            motorcycle_width * HORIZONTAL_RATIO
        )

        if horizontal_distance > max_horizontal_distance:
            continue

        # ---------------------------------------------
        # 2. Người không được ở quá xa phía trên
        # ---------------------------------------------

        top_limit = (
            my1 -
            motorcycle_height * TOP_RATIO
        )

        if person_bottom_y < top_limit:
            continue

        # ---------------------------------------------
        # 3. Người không được ở quá xa phía dưới
        # ---------------------------------------------

        bottom_limit = (
            my2 +
            motorcycle_height * BOTTOM_RATIO
        )

        if person_bottom_y > bottom_limit:
            continue

        # ---------------------------------------------
        # 4. Tính điểm khoảng cách
        # ---------------------------------------------

        horizontal_score = (
            horizontal_distance /
            motorcycle_width
        )

        # Người càng gần tâm xe → score càng nhỏ
        score = horizontal_score

        if score < best_score:

            best_score = score
            best_motorcycle = motorcycle

    return best_motorcycle


# =========================================================
# 6. MAIN LOOP
# =========================================================

print("==========================================")
print(" TRAFFIC VIOLATION - V02")
print(" PHAT HIEN CHO QUA SO NGUOI")
print("==========================================")
print()
print("Q: Thoat")
print()


while True:

    ret, frame = cap.read()

    if not ret:

        print("\n✅ Video đã kết thúc!")
        break


    # =====================================================
    # 7. YOLO + BYTETRACK
    # =====================================================

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[0, 3],       # person + motorcycle
        conf=CONFIDENCE,
        verbose=False
    )

    result = results[0]


    motorcycles = []
    persons = []


    # =====================================================
    # 8. LẤY OBJECT
    # =====================================================

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


            # ---------------------------------------------
            # MOTORCYCLE
            # ---------------------------------------------

            if class_id == 3:

                motorcycles.append({

                    "id": track_id,

                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2

                })


            # ---------------------------------------------
            # PERSON
            # ---------------------------------------------

            elif class_id == 0:

                persons.append({

                    "id": track_id,

                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2

                })


    # =====================================================
    # 9. TẠO DANH SÁCH NGƯỜI TRÊN TỪNG XE
    # =====================================================

    motorcycle_people = {}

    for motorcycle in motorcycles:

        motorcycle_people[
            motorcycle["id"]
        ] = []


    # =====================================================
    # 10. GÁN NGƯỜI → XE
    # =====================================================

    for person in persons:

        best_motorcycle = find_best_motorcycle(
            person,
            motorcycles
        )

        if best_motorcycle is not None:

            motorcycle_id = best_motorcycle["id"]

            motorcycle_people[
                motorcycle_id
            ].append(
                person["id"]
            )


    # =====================================================
    # 11. CẬP NHẬT LỊCH SỬ
    # =====================================================

    for motorcycle in motorcycles:

        motorcycle_id = motorcycle["id"]

        current_count = len(
            motorcycle_people.get(
                motorcycle_id,
                []
            )
        )

        # Lưu số người của frame hiện tại
        person_history[
            motorcycle_id
        ].append(current_count)


    # =====================================================
    # 12. TÍNH SỐ NGƯỜI ỔN ĐỊNH
    # =====================================================

    stable_counts = {}

    for motorcycle in motorcycles:

        motorcycle_id = motorcycle["id"]

        history = person_history[
            motorcycle_id
        ]

        if len(history) == 0:
            continue


        counter = Counter(history)

        most_common = counter.most_common(1)[0]

        stable_count = most_common[0]
        stable_frequency = most_common[1]


        # Chỉ chấp nhận nếu kết quả đủ ổn định
        if stable_frequency >= MIN_STABLE_COUNT:

            stable_counts[
                motorcycle_id
            ] = stable_count

        else:

            stable_counts[
                motorcycle_id
            ] = history[-1]


    # =====================================================
    # 13. VẼ XE + KẾT QUẢ
    # =====================================================

    for motorcycle in motorcycles:

        motorcycle_id = motorcycle["id"]

        x1 = motorcycle["x1"]
        y1 = motorcycle["y1"]
        x2 = motorcycle["x2"]
        y2 = motorcycle["y2"]


        people_count = stable_counts.get(
            motorcycle_id,
            0
        )


        # =================================================
        # XE CHỞ QUÁ 2 NGƯỜI
        # =================================================

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


            # ---------------------------------------------
            # Chỉ thông báo một lần cho mỗi ID
            # ---------------------------------------------

            if motorcycle_id not in reported_violations:

                print(
                    f"🔴 XE ID {motorcycle_id} "
                    f"-> {people_count} nguoi "
                    f"-> VI PHAM"
                )

                reported_violations.add(
                    motorcycle_id
                )


        else:

            box_color = (255, 255, 0)


        # =================================================
        # VẼ BOX XE
        # =================================================

        cv2.rectangle(
            frame,

            (x1, y1),

            (x2, y2),

            box_color,

            2
        )


        # =================================================
        # HIỂN THỊ ID + SỐ NGƯỜI
        # =================================================

        text = (
            f"XE ID: {motorcycle_id}"
            f" | NGUOI: {people_count}"
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


    # =====================================================
    # 14. VẼ PERSON
    # =====================================================

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


    # =====================================================
    # 15. THÔNG TIN TRÊN MÀN HÌNH
    # =====================================================

    cv2.putText(
        frame,

        "V02 - PHAT HIEN CHO QUA SO NGUOI",

        (20, 35),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (255, 255, 255),

        2
    )


    cv2.putText(
        frame,

        "Q: Thoat",

        (20, 65),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (255, 255, 255),

        2
    )


    # =====================================================
    # 16. HIỂN THỊ
    # =====================================================

    cv2.imshow(
        "Traffic Violation - V02",
        frame
    )


    key = cv2.waitKey(1) & 0xFF


    if key == ord("q") or key == ord("Q"):

        print("\n🛑 Đã dừng chương trình.")
        break


# =========================================================
# 17. CLEAN UP
# =========================================================

cap.release()
cv2.destroyAllWindows()


print()
print("==========================================")
print("       CHƯƠNG TRÌNH KẾT THÚC")
print("==========================================")