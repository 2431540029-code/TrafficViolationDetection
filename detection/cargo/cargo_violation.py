from ultralytics import YOLO
import cv2
from collections import defaultdict, deque, Counter


# =========================================================
# V03 - PHAT HIEN CHO HANG CONG KENH / QUA KHO
# =========================================================

VIDEO_PATH = "videos/videotest.mp4"
MODEL_PATH = "yolo11n.pt"

# ---------------------------------------------------------
# YOLO
# ---------------------------------------------------------

CONFIDENCE = 0.40

# ---------------------------------------------------------
# Theo dõi qua nhiều frame
# ---------------------------------------------------------

HISTORY_LENGTH = 8
MIN_STABLE_COUNT = 5

# ---------------------------------------------------------
# Ngưỡng kích thước hàng hóa
# ---------------------------------------------------------

# Hàng phải có chiều rộng ít nhất bằng bao nhiêu
# phần chiều rộng xe
MIN_CARGO_WIDTH_RATIO = 0.45

# Hàng phải có diện tích tối thiểu so với xe
MIN_CARGO_AREA_RATIO = 0.08

# ---------------------------------------------------------
# Khoảng cách hàng - xe
# ---------------------------------------------------------

MAX_HORIZONTAL_DISTANCE_RATIO = 1.0
MAX_VERTICAL_DISTANCE_RATIO = 1.5


# =========================================================
# CÁC CLASS COCO CÓ THỂ COI LÀ HÀNG HÓA
# =========================================================

CARGO_CLASSES = {
    24: "backpack",
    26: "handbag",
    28: "suitcase"
}


# =========================================================
# LOAD MODEL
# =========================================================

model = YOLO(MODEL_PATH)


# =========================================================
# MỞ VIDEO
# =========================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("❌ Không thể mở video!")
    exit()


# =========================================================
# LỊCH SỬ HÀNG HÓA CỦA TỪNG XE
# =========================================================

cargo_history = defaultdict(
    lambda: deque(maxlen=HISTORY_LENGTH)
)


# =========================================================
# ĐÃ BÁO VI PHẠM
# =========================================================

reported_violations = set()


# =========================================================
# HÀM TÍNH IOU
# =========================================================

def calculate_iou(box1, box2):

    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])

    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection_width = max(0, x2 - x1)
    intersection_height = max(0, y2 - y1)

    intersection_area = (
        intersection_width *
        intersection_height
    )

    area1 = (
        max(0, box1[2] - box1[0]) *
        max(0, box1[3] - box1[1])
    )

    area2 = (
        max(0, box2[2] - box2[0]) *
        max(0, box2[3] - box2[1])
    )

    union_area = area1 + area2 - intersection_area

    if union_area <= 0:
        return 0

    return intersection_area / union_area


# =========================================================
# TÌM XE PHÙ HỢP VỚI HÀNG
# =========================================================

def find_vehicle_for_cargo(cargo, motorcycles):

    cx1 = cargo["x1"]
    cy1 = cargo["y1"]
    cx2 = cargo["x2"]
    cy2 = cargo["y2"]

    cargo_center_x = (cx1 + cx2) / 2
    cargo_center_y = (cy1 + cy2) / 2

    best_vehicle = None
    best_score = float("inf")

    cargo_box = [
        cx1,
        cy1,
        cx2,
        cy2
    ]

    for motorcycle in motorcycles:

        mx1 = motorcycle["x1"]
        my1 = motorcycle["y1"]
        mx2 = motorcycle["x2"]
        my2 = motorcycle["y2"]

        motorcycle_width = mx2 - mx1
        motorcycle_height = my2 - my1

        if motorcycle_width <= 0:
            continue

        if motorcycle_height <= 0:
            continue

        motorcycle_center_x = (
            mx1 + mx2
        ) / 2

        motorcycle_center_y = (
            my1 + my2
        ) / 2

        # ---------------------------------------------
        # Khoảng cách ngang
        # ---------------------------------------------

        horizontal_distance = abs(
            cargo_center_x -
            motorcycle_center_x
        )

        if (
            horizontal_distance >
            motorcycle_width *
            MAX_HORIZONTAL_DISTANCE_RATIO
        ):
            continue

        # ---------------------------------------------
        # Khoảng cách dọc
        # ---------------------------------------------

        vertical_distance = abs(
            cargo_center_y -
            motorcycle_center_y
        )

        if (
            vertical_distance >
            motorcycle_height *
            MAX_VERTICAL_DISTANCE_RATIO
        ):
            continue

        # ---------------------------------------------
        # IOU
        # ---------------------------------------------

        motorcycle_box = [
            mx1,
            my1,
            mx2,
            my2
        ]

        iou = calculate_iou(
            cargo_box,
            motorcycle_box
        )

        # ---------------------------------------------
        # Điểm đánh giá
        # ---------------------------------------------

        normalized_distance = (
            horizontal_distance /
            motorcycle_width
        )

        score = (
            normalized_distance -
            iou * 0.5
        )

        if score < best_score:

            best_score = score
            best_vehicle = motorcycle

    return best_vehicle


# =========================================================
# MAIN LOOP
# =========================================================

print("==============================================")
print(" V03 - PHAT HIEN CHO HANG CONG KENH")
print("==============================================")
print()
print("Q: Thoat")
print()


while True:

    ret, frame = cap.read()

    if not ret:

        print("\n✅ Video đã kết thúc!")
        break


    # =====================================================
    # YOLO + BYTETRACK
    # =====================================================

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[
            0,      # person
            3,      # motorcycle
            24,     # backpack
            26,     # handbag
            28      # suitcase
        ],
        conf=CONFIDENCE,
        verbose=False
    )

    result = results[0]


    motorcycles = []
    cargo_objects = []


    # =====================================================
    # LẤY OBJECT
    # =====================================================

    if result.boxes.id is not None:

        boxes = result.boxes.xyxy.cpu().tolist()

        classes = (
            result.boxes.cls
            .int()
            .cpu()
            .tolist()
        )

        track_ids = (
            result.boxes.id
            .int()
            .cpu()
            .tolist()
        )

        confidences = (
            result.boxes.conf
            .cpu()
            .tolist()
        )


        for box, class_id, track_id, confidence in zip(
            boxes,
            classes,
            track_ids,
            confidences
        ):

            x1, y1, x2, y2 = map(
                int,
                box
            )


            # =================================================
            # MOTORCYCLE
            # =================================================

            if class_id == 3:

                motorcycles.append({

                    "id": track_id,

                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2

                })


            # =================================================
            # CARGO
            # =================================================

            elif class_id in CARGO_CLASSES:

                cargo_objects.append({

                    "id": track_id,

                    "class_id": class_id,

                    "name":
                        CARGO_CLASSES[class_id],

                    "confidence":
                        confidence,

                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2

                })


    # =====================================================
    # CARGO TRÊN TỪNG XE
    # =====================================================

    motorcycle_cargo = {}

    for motorcycle in motorcycles:

        motorcycle_cargo[
            motorcycle["id"]
        ] = []


    # =====================================================
    # GÁN HÀNG → XE
    # =====================================================

    for cargo in cargo_objects:

        motorcycle = find_vehicle_for_cargo(
            cargo,
            motorcycles
        )

        if motorcycle is None:
            continue


        motorcycle_id = motorcycle["id"]


        # =================================================
        # TÍNH KÍCH THƯỚC HÀNG
        # =================================================

        cargo_width = (
            cargo["x2"] -
            cargo["x1"]
        )

        cargo_height = (
            cargo["y2"] -
            cargo["y1"]
        )

        cargo_area = (
            cargo_width *
            cargo_height
        )


        # =================================================
        # KÍCH THƯỚC XE
        # =================================================

        motorcycle_width = (
            motorcycle["x2"] -
            motorcycle["x1"]
        )

        motorcycle_height = (
            motorcycle["y2"] -
            motorcycle["y1"]
        )

        motorcycle_area = (
            motorcycle_width *
            motorcycle_height
        )


        if motorcycle_area <= 0:
            continue


        # =================================================
        # TỶ LỆ
        # =================================================

        width_ratio = (
            cargo_width /
            motorcycle_width
        )

        area_ratio = (
            cargo_area /
            motorcycle_area
        )


        # =================================================
        # KIỂM TRA HÀNG CÓ ĐỦ LỚN KHÔNG
        # =================================================

        if (
            width_ratio >=
            MIN_CARGO_WIDTH_RATIO
            and
            area_ratio >=
            MIN_CARGO_AREA_RATIO
        ):

            motorcycle_cargo[
                motorcycle_id
            ].append(cargo)


    # =====================================================
    # CẬP NHẬT LỊCH SỬ
    # =====================================================

    for motorcycle in motorcycles:

        motorcycle_id = motorcycle["id"]

        cargo_count = len(
            motorcycle_cargo.get(
                motorcycle_id,
                []
            )
        )

        cargo_history[
            motorcycle_id
        ].append(cargo_count)


    # =====================================================
    # PHÂN TÍCH ỔN ĐỊNH
    # =====================================================

    stable_cargo = {}

    for motorcycle in motorcycles:

        motorcycle_id = motorcycle["id"]

        history = cargo_history[
            motorcycle_id
        ]


        if len(history) == 0:
            continue


        counter = Counter(history)

        stable_value, frequency = (
            counter.most_common(1)[0]
        )


        if frequency >= MIN_STABLE_COUNT:

            stable_cargo[
                motorcycle_id
            ] = stable_value

        else:

            stable_cargo[
                motorcycle_id
            ] = history[-1]


    # =====================================================
    # VẼ XE
    # =====================================================

    for motorcycle in motorcycles:

        motorcycle_id = motorcycle["id"]

        x1 = motorcycle["x1"]
        y1 = motorcycle["y1"]
        x2 = motorcycle["x2"]
        y2 = motorcycle["y2"]


        cargo_count = stable_cargo.get(
            motorcycle_id,
            0
        )


        # =================================================
        # VI PHẠM
        # =================================================

        if cargo_count > 0:

            # Hiện tại nếu vật thể đủ lớn
            # thì đánh dấu là hàng cồng kềnh

            box_color = (
                0,
                0,
                255
            )

            violation_text = (
                "VI PHAM: HANG CONG KENH"
            )

            cv2.putText(
                frame,

                violation_text,

                (
                    x1,
                    max(
                        y1 - 35,
                        30
                    )
                ),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.65,

                (
                    0,
                    0,
                    255
                ),

                2
            )


            if motorcycle_id not in reported_violations:

                print(
                    f"🔴 XE ID {motorcycle_id} "
                    f"-> PHAT HIEN HANG CONG KENH"
                )

                reported_violations.add(
                    motorcycle_id
                )

        else:

            box_color = (
                255,
                255,
                0
            )


        # =================================================
        # VẼ BOX XE
        # =================================================

        cv2.rectangle(
            frame,

            (
                x1,
                y1
            ),

            (
                x2,
                y2
            ),

            box_color,

            2
        )


        # =================================================
        # HIỂN THỊ THÔNG TIN XE
        # =================================================

        text = (
            f"XE ID: {motorcycle_id}"
            f" | HANG: {cargo_count}"
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
    # VẼ CARGO
    # =====================================================

    for cargo in cargo_objects:

        x1 = cargo["x1"]
        y1 = cargo["y1"]
        x2 = cargo["x2"]
        y2 = cargo["y2"]

        name = cargo["name"]

        confidence = cargo["confidence"]


        cv2.rectangle(
            frame,

            (
                x1,
                y1
            ),

            (
                x2,
                y2
            ),

            (
                0,
                255,
                255
            ),

            2
        )


        label = (
            f"{name} "
            f"{confidence:.2f}"
        )


        cv2.putText(
            frame,

            label,

            (
                x1,
                max(
                    y1 - 5,
                    20
                )
            ),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.5,

            (
                0,
                255,
                255
            ),

            2
        )


    # =====================================================
    # TIÊU ĐỀ
    # =====================================================

    cv2.putText(
        frame,

        "V03 - PHAT HIEN HANG CONG KENH",

        (
            20,
            35
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.7,

        (
            255,
            255,
            255
        ),

        2
    )


    cv2.putText(
        frame,

        "Q: Thoat",

        (
            20,
            65
        ),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.6,

        (
            255,
            255,
            255
        ),

        2
    )


    # =====================================================
    # SHOW
    # =====================================================

    cv2.imshow(
        "Traffic Violation - V03",
        frame
    )


    key = cv2.waitKey(1) & 0xFF


    if key == ord("q") or key == ord("Q"):

        print(
            "\n🛑 Đã dừng chương trình."
        )

        break


# =========================================================
# CLEAN UP
# =========================================================

cap.release()

cv2.destroyAllWindows()


print()
print("==============================================")
print("          V03 KET THUC")
print("==============================================")