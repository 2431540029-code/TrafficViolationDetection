from ultralytics import YOLO
import cv2

# ==============================
# LOAD 2 MODEL
# ==============================

# Model YOLO11 nhận diện xe máy
traffic_model = YOLO("yolo11n.pt")

# Model đã train nhận diện helmet / no_helmet
helmet_model = YOLO("runs/detect/helmet_model-2/weights/best.pt")


# ==============================
# MỞ VIDEO
# ==============================

cap = cv2.VideoCapture("videos/videotest.mp4")

if not cap.isOpened():
    print("Không thể mở video!")
    exit()


# ==============================
# XỬ LÝ VIDEO
# ==============================

while True:

    ret, frame = cap.read()

    if not ret:
        print("Video đã kết thúc!")
        break

    # --------------------------------
    # MODEL 1: NHẬN DIỆN XE MÁY
    # --------------------------------

    traffic_results = traffic_model(frame, conf=0.4)

    # Vẽ kết quả xe máy
    annotated_frame = traffic_results[0].plot()


    # --------------------------------
    # MODEL 2: NHẬN DIỆN MŨ
    # --------------------------------

    helmet_results = helmet_model(frame, conf=0.5)

    helmet_result = helmet_results[0]


    # --------------------------------
    # LẤY CÁC BOX MŨ
    # --------------------------------

    for box in helmet_result.boxes:

        confidence = float(box.conf[0])

        class_id = int(box.cls[0])

        class_name = helmet_model.names[class_id]

        x1, y1, x2, y2 = map(int, box.xyxy[0])


        # --------------------------------
        # NẾU KHÔNG ĐỘI MŨ
        # --------------------------------

        if class_name == "no_helmet":

            # Vẽ khung màu đỏ
            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 0, 255),
                3
            )

            # Hiển thị cảnh báo
            cv2.putText(
                annotated_frame,
                "VI PHAM: KHONG DOI MU",
                (x1, max(y1 - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )


        # --------------------------------
        # NẾU CÓ ĐỘI MŨ
        # --------------------------------

        elif class_name == "helmet":

            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                annotated_frame,
                "HELMET",
                (x1, max(y1 - 10, 30)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )


    # --------------------------------
    # HIỂN THỊ VIDEO
    # --------------------------------

    cv2.imshow(
        "TRAFFIC VIOLATION DETECTION",
        annotated_frame
    )


    # Nhấn Q để thoát
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==============================
# GIẢI PHÓNG
# ==============================

cap.release()
cv2.destroyAllWindows()