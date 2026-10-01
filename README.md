# Nhận diện hành vi vi phạm giao thông

Đồ án môn Xử lý ảnh và Thị giác máy tính, xây dựng hệ thống nhận diện một số hành vi vi phạm giao thông bằng Computer Vision.

## 1. Giới thiệu đề tài

Hệ thống sử dụng mô hình YOLO11 kết hợp với OpenCV và kỹ thuật theo dõi đối tượng để phát hiện các hành vi vi phạm giao thông liên quan đến xe máy.

Hệ thống được định hướng phát triển với 3 nhóm vi phạm chính:

1. Không đội mũ bảo hiểm
2. Chở quá số người quy định
3. Chở hàng cồng kềnh / quá khổ

Hệ thống ưu tiên xử lý video và có định hướng mở rộng sang camera thời gian thực và hình ảnh.

## 2. Công nghệ sử dụng

- Python
- YOLO11
- Ultralytics
- OpenCV
- PyTorch
- ByteTrack
- EasyOCR

## 3. Các chức năng chính

### 3.1. Phát hiện không đội mũ bảo hiểm

Sử dụng mô hình YOLO11 được huấn luyện riêng với 2 lớp:

- `helmet`
- `no_helmet`

Khi phát hiện người điều khiển xe máy không đội mũ bảo hiểm, hệ thống sẽ đánh dấu vùng vi phạm bằng khung màu đỏ và hiển thị thông báo.

### 3.2. Phát hiện chở quá số người

Hệ thống sử dụng:

- YOLO11 để phát hiện `person` và `motorcycle`
- ByteTrack để theo dõi từng xe
- Thuật toán ghép người với từng xe máy
- Phân tích nhiều frame để ổn định số người

Mục tiêu là xác định số người đang đi trên từng xe máy và phát hiện trường hợp chở quá số người quy định.

### 3.3. Phát hiện chở hàng cồng kềnh / quá khổ

Chức năng này đang được phát triển.

Hệ thống dự kiến sử dụng một mô hình YOLO11 riêng để phát hiện hàng hóa trên xe máy, sau đó kết hợp:

- Kích thước tương đối giữa hàng hóa và xe
- Vị trí của hàng hóa
- Theo dõi đối tượng qua nhiều frame

để xác định trường hợp có dấu hiệu chở hàng cồng kềnh / quá khổ.

## 4. Cấu trúc dự án

```text
TrafficViolationDetection/
│
├── main.py
├── train.py
├── split_dataset.py
├── check_dataset.py
│
├── helmet_test.py
├── violation_motorcycle.py
│
├── person_count_test.py
├── person_motorcycle_tracking.py
├── person_motorcycle_tracking_v2.py
│
├── plate_test.py
├── plate_detect_test.py
├── plate_ocr_test.py
├── plate_ocr_tracking.py
│
├── cargo_violation.py
│
├── yolo_test.py
├── yolo11n.pt
│
├── dataset/
├── dataset_cargo/
├── videos/
│
└── README.md