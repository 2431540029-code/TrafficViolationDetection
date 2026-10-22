# Traffic Violation Detection

Hệ thống ứng dụng thị giác máy tính và trí tuệ nhân tạo để phát hiện, theo dõi và hỗ trợ nhận diện một số hành vi vi phạm giao thông từ hình ảnh và video.

## 1. Giới thiệu

**Traffic Violation Detection** là dự án sử dụng Python, OpenCV và mô hình YOLO để nghiên cứu các bài toán nhận diện phương tiện, người tham gia giao thông và một số hành vi vi phạm.

Dự án được phát triển theo từng chức năng độc lập nhằm thuận tiện cho việc thử nghiệm, huấn luyện mô hình và đánh giá kết quả nhận diện.

Các hướng phát triển chính:

- Nhận diện người đi xe máy không đội mũ bảo hiểm.
- Phát hiện xe máy và đếm số người trong khung hình.
- Nghiên cứu nhận diện xe máy chở quá số người quy định.
- Phát hiện và theo dõi biển số xe.
- Nghiên cứu nhận diện hành vi vận chuyển hàng hóa cồng kềnh.

## 2. Công nghệ sử dụng

| Công nghệ | Vai trò |
|---|---|
| Python | Ngôn ngữ lập trình chính |
| Ultralytics YOLO | Phát hiện đối tượng và huấn luyện mô hình |
| OpenCV | Đọc, xử lý và hiển thị hình ảnh, video |
| PyTorch | Nền tảng học sâu phục vụ mô hình |
| NumPy | Hỗ trợ xử lý dữ liệu và mảng số |
| Git và GitHub | Quản lý mã nguồn và lịch sử phát triển |

Các thư viện cụ thể có thể thay đổi tùy chức năng và phiên bản môi trường thực tế.

## 3. Các chức năng của dự án

### 3.1. Nhận diện mũ bảo hiểm

Sử dụng mô hình phát hiện đối tượng để nghiên cứu việc nhận diện người đi xe máy đội hoặc không đội mũ bảo hiểm.

### 3.2. Phát hiện xe máy và đếm người

Thử nghiệm phát hiện người và xe máy trong hình ảnh hoặc video, đồng thời nghiên cứu cách liên kết người với từng phương tiện.

Việc đếm người bằng mô hình phát hiện đối tượng thông thường chưa đảm bảo xác định chính xác số người ngồi trên từng xe trong mọi tình huống.

### 3.3. Nhận diện xe máy chở quá số người (V02)

V02 tập trung xây dựng và thử nghiệm mô hình YOLO với bộ dữ liệu giao thông có nhãn liên quan đến xe máy, người đi xe và lớp `Trippling`.

Quy trình phát triển gồm:

1. Kiểm tra cấu trúc và thống kê bộ dữ liệu.
2. Chuyển đổi nhãn từ định dạng COCO sang YOLO.
3. Kiểm tra trực quan các khung nhãn.
4. Huấn luyện mô hình từ trọng số YOLO ban đầu.
5. Tiếp tục huấn luyện từ checkpoint đã lưu.
6. Đánh giá và thử nghiệm mô hình trên hình ảnh hoặc video.

Các chương trình liên quan:

- `check_v02_dataset.py`
- `check_yolo_classes.py`
- `convert_v02_coco_to_yolo.py`
- `preview_v02_labels.py`
- `train_v02.py`
- `resume_v02.py`
- `passenger_violation.py`
- `passenger_violation_image.py`

**Lưu ý:** Kết quả nhận diện phụ thuộc vào chất lượng dữ liệu, góc quay, độ che khuất và điều kiện hình ảnh. Mô hình cần được kiểm tra thực tế trước khi đánh giá khả năng phát hiện vi phạm.

### 3.4. Nhận diện biển số xe

Dự án có các chương trình thử nghiệm phát hiện biển số, nhận dạng ký tự và theo dõi biển số trong hình ảnh hoặc video.

Các file liên quan gồm:

- `plate_test.py`
- `plate_detect_test.py`
- `plate_ocr_test.py`
- `plate_ocr_tracking.py`

### 3.5. Nhận diện hàng hóa cồng kềnh

Nghiên cứu sử dụng thị giác máy tính để phát hiện các trường hợp vận chuyển hàng hóa cồng kềnh trên phương tiện giao thông.

Các file hỗ trợ dữ liệu gồm:

- `check_cargo_dataset.py`
- `filter_cargo_dataset.py`
- `cargo_violation.py`

Chức năng này đang được phát triển và cần đánh giá thêm trên các tình huống giao thông thực tế.

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
├── check_cargo_dataset.py
├── filter_cargo_dataset.py
│
├── check_triple_model.py
├── check_v02_dataset.py
├── check_yolo_classes.py
├── convert_v02_coco_to_yolo.py
├── preview_v02_labels.py
├── train_v02.py
├── resume_v02.py
├── passenger_violation.py
├── passenger_violation_image.py
│
├── yolo_test.py
├── yolo11n.pt
│
├── dataset/
├── dataset_cargo/
├── dataset_v02_yolo/
├── videos/
│
├── .gitignore
└── README.md
```

Cấu trúc trên mang tính tham khảo. Các thư mục dữ liệu, trọng số huấn luyện và kết quả chạy có thể không được đưa lên GitHub do dung lượng lớn.

## 5. Cài đặt môi trường

### Yêu cầu

- Python tương thích với các phiên bản thư viện đang sử dụng.
- Git.
- PyCharm hoặc một trình soạn thảo Python tương đương.

### Bước 1: Tải mã nguồn

```bash
git clone https://github.com/2431540029-code/TrafficViolationDetection.git
cd TrafficViolationDetection
```

### Bước 2: Tạo môi trường ảo

```bash
python -m venv .venv
```

Kích hoạt môi trường trên Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Bước 3: Cài đặt thư viện

Nếu dự án đã có file `requirements.txt`, chạy:

```bash
pip install -r requirements.txt
```

Nếu chưa có file này, cài đặt các thư viện cơ bản:

```bash
pip install ultralytics opencv-python numpy
```

PyTorch thường được cài đặt như một phần phụ thuộc của Ultralytics. Nếu cần cấu hình riêng cho CPU hoặc GPU, hãy chọn gói tương thích với máy tính và môi trường hiện tại.

## 6. Huấn luyện mô hình V02

Trước khi huấn luyện, cần chuẩn bị bộ dữ liệu và đảm bảo file cấu hình `dataset_v02_yolo/data.yaml` trỏ đúng đến các thư mục ảnh và nhãn.

Chạy huấn luyện ban đầu:

```bash
python train_v02.py
```

Mô hình và kết quả huấn luyện được lưu trong thư mục chạy do cấu hình của chương trình quy định, dự kiến tại:

```text
runs/detect/v02_trippling/
```

Các checkpoint thường được lưu tại:

```text
runs/detect/v02_trippling/weights/
├── best.pt
└── last.pt
```

- `best.pt`: trọng số tốt nhất theo chỉ số validation trong quá trình huấn luyện.
- `last.pt`: trọng số tại epoch gần nhất đã được lưu.

Nếu quá trình huấn luyện bị gián đoạn, có thể tiếp tục bằng chương trình:

```bash
python resume_v02.py
```

Chỉ tiếp tục khi checkpoint tồn tại và còn đọc được. Không xóa checkpoint nếu cần khôi phục quá trình huấn luyện.

## 7. Dữ liệu và mô hình

Bộ dữ liệu huấn luyện có thể chứa nhiều hình ảnh, nhãn và file JSON. Do dung lượng lớn, các bộ dữ liệu đầy đủ không nhất thiết được lưu trong repository.

Các file dữ liệu cần được đặt đúng đường dẫn theo cấu hình của từng chương trình.

Không nên đưa các nội dung sau lên GitHub nếu không thực sự cần thiết:

- Môi trường ảo `.venv/`.
- Bộ dữ liệu lớn.
- Kết quả huấn luyện trong `runs/`.
- Các file checkpoint dung lượng lớn như `best.pt` và `last.pt`.

Trước khi chạy chương trình, cần kiểm tra rằng các file trọng số và dữ liệu bắt buộc đã có trên máy tính.

## 8. Kết quả và đánh giá

Các mô hình cần được đánh giá trên dữ liệu validation và test phù hợp.

Những chỉ số có thể sử dụng gồm:

- **Precision:** tỷ lệ dự đoán dương tính đúng trong các dự đoán dương tính.
- **Recall:** tỷ lệ đối tượng dương tính thực tế được mô hình phát hiện.
- **mAP50:** chỉ số trung bình độ chính xác phát hiện tại ngưỡng IoU 0.5.
- **mAP50-95:** chỉ số đánh giá trên nhiều ngưỡng IoU, khắt khe hơn.

Chỉ số đánh giá trên tập dữ liệu không đồng nghĩa với tỷ lệ phát hiện vi phạm chính xác trong mọi điều kiện giao thông thực tế.

Ảnh minh họa kết quả có thể được bổ sung vào README sau khi hoàn tất thử nghiệm.

## 9. Hướng phát triển

- Cải thiện độ chính xác của mô hình nhận diện.
- Kiểm tra trên nhiều điều kiện ánh sáng và góc quay.
- Kết hợp phát hiện đối tượng với theo dõi phương tiện.
- Tối ưu tốc độ xử lý hình ảnh và video.
- Bổ sung hình ảnh minh họa và kết quả đánh giá.
- Hoàn thiện quy trình kiểm thử và tài liệu sử dụng.

## 10. Tác giả và mục đích

Dự án được xây dựng phục vụ mục đích học tập, nghiên cứu và thực hành ứng dụng thị giác máy tính trong lĩnh vực giao thông.

Các kết quả nhận diện mang tính hỗ trợ phân tích hình ảnh, không thay thế việc xác minh của con người hoặc quy trình xử lý vi phạm chính thức.
