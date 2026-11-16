# Báo cáo chức năng nhận diện hàng cồng kềnh

## 1. Mục tiêu

Tích hợp checkpoint YOLO11n Cargo v6 vào ứng dụng TrafficViolationDetection, đồng thời đánh giá kết quả bằng Validation và TEST độc lập. Việc đánh giá chỉ phục vụ báo cáo; không huấn luyện lại, không điều chỉnh ngưỡng theo TEST và không ghi đè checkpoint.

## 2. Dataset Roboflow v6

Dataset được kiểm tra tại `dataset_cargo/roboflow-v6/`, cấu hình phụ trợ là `data.train.yaml`.

| Tập | Ảnh | File nhãn | Đối tượng |
|---|---:|---:|---:|
| Train | 231 | 231 | 267 |
| Validation | 22 | 22 | 23 |
| Test | 11 | 11 | 14 |

YAML khai báo một lớp, ID `0`, tên `hang_cong_kenh`. Các nhãn quan sát được theo định dạng YOLO detection: `class_id x_center y_center width height`, tọa độ chuẩn hóa. Không phát hiện ảnh/nhãn mồ côi trong lần kiểm tra cuối. Nội dung nhãn gốc không bị thay đổi. Quy trình cụ thể mà người gán nhãn dùng để quyết định ranh giới đối tượng không được lưu trong các tệp đã kiểm tra.

## 3. Model

- Kiến trúc: Ultralytics YOLO11n, khởi tạo từ `yolo11n.pt`.
- Checkpoint được tích hợp: `runs/detect/cargo_protrusion_v6_yolo11n/weights/best.pt`.
- Checkpoint khai báo `{0: "hang_cong_kenh"}`; pipeline chuẩn hóa ID `0` thành `bulky_cargo`.
- Model có 100 layers, 2,582,347 tham số và 6.4 GFLOPs theo thông tin Ultralytics khi load.

Các checkpoint mặc định của ba chức năng sau khi tích hợp:

| Chức năng | Checkpoint | Lớp chính |
|---|---|---|
| Mũ bảo hiểm | `runs/detect/helmet_model-2/weights/best.pt` | `helmet`, `no_helmet` |
| Chở quá số người | `runs/detect/v02_trippling/weights/best.pt` | `bike`, `rider`, `trippling` |
| Hàng cồng kềnh | `runs/detect/cargo_protrusion_v6_yolo11n/weights/best.pt` | `hang_cong_kenh` → `bulky_cargo` |

## 4. Cấu hình huấn luyện đã lưu

Thông tin được đọc từ `runs/detect/cargo_protrusion_v6_yolo11n/args.yaml`:

- Data: `dataset_cargo/roboflow-v6/data.train.yaml`
- Epoch tối đa: 50; patience: 15
- Batch: 2; image size: 640
- Workers: 0; device: CUDA 0
- Optimizer: `auto`; pretrained: bật; AMP: bật
- Seed: 0; deterministic: bật
- Tên run: `cargo_protrusion_v6_yolo11n`; `exist_ok: false`

Run dừng ở epoch 29/50; epoch tốt nhất được ghi nhận là 14. Không chạy train trong lần hoàn thiện này.

## 5. Kết quả đánh giá

Checkpoint `best.pt` được đánh giá lại bằng Ultralytics trên các bản sao tạm của split; dữ liệu gốc không nhận cache hay thay đổi. Validation và TEST được giữ đúng split.

| Tập | Ảnh / đối tượng | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| Validation | 22 / 23 | 0.790 | 0.391 | 0.423 | 0.178 |
| Test | 11 / 14 | 0.253 | 0.357 | 0.227 | 0.089 |

TEST cũng được phân tích riêng ở confidence `0.25` và IoU `0.5` bằng phép ghép một-một theo IoU: 4 true positives, 9 false positives, 10 false negatives (precision 0.308, recall 0.286). Đây là thống kê ghép bbox tại ngưỡng cố định, không phải mAP. Không dùng các kết quả TEST để chọn model hay tinh chỉnh ngưỡng.

Confusion matrix Ultralytics: `runs/detect/cargo_protrusion_v6_test_audit_20261116_2011/confusion_matrix.png`. Danh sách FP/FN ở confidence `0.25`, IoU `0.5`: `runs/detect/cargo_protrusion_v6_test_audit_20261116_2011/cargo_v6_test_fp_fn.csv`. JSON metric: `cargo_v6_test_audit.json`.

## 6. Ảnh dự đoán và quan sát lỗi

Ảnh dự đoán lưu ở `runs/detect/cargo_protrusion_v6_test_audit_20261116_2011_predictions/`.

Một số bbox trên ảnh xe tải có chở hàng bao phủ một phần hàng nhưng không khớp đủ IoU với annotation lớn, dẫn đến tính là FP kèm FN ở phép ghép ngưỡng 0.5. Đây là trường hợp cần xem lại tính nhất quán ranh giới bbox, không tự động sửa nhãn.

Có false positive rõ trên vùng trời, cùng với một đối tượng hàng hóa thực tế không được phát hiện:

![Ví dụ false positive trên vùng trời](runs/detect/cargo_protrusion_v6_test_audit_20261116_2011_predictions/suggested-uKwJei3QZRFJtf2zjbCT_jpg.rf.0de55753002356779bbaa3742fbb2f08.jpg)

![Ví dụ bỏ sót hàng cồng kềnh](runs/detect/cargo_protrusion_v6_test_audit_20261116_2011_predictions/suggested-di8gyJyJwybW05qmI2Dd_jpg.rf.3658ad3b4fcda45f6d271771c1c55b9d.jpg)

Kết quả TEST còn thấp; model chưa thể được xem là đáng tin cậy để tự động kết luận vi phạm. Tập TEST chỉ có 11 ảnh nên metric có độ bất định cao.

## 7. Tích hợp vào ứng dụng

- `latest_cargo_checkpoint()` ưu tiên run v6, sau đó giữ nguyên fallback relabel v1, v4-only và v3-fixed.
- CLI video (`main.py`) và web (`web/app.py`) cùng dùng bộ chọn checkpoint này. CLI khởi tạo pipeline trước vòng lặp frame; web cache pipeline trong `app.config["PIPELINE"]`, tránh nạp lại model mỗi frame/request.
- Pipeline nhận class gốc `hang_cong_kenh`, chuẩn hóa thành `bulky_cargo`, vẽ bbox và confidence.
- Model Cargo quét toàn ảnh và không lọc theo phương tiện; mọi bbox được vẽ và thống kê dù không ghép được với xe máy. Bước liên kết hiện ưu tiên xe máy; model liên kết không phân loại riêng ô tô/xe tải. Chỉ vùng Cargo gán được với xe máy mới tạo event, không suy diễn vi phạm cho loại xe khác.
- Cargo hiện dùng ngưỡng confidence chung của pipeline (mặc định `0.35`); chưa có ngưỡng Cargo riêng. Không điều chỉnh threshold trong lần này.
- Không sửa logic mũ bảo hiểm hay chở quá số người. Giao diện được cập nhật để ghi rõ đang dùng Roboflow v6 và giới hạn đánh giá.
- CLI video từ chối ghi đè video/CSV đã tồn tại; CSV mặc định có timestamp. Web dùng UUID cho mỗi lượt upload.
- Web không đưa exception/path nội bộ vào thông báo lỗi; chi tiết vẫn được ghi server-side. Website hỗ trợ ảnh; xử lý video qua CLI.

## 8. Kiểm tra thực tế

- PyTorch: `2.14.1+cu126`; Ultralytics: `8.4.153`.
- GPU: NVIDIA GeForce RTX 3050 Laptop GPU, 4 GiB VRAM; trước khi đánh giá TEST còn khoảng 3.23 GiB.
- Đánh giá model trên GPU với batch 2 hoàn tất; 11 ảnh dự đoán được lưu.
- Nghiệm thu ảnh qua Flask: ảnh Passenger thật tạo được kết quả; ảnh Cargo TEST `suggested-uKwJei3QZRFJtf2zjbCT_jpg.rf.0de55753002356779bbaa3742fbb2f08.jpg` tạo 2 phát hiện Cargo không gán với xe máy, vẫn được vẽ/hiển thị và ghi nhận 0 sự kiện vi phạm. Ảnh Cargo đầu ra và ảnh gốc tải được; một lượt upload mới có UUID riêng và không dùng lại kết quả cũ.
- Kiểm tra bố cục web ở viewport 375 px và desktop 1350 px: không có tràn ngang; ảnh gốc/kết quả tải hoàn tất; khu vực kết quả chuyển từ một cột ảnh trên mobile sang hai cột trên desktop.
- Theo kiểm tra video lịch sử, CLI từng đọc, xử lý, ghi và mở lại video đủ **20/20 frame**, CSV có đúng header sự kiện, và `violations_detected.csv` cũ không đổi. **Không kiểm thử video lại trong đợt nghiệm thu ảnh này.**
- Web upload ảnh thật và download route trả HTTP 200. Kiểm tra model thiếu trả thông báo chung, không lộ đường dẫn nội bộ.
- Import `main`/`web.app`, CLI `--help`, compile và toàn bộ unit tests đều thành công: **58 tests passed**. Trang chính phân biệt rõ tệp checkpoint tồn tại với khả năng tải model; ngưỡng `no_helmet` hiển thị theo cấu hình. Tên ảnh tải xuống gồm một phần UUID; link kết quả đã mất trả thông báo 404.
- SHA-256 của helmet, passenger, Cargo relabel v1, Cargo v6 `best.pt` và `last.pt` khớp các giá trị đã ghi nhận trước khi đánh giá; không checkpoint nào bị ghi đè.
- Các lượt evaluation chỉ tạo output dưới `runs/detect/cargo_protrusion_v6_valid_audit_20261116_2014/`, `runs/detect/cargo_protrusion_v6_test_audit_20261116_2011/` và thư mục prediction tương ứng. Không sửa YAML, dataset hoặc annotation.

## 9. Hạn chế và hướng tiếp theo

1. Recall còn thấp, đặc biệt trên TEST; cần thu thập thêm ví dụ đa dạng và kiểm tra chất lượng ranh giới bbox trước khi quyết định huấn luyện tiếp.
2. Nên rà soát thủ công các ảnh FP/FN trong CSV và các ví dụ minh họa, nhưng không sửa nhãn gốc hàng loạt.
3. Giữ TEST làm tập đánh giá cuối; mọi thử nghiệm dữ liệu/hyperparameter tiếp theo chỉ dùng TRAIN và VALID.
4. Hệ thống phát hiện hàng trong bbox, không đo kích thước/độ nhô và không tự kết luận tính hợp pháp theo quy định. Việc gán bbox Cargo vào xe máy hiện có chỉ là logic ứng dụng đã có.
5. Trước khi trình diễn, nên kiểm tra trực tiếp trên các ảnh mục tiêu và trình bày rõ rằng đây là model thử nghiệm, không phải hệ thống xác định vi phạm pháp lý.
6. Trong lần này, Helmet và Passenger được load và smoke-test qua pipeline nhưng không có đánh giá độc lập mới. README ghi ngưỡng `no_helmet=0.48` đã được chọn sau khi xem một tập test cũ; tập đó không còn là test độc lập. Không tuyên bố độ chính xác của hai model từ smoke test.
7. `gui.py` là launcher Tkinter cũ chỉ dùng checkpoint Passenger; dùng Flask để trình diễn pipeline ba chức năng. Nhãn bbox trên cảnh đông vẫn có thể chồng nhau; không có thuật toán tự dời nhãn vì có thể che đối tượng hoặc làm sai cách đọc.
