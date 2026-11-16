# Traffic Violation Detection

Hệ thống ứng dụng thị giác máy tính và trí tuệ nhân tạo để phát hiện, theo dõi và hỗ trợ nhận diện một số hành vi vi phạm giao thông từ hình ảnh và video.

## 1. Giới thiệu

**Traffic Violation Detection** là ứng dụng Python, OpenCV, Ultralytics YOLO và Flask phục vụ nghiên cứu nhận diện một số đối tượng và hành vi giao thông. Bản demo cuối kỳ tập trung vào luồng tải và phân tích ảnh trên web; các công cụ video cũ vẫn nằm trong repository nhưng không thuộc phạm vi nghiệm thu ảnh của phiên bản này.

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

Model mặc định ở `runs/detect/helmet_model-2/weights/best.pt`, gồm class `helmet` và `no_helmet`. Hộp `no_helmet` chỉ được ghi nhận là vi phạm khi gán được với xe máy; phát hiện chưa ghép được vẫn được đánh dấu riêng, không tính là vi phạm.

### 3.2. Phát hiện xe máy và đếm người

Thử nghiệm phát hiện người và xe máy trong hình ảnh hoặc video, đồng thời nghiên cứu cách liên kết người với từng phương tiện.

Việc đếm người bằng mô hình phát hiện đối tượng thông thường chưa đảm bảo xác định chính xác số người ngồi trên từng xe trong mọi tình huống.

### 3.3. Nhận diện xe máy chở quá số người (V02)

Model mặc định ở `runs/detect/v02_trippling/weights/best.pt`, gồm class `bike`, `rider` và `trippling`. Ứng dụng đếm Rider được gán vào từng Bike; chỉ xe có hơn hai Rider được gán hoặc có class `trippling` được đánh dấu. Rider/Trippling không ghép được với Bike không tự tạo vi phạm.

Quy trình phát triển gồm:

1. Kiểm tra cấu trúc và thống kê bộ dữ liệu.
2. Chuyển đổi nhãn từ định dạng COCO sang YOLO.
3. Kiểm tra trực quan các khung nhãn.
4. Huấn luyện mô hình từ trọng số YOLO ban đầu.
5. Tiếp tục huấn luyện từ checkpoint đã lưu.
6. Đánh giá và thử nghiệm mô hình trên hình ảnh hoặc video.

Các chương trình liên quan:

- `tools/datasets/`: kiểm tra, chuyển đổi và xem nhãn V02.
- `training/`: huấn luyện và tiếp tục checkpoint V02.
- `detection/passenger/`: đếm người và thử nghiệm gán người vào xe.

**Lưu ý:** Kết quả nhận diện phụ thuộc vào chất lượng dữ liệu, góc quay, độ che khuất và điều kiện hình ảnh. Mô hình cần được kiểm tra thực tế trước khi đánh giá khả năng phát hiện vi phạm.

### 3.4. Nhận diện biển số xe

Dự án có các chương trình thử nghiệm phát hiện biển số, nhận dạng ký tự và theo dõi biển số trong hình ảnh hoặc video.

Các file liên quan nằm trong `detection/license_plate/`:

- `plate_test.py`
- `plate_detect_test.py`
- `plate_ocr_test.py`
- `plate_ocr_tracking.py`

Các thử nghiệm OCR cần cài thêm `easyocr`:

```powershell
.\.venv\Scripts\python.exe -m pip install easyocr
```

### 3.5. Nhận diện hàng hóa cồng kềnh

Checkpoint đang tích hợp là YOLO11n v6 tại `runs/detect/cargo_protrusion_v6_yolo11n/weights/best.pt`. Class dataset `hang_cong_kenh` được chuẩn hóa thành `bulky_cargo`. Hệ thống vẫn vẽ và thống kê phát hiện chưa ghép được với xe, nhưng chỉ phát sinh sự kiện khi gán Cargo với một xe máy đã phát hiện.

Các file kiểm tra và huấn luyện dữ liệu nằm trong `detection/cargo/`:

- `check_cargo_dataset.py`
- `filter_cargo_dataset.py`
- `cargo_violation.py`

Độ tin cậy của Cargo v6 còn hạn chế; kết quả cần được kiểm chứng trong điều kiện thực tế. Hệ thống chưa đo kích thước/độ nhô và không thay thế việc xác minh của con người.

## 4. Chạy ứng dụng tích hợp

Ứng dụng chính dùng checkpoint mũ, V02 và ưu tiên Cargo v6 tại
`runs/detect/cargo_protrusion_v6_yolo11n/weights/best.pt`. Nếu checkpoint này
không có, hệ thống fallback lần lượt về Cargo relabel v1, v4-only rồi
v3-fixed. Lớp `hang_cong_kenh`, `xe_cong_kenh`, `bulky_cargo` và `protrusion`
được chuẩn hóa thành `bulky_cargo` cho pipeline hiển thị. Nếu không có checkpoint
Cargo nào, ứng dụng vẫn chạy hai chức năng kia và báo rõ nhận diện hàng cồng kềnh đang tắt;
không thay thế bằng các lớp ba lô/vali COCO.
Phát hiện Cargo luôn được vẽ và thống kê, kể cả khi chưa ghép được với xe máy.
Model Cargo quét toàn ảnh và không lọc theo loại phương tiện, nên các vùng trên
ô tô hoặc loại xe khác vẫn được giữ lại và hiển thị. Bước liên kết hiện ưu tiên
xe máy; model liên kết không phân loại riêng ô tô/xe tải. Sự kiện vi phạm Cargo
chỉ được tạo khi có liên kết xe máy, không tự suy diễn kết luận cho phương tiện
khác. Nếu model xe máy bỏ sót một xe thật thì Cargo trên xe đó vẫn được hiển thị
nhưng không có thông tin phân loại phương tiện.

Chạy video mặc định:

```powershell
.\.venv\Scripts\python.exe main.py
```

Chọn video, xuất video kết quả và CSV sự kiện:

```powershell
.\.venv\Scripts\python.exe main.py --source videos\videotest.mp4 --output results\annotated.mp4
```

Mỗi lần chạy phải dùng đường dẫn video và CSV chưa tồn tại; CLI từ chối ghi đè
kết quả đã có. Nếu không truyền `--output`, CSV được tạo với tên
`violations_detected_<timestamp>.csv`.

Chạy webcam số 0 hoặc kiểm tra một số frame không mở cửa sổ:

```powershell
.\.venv\Scripts\python.exe main.py --source 0
.\.venv\Scripts\python.exe main.py --source videos\videotest.mp4 --no-display --max-frames 100
```

Chạy giao diện desktop cũ để chọn và nhận diện ảnh:

```powershell
.\.venv\Scripts\python.exe gui.py
```

`gui.py` là giao diện Tkinter cũ, hiện chỉ dùng checkpoint passenger V02;
dùng website Flask bên dưới để trình diễn pipeline tích hợp cả ba chức năng.

Logic chở quá số người dùng số Rider đã gán vào một xe và class `Trippling` của
V02; gán người/hàng vào xe dựa trên tâm hộp và vùng mở rộng quanh xe nên vẫn cần
kiểm tra thủ công với các xe sát nhau hoặc bị che khuất. Đây là công cụ demo,
không phải bằng chứng xử phạt tự động.

### Kiến trúc và luồng xử lý ảnh

`app.py` ở thư mục gốc gọi `web.app.main()`. Flask tạo pipeline dùng chung; ở
lần nhận diện đầu tiên pipeline nạp các checkpoint Helmet, Passenger và Cargo,
sau đó giữ pipeline để tái sử dụng. `POST /detect` đọc ảnh bằng Pillow (có
hiệu chỉnh EXIF), chạy các model và luật gán trong `core/violation_pipeline.py`,
rồi lưu ảnh gốc/ảnh đã vẽ cùng lịch sử vào `web_results/`. Tên file kết quả
dùng UUID để tránh va chạm tên upload.

Các route ảnh chính:

| Route | Chức năng |
|---|---|
| `GET /` | Trang chính và trạng thái các model |
| `POST /detect` | Nhận ảnh và trả trang kết quả |
| `GET /media/<uuid>/<kind>` | Đọc ảnh nguồn hoặc ảnh có bounding box |
| `GET /download/<uuid>` | Tải ảnh kết quả |
| `GET /history/<uuid>` | Mở kết quả đã lưu |
| `POST /history/<uuid>/delete` | Xóa một mục lịch sử cùng hai ảnh thuộc mục đó |

Model mặc định của web: Helmet `runs/detect/helmet_model-2/weights/best.pt`,
Passenger `runs/detect/v02_trippling/weights/best.pt`, Cargo v6
`runs/detect/cargo_protrusion_v6_yolo11n/weights/best.pt`. Ngưỡng web hiện là
confidence chung `0.35`, riêng `no_helmet` `0.48`, và Passenger trên ảnh tĩnh
`0.30`. Cargo dùng ngưỡng chung, không có threshold riêng đã được hiệu chỉnh.
Trạng thái trên trang chính chỉ xác nhận checkpoint tồn tại trên đĩa; model được
nạp khi nhận diện lần đầu, vì vậy nhãn này không cam kết model tải thành công
hay có độ chính xác nhất định.

## 5. Dataset và lịch sử huấn luyện hàng cồng kềnh

Model Cargo được chọn cho luồng nhận diện chính là checkpoint v6 nêu ở mục 3.5.
Dataset là `dataset_cargo/roboflow-v6/`, với 231 ảnh train/267 đối tượng,
22 ảnh validation/23 đối tượng và 11 ảnh test/14 đối tượng; class ID 0 có tên
`hang_cong_kenh`. YAML phụ trợ dùng khi huấn luyện là
`dataset_cargo/roboflow-v6/data.train.yaml`. Không có train lại trong lần hoàn
thiện phiên bản này. Các run v3, v4-only và relabel cũ bên dưới được giữ như
lịch sử; không xóa checkpoint hay kết quả audit.

Thông tin audit dữ liệu và metric v6 được tổng hợp ở
[`REPORT_FINAL_CARGO.md`](./REPORT_FINAL_CARGO.md). Dữ liệu test nhỏ, metric
không phải độ chính xác thực tế và không đủ căn cứ để tự động kết luận vi phạm.

Trích frame từ video mới để gán nhãn thủ công (các frame trích ra chưa có label):

```powershell
.\.venv\Scripts\python.exe -m detection.cargo.collect_cargo_frames --video path\to\new_video.mp4 --interval 2
```

Gán nhãn một class `bulky_cargo` theo YOLO, sau đó bổ sung ảnh/label vào các
thư mục `train`, `valid`, `test` trong dataset tương ứng với YAML truyền vào.
Chia tập theo **video nguồn**, không chia ngẫu nhiên frame của cùng video giữa
train và test, để tránh rò rỉ các cảnh gần như trùng lặp.

Kiểm tra các nhãn và tạo ảnh xem trước trên dataset Roboflow hiện tại:

```powershell
.\.venv\Scripts\python.exe -m detection.cargo.check_cargo_labels
```

Huấn luyện model riêng (mặc định tối đa 60 epoch, dừng sớm nếu validation không cải thiện):

```powershell
.\.venv\Scripts\python.exe -m detection.cargo.train_cargo --device 0
```

### 5.1. Dataset cargo Roboflow v4

Dataset nguồn `dataset_cargo/roboflow-v4` chỉ có 90 ảnh train, 90 nhãn ghép
đúng, một class ID `0` tên `xe_cong_kenh`; chưa có valid/test. Roboflow ghi rõ
mỗi ảnh nguồn có ba phiên bản, và tên file xác nhận 90 ảnh thuộc 30 nhóm nguồn.
Không chia ngẫu nhiên từng ảnh vì sẽ làm các phiên bản gần trùng rơi vào cả
train và test.

Bản chuẩn bị ở `dataset_cargo/roboflow-v4-prepared` giữ nguyên dataset gốc và
chép từng nhóm nguồn trọn vẹn vào split: train 63 ảnh/66 đối tượng từ 21 cảnh,
valid 12 ảnh/15 đối tượng từ 4 cảnh, test 15 ảnh/15 đối tượng từ 5 cảnh.
`data.yaml` dùng đường dẫn tuyệt đối đã được Ultralytics kiểm tra. Manifest nhóm
và split nằm trong `split_manifest.csv`; ảnh và nhãn trong bản sao giống byte
với nguồn.

Các ảnh xem trước cho thấy lớp được gán quanh xe máy cùng phần hàng cồng kềnh.
Checkpoint v3 `protrusion` được train trên các cảnh xe tải/chở gỗ, nên dù cùng
một class ID, miền dữ liệu và đối tượng không khớp với xe máy v4; dùng
`yolo11n.pt` pretrained thay vì fine-tune từ v3. Chỉ có 30 cảnh nguồn độc lập
(4 cảnh validation và 5 test), vì vậy kết quả chỉ là thử nghiệm ban đầu, metric
validation/test sẽ dao động mạnh và không đại diện cho hiệu năng ngoài dữ liệu
này. Cần bổ sung thêm ảnh thực tế độc lập trước khi kết luận.

Lệnh train ban đầu:

```powershell
.\.venv\Scripts\python.exe -m detection.cargo.train_cargo --data .\dataset_cargo\roboflow-v4-prepared\data.yaml --weights .\yolo11n.pt --epochs 30 --imgsz 640 --batch 4 --device 0 --name cargo_protrusion_v4
```

Kết quả mới được lưu tại `runs/detect/cargo_protrusion_v4`; script từ chối ghi
đè run có sẵn. Nếu bị gián đoạn, tiếp tục chính xác từ checkpoint:

```powershell
.\.venv\Scripts\python.exe -m detection.cargo.train_cargo --data .\dataset_cargo\roboflow-v4-prepared\data.yaml --weights .\runs\detect\cargo_protrusion_v4\weights\last.pt --name cargo_protrusion_v4 --resume --device 0
```

Đánh giá split sau khi chọn checkpoint:

```powershell
.\.venv\Scripts\python.exe -m detection.cargo.evaluate_cargo --model .\runs\detect\cargo_protrusion_v4\weights\best.pt --data .\dataset_cargo\roboflow-v4-prepared\data.yaml --split val --name cargo_protrusion_v4_validation_evaluation --device 0
.\.venv\Scripts\python.exe -m detection.cargo.evaluate_cargo --model .\runs\detect\cargo_protrusion_v4\weights\best.pt --data .\dataset_cargo\roboflow-v4-prepared\data.yaml --split test --name cargo_protrusion_v4_test_evaluation --device 0
```

Ultralytics validation dùng `conf=0.001` để tính P/R/mAP theo đường cong
precision-recall; danh sách FP/FN riêng dùng confidence `0.25` và IoU `0.5`
theo mặc định. Hai cách tính P/R có thể khác nhau; báo cáo JSON lưu cả hai.

Lần chạy `cargo_protrusion_v4` đã hoàn thành 30 epoch trên RTX 3050; checkpoint
tốt nhất ở epoch 26. Đánh giá Ultralytics (`conf=0.001`): validation P=0.933,
R=0.927, mAP50=0.979, mAP50-95=0.655; test P=0.991, R=1.000, mAP50=0.995,
mAP50-95=0.678. Test chỉ có 5 cảnh nguồn độc lập (15 ảnh biến thể), nên các
metric cao này chưa đáng tin để khái quát. Ở confidence 0.25, danh sách test
ghi nhận 15 TP, 12 FP và 0 FN (precision từ ghép hộp IoU 0.5 là 0.556); cần
kiểm tra thêm dữ liệu mới trước khi kết luận hoặc dùng thực tế. Tất cả ảnh hiện
có đều chứa đối tượng dương tính, nên chưa đánh giá được báo động nhầm trên xe
máy không chở hàng cồng kềnh. Không so sánh trực tiếp các metric với model v3
vì tập v3 chủ yếu là xe tải, khác miền đối tượng.

Sau khi train, đánh giá trên test set:

```powershell
.\.venv\Scripts\python.exe -m detection.cargo.evaluate_cargo
```

Báo cáo và danh sách false positive / bỏ sót được lưu trong
`runs/detect/cargo_protrusion_test_evaluation`. Video đã dùng để phát triển không
thay thế video test mới, độc lập và có nhãn thực tế.

### 5.2. Kiểm tra và gộp dataset cargo v3/v4

Script `detection/cargo/merge_and_train.py` kiểm tra cấu trúc, class ID, nhãn,
ảnh hỏng, tên file, ảnh trùng SHA-256 và nhóm augmentation theo tên; sau đó mới
có thể tạo `dataset_cargo/roboflow-combined` và train YOLO11n. Chạy audit trước,
không gộp hoặc train:

```powershell
.\.venv\Scripts\python.exe -m detection.cargo.merge_and_train --audit-only
```

Audit thực tế ghi nhận v3 có 430/64/32 ảnh ở train/valid/test, nhưng có 8 nhãn
train và 1 nhãn test không ghép được với ảnh; README Roboflow khai báo 535 ảnh,
trong khi các split hiện có 526 ảnh. v4 có 90 ảnh train nhưng không có valid/test;
đường dẫn trong YAML nguồn cũng không trỏ đến thư mục thực tế. Không tự ý xóa
hay sửa các file nguồn này. Hãy xem danh sách cụ thể trong
`dataset_cargo/roboflow-combined_audit.json`.

Quan trọng: v3 class `protrusion` được gán trên hàng hóa của xe tải và các hộp
mẫu bao quanh phần hàng; v4 class `xe_cong_kenh` là hàng cồng kềnh trên xe máy
và hộp mẫu bao gồm cả xe/người/hàng. Không tự động gộp hai quy ước nhãn này.
Chỉ sau khi kiểm tra và xác nhận chúng thực sự là cùng mục tiêu/kiểu bounding
box mới chạy script không có `--audit-only`. Nếu các nhãn mồ côi đã được xem xét
và xác nhận là file không dùng được do không có ảnh tương ứng, có thể cho phép
bỏ chúng khỏi **bản sao đầu ra** bằng `--confirm-ignore-orphan-labels`; file
nguồn vẫn được giữ nguyên. Không dùng xác nhận này nếu nhãn mồ côi cần được
khôi phục.

v4 hiện chỉ có tập train. Nếu tiếp tục gộp, validation/test sẽ đến từ v3 và
chủ yếu thể hiện cảnh xe tải, không đủ đại diện cho bài toán xe máy chở hàng.
Script chỉ đưa test vào đánh giá khi có xác nhận độc lập rõ ràng. Mặc định
training dùng `yolo11n.pt`, 50 epoch, ảnh 640, batch 4, tự chọn CUDA; nếu CUDA
hết bộ nhớ sẽ thử batch nhỏ hơn. Nếu máy không có CUDA, cần xác nhận CPU qua
`--allow-cpu` hoặc trả lời prompt tương tác. Script không ghi đè dataset hay
run đã tồn tại và lưu audit, cấu hình, log cùng metric thực tế.

Với mục tiêu **xe máy** của đồ án, nên bỏ v3 và dùng riêng v4. Chạy kiểm tra
v4, rồi chuẩn bị dataset theo nhóm cảnh (dùng manifest v4 đã tạo trước đó nếu
còn hợp lệ; nếu không, chia nhóm tên nguồn với seed 42):

```powershell
.\.venv\Scripts\python.exe -m detection.cargo.merge_and_train --v4-only --audit-only
.\.venv\Scripts\python.exe -m detection.cargo.merge_and_train --v4-only --prepare-only
```

Sau khi xem lại tập đã chuẩn bị, có thể chạy huấn luyện bằng:

```powershell
.\.venv\Scripts\python.exe -m detection.cargo.merge_and_train --v4-only --train-existing
```

Chế độ này tạo `dataset_cargo/roboflow-v4-only-prepared` và run
`runs/detect/cargo_protrusion_v4_only`, không sử dụng các nhãn xe tải v3.

Lượt huấn luyện v4-only đã chạy trên CUDA với batch 4, tối đa 50 epoch và dừng
sớm sau 48 epoch; epoch có mAP50-95 huấn luyện tốt nhất là 33. Đánh giá lại
checkpoint tốt nhất trên validation (12 ảnh từ 4 cảnh): Precision 0.9641,
Recall 0.8667, mAP50 0.9763, mAP50-95 0.6152. Đây là metric trên tập nhỏ, không
phải độ chính xác tổng quát. Không đánh giá lại test vì tập 15 ảnh/5 cảnh đã
được dùng trong lần đánh giá v4 trước đó. Smoke test lịch sử trên một ảnh validation cho class `bulky_cargo` confidence
0.829; đây chỉ xác nhận checkpoint chạy được, không phải thước đo hiệu năng.
Luồng nhận diện chính hiện ưu tiên checkpoint v6 theo mục 3.5.

### 5.3. Dataset cargo Roboflow đã gắn nhãn lại

Dataset mới trong `dataset_cargo/roboflow-v5` là **version 2** của project
Motorcycle Bulky Cargo Detection, theo `data.yaml` và README Roboflow (xuất ngày
28/10/2026, license CC BY 4.0). Tên thư mục `v5` không phải version do Roboflow
công bố. Bản export có 90 ảnh train, 90 nhãn YOLO detection, một class
`xe_cong_kenh` và 105 bounding box. Kiểm tra định dạng không phát hiện nhãn
rỗng, class lạ, tọa độ sai hoặc ảnh thiếu nhãn; kiểm tra trực quan cho thấy
các hộp mới chủ yếu bao quanh hàng hóa thay vì cả người và xe. Dataset không có
ảnh âm tính (không chở hàng), nên vẫn cần bổ sung loại ảnh này để đánh giá và
giảm false positive.

`data.yaml` nguồn trỏ tới các thư mục `../train`, `../valid`, `../test` ở bên
ngoài `roboflow-v5`, không phải các file trong bản export mới. Không sửa dữ liệu
nguồn; bản làm việc riêng `dataset_cargo/roboflow-v5-prepared` được tạo từ
`roboflow-v5/train` và dùng YAML có đường dẫn tuyệt đối hợp lệ. README cho biết
90 ảnh là 3 phiên bản augmentation của 30 cảnh nguồn. Chia theo nhóm tên nguồn
với seed 42 theo tỷ lệ 70/20/10: train 63 ảnh/21 cảnh, validation 18 ảnh/6
cảnh, test 9 ảnh/3 cảnh. Không có nhóm cảnh hoặc ảnh trùng byte đi qua nhiều
split. Xem `split_manifest.csv` và `README.split.txt` trong bản chuẩn bị để
kiểm tra phân chia. Do chỉ có 30 cảnh độc lập, đặc biệt test chỉ có 3 cảnh,
các chỉ số đánh giá có độ bất định cao và không đại diện cho hiệu năng ngoài
dataset.

Huấn luyện mới từ pretrained YOLO11n, không tiếp tục checkpoint cũ:

```powershell
.\.venv\Scripts\python.exe -m detection.cargo.train_cargo --data .\dataset_cargo\roboflow-v5-prepared\data.yaml --weights .\yolo11n.pt --epochs 50 --imgsz 640 --batch 2 --workers 2 --device 0 --name cargo_protrusion_relabel_v1
```

Batch 2 được chọn vì GPU hiện có 4 GB VRAM. Run mới được lưu tại
`runs/detect/cargo_protrusion_relabel_v1`; script không ghi đè run đã tồn tại.
Huấn luyện thực tế dừng sớm ở epoch 39/50; checkpoint tốt nhất ở epoch 24.
Validation (18 ảnh/6 cảnh): Precision 0.668, Recall 0.571, mAP50 0.613,
mAP50-95 0.247. Test độc lập đã chạy một lần (9 ảnh/3 cảnh): Precision 1.000,
Recall 0.427, mAP50 0.504, mAP50-95 0.195 theo metric Ultralytics
(`conf=0.001`). Ở ngưỡng liệt kê lỗi 0.25, test có 3 TP, 6 FP và 6 FN; P/R
theo cách ghép hộp IoU 0.5 lần lượt là 0.333/0.333. Các số liệu này thấp và
dao động mạnh vì test chỉ có 3 cảnh nguồn; chúng không phải độ chính xác tổng
quát. Cần thêm cảnh độc lập và ảnh xe không chở hàng trước khi đánh giá triển
khai. Báo cáo và danh sách lỗi nằm trong các thư mục validation/test riêng
bên dưới `runs/detect/`.

### 5.4. Roboflow Cargo v6 đang tích hợp

Checkpoint `runs/detect/cargo_protrusion_v6_yolo11n/weights/best.pt` được train
từ YOLO11n pretrained với `imgsz=640`, batch 2, workers 0, tối đa 50 epoch,
patience 15 và CUDA 0. Run dừng sớm ở epoch 29; epoch tốt nhất là 14. Kết quả
đánh giá đã lưu:

| Split | Ảnh / đối tượng | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| Validation | 22 / 23 | 0.790 | 0.391 | 0.423 | 0.178 |
| Test | 11 / 14 | 0.253 | 0.357 | 0.227 | 0.089 |

Test có false positive và false negative; các ảnh minh họa, confusion matrix
và file phân tích giữ tại các thư mục audit trong `runs/detect/`. Không dùng
test để chỉnh ngưỡng; Cargo đang dùng confidence chung `0.35`. Độ tin cậy còn
hạn chế và cần kiểm chứng thực tế. Xem [`REPORT_FINAL_CARGO.md`](./REPORT_FINAL_CARGO.md)
để biết cách đo FP/FN, giới hạn bbox và chi tiết lần đánh giá.

## 6. Chạy website Flask

Cài thư viện (nếu môi trường hiện tại chưa có Flask):

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Khởi động website:

```powershell
.\.venv\Scripts\python.exe app.py
```

Mở `http://127.0.0.1:5000`, chọn ảnh (tối đa 12 MB), rồi
nhấn **Phân tích ảnh**. Website hiển thị ảnh đầu vào, ảnh kết quả, thống kê và
các hành vi được đánh dấu. Nút **Lưu ảnh kết quả** tải ảnh đã vẽ khung xuống
máy. Ảnh trên máy chủ được lưu trong `web_results/`.
Ảnh tĩnh có chiều rộng hoặc chiều cao trên 1536 px được phân tích thêm theo các
vùng ảnh chồng lấp để tăng cơ hội nhận ra đối tượng nhỏ; ảnh thông thường chỉ
phân tích toàn khung hình nhằm hạn chế phát hiện trùng. Kết quả vẫn phụ thuộc
vào dữ liệu huấn luyện và có thể bỏ sót xe trong cảnh quá đông. Ngưỡng nhận diện
xe cho ảnh tĩnh cấu hình bằng `IMAGE_PASSENGER_CONFIDENCE` trong `web/app.py`;
video dùng ngưỡng `CONFIDENCE`.
Phát hiện lớp `no_helmet` dùng ngưỡng riêng `NO_HELMET_CONFIDENCE=0.48`.
Trong lần rà soát 65 ảnh test, ngưỡng này loại các hộp mũ bảo hiểm bị gán nhầm
`no_helmet` (2 xuống 0 ở IoU 0.5), đồng thời giảm dự đoán `no_helmet` không
khớp nhãn từ 20 xuống 10; số phát hiện đúng lớp `no_helmet` giảm từ 78 xuống
68. Vì test set đã được dùng để chọn ngưỡng, các con số này chỉ là chẩn đoán,
không còn là kết quả đánh giá test độc lập. Ngưỡng là đánh đổi precision/recall,
không phải sửa lỗi toàn bộ checkpoint. Dự đoán `no_helmet` chưa ghép được với
xe được hiển thị bằng khung vàng có dấu `?`, không tính vào số xe vi phạm; chỉ
phát hiện gắn được với xe mới được đánh dấu đỏ. Cần bổ sung ảnh giao thông Việt
Nam có nhãn chuẩn và đánh giá lại trên một tập độc lập để cải thiện độ tổng quát.

Trang web nhận ảnh raster thông dụng dựa trên nội dung thực của tệp, không chỉ
phần mở rộng: JPEG, PNG, GIF, TIFF, BMP, WEBP và HEIC/HEIF. Ảnh GIF động sử
dụng khung đầu tiên. SVG/PDF không phải ảnh raster. Mỗi lần
phân tích được ghi vào `web_results/history.sqlite3`; trang lịch sử mở lại kết
quả hoặc xóa bản ghi cùng ảnh đã lưu. Lịch sử hiển thị 50 lần gần nhất.
Ảnh tải xuống có tên dựa trên tên upload kèm một phần UUID để tránh trùng tên;
liên kết lịch sử hoặc ảnh kết quả không còn tồn tại sẽ trả thông báo 404 rõ ràng.

Mặc định website chỉ lắng nghe trên máy hiện tại (`127.0.0.1`), không bật debug
server và không yêu cầu tải ảnh lên dịch vụ bên ngoài. Pipeline và model được
nạp khi có yêu cầu nhận diện đầu tiên rồi dùng lại. Cần có checkpoint mũ và V02
trong thư mục `runs/detect` như cấu hình trong `web/app.py`; Cargo được chọn theo
thứ tự ưu tiên nêu trên. Model Cargo quét toàn ảnh, không loại vùng phát hiện
trên ô tô hay phương tiện khác; các bbox được giữ và thống kê, còn bước liên kết
xe máy là một thông tin bổ sung. Website nhận ảnh tĩnh;
video được xử lý bằng CLI `main.py`.

### 6.1. Nghiệm thu phiên bản ảnh

Lệnh kiểm thử đầy đủ:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Upload giới hạn 12 MB; web kiểm tra nội dung ảnh thay vì chỉ tin phần mở rộng.
Ảnh lỗi hoặc vượt giới hạn nhận thông báo dễ hiểu; ảnh đầu vào và ảnh kết quả
được lưu riêng theo UUID, vì vậy upload trùng tên không ghi đè lượt trước.
Tổng bounding box, số phát hiện từng chức năng, xe/Rider đã gán, Cargo chưa
ghép và số sự kiện lấy từ kết quả pipeline; tổng bbox không phải số đối tượng
duy nhất vì các model có thể cùng phát hiện một thực thể.

Lần nghiệm thu ảnh cuối kỳ kiểm tra upload JPEG từ các split Helmet, Passenger
và Cargo; web tạo được ảnh đầu vào/kết quả cùng link tải. Đây là smoke test tích
hợp, không phải đánh giá độ chính xác. Không chạy video trong nghiệm thu này.
Độ tin cậy Cargo v6 còn hạn chế; các kết quả và ảnh minh họa được ghi riêng ở
[`REPORT_FINAL_CARGO.md`](./REPORT_FINAL_CARGO.md).

## 7. Cấu trúc dự án

```text
TrafficViolationDetection/
├── app.py, main.py, gui.py       # Lệnh chạy tương thích ở thư mục gốc
├── core/                         # Pipeline dùng chung
├── web/                          # Flask, templates và static assets
├── detection/
│   ├── helmet/
│   ├── passenger/
│   ├── cargo/
│   └── license_plate/
├── training/                     # Huấn luyện và resume YOLO
├── tools/
│   ├── datasets/                 # Kiểm tra/chuyển đổi/chia dataset
│   └── experiments/              # Debug model và thử nghiệm YOLO
├── desktop/                      # GUI CustomTkinter cũ
├── tests/
├── dataset/, dataset_cargo/, dataset_v02_yolo/
├── runs/, videos/, web_results/
├── requirements.txt
└── README.md
```

Các lệnh huấn luyện và công cụ trong package chạy từ thư mục gốc bằng `python -m`, ví dụ:

```powershell
.\.venv\Scripts\python.exe -m training.train_v02
.\.venv\Scripts\python.exe -m detection.cargo.train_cargo --device cpu
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Dataset và kết quả huấn luyện cũ không được đưa lên GitHub. Repository chỉ theo
dõi ba checkpoint cần cho ứng dụng web; các file `last.pt` và checkpoint thí
nghiệm khác vẫn bị loại trừ.

## 8. Cài đặt môi trường

### Yêu cầu

- Python 3.13.14 trong môi trường đã nghiệm thu; máy khác cần chọn phiên bản
  tương thích với PyTorch và Ultralytics trong `requirements.txt`.
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

```bash
pip install -r requirements.txt
```

PyTorch thường được cài đặt như một phần phụ thuộc của Ultralytics. Nếu cần cấu hình riêng cho CPU hoặc GPU, hãy chọn gói tương thích với máy tính và môi trường hiện tại.

### Bước 4: Kiểm tra checkpoint ứng dụng

Ba checkpoint dưới đây được lưu cùng repository tại đúng đường dẫn mà web sử
dụng:

| Chức năng | Đường dẫn |
|---|---|
| Mũ bảo hiểm | `runs/detect/helmet_model-2/weights/best.pt` |
| Chở quá số người | `runs/detect/v02_trippling/weights/best.pt` |
| Hàng cồng kềnh | `runs/detect/cargo_protrusion_v6_yolo11n/weights/best.pt` |

Kiểm tra các file sau khi clone trên Windows PowerShell:

```powershell
Test-Path .\runs\detect\helmet_model-2\weights\best.pt
Test-Path .\runs\detect\v02_trippling\weights\best.pt
Test-Path .\runs\detect\cargo_protrusion_v6_yolo11n\weights\best.pt
```

Ba lệnh cần trả về `True`. Không cần tải checkpoint riêng; dataset và các
checkpoint huấn luyện khác không nằm trong bản clone.

### Bước 5: Chạy thử bằng ảnh tĩnh

Khởi động web bằng `python app.py`, mở `http://127.0.0.1:5000` và tải lần lượt
các ảnh JPG/JPEG hoặc PNG phù hợp:

| Chức năng cần kiểm tra | Ảnh nên chọn | Kết quả cần xem |
|---|---|---|
| Mũ bảo hiểm | Ảnh giao thông có người đội và/hoặc không đội mũ | Hộp `helmet`/`no_helmet`; chỉ phát hiện không đội mũ được ghép với xe mới được tính là vi phạm |
| Chở quá số người | Ảnh có xe máy và người ngồi trên xe | Hộp Bike/Rider/Trippling, số Rider được ghép và trạng thái vi phạm; không kết luận nếu thiếu thông tin ghép |
| Hàng cồng kềnh | Ảnh xe chở hàng, gồm xe máy hoặc phương tiện khác | Hộp Cargo được hiển thị trên toàn ảnh; sự kiện vi phạm chỉ được tạo khi có liên kết với xe máy |

Mỗi lượt upload tạo kết quả riêng. Dùng ảnh có sẵn của bạn; dataset không được
đóng gói trong repository. Kết quả phụ thuộc vào góc chụp, độ che khuất và chất
lượng model; Cargo v6 có độ tin cậy còn hạn chế, nên cần kiểm chứng bằng mắt.

## 9. Huấn luyện mô hình V02

Trước khi huấn luyện, cần chuẩn bị bộ dữ liệu và đảm bảo file cấu hình `dataset_v02_yolo/data.yaml` trỏ đúng đến các thư mục ảnh và nhãn.

Chạy huấn luyện ban đầu:

```bash
python -m training.train_v02
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
python -m training.resume_v02
```

Chỉ tiếp tục khi checkpoint tồn tại và còn đọc được. Không xóa checkpoint nếu cần khôi phục quá trình huấn luyện.

## 10. Dữ liệu và mô hình

Bộ dữ liệu huấn luyện có thể chứa nhiều hình ảnh, nhãn và file JSON. Do dung lượng lớn, các bộ dữ liệu đầy đủ không nhất thiết được lưu trong repository.

Các file dữ liệu cần được đặt đúng đường dẫn theo cấu hình của từng chương trình.

Không nên đưa các nội dung sau lên GitHub nếu không thực sự cần thiết:

- Môi trường ảo `.venv/`.
- Bộ dữ liệu lớn.
- Kết quả huấn luyện trong `runs/`.
- Các file checkpoint dung lượng lớn như `best.pt` và `last.pt`.

Trước khi chạy chương trình, cần kiểm tra rằng các file trọng số và dữ liệu bắt buộc đã có trên máy tính.

## 11. Kết quả và đánh giá

Các mô hình cần được đánh giá trên dữ liệu validation và test phù hợp.

Những chỉ số có thể sử dụng gồm:

- **Precision:** tỷ lệ dự đoán dương tính đúng trong các dự đoán dương tính.
- **Recall:** tỷ lệ đối tượng dương tính thực tế được mô hình phát hiện.
- **mAP50:** chỉ số trung bình độ chính xác phát hiện tại ngưỡng IoU 0.5.
- **mAP50-95:** chỉ số đánh giá trên nhiều ngưỡng IoU, khắt khe hơn.

Chỉ số đánh giá trên tập dữ liệu không đồng nghĩa với tỷ lệ phát hiện vi phạm chính xác trong mọi điều kiện giao thông thực tế.

Ảnh minh họa kết quả có thể được bổ sung vào README sau khi hoàn tất thử nghiệm.

## 12. Hướng phát triển

- Cải thiện độ chính xác của mô hình nhận diện.
- Kiểm tra trên nhiều điều kiện ánh sáng và góc quay.
- Kết hợp phát hiện đối tượng với theo dõi phương tiện.
- Tối ưu tốc độ xử lý hình ảnh và video.
- Bổ sung hình ảnh minh họa và kết quả đánh giá.
- Hoàn thiện quy trình kiểm thử và tài liệu sử dụng.

## 13. Tác giả và mục đích

Dự án được xây dựng phục vụ mục đích học tập, nghiên cứu và thực hành ứng dụng thị giác máy tính trong lĩnh vực giao thông.

Các kết quả nhận diện mang tính hỗ trợ phân tích hình ảnh, không thay thế việc xác minh của con người hoặc quy trình xử lý vi phạm chính thức.
