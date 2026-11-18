# HỌC VIỆN HÀNG KHÔNG VIỆT NAM
## KHOA CÔNG NGHỆ THÔNG TIN

<br><br><br>

# BÁO CÁO
## XỬ LÝ ẢNH VÀ THỊ GIÁC MÁY TÍNH
### NHẬN DIỆN HÀNH VI VI PHẠM GIAO THÔNG

<br><br>

**Giảng viên hướng dẫn:** Hồ Văn Quí  
**Sinh viên/Nhóm sinh viên thực hiện:** Nhóm 02  
**Lớp:** 24ĐHTT01

<br><br><br><br>

**Thành phố Hồ Chí Minh, tháng 12 năm 2026**

---

## DANH SÁCH NHÓM

| STT | Họ và tên | MSSV | Lớp | Ghi chú |
|---:|---|---|---|---|
| 1 | Đỗ Trần Quang Anh | `[BỔ SUNG MSSV ĐẦY ĐỦ]` | 24ĐHTT01 | Nhóm trưởng |
| 2 | Trần Bảo An | 2431540029 | 24ĐHTT01 | Thành viên |
| 3 | Nguyễn Trần Trung Khoa | `[BỔ SUNG MSSV ĐẦY ĐỦ]` | 24ĐHTT01 | Thành viên |
| 4 | Đinh Quỳnh Anh | `[BỔ SUNG MSSV ĐẦY ĐỦ]` | 24ĐHTT01 | Thành viên |
| 5 | Phan Trung Hiếu | `[BỔ SUNG MSSV ĐẦY ĐỦ]` | 24ĐHTT01 | Thành viên |

> **Lưu ý trước khi nộp:** Các MSSV được che một phần trong thông tin đầu vào nên báo cáo không tự suy đoán các chữ số còn thiếu. Nhóm cần điền lại MSSV chính xác, kiểm tra tên giảng viên, lớp và tháng nộp theo yêu cầu học phần.

---

## DANH MỤC CÁC KÝ HIỆU, CHỮ VIẾT TẮT

| Ký hiệu | Chữ viết đầy đủ | Ý nghĩa trong báo cáo |
|---|---|---|
| AI | Artificial Intelligence | Trí tuệ nhân tạo |
| API | Application Programming Interface | Giao diện lập trình ứng dụng giữa trình duyệt và máy chủ |
| CNN | Convolutional Neural Network | Mạng nơ-ron tích chập |
| CUDA | Compute Unified Device Architecture | Nền tảng tính toán GPU của NVIDIA |
| CV | Computer Vision | Thị giác máy tính |
| FN | False Negative | Đối tượng có thật nhưng mô hình bỏ sót |
| FP | False Positive | Mô hình phát hiện nhầm đối tượng |
| GPU | Graphics Processing Unit | Bộ xử lý đồ họa |
| IoU | Intersection over Union | Tỷ lệ giao trên hợp giữa hai hộp giới hạn |
| mAP | mean Average Precision | Trung bình độ chính xác trung bình |
| MSSV | Mã số sinh viên | Mã định danh sinh viên |
| NMS | Non-Maximum Suppression | Lọc các hộp dự đoán trùng lặp |
| P/R | Precision/Recall | Độ chính xác dự đoán / độ bao phủ đối tượng |
| SQL | Structured Query Language | Ngôn ngữ truy vấn cơ sở dữ liệu |
| SQLite | SQLite Database | Cơ sở dữ liệu quan hệ gọn nhẹ, lưu trong một tệp |
| YOLO | You Only Look Once | Họ mô hình phát hiện đối tượng một giai đoạn |

---

## MỤC LỤC

1. [Mở đầu](#mở-đầu)
2. [Chương 1. Giới thiệu](#chương-1-giới-thiệu)
3. [Chương 2. Cơ sở lý thuyết](#chương-2-cơ-sở-lý-thuyết)
4. [Chương 3. Phân tích hệ thống và xây dựng sản phẩm](#chương-3-phân-tích-hệ-thống-và-xây-dựng-sản-phẩm)
5. [Kết luận và hướng phát triển](#kết-luận-và-hướng-phát-triển)
6. [Danh mục tài liệu tham khảo](#danh-mục-tài-liệu-tham-khảo)
7. [Phụ lục](#phụ-lục)

> Khi đưa nội dung sang Microsoft Word, nên tạo mục lục tự động từ các Heading để cập nhật số trang và tên chương.

## DANH MỤC BẢNG, HÌNH VẼ VÀ ĐỒ THỊ

### Danh mục bảng

| Số hiệu | Tên bảng |
|---|---|
| Bảng 3.1 | Thống kê số ảnh, cảnh nguồn và hộp nhãn của dataset cargo |
| Bảng 3.2 | Kết quả đánh giá model cargo trên validation và test |

### Danh mục hình

| Số hiệu | Tên hình dự kiến |
|---|---|
| Hình 3.1 | Sơ đồ kiến trúc tổng quát của hệ thống |
| Hình 3.2 | Giao diện trang chủ và tải ảnh |
| Hình 3.3 | Kết quả nhận diện ảnh sau khi xử lý |
| Hình 3.4 | Giao diện so sánh hiệu năng mô hình |
| Hình 3.5 | Một số dự đoán cargo trên tập test |

> Hình 3.2–3.5 cần được nhóm chèn ảnh chụp thực tế từ ứng dụng và thư mục dự đoán trước khi nộp. Không sử dụng ảnh minh họa không lấy từ sản phẩm.

---

# MỞ ĐẦU

Sự phát triển của camera, thiết bị di động và các phương pháp học sâu tạo điều kiện xây dựng những công cụ hỗ trợ phân tích hình ảnh giao thông. Trong phạm vi học phần Xử lý ảnh và Thị giác máy tính, nhóm thực hiện đề tài “Nhận diện hành vi vi phạm giao thông” nhằm khảo sát cách mô hình phát hiện đối tượng có thể hỗ trợ nhận diện một số tình huống liên quan đến xe máy.

Sản phẩm hiện tại là một ứng dụng web chạy bằng Flask. Người dùng tải ảnh lên; chương trình sử dụng các checkpoint YOLO đã huấn luyện để phát hiện mũ bảo hiểm, xe máy, người đi xe và hàng hóa cồng kềnh. Sau khi mô hình trả về các hộp và lớp dự đoán, pipeline xử lý kết hợp một số kết quả theo vị trí hình học, vẽ chú thích lên ảnh và trình bày thống kê. Ứng dụng cũng lưu ảnh đầu vào, ảnh đầu ra và lịch sử xử lý trên máy cục bộ.

Đề tài tập trung vào việc xây dựng một nguyên mẫu phục vụ học tập và thử nghiệm, không thay thế lực lượng chức năng và không tự động đưa ra kết luận xử phạt. Độ tin cậy phụ thuộc vào dữ liệu, chất lượng nhãn, góc chụp, độ phân giải, vật thể bị che khuất và khả năng phát hiện của từng mô hình.

Trong quá trình kiểm thử chức năng hàng hóa cồng kềnh, nhóm nhận thấy mô hình có thể dự đoán nhầm một số vùng không phải hàng hóa, chẳng hạn tán cây. Nhóm đã chuẩn bị phiên bản dữ liệu được gắn nhãn lại, huấn luyện lại YOLO11n và cập nhật pipeline để chỉ ghi nhận hàng hóa khi phát hiện đó liên kết được với một xe máy đã phát hiện. Kết quả hiện tại vẫn còn hạn chế do số cảnh độc lập trong validation và test ít; báo cáo trình bày rõ kết quả thực nghiệm và các giới hạn này.

---

# CHƯƠNG 1. GIỚI THIỆU

## 1.1. Lý do chọn đề tài

Giao thông đường bộ là môi trường có nhiều đối tượng chuyển động, thay đổi nhanh về vị trí, kích thước và mức độ che khuất. Việc quan sát thủ công từng hình ảnh hoặc video có thể tốn thời gian, đặc biệt khi cần chú ý đồng thời đến người tham gia giao thông, xe máy, mũ bảo hiểm và số người trên xe. Một số tình huống như xe chở nhiều người hoặc chở hàng có kích thước lớn cũng khó đánh giá chỉ bằng một dấu hiệu đơn lẻ.

Thị giác máy tính cung cấp phương pháp để máy tính trích xuất thông tin từ ảnh và video. Các mô hình phát hiện đối tượng có thể dự đoán vị trí và loại đối tượng, từ đó tạo dữ liệu đầu vào cho các bước xử lý tiếp theo. Việc xây dựng một hệ thống thử nghiệm giúp nhóm vận dụng kiến thức xử lý ảnh, học sâu, thiết kế ứng dụng web và kiểm thử phần mềm vào một bài toán gần với thực tế.

Đề tài được lựa chọn nhằm nghiên cứu khả năng xây dựng công cụ hỗ trợ quan sát một số tình huống giao thông bằng ảnh. Lợi ích mong muốn là tạo một nguyên mẫu có thể trình bày kết quả trực quan, ghi lại các lần xử lý và hỗ trợ so sánh mô hình trong phạm vi học tập. Đề tài không khẳng định mô hình có thể thay thế việc đánh giá của con người hoặc cơ quan có thẩm quyền.

## 1.2. Mục tiêu đề tài

### 1.2.1. Mục tiêu tổng quát

Xây dựng một ứng dụng web nguyên mẫu sử dụng YOLO và các kỹ thuật xử lý ảnh để phân tích ảnh giao thông, hiển thị một số đối tượng và tình huống liên quan đến xe máy, đồng thời lưu trữ kết quả để phục vụ kiểm tra.

### 1.2.2. Mục tiêu cụ thể

1. Sử dụng mô hình phát hiện các lớp `helmet` và `no_helmet`.
2. Sử dụng mô hình phát hiện xe máy, người đi xe và lớp `Trippling`; kết hợp kết quả để ước lượng số người gắn với xe.
3. Thử nghiệm mô hình phát hiện hàng hóa cồng kềnh trên xe máy.
4. Xây dựng pipeline dùng OpenCV để đọc ảnh, vẽ hộp phát hiện và thông tin vi phạm.
5. Xây dựng trang web Flask nhận ảnh, trả về ảnh đã chú thích và thống kê.
6. Lưu lịch sử phân tích và kết quả đánh giá mô hình bằng SQLite.
7. Cung cấp trang so sánh các checkpoint theo cùng tác vụ và tập dữ liệu.
8. Đánh giá mô hình cargo bằng Precision, Recall, mAP50 và mAP50–95, đồng thời phân tích lỗi và nêu giới hạn của tập dữ liệu.

## 1.3. Phạm vi đề tài

### 1.3.1. Phạm vi chức năng

Phần ứng dụng web tập trung vào xử lý ảnh tĩnh và ba nhóm chức năng: mũ bảo hiểm, xe máy/chở quá số người và hàng hóa cồng kềnh. Dự án có các chương trình thử nghiệm video, tracking và nhận diện biển số riêng; các chức năng này không được xem là đã tích hợp hoàn chỉnh vào luồng web chính trong báo cáo này.

Phần so sánh mô hình cho phép chọn tác vụ, checkpoint sẵn sàng, split dữ liệu, ngưỡng confidence và số lượt đo. Các metric được tính trên dataset đã cấu hình cho từng tác vụ; không so sánh trực tiếp mAP giữa các tác vụ khác nhau.

### 1.3.2. Phạm vi dữ liệu

Ba bộ dữ liệu cấu hình trong dự án nằm tại:

- Mũ bảo hiểm: `dataset/`, có 446 ảnh train, 127 ảnh validation và 65 ảnh test.
- V02: `dataset_v02_yolo/`, có 4.644 ảnh train, 117 ảnh validation và 78 ảnh test.
- Hàng cồng kềnh: dữ liệu nguồn mới trong `dataset_cargo/roboflow-v5/` được kiểm tra và sao chép thành bản làm việc `dataset_cargo/roboflow-v5-prepared/`, gồm 63 ảnh train, 18 ảnh validation và 9 ảnh test.

Các con số trên là số ảnh theo thư mục cấu hình tại thời điểm kiểm tra; chúng không thể hiện số cảnh độc lập. Đối với cargo, 90 ảnh nguồn gồm ba phiên bản augmentation của 30 cảnh nguồn, nên bản chia theo nhóm có 21 cảnh train, 6 cảnh validation và 3 cảnh test.

### 1.3.3. Phạm vi triển khai

Website mặc định chạy trên máy cục bộ tại `127.0.0.1`. Ảnh tải lên được giới hạn tối đa 12 MB. Đây là môi trường demo, chưa được thiết kế hay kiểm định như một dịch vụ xử lý vi phạm giao thông triển khai diện rộng.

## 1.4. Đối tượng nghiên cứu

Đối tượng nghiên cứu của đề tài gồm:

- Ảnh giao thông có xe máy, người tham gia giao thông, mũ bảo hiểm và hàng hóa.
- Các lớp đối tượng trong dataset mũ bảo hiểm: `helmet`, `no_helmet`.
- Các lớp trong dataset V02: `Bike`, `Helmet_On`, `No_helmet`, `Rider`, `Trippling`.
- Lớp hàng cồng kềnh `xe_cong_kenh`, được chuẩn hóa tên hiển thị thành `bulky_cargo` trong pipeline.
- Mô hình YOLO11n và các checkpoint YOLOv8s, YOLO26n được dùng trong thử nghiệm/so sánh.
- Quy tắc liên kết các hộp người, hàng hóa và xe dựa trên tọa độ hình học.
- Ứng dụng Flask, lưu lịch sử SQLite và giao diện hiển thị ảnh kết quả.

## 1.5. Phương pháp nghiên cứu

### 1.5.1. Phương pháp thu thập thông tin

Nhóm khảo sát cấu trúc mã nguồn và cấu hình dataset, đọc tài liệu mô hình YOLO, thư viện Flask, PyTorch và OpenCV. Dataset cargo mới được xác định theo metadata trong `data.yaml` và README xuất từ Roboflow thay vì chỉ dựa vào tên thư mục. README ghi đây là version 2 của project Motorcycle Bulky Cargo Detection, xuất ngày 28/10/2026, giấy phép CC BY 4.0.

### 1.5.2. Phương pháp xử lý thông tin

Nhóm kiểm tra số lượng ảnh, nhãn, class ID, số trường mỗi dòng nhãn, phạm vi tọa độ chuẩn hóa, ảnh không có nhãn, nhãn không có ảnh và ảnh trùng byte. Các ảnh augment có chung tiền tố nguồn trước `.rf.` được gom thành nhóm để chia tập, tránh đưa các bản gần giống nhau sang nhiều split.

### 1.5.3. Phương pháp thực nghiệm

YOLO11n pretrained được fine-tune trên tập train. Trong quá trình huấn luyện, validation được dùng để theo dõi các metric và chọn checkpoint. Sau khi huấn luyện, `best.pt` được đánh giá trên validation và test. Nhóm cũng lưu ảnh dự đoán trên test để quan sát hộp dự đoán so với nhãn.

### 1.5.4. Phương pháp kiểm thử phần mềm

Các kiểm thử tự động dùng `unittest` để kiểm tra xử lý pipeline, định tuyến web, kiểm tra đầu vào, lịch sử và các quy tắc gán đối tượng. Lần chạy kiểm thử gần nhất trong môi trường dự án có 43 test đạt. Ngoài ra, nhóm chạy trực tiếp pipeline trên ảnh đã gây báo nhầm tán cây để kiểm tra thay đổi lọc cargo không gán được cho xe.

## 1.6. Bố cục đề tài

Chương 1 giới thiệu lý do chọn đề tài, mục tiêu, phạm vi, đối tượng và phương pháp nghiên cứu. Chương 2 trình bày các kiến thức về thị giác máy tính, phát hiện đối tượng, YOLO, dữ liệu gán nhãn, các chỉ số đánh giá và nền tảng xây dựng ứng dụng. Chương 3 phân tích yêu cầu, mô tả kiến trúc, quá trình chuẩn bị dữ liệu, huấn luyện, xây dựng website và kiểm thử thực nghiệm. Phần kết luận tổng hợp kết quả đạt được, hạn chế và các hướng phát triển tiếp theo.

---

# CHƯƠNG 2. CƠ SỞ LÝ THUYẾT

## 2.1. Thị giác máy tính và phát hiện đối tượng

Thị giác máy tính là lĩnh vực nghiên cứu cách máy tính tiếp nhận và phân tích dữ liệu hình ảnh hoặc video. Trong bài toán phát hiện đối tượng, hệ thống cần xác định đồng thời loại đối tượng và vị trí của đối tượng trong ảnh. Kết quả thường gồm một hộp giới hạn, nhãn lớp và điểm confidence.

Khác với phân loại ảnh, vốn chỉ gán một hay nhiều nhãn cho toàn ảnh, phát hiện đối tượng có thể trả về nhiều vùng khác nhau. Trong ảnh giao thông, một mô hình có thể phát hiện nhiều xe máy, người hoặc vật thể trong cùng một khung hình. Hệ thống sau đó cần xử lý các quan hệ giữa các hộp, chẳng hạn người nào thuộc về xe nào.

## 2.2. Mạng nơ-ron tích chập

Mạng nơ-ron tích chập (CNN) là một dạng mạng học sâu thường được dùng cho dữ liệu ảnh. Các lớp tích chập học đặc trưng từ đơn giản đến phức tạp: cạnh và kết cấu ở các tầng đầu, đến hình dạng hoặc bộ phận đối tượng ở các tầng sâu hơn. Các đặc trưng này được dùng để dự đoán lớp và vị trí.

Trong dự án, nhóm không tự thiết kế và huấn luyện một kiến trúc CNN từ đầu. Thay vào đó, nhóm dùng mô hình YOLO pretrained, sau đó fine-tune trên dataset giao thông đã gán nhãn để điều chỉnh mô hình cho các lớp cần nhận diện.

## 2.3. Mô hình YOLO11n

YOLO thuộc nhóm mô hình phát hiện đối tượng một giai đoạn: mô hình xử lý ảnh và tạo dự đoán hộp/lớp trong một luồng suy luận, hướng tới cân bằng giữa tốc độ và khả năng phát hiện. YOLO11 là một thế hệ mô hình của Ultralytics, hỗ trợ nhiều kích thước và tác vụ; biến thể `n` là cấu hình nhỏ gọn được sử dụng trong đề tài [1], [2].

Nhóm sử dụng `yolo11n.pt` làm trọng số pretrained cho lần huấn luyện cargo mới. Trọng số pretrained là điểm khởi đầu; nó không phải bộ dữ liệu và cũng không mặc nhiên nhận diện đúng các khái niệm vi phạm của đề tài. Fine-tune sử dụng ảnh và nhãn của tác vụ để cập nhật trọng số.

Trong Ultralytics, các kết quả huấn luyện thường được lưu dưới dạng checkpoint. `best.pt` là checkpoint được chọn theo kết quả validation tốt nhất trong run; `last.pt` là checkpoint được lưu gần cuối quá trình chạy. Dự án lưu kết quả cargo mới trong `runs/detect/cargo_protrusion_relabel_v1/`.

## 2.4. Dataset và định dạng nhãn YOLO

Dataset cho phát hiện đối tượng gồm ảnh và file nhãn tương ứng. Mỗi dòng nhãn detection trong định dạng YOLO có năm trường:

```text
class_id center_x center_y width height
```

Trong đó tọa độ tâm, chiều rộng và chiều cao thường được chuẩn hóa về khoảng `[0, 1]` theo kích thước ảnh. Class ID là số nguyên ánh xạ sang tên lớp được khai báo trong `data.yaml`.

Dataset thường được chia thành:

- **Train:** dùng cập nhật trọng số mô hình.
- **Validation:** dùng theo dõi quá trình huấn luyện và hỗ trợ chọn mô hình/cấu hình.
- **Test:** dành để ước lượng cuối cùng trên dữ liệu chưa dùng trong quá trình học hoặc chọn mô hình.

Nếu nhiều ảnh là các bản augmentation của cùng một cảnh, các ảnh này không nên bị tách ngẫu nhiên sang những split khác nhau. Nếu không, mô hình có thể nhìn thấy ảnh gần giống trong train và test, làm kết quả đánh giá lạc quan hơn khả năng tổng quát hóa.

Tệp `data.yaml` khai báo đường dẫn dataset, split, số lớp và tên lớp. Trong bản làm việc cargo của đề tài, cấu hình trỏ đến các thư mục train/valid/test bên trong `roboflow-v5-prepared` và khai báo class ID `0` là `xe_cong_kenh`.

## 2.5. Các chỉ số đánh giá

### 2.5.1. Intersection over Union

IoU đo mức chồng lấp giữa hộp dự đoán \(B_p\) và hộp nhãn chuẩn \(B_g\):

\[
IoU = \frac{|B_p \cap B_g|}{|B_p \cup B_g|}
\]

IoU càng cao thì hộp dự đoán càng khớp với hộp chuẩn. Khi liệt kê FP/FN trong script cargo của dự án, một dự đoán được ghép với nhãn nếu IoU đạt ngưỡng 0,5 và mỗi hộp chỉ được ghép tối đa một lần.

### 2.5.2. Precision và Recall

\[
Precision = \frac{TP}{TP + FP}
\]

\[
Recall = \frac{TP}{TP + FN}
\]

Precision phản ánh tỷ lệ dự đoán đúng trong số các dự đoán mà mô hình đưa ra. Recall phản ánh tỷ lệ đối tượng có nhãn mà mô hình tìm được. Tăng ngưỡng confidence thường làm giảm một số dự đoán yếu, nhưng có thể khiến mô hình bỏ sót thêm đối tượng.

### 2.5.3. Average Precision và mAP

Average Precision (AP) tổng hợp Precision–Recall trên nhiều ngưỡng confidence. mAP là trung bình AP theo lớp; `mAP50` dùng ngưỡng IoU 0,50, còn `mAP50–95` trung bình qua nhiều ngưỡng IoU từ 0,50 đến 0,95. mAP không đồng nghĩa với “độ chính xác tổng quát” của toàn hệ thống; giá trị phụ thuộc dataset, quy tắc nhãn và cấu hình đánh giá.

Trong báo cáo này, metric Ultralytics trên validation/test được tính theo chế độ validation với confidence thấp để xây dựng đường Precision–Recall. Script cũng liệt kê FP/FN riêng tại confidence 0,25 và IoU 0,5. Vì đây là hai phép tính khác nhau, các giá trị Precision/Recall của chúng không nhất thiết trùng nhau.

## 2.6. Liên kết đối tượng bằng hình học

Một mô hình phát hiện người và một mô hình phát hiện xe không tự cung cấp quan hệ “người này ngồi trên xe nào”. Pipeline của dự án dùng tâm hộp của người/hàng để kiểm tra xem tâm đó có nằm trong một vùng mở rộng quanh hộp xe hay không. Nếu thuộc nhiều xe, hệ thống chọn xe có điểm khoảng cách chuẩn hóa nhỏ hơn.

Đây là quy tắc hình học đơn giản, không phải theo dõi quan hệ vật lý chính xác. Xe sát nhau, hộp bị lệch, người bị che hoặc ảnh có phối cảnh mạnh có thể khiến gán nhầm hoặc không gán được. Trong video, YOLO tracking và ByteTrack được dùng cho nhánh theo dõi; trong xử lý ảnh tải lên, chương trình chạy nhánh ảnh tĩnh.

## 2.7. Flask, OpenCV, PyTorch và SQLite

Flask là framework web Python cung cấp định tuyến, tiếp nhận request và render template [3]. Trong dự án, Flask nhận ảnh tải lên, gọi pipeline, trả trang kết quả và cung cấp API cho trang so sánh model.

OpenCV hỗ trợ đọc, chuyển đổi và vẽ khung lên ảnh. PyTorch là nền tảng tensor và học sâu mà Ultralytics sử dụng trong môi trường triển khai; CUDA được dùng khi GPU NVIDIA khả dụng [4], [5].

SQLite là cơ sở dữ liệu quan hệ nằm trong một tệp. Dự án dùng SQLite để lưu lịch sử phân tích và kết quả so sánh, không dùng SQLite làm dữ liệu huấn luyện. Ảnh/nhãn huấn luyện nằm trong các thư mục dataset; kiến thức mô hình sau huấn luyện nằm trong checkpoint `.pt`.

---

# CHƯƠNG 3. PHÂN TÍCH HỆ THỐNG VÀ XÂY DỰNG SẢN PHẨM

## 3.1. Phân tích hệ thống

### 3.1.1. Người dùng và chức năng chính

Người dùng chính của nguyên mẫu là sinh viên/giảng viên cần thử nghiệm mô hình trên ảnh giao thông. Luồng sử dụng gồm tải ảnh lên, chờ pipeline xử lý, xem ảnh đầu vào và ảnh đã chú thích, xem thống kê, tải ảnh kết quả hoặc mở lại lịch sử.

Trang so sánh AI hỗ trợ chọn tác vụ, split, confidence, số lần đo và các checkpoint đã sẵn sàng. Backend đánh giá model ở tiến trình nền và lưu kết quả để giao diện hiển thị biểu đồ, bảng và lịch sử.

### 3.1.2. Yêu cầu chức năng

1. Nhận ảnh hợp lệ và giới hạn dung lượng tải lên.
2. Nạp checkpoint theo cấu hình của từng tác vụ.
3. Phát hiện hộp/lớp bằng YOLO.
4. Kết hợp các dự đoán liên quan đến xe máy bằng quy tắc hình học.
5. Vẽ kết quả và trả thống kê cho người dùng.
6. Lưu ảnh đầu vào, ảnh đầu ra và bản ghi lịch sử.
7. Cho phép xem hoặc xóa một bản ghi lịch sử.
8. Kiểm tra yêu cầu đánh giá model và từ chối tác vụ/split/model không hợp lệ.
9. Ghi các metric đánh giá vào SQLite.

### 3.1.3. Yêu cầu phi chức năng và giới hạn vận hành

Ứng dụng cần thông báo lỗi đọc ảnh hoặc lỗi suy luận thay vì trả kết quả thành công giả. Tác vụ so sánh model được tuần tự hóa, tránh đồng thời khởi chạy nhiều lượt đánh giá trong cùng ứng dụng. Website mặc định chỉ lắng nghe `127.0.0.1`; đây là cấu hình thuận tiện cho demo trên máy phát triển, không phải cấu hình triển khai Internet.

Ảnh tải lên và ảnh kết quả được lưu trên máy chủ cục bộ để phục vụ lịch sử. Vì ảnh giao thông có thể chứa người và phương tiện, người vận hành cần cân nhắc quyền riêng tư, thời hạn lưu trữ và việc xóa dữ liệu khi không còn cần thiết.

## 3.2. Kiến trúc hệ thống

```text
Trình duyệt
    │  Ảnh tải lên / lựa chọn đánh giá
    ▼
Flask (web/app.py)
    ├── Nhận ảnh, kiểm tra dung lượng/định dạng
    ├── Gọi ViolationPipeline
    │       ├── YOLO mũ bảo hiểm
    │       ├── YOLO xe máy/người (V02)
    │       ├── YOLO hàng cồng kềnh
    │       ├── Liên kết hộp theo vị trí
    │       └── OpenCV vẽ kết quả
    ├── Lưu ảnh và lịch sử phân tích
    └── API so sánh model
            ├── Ultralytics validation/predict
            └── Lưu ai_model_evaluations
                    │
                    ▼
                 SQLite
```

**Hình 3.1. Sơ đồ kiến trúc tổng quát của hệ thống**

Các thành phần xử lý chính nằm trong `web/app.py`, `web/model_comparison.py` và `core/violation_pipeline.py`. Các checkpoint được lưu dưới `runs/detect/.../weights/`; dataset và nhãn lưu trong thư mục `dataset.../`.

## 3.3. Chuẩn bị và kiểm tra dữ liệu cargo

### 3.3.1. Xác định phiên bản dataset

Dataset mới trong `dataset_cargo/roboflow-v5/` được xác định theo metadata, không chỉ theo tên thư mục. `data.yaml` cho biết project `motorcycle-bulky-cargo-detection`, version 2, license CC BY 4.0 và URL project [6]. README ghi bản export có 90 ảnh và ba phiên bản augmentation cho mỗi ảnh gốc. Vì vậy tên thư mục `roboflow-v5` là tên lưu trữ cục bộ; version metadata của Roboflow là 2.

### 3.3.2. Kiểm tra cấu trúc và nhãn

Bản export mới chứa 90 ảnh trong `train/images` và 90 nhãn ghép theo tên file trong `train/labels`. Kiểm tra cho thấy:

- Tất cả ảnh có nhãn cùng stem; không có nhãn rỗng hoặc nhãn mồ côi.
- Mỗi dòng nhãn có năm trường detection YOLO.
- Tất cả class ID là 0 và khớp lớp `xe_cong_kenh`.
- Tọa độ nằm trong miền chuẩn hóa hợp lệ; chiều rộng và chiều cao hộp dương.
- Tổng số hộp gắn nhãn là 105.
- Không có ảnh trùng byte trong bản export.
- Dữ liệu là bounding box detection, không phải segmentation.

Quan sát trực quan các nhãn cho thấy các hộp được gắn quanh phần hàng hóa trong nhiều ảnh, thay vì bao toàn bộ xe và người như quy ước nhãn cũ. Việc kiểm tra trực quan giúp phát hiện lỗi phạm vi hộp, nhưng không thay thế việc rà soát thủ công từng annotation.

### 3.3.3. Sửa đường dẫn và chia train/validation/test

`data.yaml` nguồn trỏ tới `../train/images`, `../valid/images` và `../test/images` nằm ngoài thư mục export mới. Nhóm không sửa dữ liệu nguồn. Thay vào đó, nhóm tạo bản làm việc riêng `dataset_cargo/roboflow-v5-prepared`, sao chép các cặp ảnh/nhãn vào cấu trúc nhất quán và tạo `data.yaml` với đường dẫn tuyệt đối đến bản làm việc.

Ba biến thể augmentation có chung tiền tố tên nguồn được gom thành một nhóm. Dùng seed 42, 30 nhóm nguồn được chia theo tỷ lệ 70/20/10:

**Bảng 3.1. Thống kê split dataset hàng cồng kềnh**

| Split | Số ảnh | Số cảnh nguồn | Số hộp gắn nhãn |
|---|---:|---:|---:|
| Train | 63 | 21 | 75 |
| Validation | 18 | 6 | 21 |
| Test | 9 | 3 | 9 |
| **Tổng** | **90** | **30** | **105** |

Không có nhóm augmentation hoặc ảnh trùng byte đi qua nhiều split. Manifest `split_manifest.csv` lưu nhóm nguồn, split, tên ảnh/nhãn và SHA-256 để có thể kiểm tra lại.

### 3.3.4. Giới hạn dataset

Dataset hiện tại chỉ có 30 cảnh nguồn độc lập. Test chỉ gồm ba cảnh, vì vậy một vài dự đoán sai có thể làm metric thay đổi đáng kể. Ngoài ra, ảnh đều có đối tượng hàng hóa dương tính; dataset chưa có đủ ảnh xe máy không chở hàng để định lượng tỷ lệ báo nhầm trên tình huống âm tính. Những hạn chế này làm cho metric chưa thể đại diện cho điều kiện giao thông thực tế.

## 3.4. Huấn luyện mô hình hàng cồng kềnh

### 3.4.1. Cấu hình huấn luyện

Nhóm khởi tạo từ `yolo11n.pt` pretrained, không tiếp tục checkpoint cũ vì quy ước nhãn đã thay đổi. Môi trường kiểm tra gồm Python 3.13.14, Ultralytics 8.4.153, PyTorch 2.14.1+cu126 và GPU NVIDIA GeForce RTX 3050 Laptop GPU với 4 GB VRAM.

Lệnh huấn luyện thực tế:

```powershell
.\.venv\Scripts\python.exe -m detection.cargo.train_cargo --data .\dataset_cargo\roboflow-v5-prepared\data.yaml --weights .\yolo11n.pt --epochs 50 --imgsz 640 --batch 2 --workers 2 --device 0 --name cargo_protrusion_relabel_v1
```

Batch 2 được chọn để phù hợp dung lượng GPU 4 GB. Số epoch tối đa là 50; cơ chế early stopping kết thúc ở epoch 39 do không cải thiện trong khoảng patience cấu hình. Epoch 24 là epoch có mAP50–95 validation cao nhất trong run. `best.pt` và `last.pt` đều được lưu trong thư mục run mới; các checkpoint cũ không bị ghi đè.

### 3.4.2. Kết quả đánh giá

Kết quả validation và test do script đánh giá cargo lưu lại:

**Bảng 3.2. Kết quả đánh giá YOLO11n cargo**

| Tập đánh giá | Ảnh | Precision | Recall | mAP50 | mAP50–95 |
|---|---:|---:|---:|---:|---:|
| Validation | 18 | 0,668 | 0,571 | 0,613 | 0,247 |
| Test | 9 | 1,000 | 0,427 | 0,504 | 0,195 |

Các metric ở bảng trên là metric Ultralytics, được tính ở cấu hình validation confidence thấp (`conf=0.001`) để tổng hợp đường Precision–Recall. Khi script liệt kê lỗi ở confidence 0,25 và IoU 0,5, kết quả test là 3 TP, 6 FP và 6 FN; precision/recall của phép ghép cố định này lần lượt là 0,333 và 0,333. Sự khác nhau giữa các con số là do cách tính và ngưỡng khác nhau, không phải một bộ metric duy nhất.

Kết quả test có Precision Ultralytics bằng 1,000 nhưng Recall chỉ 0,427 và mAP50–95 là 0,195. Không nên diễn giải Precision 1,000 như “mô hình chính xác 100%”: test có 9 ảnh và chỉ 3 cảnh độc lập; ở ngưỡng liệt kê FP/FN 0,25 vẫn xuất hiện sáu dự đoán nhầm và sáu đối tượng bị bỏ sót. Các kết quả này chỉ là đánh giá sơ bộ trên dataset hiện có.

### 3.4.3. Sản phẩm huấn luyện và báo cáo

- Checkpoint tốt nhất: `runs/detect/cargo_protrusion_relabel_v1/weights/best.pt`.
- Checkpoint cuối: `runs/detect/cargo_protrusion_relabel_v1/weights/last.pt`.
- Kết quả theo epoch: `runs/detect/cargo_protrusion_relabel_v1/results.csv`.
- Báo cáo validation: `runs/detect/cargo_protrusion_relabel_v1_validation/cargo_val_report.json`.
- Báo cáo test: `runs/detect/cargo_protrusion_relabel_v1_test/cargo_test_report.json`.
- Danh sách FP/FN: các tệp `cargo_false_positives_and_misses.csv` trong thư mục đánh giá tương ứng.
- Ảnh dự đoán test: `runs/detect/cargo_protrusion_relabel_v1_predictions/`.

*[Chèn Hình 3.5: một số ảnh dự đoán cargo trên tập test, thể hiện hộp dự đoán và nhãn chuẩn.]*

## 3.5. Pipeline nhận diện và luật nghiệp vụ

### 3.5.1. Nhận diện ảnh

Khi nhận ảnh, pipeline nạp các checkpoint mũ bảo hiểm và V02; checkpoint cargo được nạp nếu tồn tại. Mô hình YOLO trả về class, confidence và tọa độ từng hộp. Với ảnh lớn, pipeline có thể bổ sung các vùng ảnh chồng lấp để tăng cơ hội phát hiện vật thể nhỏ; các hộp trùng cùng lớp được gộp.

### 3.5.2. Nhận diện tình huống

- **Mũ bảo hiểm:** model phát hiện `helmet` và `no_helmet`. Dự đoán `no_helmet` cần được liên kết với xe bằng quy tắc vị trí trước khi tính là vi phạm; dự đoán chưa gán được được hiển thị phân biệt.
- **Chở quá số người:** model V02 phát hiện `bike`, `rider` và `trippling`. Người được gán vào xe dựa trên tâm hộp và vùng mở rộng quanh xe. Pipeline đánh dấu quá số người nếu số rider gán được lớn hơn hai hoặc có dự đoán `trippling` được gán vào xe.
- **Hàng cồng kềnh:** cargo detector dự đoán hộp `xe_cong_kenh`, được chuẩn hóa tên hiển thị thành `bulky_cargo`. Pipeline chỉ giữ cargo được liên kết với một xe đã phát hiện; cargo không gán được sẽ không được vẽ và không tạo sự kiện vi phạm.

Quy tắc lọc cargo không làm cho bộ phân loại tự hiểu cây hay hàng hóa. Quy tắc chỉ ngăn một hộp cargo không liên quan tới xe nào trở thành sự kiện. Nếu mô hình phát hiện xe bỏ sót xe thật, hàng trên xe đó cũng có thể bị lọc. Do vậy đây là biện pháp giảm báo nhầm ở tầng nghiệp vụ, không phải bằng chứng rằng cargo model đã đạt độ tin cậy cao.

## 3.6. Xây dựng giao diện sản phẩm

### 3.6.1. Trang chủ và tải ảnh

Trang chủ hiển thị trạng thái checkpoint cho từng chức năng, biểu mẫu tải ảnh và lịch sử gần đây. Máy chủ giới hạn ảnh 12 MB, chuyển ảnh về định dạng xử lý phù hợp, gọi pipeline và trả lại kết quả. Giao diện trình bày ảnh đầu vào, ảnh đã vẽ hộp, các chỉ số tóm tắt và danh sách sự kiện. Người dùng có thể tải ảnh kết quả.

*[Chèn Hình 3.2: ảnh chụp trang chủ và biểu mẫu tải ảnh.]*

*[Chèn Hình 3.3: ảnh đầu vào và kết quả nhận diện có hộp chú thích.]*

### 3.6.2. Lịch sử phân tích

Ảnh đầu vào và ảnh kết quả được lưu dưới `web_results/`. Bản ghi lịch sử trong `web_results/history.sqlite3` gồm tên ảnh, thời điểm, tên tệp ảnh và JSON tóm tắt. SQLite không lưu trọng số huấn luyện và không dùng để dạy model.

### 3.6.3. Trang so sánh hiệu năng AI

Trang `/ai-comparison` cho chọn bài toán, model, split, confidence và số lần đo. Backend kiểm tra model có checkpoint sẵn sàng, chạy validation và đo thời gian suy luận, sau đó lưu vào bảng `ai_model_evaluations` trong SQLite. Giao diện hiển thị precision, recall, mAP, latency, kích thước checkpoint, biểu đồ và lịch sử.

Tập test cargo hiện đã được dùng trong lần đánh giá cuối nên bị khóa khỏi các phép so sánh mới nhằm tránh xem cùng một test set như holdout chưa dùng. Các metric giữa các bài toán không được so trực tiếp vì class và dataset khác nhau.

*[Chèn Hình 3.4: trang so sánh hiệu năng mô hình và bảng/biểu đồ kết quả.]*

> **Ghi chú cập nhật giao diện:** Một số dòng mô tả model cargo trong bảng giao diện chính có thể còn ghi thông tin Roboflow v4 từ nội dung cũ. Khi hoàn thiện bản nộp, nhóm cần kiểm tra và cập nhật lại chữ hiển thị để thống nhất với checkpoint relabel Roboflow version 2 đang được cấu hình trong backend.

## 3.7. Kiểm thử và kết quả

### 3.7.1. Kiểm thử tự động

Lần chạy đầy đủ gần nhất bằng `unittest` có **43 kiểm thử đạt**. Các kiểm thử gồm:

- Kiểm tra các route chính của website và hiển thị trang so sánh.
- Kiểm tra yêu cầu đánh giá thiếu model, sai model hoặc dùng split không được phép.
- Kiểm tra cấu hình lịch sử và lưu kết quả đánh giá.
- Kiểm tra chuẩn hóa tên lớp cargo.
- Kiểm tra lựa chọn checkpoint cargo relabel mới và fallback model cũ.
- Kiểm tra gán người/hàng với xe.
- Kiểm tra cargo không gán được thì không tạo vi phạm.
- Kiểm tra cargo gán được với xe vẫn được giữ lại.
- Kiểm tra các tiện ích audit và chuẩn bị dataset cargo.

Kiểm thử tự động xác nhận các nhánh logic được kiểm tra hoạt động theo mong đợi; chúng không thay thế đánh giá bằng nhiều ảnh/video ngoài thực tế.

### 3.7.2. Kiểm tra ảnh cây bị báo nhầm

Nhóm dùng lại ảnh đã gây báo nhầm. Trước khi sửa pipeline, bản ghi lịch sử cho ảnh cho thấy 0 xe máy nhưng 1 sự kiện hàng cồng kềnh với confidence 0,3625. Sau khi thay đổi logic, chạy pipeline trực tiếp trên ảnh đó cho kết quả 0 xe máy, 0 cargo được ghi nhận và không có sự kiện cargo. Kiểm thử hồi quy cũng kiểm tra trường hợp cargo thực sự gán được vào xe vẫn tạo sự kiện.

Kết quả này chứng minh tình huống cụ thể không còn tạo vi phạm trong pipeline đã sửa. Nó không chứng minh mô hình cargo không còn nhầm cây trên các ảnh khác.

### 3.7.3. Đánh giá mức độ hoàn thành

Ở thời điểm lập báo cáo, các thành phần đã hoàn thành ở mức nguyên mẫu gồm luồng tải ảnh, gọi model, hiển thị kết quả, lịch sử SQLite, trang so sánh, huấn luyện checkpoint cargo mới và kiểm thử phần mềm. Phần chưa hoàn thiện đáng kể là độ tổng quát của mô hình cargo và đánh giá độc lập trên đủ nhiều cảnh, đặc biệt là ảnh âm tính không chở hàng.

Do chưa có tiêu chí nghiệm thu chính thức và bộ test đại diện đủ lớn, báo cáo không quy đổi tiến độ thành một tỷ lệ phần trăm mang tính đo lường khoa học. Mức đánh giá phù hợp là **đã hoàn thiện luồng demo chính nhưng đang ở giai đoạn thử nghiệm mô hình**, chưa sẵn sàng sử dụng làm căn cứ xử phạt.

## 3.8. Hạn chế và rủi ro

1. Dataset cargo nhỏ, chỉ có 30 cảnh nguồn; test chỉ có ba cảnh.
2. Không có đủ ảnh âm tính để đo false positive trên xe không chở hàng.
3. Metric test dao động mạnh và recall còn thấp; nhiều đối tượng cargo bị bỏ sót.
4. Gán người/hàng vào xe dựa trên vùng hình học, nên có thể nhầm khi xe sát nhau hoặc bị che.
5. Pipeline lọc cargo không gán được có thể giảm báo nhầm ngoài xe nhưng cũng có thể bỏ qua hàng thật khi model xe phát hiện thiếu.
6. Các ngưỡng confidence ảnh hưởng đến cân bằng precision/recall và chưa được đánh giá trên tập độc lập đủ lớn.
7. Nhận diện đối tượng không đồng nghĩa xác định đầy đủ căn cứ pháp lý của một vi phạm.
8. Website hiện là ứng dụng cục bộ phục vụ học tập; chưa có xác thực người dùng, quản trị truy cập hay triển khai dịch vụ production.
9. Ảnh tải lên được lưu phục vụ lịch sử nên cần quy định rõ thời hạn lưu, quyền truy cập và xóa dữ liệu.

---

# KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN

## Kết luận

Đề tài đã xây dựng được nguyên mẫu web xử lý ảnh giao thông bằng YOLO, Flask, OpenCV và SQLite. Ứng dụng có thể tiếp nhận ảnh, gọi các model theo từng tác vụ, kết hợp dự đoán bằng pipeline, vẽ kết quả, lưu lịch sử và cung cấp trang so sánh model.

Với chức năng hàng hóa cồng kềnh, nhóm đã xác định dataset Roboflow project version 2, kiểm tra định dạng 90 ảnh và 90 nhãn, tạo bản làm việc riêng chia theo 30 nhóm cảnh, rồi fine-tune YOLO11n từ pretrained checkpoint. Checkpoint mới được cấu hình làm model cargo ưu tiên. Nhóm cũng sửa pipeline để cargo không liên kết được với xe không trở thành sự kiện vi phạm.

Các mục tiêu xây dựng luồng demo, huấn luyện và kiểm thử cơ bản đã đạt được. Tuy nhiên, kết quả cargo còn hạn chế: trên test 9 ảnh/3 cảnh, mAP50–95 là 0,195 theo phép đánh giá Ultralytics và recall 0,427. Số cảnh độc lập quá ít để khẳng định khả năng tổng quát. Vì thế, sản phẩm hiện phù hợp cho mục đích trình diễn và nghiên cứu ban đầu; chưa phù hợp để tự động kết luận vi phạm hoặc thay thế con người.

Qua quá trình thực hiện, nhóm rút ra rằng chất lượng nhãn và cách chia dữ liệu có ảnh hưởng trực tiếp đến chất lượng mô hình; ảnh augmentation cùng nguồn cần được giữ cùng split; metric phải được giải thích theo ngưỡng và phương pháp tính; và cần phân biệt lỗi của model với lỗi trong logic hậu xử lý.

## Hướng phát triển

1. Thu thập thêm ảnh từ nhiều tuyến đường, thời điểm, điều kiện ánh sáng, loại xe và góc camera khác nhau.
2. Bổ sung ảnh âm tính: xe máy không chở hàng, cây, biển quảng cáo, mái che và các vật thể dễ gây nhầm.
3. Rà soát thủ công bounding box cargo, xây dựng hướng dẫn gắn nhãn nhất quán và kiểm tra chéo giữa các thành viên.
4. Tạo test set độc lập theo video/cảnh nguồn, tránh dùng lại khi điều chỉnh ngưỡng hoặc chọn checkpoint.
5. Cải thiện khả năng phát hiện xe máy trong ảnh đông người; đánh giá riêng độ chính xác của bước liên kết người/hàng với xe.
6. Thiết kế quy trình đánh giá nhiều lần và báo cáo khoảng bất định hoặc thống kê theo cảnh, không chỉ theo ảnh.
7. Tối ưu giao diện thông báo mức độ tin cậy và phân biệt dự đoán AI với kết luận do người dùng xác nhận.
8. Cập nhật các nhãn/mô tả model trong giao diện để khớp với checkpoint cargo relabel mới.
9. Bổ sung kiểm soát truy cập, chính sách lưu/xóa ảnh và cấu hình triển khai an toàn nếu đưa web ra khỏi máy cục bộ.
10. Chỉ xem xét thử nghiệm thực địa sau khi có dữ liệu đánh giá độc lập, kiểm tra sai số và quy trình con người xác nhận kết quả.

---

# DANH MỤC TÀI LIỆU THAM KHẢO

[1] J. Redmon, S. Divvala, R. Girshick và A. Farhadi, “You Only Look Once: Unified, Real-Time Object Detection,” *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016, DOI: 10.1109/CVPR.2016.91.

[2] Ultralytics, “YOLO11,” tài liệu chính thức, https://docs.ultralytics.com/models/yolo11/; và “Model Training with Ultralytics YOLO,” https://docs.ultralytics.com/modes/train/. Truy cập ngày 28/10/2026.

[3] Pallets Projects, “Welcome to Flask,” tài liệu Flask, https://flask.palletsprojects.com/en/stable/. Truy cập ngày 28/10/2026.

[4] PyTorch, “PyTorch Documentation,” https://pytorch.org/docs/stable/. Truy cập ngày 28/10/2026.

[5] OpenCV, “OpenCV Documentation,” https://docs.opencv.org/. Truy cập ngày 28/10/2026.

[6] Roboflow Universe, “Motorcycle Bulky Cargo Detection, version 2,” xuất dữ liệu ngày 28/10/2026, giấy phép CC BY 4.0, https://universe.roboflow.com/tran-bao-an/motorcycle-bulky-cargo-detection/dataset/2.

[7] Ultralytics, “Model Validation with Ultralytics YOLO,” tài liệu chính thức, https://docs.ultralytics.com/modes/val/. Truy cập ngày 28/10/2026.

[8] 2431540029-code, “TrafficViolationDetection,” mã nguồn và tài liệu dự án, GitHub, https://github.com/2431540029-code/TrafficViolationDetection. Truy cập ngày 28/10/2026.

---

# PHỤ LỤC

## Phụ lục A. Cấu trúc thư mục liên quan

```text
TrafficViolationDetection/
├── core/
│   └── violation_pipeline.py
├── detection/
│   └── cargo/
│       ├── evaluate_cargo.py
│       ├── paths.py
│       └── train_cargo.py
├── dataset/
├── dataset_v02_yolo/
├── dataset_cargo/
│   ├── roboflow-v5/
│   └── roboflow-v5-prepared/
├── runs/
│   └── detect/
│       └── cargo_protrusion_relabel_v1/
│           └── weights/
│               ├── best.pt
│               └── last.pt
├── tests/
├── training/
├── web/
│   ├── app.py
│   ├── model_comparison.py
│   ├── templates/
│   └── static/
└── web_results/
    └── history.sqlite3
```

## Phụ lục B. Lệnh khởi chạy và kiểm thử

Khởi chạy website từ thư mục gốc dự án:

```powershell
.\.venv\Scripts\python.exe app.py
```

Sau đó mở `http://127.0.0.1:5000`.

Chạy toàn bộ kiểm thử:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
```

Huấn luyện lại cargo từ pretrained checkpoint cần dataset và tên run chưa tồn tại:

```powershell
.\.venv\Scripts\python.exe -m detection.cargo.train_cargo --data .\dataset_cargo\roboflow-v5-prepared\data.yaml --weights .\yolo11n.pt --epochs 50 --imgsz 640 --batch 2 --workers 2 --device 0 --name cargo_protrusion_relabel_v1
```

Lệnh trên chỉ ghi lại cấu hình đã thực hiện; không chạy lại nếu thư mục run đó đã tồn tại. Checkpoint và dataset lớn thường bị loại khỏi Git theo `.gitignore`, nên cần lưu trữ riêng khi chuyển máy.

## Phụ lục C. Những nội dung cần hoàn thiện trước khi nộp

- Điền MSSV đầy đủ cho các thành viên còn thiếu.
- Cập nhật mục lục và số trang tự động sau khi dàn trang trong Word.
- Chèn ảnh chụp giao diện trang chủ, kết quả nhận diện, trang so sánh và ảnh dự đoán test; đặt tên hình theo quy định của giảng viên.
- Xác nhận với giảng viên định dạng bìa, lề, cỡ chữ, giãn dòng và yêu cầu về phân công đóng góp của từng thành viên.
- Rà soát dòng mô tả cargo còn nhắc Roboflow v4 trong giao diện chính để khớp phiên bản model hiện hành.
- Bảo đảm mọi số liệu trong bản nộp khớp với báo cáo `cargo_val_report.json` và `cargo_test_report.json` đi kèm.
