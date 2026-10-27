import os
import cv2
import customtkinter as ctk

from tkinter import filedialog, messagebox
from PIL import Image
from ultralytics import YOLO


# =========================
# 1. CẤU HÌNH GIAO DIỆN
# =========================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

APP_TITLE = "HỆ THỐNG NHẬN DIỆN VI PHẠM GIAO THÔNG"

MODEL_PATH = os.path.join(
    "runs",
    "detect",
    "v02_trippling",
    "weights",
    "best.pt"
)

RESULT_DIR = "gui_results"


class TrafficViolationApp(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title(APP_TITLE)
        self.geometry("1200x750")
        self.minsize(1000, 650)

        self.image_path = None
        self.result_path = None
        self.original_image = None
        self.result_image = None
        self.model = None

        self.create_widgets()

    # =========================
    # 2. TẠO GIAO DIỆN
    # =========================

    def create_widgets(self):

        # Tiêu đề
        title = ctk.CTkLabel(
            self,
            text="HỆ THỐNG NHẬN DIỆN VI PHẠM GIAO THÔNG",
            font=("Arial", 24, "bold")
        )
        title.pack(pady=(20, 5))

        subtitle = ctk.CTkLabel(
            self,
            text="Ứng dụng xử lý ảnh và mô hình YOLO",
            font=("Arial", 14)
        )
        subtitle.pack(pady=(0, 15))

        # Khu vực hiển thị ảnh
        image_frame = ctk.CTkFrame(self)
        image_frame.pack(
            fill="both",
            expand=True,
            padx=20,
            pady=10
        )

        image_frame.grid_columnconfigure(0, weight=1)
        image_frame.grid_columnconfigure(1, weight=1)
        image_frame.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            image_frame,
            text="ẢNH ĐẦU VÀO",
            font=("Arial", 16, "bold")
        ).grid(row=0, column=0, pady=10)

        ctk.CTkLabel(
            image_frame,
            text="KẾT QUẢ NHẬN DIỆN",
            font=("Arial", 16, "bold")
        ).grid(row=0, column=1, pady=10)

        self.input_label = ctk.CTkLabel(
            image_frame,
            text="Chưa chọn ảnh",
            font=("Arial", 14)
        )
        self.input_label.grid(
            row=1,
            column=0,
            padx=10,
            pady=10,
            sticky="nsew"
        )

        self.output_label = ctk.CTkLabel(
            image_frame,
            text="Chưa có kết quả",
            font=("Arial", 14)
        )
        self.output_label.grid(
            row=1,
            column=1,
            padx=10,
            pady=10,
            sticky="nsew"
        )

        # Khu vực nút chức năng
        button_frame = ctk.CTkFrame(self)
        button_frame.pack(fill="x", padx=20, pady=10)

        ctk.CTkButton(
            button_frame,
            text="1. Chọn ảnh",
            command=self.select_image,
            width=170,
            height=40
        ).pack(side="left", padx=10, pady=15)

        ctk.CTkButton(
            button_frame,
            text="2. Nhận diện",
            command=self.detect_image,
            width=170,
            height=40
        ).pack(side="left", padx=10, pady=15)

        ctk.CTkButton(
            button_frame,
            text="3. Lưu kết quả",
            command=self.save_result,
            width=170,
            height=40
        ).pack(side="left", padx=10, pady=15)

        ctk.CTkButton(
            button_frame,
            text="Xóa ảnh",
            command=self.clear_images,
            width=120,
            height=40,
            fg_color="#555555",
            hover_color="#444444"
        ).pack(side="left", padx=10, pady=15)

        # Trạng thái
        self.status_label = ctk.CTkLabel(
            self,
            text="Trạng thái: Sẵn sàng",
            font=("Arial", 13)
        )
        self.status_label.pack(pady=(0, 15))

    # =========================
    # 3. HIỂN THỊ ẢNH
    # =========================

    def show_image(self, image, label):

        if image is None:
            return

        # OpenCV dùng BGR, Pillow dùng RGB
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        pil_image = Image.fromarray(image)

        # Giới hạn kích thước ảnh hiển thị
        pil_image.thumbnail((500, 450))

        ctk_image = ctk.CTkImage(
            light_image=pil_image,
            dark_image=pil_image,
            size=pil_image.size
        )

        label.configure(image=ctk_image, text="")
        label.image = ctk_image

    # =========================
    # 4. CHỌN ẢNH ĐẦU VÀO
    # =========================

    def select_image(self):

        file_path = filedialog.askopenfilename(
            title="Chọn ảnh giao thông",
            filetypes=[
                ("File hình ảnh", "*.jpg *.jpeg *.png *.bmp"),
                ("Tất cả các file", "*.*")
            ]
        )

        if not file_path:
            return

        image = cv2.imread(file_path)

        if image is None:
            messagebox.showerror(
                "Lỗi",
                "Không thể đọc ảnh đã chọn."
            )
            return

        self.image_path = file_path
        self.original_image = image
        self.result_image = None
        self.result_path = None

        self.show_image(self.original_image, self.input_label)

        self.output_label.configure(
            image=None,
            text="Nhấn Nhận diện để xử lý ảnh"
        )

        self.status_label.configure(
            text=f"Đã chọn ảnh: {os.path.basename(file_path)}"
        )

    # =========================
    # 5. NHẬN DIỆN BẰNG YOLO
    # =========================

    def detect_image(self):

        if self.original_image is None:
            messagebox.showwarning(
                "Chưa có ảnh",
                "Vui lòng chọn ảnh trước khi nhận diện."
            )
            return

        if not os.path.isfile(MODEL_PATH):
            messagebox.showerror(
                "Không tìm thấy mô hình",
                "Không tìm thấy file best.pt tại:\n"
                + os.path.abspath(MODEL_PATH)
            )
            return

        try:
            self.status_label.configure(
                text="Đang tải mô hình và nhận diện..."
            )
            self.update_idletasks()

            # Chỉ tải mô hình một lần
            if self.model is None:
                self.model = YOLO(MODEL_PATH)

            # Nhận diện ảnh
            results = self.model.predict(
                source=self.original_image,
                conf=0.25,
                verbose=False
            )

            # Lấy ảnh đã vẽ khung nhận diện
            self.result_image = results[0].plot()

            # Hiển thị kết quả
            self.show_image(
                self.result_image,
                self.output_label
            )

            # Tạo thư mục lưu kết quả
            os.makedirs(RESULT_DIR, exist_ok=True)

            filename = os.path.splitext(
                os.path.basename(self.image_path)
            )[0]

            self.result_path = os.path.join(
                RESULT_DIR,
                filename + "_result.jpg"
            )

            cv2.imwrite(
                self.result_path,
                self.result_image
            )

            count = len(results[0].boxes)

            self.status_label.configure(
                text=f"Hoàn tất! Phát hiện {count} đối tượng."
            )

        except Exception as error:
            messagebox.showerror(
                "Lỗi nhận diện",
                str(error)
            )

            self.status_label.configure(
                text="Nhận diện thất bại."
            )

    # =========================
    # 6. LƯU KẾT QUẢ
    # =========================

    def save_result(self):

        if self.result_image is None:
            messagebox.showwarning(
                "Chưa có kết quả",
                "Bạn cần nhận diện ảnh trước."
            )
            return

        save_path = filedialog.asksaveasfilename(
            title="Lưu ảnh kết quả",
            defaultextension=".jpg",
            filetypes=[
                ("JPEG", "*.jpg"),
                ("PNG", "*.png")
            ],
            initialfile="ket_qua_nhan_dien.jpg"
        )

        if not save_path:
            return

        success = cv2.imwrite(
            save_path,
            self.result_image
        )

        if success:
            messagebox.showinfo(
                "Thành công",
                f"Đã lưu ảnh tại:\n{save_path}"
            )
        else:
            messagebox.showerror(
                "Lỗi",
                "Không thể lưu ảnh."
            )

    # =========================
    # 7. XÓA ẢNH
    # =========================

    def clear_images(self):

        self.image_path = None
        self.result_path = None
        self.original_image = None
        self.result_image = None

        self.input_label.configure(
            image=None,
            text="Chưa chọn ảnh"
        )

        self.output_label.configure(
            image=None,
            text="Chưa có kết quả"
        )

        self.status_label.configure(
            text="Trạng thái: Sẵn sàng"
        )


# =========================
# 8. KHỞI CHẠY ỨNG DỤNG
# =========================

if __name__ == "__main__":
    app = TrafficViolationApp()
    app.mainloop()
