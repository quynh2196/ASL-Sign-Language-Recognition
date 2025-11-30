# 📂 Dữ liệu ASL Signs

Dữ liệu của dự án này được lấy từ cuộc thi **Google - Isolated Sign Language Recognition** trên Kaggle.

- **🔗 Link gốc:** [Kaggle ASL Signs Dataset](https://www.kaggle.com/competitions/asl-signs/data)
- **📝 Tổng quan:** Bộ dữ liệu bao gồm các file video đã được chuyển đổi thành tọa độ (landmarks) bằng MediaPipe và file nhãn tương ứng.

---

## 1. Cấu trúc thư mục
```text
data/
├── train.csv                # File chứa nhãn và đường dẫn
└── train_landmark_files/    # Thư mục chứa các file .parquet
    └── [participant_id]/
        └── [sequence_id].parquet
```

---

## 2. Chi tiết dữ liệu

### A. File `train_landmark_files` (Dữ liệu tọa độ)
Mỗi file `.parquet` chứa dữ liệu về các điểm mốc (landmarks) được trích xuất từ video thô thông qua mô hình **MediaPipe Holistic**.
*Lưu ý: Không phải frame nào cũng phát hiện được bàn tay.*

Các trường thông tin trong file Parquet:

| Tên cột | Mô tả |
| :--- | :--- |
| **`frame`** | Số thứ tự của khung hình (frame) trong video gốc. |
| **`row_id`** | Mã định danh duy nhất cho hàng đó (Ví dụ: `5414471_face_0`). |
| **`type`** | Loại điểm mốc. Gồm 4 loại: `'face'`, `'left_hand'`, `'pose'`, `'right_hand'`. |
| **`landmark_index`** | Số thứ tự của điểm mốc trong nhóm đó (Ví dụ: ngón cái, ngón trỏ...). |
| **`[x/y/z]`** | **Tọa độ không gian đã chuẩn hóa**. <br>🔹 Đây là dữ liệu đầu vào chính cho mô hình.<br>🔹 **Lưu ý:** MediaPipe dự đoán độ sâu (z) chưa tốt, bạn có thể cân nhắc bỏ qua giá trị `z`. |

### B. File `train.csv` (Metadata & Nhãn)
File này đóng vai trò như mục lục, ánh xạ giữa file dữ liệu và ý nghĩa của nó.

| Tên cột | Mô tả |
| :--- | :--- |
| **`path`** | Đường dẫn tương đối đến file `.parquet` (Ví dụ: `train_landmark_files/26734/1000035562.parquet`). |
| **`participant_id`** | ID định danh người thực hiện ký hiệu (dùng để chia tập train/test tránh data leakage). |
| **`sequence_id`** | ID định danh duy nhất cho chuỗi hành động đó. |
| **`sign`** | **Nhãn (Label)** của ký hiệu (Ví dụ: `book`, `bird`, `up`...). Đây là giá trị model cần dự đoán. |

---

> **⚠️ Lưu ý quan trọng:**
> Dữ liệu đầu vào cho mô hình của bạn chỉ nên là các cột tọa độ **`x`, `y`, `z`** (hoặc chỉ `x, y`). Các cột khác như `row_id` hay `type` chỉ dùng để lọc và xử lý dữ liệu.
