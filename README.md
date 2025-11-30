# Hệ thống Nhận diện Ngôn ngữ Ký hiệu ASL (ASL Recognition System)


Dự án xây dựng hệ thống nhận diện ngôn ngữ ký hiệu Mỹ (ASL) theo thời gian thực. Hệ thống sử dụng **MediaPipe** để trích xuất điểm mốc xương khớp (landmarks) và mô hình **Transformer** để phân loại chuỗi hành động.

---

## 📂 1. Cấu trúc Dự án

Dưới đây là sơ đồ cây thư mục và chức năng chi tiết:
```text
asl-recognition-project/
├── data/                      # Chứa dữ liệu (train.csv, thư mục train_landmark_files)
├── models/                    # Chứa file model sau khi train (.keras)
├── notebooks/                 # Jupyter Notebooks nghiên cứu
├── src/                       # MÃ NGUỒN (CORE MODULES & TRAINING)
│   ├── __init__.py            # Đánh dấu package
│   ├── config.py              # Cấu hình (MAX_FRAMES, LANDMARKS, Labels...)
│   ├── dataset.py             # Pipeline xử lý dữ liệu
│   ├── layers.py              # Custom Layers (Preprocess, Embedding)
│   ├── model.py               # Kiến trúc Transformer
│   └── train.py               # SCRIPT HUẤN LUYỆN (Training Loop)
├── demo.py                    # Ứng dụng Demo (Webcam + UI)
├── requirements.txt           # Danh sách thư viện cần thiết
└── README.md                  # Tài liệu hướng dẫn này
```
---

## 📄 Giải thích chi tiết Module (`src/`)

| File Module | Chức năng & Nhiệm vụ |
| :--- | :--- |
| **`src/config.py`** | **Cấu hình:** Lưu các hằng số dùng chung cho cả lúc train và demo (Số frame tối đa, danh sách khớp xương, danh sách nhãn...). |
| **`src/dataset.py`** | **Dữ liệu:** Hàm đọc file `.parquet` và tạo `tf.data.Dataset`. |
| **`src/layers.py`** | **Custom Layers:** Chứa lớp `Preprocess` (xử lý NaN, chuẩn hóa input) và `PositionalEmbedding` (mã hóa vị trí cho Transformer). Bắt buộc phải có để load model. |
| **`src/model.py`** | **Kiến trúc:** Định nghĩa mạng Transformer Encoder. |
| **`src/train.py`** | **Dữ liệu:** Load dữ liệu từ `data/`, Chia tập Train/Val/Test, Xây dựng và huấn luyện model, Lưu model tốt nhất vào `models/` và vẽ biểu đồ kết quả|
| **`demo.py`** | **Chạy ứng dụng:** File duy nhất cần chạy để bật Webcam. Nó chứa cả logic vẽ giao diện (UI) và xử lý luồng video. |

---

## 🛠 2. Hướng dẫn Cài đặt

### Bước 1: Chuẩn bị môi trường
# Clone dự án
git clone https://github.com/quynh2196/ASL-Sign-Language-Recognition.git
cd ASL-Sign-Language-Recognition

# Tạo môi trường ảo (Khuyên dùng)
python -m venv venv

# Kích hoạt (Windows)
.\venv\Scripts\activate
# Kích hoạt (Mac/Linux)
source venv/bin/activate

### Bước 2: Cài đặt thư viện
pip install -r requirements.txt

---

## 🚀 3. Hướng dẫn Sử dụng

### A. Chuẩn bị Dữ liệu
Tải dữ liệu từ Kaggle hướng dẫn trực tiêps trong `data/README.md`:
- `data/train.csv`
- `data/train_landmark_files/`

### B. Chạy Demo (Webcam)
Chạy lệnh sau để bật webcam và nhận diện:
```text
python demo.py
```

*Phím tắt:*
- **Q**: Thoát chương trình.
- **S**: Bật/Tắt đọc giọng nói (TTS).

### C. Huấn luyện (Training)
Để train lại model với dữ liệu trong thư mục `data/`:
```text
python src/train.py
```
---

## 📜 Giấy phép
Dự án được phân phối dưới giấy phép MIT.
