import numpy as np

# Đường dẫn dữ liệu
DATA_DIR = '/kaggle/input/asl-signs'
TRAIN_CSV = f'{DATA_DIR}/train.csv'

# Tham số mô hình & Huấn luyện
MAX_FRAMES = 384
TOTAL_LANDMARKS = 543
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-4

# Chọn đặc trưng (Features) các điểm Landmarks quan trọng (MediaPipe)
LIP = [
    61, 185, 40, 39, 37, 267, 269, 270, 409,
    291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    95, 88, 178, 87, 14, 317, 402, 318, 324, 308,
]
LHAND = np.arange(468, 489).tolist()
RHAND = np.arange(522, 543).tolist()
NOSE = [1, 2, 98, 327]
REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# Tổng hợp các điểm landmarks sẽ sử dụng
POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE + LEFT_EYEBROW + RIGHT_EYEBROW + FACE_OVAL