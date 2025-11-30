import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import os
import argparse


try:
    from src.config import MAX_FRAMES, TOTAL_LANDMARKS
    from src.layers import Preprocess, PositionalEmbedding, tf_nan_mean, tf_nan_std
except ImportError as e:
    print("Lỗi Import: Không tìm thấy thư mục 'src' hoặc các file cần thiết.")
    print(f"Chi tiết: {e}")
    exit(1)

# --- CẤU HÌNH NHÃN (LABELS) ---
MY_10_SIGNS = sorted(['bird', 'donkey', 'duck', 'hear', 'listen', 'look', 'mouse', 'pretend', 'shhh', 'uncle'])
ID_TO_SIGN = {idx: sign for idx, sign in enumerate(MY_10_SIGNS)}

class ModernUI:
    """Class quản lý toàn bộ việc vẽ giao diện để code chính gọn gàng"""
    def __init__(self):
        self.last_spoken = None
        self.last_speak_time = 0
        
    def speak(self, text, enabled=True):
        """Text to Speech (macOS only)"""
        curr = time.time()
        if enabled and text not in ["NO HAND", "..."] and text != self.last_spoken:
            if (curr - self.last_speak_time) > 2.0:
                os.system(f'say "{text.lower()}" &') # Dấu & để không bị delay hình
                self.last_spoken = text
                self.last_speak_time = curr

    def draw(self, image, text, conf, fps, stable_cnt, tts_on, seq_len):
        h, w = image.shape[:2]
        
        # 1. Header Gradient Background
        overlay = image.copy()
        cv2.rectangle(overlay, (0, 0), (w, 120), (30, 30, 40), -1)
        cv2.addWeighted(overlay, 0.85, image, 0.15, 0, image)
        cv2.line(image, (0, 120), (w, 120), (100, 200, 255), 2)

        # 2. Main Text Status
        color = (100, 255, 100) if conf > 0.6 else ((100, 255, 255) if conf > 0.3 else (100, 150, 255))
        status_txt = "RECOGNIZED" if conf > 0.6 else "DETECTING..."
        
        # Shadow & Text
        cv2.putText(image, text, (32, 72), cv2.FONT_HERSHEY_DUPLEX, 2, (0,0,0), 5)
        cv2.putText(image, text, (30, 70), cv2.FONT_HERSHEY_DUPLEX, 2, color, 2)
        
        # Small Status Badge
        cv2.rectangle(image, (30, 85), (30 + len(status_txt)*12, 110), color, -1)
        cv2.putText(image, status_txt, (35, 103), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

        # 3. Confidence Bar
        bar_x, bar_y, bar_w = 250, 90, 200
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + 15), (60,60,80), -1)
        fill_w = int(bar_w * conf)
        cv2.rectangle(image, (bar_x, bar_y), (bar_x + fill_w, bar_y + 15), color, -1)
        cv2.putText(image, f"{int(conf*100)}%", (bar_x + bar_w + 10, bar_y + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

        # 4. Info Panel (Bottom)
        info = f"FPS: {int(fps)} | Frames: {seq_len} | TTS: {'ON' if tts_on else 'OFF'}"
        cv2.putText(image, info, (w - 350, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        
        # 5. Stability Dots
        for i in range(3):
            c = (100, 255, 100) if i < stable_cnt else (80, 80, 80)
            cv2.circle(image, (30 + i*25, 140), 6, c, -1)
        
        return image

    def draw_skeleton(self, image, results):
        """Vẽ khung xương tay đơn giản"""
        if not results: return
        mp.solutions.drawing_utils.draw_landmarks(
            image, results.left_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)
        mp.solutions.drawing_utils.draw_landmarks(
            image, results.right_hand_landmarks, mp.solutions.holistic.HAND_CONNECTIONS)

def extract_landmarks(results):
    """Trích xuất tọa độ xyz từ kết quả MediaPipe"""
    def get_arr(objs, count):
        if objs: 
            return np.array([[res.x, res.y, res.z] for res in objs.landmark], dtype=np.float32)
        return np.zeros((count, 3), dtype=np.float32)
    
    return np.concatenate([
        get_arr(results.face_landmarks, 468),
        get_arr(results.left_hand_landmarks, 21),
        get_arr(results.pose_landmarks, 33),
        get_arr(results.right_hand_landmarks, 21)
    ])

def main():
    # 1. Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/asl_transformer_model.keras', help='Đường dẫn file model')
    args = parser.parse_args()

    # 2. Load Model (Sử dụng Custom Objects từ src)
    print(f"Đang tải mô hình từ: {args.model}")
    try:
        custom_objects = {
            "Preprocess": Preprocess,
            "PositionalEmbedding": PositionalEmbedding,
            "tf_nan_mean": tf_nan_mean,
            "tf_nan_std": tf_nan_std
        }
        model = tf.keras.models.load_model(args.model, custom_objects=custom_objects, compile=False)
        # Warmup chạy thử 1 lần
        model.predict(np.zeros((1, MAX_FRAMES, TOTAL_LANDMARKS, 3)), verbose=0)
        print("Đã tải mô hình thành công!")
    except Exception as e:
        print(f"Không tải được model. Lỗi: {e}")
        return

    # 3. Khởi tạo Camera & Biến
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    ui = ModernUI()
    holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=0)
    
    sequence = []
    current_text = "..."
    current_conf = 0.0
    stable_count = 0
    last_pred = None
    tts_enabled = True
    prev_time = 0

    print("Đang chạy... Nhấn 'Q' để thoát, 'S' để bật/tắt đọc giọng nói.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        # Tính FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # Xử lý MediaPipe
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Logic Dự Đoán
        if results.left_hand_landmarks or results.right_hand_landmarks:
            keypoints = extract_landmarks(results)
            sequence.append(keypoints)
            sequence = sequence[-30:] # Giữ 30 frame cuối

            if len(sequence) >= 10: # Chỉ dự đoán khi có ít nhất 10 frame
                # Chuẩn bị input (Padding nếu cần)
                input_data = np.expand_dims(sequence, axis=0)
                if input_data.shape[1] < MAX_FRAMES:
                    pad = np.zeros((1, MAX_FRAMES - input_data.shape[1], TOTAL_LANDMARKS, 3))
                    input_data = np.concatenate([input_data, pad], axis=1)

                # Predict
                res = model.predict(input_data, verbose=0)[0]
                idx = np.argmax(res)
                conf = res[idx]
                sign = ID_TO_SIGN[idx]

                # Logic ổn định kết quả (Debounce)
                if conf > 0.6:
                    if sign == last_pred:
                        stable_count += 1
                    else:
                        stable_count = 0
                        last_pred = sign
                    
                    if stable_count > 3: # Ổn định trong 3 frame liên tiếp
                        current_text = sign.upper()
                        current_conf = conf
                else:
                    current_conf *= 0.9 # Giảm dần confidence khi không chắc
        else:
            sequence = [] # Reset nếu bỏ tay ra
            stable_count = 0
            if current_conf < 0.5: current_text = "NO HAND"

        # Vẽ UI & TTS
        ui.speak(current_text, tts_enabled)
        ui.draw_skeleton(image, results)
        image = ui.draw(image, current_text, current_conf, fps, stable_count, tts_enabled, len(sequence))

        cv2.imshow('ASL Pro Demo', image)

        # Phím tắt
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        if key == ord('s'): 
            tts_enabled = not tts_enabled
            print(f"🔊 TTS: {tts_enabled}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()