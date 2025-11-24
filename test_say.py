import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import keras
import time
import os

def speak_text(text):
    """Đọc văn bản bằng giọng nói (macOS)"""
    os.system(f'say "{text}"')

def draw_rounded_rectangle(img, pt1, pt2, color, thickness, radius=20):
    """Vẽ hình chữ nhật bo góc"""
    x1, y1 = pt1
    x2, y2 = pt2
    
    # Vẽ các cạnh
    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
    
    # Vẽ các góc
    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

def draw_gradient_background(img, y1, y2, color1, color2):
    """Vẽ background gradient"""
    for y in range(y1, y2):
        alpha = (y - y1) / (y2 - y1)
        color = tuple([int(c1 * (1 - alpha) + c2 * alpha) for c1, c2 in zip(color1, color2)])
        cv2.line(img, (0, y), (img.shape[1], y), color, 1)

def draw_modern_ui(image, text_display, current_conf, fps, sequence_len, stable_count, tts_enabled):
    """Vẽ giao diện hiện đại"""
    h, w = image.shape[:2]
    
    # === HEADER PANEL - Gradient ===
    header_height = 140
    overlay = image.copy()
    
    # Gradient background cho header
    draw_gradient_background(overlay, 0, header_height, (30, 30, 40), (50, 50, 70))
    cv2.addWeighted(overlay, 0.85, image, 0.15, 0, image)
    
    # Border dưới header
    cv2.line(image, (0, header_height), (w, header_height), (100, 200, 255), 2)
    
    # === MAIN TEXT - Prediction ===
    if current_conf > 0.6:
        main_color = (100, 255, 100)  # Xanh lá
        status = "RECOGNIZED"
        status_color = (100, 255, 100)
    elif current_conf > 0.3:
        main_color = (100, 255, 255)  # Vàng
        status = "DETECTING..."
        status_color = (100, 255, 255)
    else:
        main_color = (100, 150, 255)  # Đỏ nhạt
        status = "WAITING"
        status_color = (150, 150, 255)
    
    # Text chính với shadow effect
    text = text_display if text_display not in ["NO HAND", "..."] else "Ready"
    
    # Shadow
    cv2.putText(image, text, (22, 62), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 0), 8)
    # Main text
    cv2.putText(image, text, (20, 60), cv2.FONT_HERSHEY_DUPLEX, 2, main_color, 6)
    
    # Status badge
    status_x = 20
    status_y = 80
    status_w = len(status) * 11 + 30
    
    # Badge background
    cv2.rectangle(image, (status_x, status_y), (status_x + status_w, status_y + 30), 
                  status_color, -1)
    cv2.rectangle(image, (status_x, status_y), (status_x + status_w, status_y + 30), 
                  (255, 255, 255), 2)
    
    # Status text
    cv2.putText(image, status, (status_x + 15, status_y + 21), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    # === CONFIDENCE BAR ===
    bar_x = status_x + status_w + 20
    bar_y = 85
    bar_width = 200
    bar_height = 20
    
    # Background bar
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                  (60, 60, 80), -1)
    cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                  (150, 150, 200), 2)
    
    # Confidence fill
    fill_width = int(bar_width * min(current_conf, 1.0))
    if fill_width > 0:
        # Gradient fill
        for i in range(fill_width):
            alpha = i / bar_width
            color = (
                int(100 + alpha * 155),
                int(150 + alpha * 105),
                int(255 - alpha * 155)
            )
            cv2.line(image, (bar_x + i, bar_y), (bar_x + i, bar_y + bar_height), color, 1)
    
    # Confidence percentage
    cv2.putText(image, f"{int(current_conf * 100)}%", 
                (bar_x + bar_width + 10, bar_y + 16), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    # === STABILITY INDICATOR ===
    stab_x = 20
    stab_y = 120
    for i in range(3):
        if i < stable_count:
            color = (100, 255, 100)
            cv2.circle(image, (stab_x + i * 25, stab_y), 8, color, -1)
            cv2.circle(image, (stab_x + i * 25, stab_y), 8, (255, 255, 255), 2)
        else:
            color = (80, 80, 100)
            cv2.circle(image, (stab_x + i * 25, stab_y), 8, color, -1)
            cv2.circle(image, (stab_x + i * 25, stab_y), 8, (120, 120, 140), 2)
    
    cv2.putText(image, "Stability", (stab_x + 80, stab_y + 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    # === INFO PANEL - Bottom Right ===
    info_x = w - 200
    info_y = h - 100
    
    # Semi-transparent panel
    overlay = image.copy()
    cv2.rectangle(overlay, (info_x - 10, info_y - 10), (w - 10, h - 10), 
                  (40, 40, 60), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    
    # Border
    cv2.rectangle(image, (info_x - 10, info_y - 10), (w - 10, h - 10), 
                  (100, 150, 255), 2)
    
    # Info text
    cv2.putText(image, f"FPS: {int(fps)}", (info_x, info_y + 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(image, f"Frames: {sequence_len}", (info_x, info_y + 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # TTS Icon
    tts_icon = "🔊" if tts_enabled else "🔇"
    tts_color = (100, 255, 100) if tts_enabled else (150, 150, 150)
    cv2.putText(image, "TTS", (info_x, info_y + 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, tts_color, 1)
    
    # === CONTROLS INFO - Bottom Left ===
    controls_y = h - 60
    overlay = image.copy()
    cv2.rectangle(overlay, (10, controls_y - 10), (250, h - 10), 
                  (40, 40, 60), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    
    cv2.rectangle(image, (10, controls_y - 10), (250, h - 10), 
                  (100, 150, 255), 2)
    
    cv2.putText(image, "Q - Quit  |  S - TTS", (20, controls_y + 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    return image

def draw_hands(image, results):
    """Vẽ bàn tay với style hiện đại"""
    h, w = image.shape[:2]
    
    HAND_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17)
    ]
    
    # Left hand - Gradient xanh lá
    if results.left_hand_landmarks:
        for idx, (start, end) in enumerate(HAND_CONNECTIONS):
            pt1 = results.left_hand_landmarks.landmark[start]
            pt2 = results.left_hand_landmarks.landmark[end]
            
            # Màu gradient
            color = (50 + idx * 10, 255, 150 - idx * 5)
            
            cv2.line(image, (int(pt1.x*w), int(pt1.y*h)), 
                    (int(pt2.x*w), int(pt2.y*h)), color, 3)
        
        # Vẽ các điểm landmark
        for landmark in results.left_hand_landmarks.landmark:
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (cx, cy), 4, (100, 255, 100), -1)
            cv2.circle(image, (cx, cy), 4, (255, 255, 255), 1)
    
    # Right hand - Gradient tím/hồng
    if results.right_hand_landmarks:
        for idx, (start, end) in enumerate(HAND_CONNECTIONS):
            pt1 = results.right_hand_landmarks.landmark[start]
            pt2 = results.right_hand_landmarks.landmark[end]
            
            # Màu gradient
            color = (255, 50 + idx * 10, 255 - idx * 10)
            
            cv2.line(image, (int(pt1.x*w), int(pt1.y*h)), 
                    (int(pt2.x*w), int(pt2.y*h)), color, 3)
        
        # Vẽ các điểm landmark
        for landmark in results.right_hand_landmarks.landmark:
            cx, cy = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (cx, cy), 4, (255, 100, 255), -1)
            cv2.circle(image, (cx, cy), 4, (255, 255, 255), 1)

MODEL_PATH = '/Users/phong/Code/vsCode/AI/asl_transformer_model (1).keras'
THRESHOLD = 0.6
MAX_FRAMES = 384
TOTAL_LANDMARKS = 543

MY_10_SIGNS = sorted(['bird', 'donkey', 'duck', 'hear', 'listen', 'look', 'mouse', 'pretend', 'shhh', 'uncle'])
id_to_sign = {idx: sign for idx, sign in enumerate(MY_10_SIGNS)}

LIP = [61, 185, 40, 39, 37, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]
LHAND = list(range(468, 489))
RHAND = list(range(522, 543))
NOSE = [1, 2, 98, 327]
REYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
LEYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
RIGHT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

POINT_LANDMARKS = LIP + LHAND + RHAND + NOSE + REYE + LEYE + LEFT_EYEBROW + RIGHT_EYEBROW + FACE_OVAL

@keras.saving.register_keras_serializable()
def tf_nan_mean(x, axis=0, keepdims=False):
    return tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), x), axis=axis, keepdims=keepdims) / \
           tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), tf.ones_like(x)), axis=axis, keepdims=keepdims)

@keras.saving.register_keras_serializable()
def tf_nan_std(x, center=None, axis=0, keepdims=False):
    if center is None: 
        center = tf_nan_mean(x, axis=axis, keepdims=True)
    d = x - center
    return tf.math.sqrt(tf_nan_mean(d * d, axis=axis, keepdims=keepdims))

@keras.saving.register_keras_serializable()
class Preprocess(keras.layers.Layer):
    def __init__(self, max_len=384, point_landmarks=None, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.point_landmarks = point_landmarks if point_landmarks is not None else POINT_LANDMARKS

    def call(self, inputs):
        x = inputs
        mean = tf_nan_mean(tf.gather(x, [17], axis=2), axis=[1,2], keepdims=True)
        mean = tf.where(tf.math.is_nan(mean), tf.constant(0.5, x.dtype), mean)
        x = tf.gather(x, self.point_landmarks, axis=2)
        std = tf_nan_std(x, center=mean, axis=[1,2], keepdims=True)
        x = (x - mean) / std
        if self.max_len is not None:
            x = x[:, :self.max_len]
        length = tf.shape(x)[1]
        x = x[..., :2]
        x = tf.reshape(x, (-1, length, 2 * len(self.point_landmarks)))
        x = tf.where(tf.math.is_nan(x), tf.constant(0., x.dtype), x)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({"max_len": self.max_len})
        return config

def positional_encoding(length, depth):
    depth = depth / 2
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :] / depth
    angle_rates = 1 / (10000 ** depths)
    angle_rads = positions * angle_rates
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)

@keras.saving.register_keras_serializable()
class PositionalEmbedding(keras.layers.Layer):
    def __init__(self, d_model, max_len=384, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.max_len = max_len
        self.pos_encoding = positional_encoding(length=max_len, depth=d_model)
        self.supports_masking = True
    
    def call(self, x):
        length = tf.shape(x)[1]
        return x + self.pos_encoding[:length, :]
    
    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "max_len": self.max_len})
        return config

keras.saving.get_custom_objects().update({
    "Preprocess": Preprocess,
    "PositionalEmbedding": PositionalEmbedding,
    "tf_nan_mean": tf_nan_mean,
    "tf_nan_std": tf_nan_std
})

print("🚀 Đang tải mô hình...")
try:
    model = keras.models.load_model(MODEL_PATH, compile=False)
    dummy = np.zeros((1, 384, 543, 3), dtype=np.float32)
    model.predict(dummy, verbose=0)
    print("✅ Đã tải mô hình thành công!\n")
except Exception as e:
    print(f"❌ Lỗi: {e}")
    exit()

def extract_landmarks(results):
    def get_arr(objs, count):
        if objs: 
            return np.array([[res.x, res.y, res.z] for res in objs.landmark], dtype=np.float32)
        return np.zeros((count, 3), dtype=np.float32)
    
    face = get_arr(results.face_landmarks, 468)
    lh = get_arr(results.left_hand_landmarks, 21)
    pose = get_arr(results.pose_landmarks, 33)
    rh = get_arr(results.right_hand_landmarks, 21)
    return np.concatenate([face, lh, pose, rh])

SEQUENCE_MAX_LEN = 30
MIN_FRAMES = 10
PREDICT_INTERVAL = 1
STABLE_THRESHOLD = 3
CONFIDENCE_DECAY = 0.95
SPEAK_COOLDOWN = 2.0

sequence = []
current_pred = "NO HAND"
current_conf = 0.0
no_hands_count = 0
prev_time = time.time()
last_spoken_word = None
last_speak_time = 0
pred_history = []
last_confident_pred = None
stable_count = 0
tts_enabled = True

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

print("="*60)
print("🎯 ASL RECOGNITION - MODERN UI")
print("="*60)
print("📌 Controls:")
print("   Q - Quit")
print("   S - Toggle TTS")
print("="*60 + "\n")

mp_holistic = mp.solutions.holistic

with mp_holistic.Holistic(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5, 
    model_complexity=0
) as holistic:
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: 
            break

        frame_count += 1
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Vẽ bàn tay với style hiện đại
        draw_hands(image, results)

        hands_present = (results.left_hand_landmarks or results.right_hand_landmarks)

        if hands_present:
            no_hands_count = 0
            keypoints = extract_landmarks(results)
            sequence.append(keypoints)
            sequence = sequence[-SEQUENCE_MAX_LEN:]

            if len(sequence) >= MIN_FRAMES and frame_count % PREDICT_INTERVAL == 0:
                input_data = np.expand_dims(sequence, axis=0)
                
                if input_data.shape[1] < MAX_FRAMES:
                    pad = np.zeros((1, MAX_FRAMES - input_data.shape[1], TOTAL_LANDMARKS, 3), dtype=np.float32)
                    input_data = np.concatenate([input_data, pad], axis=1)
                
                try:
                    res = model.predict(input_data, verbose=0)[0]
                    best_idx = np.argmax(res)
                    conf = res[best_idx]
                    predicted_sign = id_to_sign[best_idx]

                    pred_history.append(predicted_sign)
                    pred_history = pred_history[-5:]
                    
                    if conf > THRESHOLD:
                        if predicted_sign == last_confident_pred:
                            stable_count += 1
                        else:
                            stable_count = 1
                            last_confident_pred = predicted_sign
                        
                        if stable_count >= STABLE_THRESHOLD or conf > 0.85:
                            current_pred = predicted_sign
                            current_conf = conf
                    else:
                        current_conf *= CONFIDENCE_DECAY
                        if current_conf < THRESHOLD:
                            current_pred = "..."
                            stable_count = 0
                            
                except Exception as e:
                    print(f"❌ Lỗi dự đoán: {e}")
        else:
            no_hands_count += 1
            if no_hands_count > 5:
                sequence = []
                current_pred = "NO HAND"
                current_conf = 0.0
                pred_history = []
                last_confident_pred = None
                stable_count = 0

        # Xác định text hiển thị
        if current_conf > THRESHOLD:
            text_display = current_pred.upper()
        else:
            text_display = "NO HAND" if current_pred == "NO HAND" else "..."
        
        # Text-to-Speech
        current_time = time.time()
        if (text_display != last_spoken_word and 
            text_display not in ["NO HAND", "...", ""] and
            (current_time - last_speak_time) > SPEAK_COOLDOWN and
            tts_enabled):
            
            speak_text(text_display.lower())
            last_spoken_word = text_display
            last_speak_time = current_time
        
        # Vẽ UI hiện đại
        image = draw_modern_ui(image, text_display, current_conf, fps, 
                               len(sequence), stable_count, tts_enabled)

        cv2.imshow('ASL Recognition - Modern UI', image)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') or key == ord('S'):
            tts_enabled = not tts_enabled
            print(f"🔊 TTS: {'ON' if tts_enabled else 'OFF'}")

cap.release()
cv2.destroyAllWindows()
print("\n👋 Đã đóng ứng dụng!")