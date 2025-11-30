import pandas as pd
import numpy as np
import tensorflow as tf
from src.config import MAX_FRAMES, TOTAL_LANDMARKS, DATA_DIR

def load_and_preprocess_data(data_path):
    """Đọc file parquet và chuẩn hóa kích thước frames."""
    path_str = data_path.numpy().decode('utf-8')
    df = pd.read_parquet(path_str)
    
    landmarks = df[['x', 'y', 'z']].values
    
    n_frames = len(landmarks) // TOTAL_LANDMARKS
    if n_frames == 0:
        return np.zeros((MAX_FRAMES, TOTAL_LANDMARKS, 3), dtype=np.float32)
        
    landmarks = landmarks.reshape(n_frames, TOTAL_LANDMARKS, 3)
    
    # Padding hoặc cắt bớt frames
    if n_frames < MAX_FRAMES:
        padding = np.zeros((MAX_FRAMES - n_frames, TOTAL_LANDMARKS, 3))
        landmarks = np.vstack([landmarks, padding])
    else:
        landmarks = landmarks[:MAX_FRAMES, :, :]
    
    return landmarks.astype(np.float32)

def create_dataset(dataframe, sign_to_id, num_classes, batch_size=32, shuffle=True):
    """Tạo TensorFlow Dataset từ dataframe."""
    paths = [f"{DATA_DIR}/{path}" for path in dataframe['path']]
    labels = [sign_to_id[sign] for sign in dataframe['sign']]
    
    @tf.function
    def process_path(path, label):
        landmarks = tf.py_function(
            func=load_and_preprocess_data,
            inp=[path],
            Tout=tf.float32
        )
        landmarks.set_shape((MAX_FRAMES, TOTAL_LANDMARKS, 3))
        label = tf.one_hot(label, depth=num_classes)
        return landmarks, label
    
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(dataframe))
    
    dataset = dataset.map(process_path, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset