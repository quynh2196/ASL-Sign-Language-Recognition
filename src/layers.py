import tensorflow as tf
import numpy as np
from src.config import POINT_LANDMARKS

def tf_nan_mean(x, axis=0, keepdims=False):
    """Tính trung bình bỏ qua giá trị NaN."""
    return tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), x), axis=axis, keepdims=keepdims) / \
           tf.reduce_sum(tf.where(tf.math.is_nan(x), tf.zeros_like(x), tf.ones_like(x)), axis=axis, keepdims=keepdims)

def tf_nan_std(x, center=None, axis=0, keepdims=False):
    """Tính độ lệch chuẩn bỏ qua giá trị NaN."""
    if center is None:
        center = tf_nan_mean(x, axis=axis, keepdims=True)
    d = x - center
    return tf.math.sqrt(tf_nan_mean(d * d, axis=axis, keepdims=keepdims))

class Preprocess(tf.keras.layers.Layer):
    def __init__(self, max_len=384, point_landmarks=POINT_LANDMARKS, **kwargs):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.point_landmarks = point_landmarks

    def call(self, inputs):
        x = inputs
        # Xử lý NaN và chuẩn hóa
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

def positional_encoding(length, depth):
    """Tạo positional encoding cho Transformer."""
    depth = depth/2
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :]/depth
    angle_rates = 1 / (10000**depths)
    angle_rads = positions * angle_rates
    pos_encoding = np.concatenate(
        [np.sin(angle_rads), np.cos(angle_rads)], axis=-1) 
    return tf.cast(pos_encoding, dtype=tf.float32)

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_len=384):
        super().__init__()
        self.d_model = d_model
        self.pos_encoding = positional_encoding(length=max_len, depth=d_model)
        self.supports_masking = True

    def call(self, x):
        length = tf.shape(x)[1]
        return x + self.pos_encoding[:length, :]