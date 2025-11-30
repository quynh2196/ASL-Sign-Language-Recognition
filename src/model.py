import tensorflow as tf
from tensorflow.keras import layers
from src.layers import Preprocess, PositionalEmbedding
from src.config import MAX_FRAMES, TOTAL_LANDMARKS

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    attn_output = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Add()([inputs, attn_output])
    
    # Feed Forward Network
    y = layers.LayerNormalization(epsilon=1e-6)(x)
    y = layers.Dense(ff_dim, activation='gelu')(y)
    y = layers.Dropout(dropout)(y)
    y = layers.Dense(inputs.shape[-1])(y)
    y = layers.Dropout(dropout)(y)
    return layers.Add()([x, y])

def build_transformer_model(num_classes):
    inputs = tf.keras.Input(shape=(MAX_FRAMES, TOTAL_LANDMARKS, 3))
    
    # Lớp tiền xử lý tùy chỉnh
    x = Preprocess(max_len=MAX_FRAMES)(inputs) 
    x = layers.Masking(mask_value=0.0)(x)

    # Cấu hình Transformer
    head_size = 64
    num_heads = 4
    d_model = head_size * num_heads
    ff_dim = 128 
    transformer_layers = 4
    mlp_units = [128]
    dropout = 0.2
    mlp_dropout = 0.4
    
    # Embedding
    x = layers.Dense(d_model)(x)
    x = PositionalEmbedding(d_model=d_model)(x)
    x = layers.Dropout(dropout)(x)
    
    # Encoder blocks
    for _ in range(transformer_layers):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    
    # Classification head
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x) 
    for dim in mlp_units:
        x = layers.Dense(dim, activation="gelu")(x)
        x = layers.Dropout(mlp_dropout)(x)
        
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    
    return tf.keras.Model(inputs, outputs)