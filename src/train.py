import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from src.config import TRAIN_CSV, BATCH_SIZE, EPOCHS, LEARNING_RATE
from src.dataset import create_dataset
from src.model import build_transformer_model

def main():
    #Tải dữ liệu và lọc
    train_df = pd.read_csv(TRAIN_CSV)
    top_signs = train_df['sign'].value_counts().head(10).index.tolist()
    filtered_train_df = train_df[train_df['sign'].isin(top_signs)]
    
    # Tạo ánh xạ nhãn
    unique_signs = sorted(top_signs)
    sign_to_id = {sign: idx for idx, sign in enumerate(unique_signs)}
    num_classes = len(unique_signs)
    
    print(f"Huấn luyện trên {num_classes} lớp: {unique_signs}")
    
    # Chia tập Train/Val
    train_data, val_data = train_test_split(
        filtered_train_df, test_size=0.2, random_state=2176, stratify=filtered_train_df['sign']
    )

    # Tạo Datasets
    train_dataset = create_dataset(train_data, sign_to_id, num_classes, batch_size=BATCH_SIZE)
    val_dataset = create_dataset(val_data, sign_to_id, num_classes, batch_size=BATCH_SIZE, shuffle=True)

    # Xây dựng mô hình
    model = build_transformer_model(num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=["accuracy"]
    )
    model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint("best_model.keras", monitor="val_loss", save_best_only=True)
    ]

    # Huấn luyện
    history = model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset,
        callbacks=callbacks
    )

if __name__ == "__main__":
    main()