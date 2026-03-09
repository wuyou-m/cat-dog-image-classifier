# src/train.py

import tensorflow as tf
from pathlib import Path
from src.model import build_baseline_cnn, build_transfer_model, compile_model

# ------------------------
# Config
# ------------------------

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
FINE_TUNE_EPOCHS = 5

TRAIN_DIR = "data/train"
VAL_DIR = "data/validation"

# "baseline" or "transfer"
# MODEL_TYPE = "baseline"
MODEL_TYPE = "transfer"

USE_AUGMENTATION = True
USE_FINE_TUNING = True
FINE_TUNE_AT = 120

EXPERIMENT_NAME = f"{MODEL_TYPE}"
if USE_AUGMENTATION:
    EXPERIMENT_NAME += "_with_aug"
else:
    EXPERIMENT_NAME += "_no_aug"
if USE_FINE_TUNING:
    EXPERIMENT_NAME += "_fine_tuned"


# ------------------------
# Dataset
# ------------------------

def load_datasets():

    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary"
    )

    return train_ds, val_ds

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])

# 减弱版本
# data_augmentation = tf.keras.Sequential([
#     tf.keras.layers.RandomFlip("horizontal"),
#     tf.keras.layers.RandomRotation(0.05),
#     tf.keras.layers.RandomZoom(0.05),
# ])

# ------------------------
# Build Model
# ------------------------

def build_model():

    base_model = None

    if MODEL_TYPE == "baseline":
        model = build_baseline_cnn()
        model = compile_model(model, learning_rate=1e-4)
    # 迁移学习版本，默认冻结预训练模型的权重，只训练新添加的输出层。后续可以解冻部分预训练模型进行微调。
    elif MODEL_TYPE == "transfer":
        model, base_model = build_transfer_model(base_trainable=False)
        model = compile_model(model, learning_rate=1e-4)

    else:
        raise ValueError("Unknown model type")

    return model, base_model


# ------------------------
# Training
# ------------------------

def train():

    train_ds, val_ds = load_datasets()

    if USE_AUGMENTATION:
        train_ds = train_ds.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    model, base_model = build_model()

    model.summary()

    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            f"models/{EXPERIMENT_NAME}.keras",
            monitor="val_loss",
            save_best_only=True
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True
        )

    ]

    print("\nStage 1: Training classifier head...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    if MODEL_TYPE == "transfer" and USE_FINE_TUNING:
        print("\nStage 2: Fine-tuning the last layers of the backbone...")

        base_model.trainable = True

        # 冻结前面的层，只解冻最后一部分
        for layer in base_model.layers[:FINE_TUNE_AT]:
            layer.trainable = False

        # fine-tuning 要用更小学习率
        model = compile_model(model, learning_rate=1e-5)

        fine_tune_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS + FINE_TUNE_EPOCHS,
            initial_epoch=history.epoch[-1] + 1,
            callbacks=callbacks
        )

        return history, fine_tune_history

    return history


if __name__ == "__main__":
    train()