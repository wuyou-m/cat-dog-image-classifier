# src/evaluate.py

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


# ------------------------
# Config
# ------------------------

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
VAL_DIR = "data/validation"

# MODEL_NAME = "transfer_with_aug.keras"
MODEL_NAME = "transfer_with_aug_fine_tuned.keras"
MODEL_PATH = f"models/{MODEL_NAME}"

OUTPUT_DIR = "outputs/figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MISCLASSIFIED_DIR = "outputs/misclassified"
os.makedirs(MISCLASSIFIED_DIR, exist_ok=True)


# ------------------------
# Load validation dataset
# ------------------------

def load_validation_dataset():
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="binary",
        shuffle=False # 评估时不需要打乱数据，保持顺序以便后续分析
    )
    return val_ds

# ------------------------
# Save misclassified images
# ------------------------

def save_misclassified_images(val_ds, y_true, y_pred, class_names, max_save=20):
    """
    Save a subset of misclassified images to disk.
    """
    saved_count = 0
    global_index = 0

    for batch_images, batch_labels in val_ds:
        batch_size = batch_images.shape[0]

        for i in range(batch_size):
            true_label = int(y_true[global_index])
            pred_label = int(y_pred[global_index])

            if true_label != pred_label:
                img = batch_images[i].numpy().astype("uint8")
                true_name = class_names[true_label]
                pred_name = class_names[pred_label]

                filename = f"{true_name}_pred_{pred_name}_{saved_count+1:03d}.png"
                filepath = os.path.join(MISCLASSIFIED_DIR, filename)

                plt.figure(figsize=(4, 4))
                plt.imshow(img)
                plt.axis("off")
                plt.title(f"True: {true_name} | Pred: {pred_name}")
                plt.tight_layout()
                plt.savefig(filepath)
                plt.close()

                saved_count += 1

            global_index += 1

            if saved_count >= max_save:
                print(f"\nSaved {saved_count} misclassified images to: {MISCLASSIFIED_DIR}")
                return

    print(f"\nSaved {saved_count} misclassified images to: {MISCLASSIFIED_DIR}")

# ------------------------
# Evaluate model
# ------------------------

def evaluate():
    val_ds = load_validation_dataset()

    print(f"\nLoading model from: {MODEL_PATH}")
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Evaluating model: {MODEL_NAME}")

    # Basic evaluation
    results = model.evaluate(val_ds, verbose=1)
    metric_names = model.metrics_names

    print("\nEvaluation Results:")
    for name, value in zip(metric_names, results):
        print(f"{name}: {value:.4f}")

    # Collect labels
    y_true = np.concatenate([y.numpy() for _, y in val_ds]).astype(int).flatten()

    # Predict probabilities
    y_prob = model.predict(val_ds)
    y_pred = (y_prob > 0.5).astype(int).flatten()

    # Class names from dataset
    class_names = val_ds.class_names
    print("\nClass names:", class_names)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix - {MODEL_NAME}")
    plt.tight_layout()

    cm_filename = f"confusion_matrix_{MODEL_NAME.replace('.keras', '')}.png"
    cm_path = os.path.join(OUTPUT_DIR, cm_filename)
    plt.savefig(cm_path)
    plt.show()

    print(f"\nConfusion matrix saved to: {cm_path}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Save misclassified images
    save_misclassified_images(val_ds, y_true, y_pred, class_names, max_save=20)


if __name__ == "__main__":
    evaluate()