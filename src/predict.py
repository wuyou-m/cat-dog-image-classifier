# src/predict.py

import argparse
import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from PIL import Image


IMG_SIZE = (224, 224)
DEFAULT_MODEL_PATH = "models/transfer_with_aug.keras"


def load_image(image_path, target_size=IMG_SIZE):
    """
    Load and preprocess a single image.
    Returns:
        img_array: shape (1, height, width, 3)
        original_img: PIL image for display
    """
    original_img = Image.open(image_path).convert("RGB")
    resized_img = original_img.resize(target_size)

    img_array = np.array(resized_img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, H, W, 3)

    return img_array, original_img


def predict_image(model, img_array):
    """
    Predict whether the image is cat or dog.
    Returns:
        predicted_label: str
        confidence: float
        dog_probability: float
    """
    prob = model.predict(img_array, verbose=0)[0][0]

    if prob >= 0.5:
        predicted_label = "dog"
        confidence = prob
    else:
        predicted_label = "cat"
        confidence = 1 - prob

    return predicted_label, confidence, prob


def show_prediction(original_img, predicted_label, confidence, dog_probability, image_path):
    """
    Display the image with prediction result.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(original_img)
    plt.axis("off")
    plt.title(
        f"Prediction: {predicted_label} | Confidence: {confidence:.4f}\n"
        f"Dog probability: {dog_probability:.4f}"
    )
    plt.tight_layout()
    plt.show()

    print(f"\nImage: {image_path}")
    print(f"Prediction: {predicted_label}")
    print(f"Confidence: {confidence:.4f}")
    print(f"Dog probability: {dog_probability:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Predict cat or dog from an image.")
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the input image"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained .keras model"
    )

    args = parser.parse_args()

    if not os.path.exists(args.image):
        raise FileNotFoundError(f"Image not found: {args.image}")

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model not found: {args.model}")

    print(f"Loading model from: {args.model}")
    model = tf.keras.models.load_model(args.model)

    img_array, original_img = load_image(args.image)
    predicted_label, confidence, dog_probability = predict_image(model, img_array)

    show_prediction(
        original_img,
        predicted_label,
        confidence,
        dog_probability,
        args.image
    )


if __name__ == "__main__":
    main()