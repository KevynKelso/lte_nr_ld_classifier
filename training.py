import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix

from constants import (BATCH_SIZE, DATA_DIR, EPOCHS, IMG_SIZE, TEST_SPLIT,
                       VAL_SPLIT)
from dataset import load_data
from networks import (classification_network, deeper_cnn_network, load_network,
                      residual_style_network)


def train_network(model):
    """Train a given model

    Args:
        model ():
    """
    X_train, X_test, y_train, y_test, class_names = load_data(
        DATA_DIR, IMG_SIZE, TEST_SPLIT
    )

    y_train_onehot = tf.keras.utils.to_categorical(
        y_train, num_classes=len(class_names)
    )
    y_test_onehot = tf.keras.utils.to_categorical(y_test, num_classes=len(class_names))

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=20,
            verbose=1,
            mode="max",
            restore_best_weights=True,
        ),
    ]
    history = model.fit(
        X_train,
        y_train_onehot,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VAL_SPLIT,
        verbose=1,
        callbacks=callbacks,
    )
    model.save(f"{model.friendly_name}.keras")

    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"{model.friendly_name}-accuracy.png")
    plt.clf()

    plt.plot(history.history["loss"], label="Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"{model.friendly_name}-loss.png")
    plt.clf()


def plot_confusion_matrix(y_true, y_pred, class_names, name):
    """
    Creates a labeled confusion matrix plot suitable for reports.
    """
    cm = confusion_matrix(y_true, y_pred)

    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=False,
    )

    plt.xlabel("Predicted Labels", fontsize=12)
    plt.ylabel("True Labels", fontsize=12)
    plt.title("Confusion Matrix", fontsize=14, pad=20)

    # Rotate tick labels for better readability
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(f"{name}-confusion-matrix.png")
    plt.clf()


def evaluate_model_performance(model, sample_size=500, target_size=(128, 128)):
    """
    Evaluates model performance on a random sample of images from a directory structure.
    """
    class_names = sorted(
        [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    )
    num_classes = len(class_names)

    images = []
    y_true = []

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(DATA_DIR, class_name)
        all_files = [f for f in os.listdir(class_dir) if f.endswith(".png")]

        sampled_files = random.sample(all_files, min(sample_size, len(all_files)))

        for filename in sampled_files:
            img_path = os.path.join(class_dir, filename)
            img = Image.open(img_path).convert("RGB")

            if img.size != target_size:
                img = img.resize(target_size)

            img_array = np.array(img) / 255.0
            images.append(img_array)
            y_true.append(class_idx)

    X_test = np.array(images)  # Shape: (n_samples, 128, 128, 3)
    y_true = np.array(y_true)

    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print(f"Evaluation for {model.friendly_name}")
    print("Classification Report:")
    # class_names.remove("Unknown")  # TODO(kkelso): unknown data
    # print(class_names)
    print(classification_report(y_true, y_pred, target_names=class_names))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    plot_confusion_matrix(y_true, y_pred, class_names, model.friendly_name)

    return y_true, y_pred, class_names


def train_all_nets():
    model_funs = [classification_network, deeper_cnn_network, residual_style_network]
    for fun in model_funs:
        model = fun()
        train_network(model)

    for fun in model_funs:
        model = fun()
        model = load_network(model)
        evaluate_model_performance(model)


def main():
    train_all_nets()


if __name__ == "__main__":
    main()
