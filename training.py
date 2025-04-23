import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from constants import (BATCH_SIZE, DATA_DIR, EPOCHS, IMG_SIZE, TEST_SPLIT,
                       VAL_SPLIT)
from dataset import load_data


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

    history = model.fit(
        X_train,
        y_train_onehot,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VAL_SPLIT,
        verbose=1,
    )
    model.save(f"{model.friendly_name}.keras")

    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
    plt.savefig("history.png")

    test_loss, test_acc = model.evaluate(X_test, y_test_onehot, verbose=0)
    print(f"\nTest Accuracy: {test_acc:.4f}")

    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    print("\nClassification Report:")
    # TODO: target names add noise and uknown when we have that data
    print(classification_report(y_test, y_pred_classes, target_names=["LTE", "NR"]))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred_classes))
