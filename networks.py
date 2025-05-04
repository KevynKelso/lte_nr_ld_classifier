import os
from os.path import isfile

from tensorflow.keras import Model, layers, models, optimizers

from constants import DATA_DIR, IMG_SIZE


def tutorial_network():
    pass


def pretrained_network():
    pass


def classification_network():
    class_names = sorted(os.listdir(DATA_DIR))
    model = models.Sequential(
        [
            layers.Conv2D(
                32, (3, 3), activation="relu", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
            ),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(len(class_names), activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    # Note, model name is set for saving
    model.my_class_names = sorted(os.listdir(DATA_DIR))
    model.friendly_name = "classifier_v2"

    return model


def deeper_cnn_network():
    class_names = sorted(os.listdir(DATA_DIR))

    model = models.Sequential(
        [
            # Block 1
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # Block 2
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # Block 3
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            # Block 4 (Deeper)
            layers.Conv2D(256, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling2D(),  # Reduces spatial dimensions
            # Classifier
            layers.Dense(256, activation="relu"),
            layers.Dropout(0.5),
            layers.Dense(len(class_names), activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    model.my_class_names = class_names
    model.friendly_name = "deeper_cnn_v1"
    return model


def residual_style_network():
    class_names = sorted(os.listdir(DATA_DIR))
    inputs = layers.Input(shape=(128, 128, 3))

    # Initial Conv Block
    x = layers.Conv2D(32, (7, 7), strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding="same")(x)

    # Residual Block 1
    residual = x
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([x, residual])  # Skip connection

    # Residual Block 2
    residual = layers.Conv2D(128, (1, 1), strides=2)(x)  # Match dimensions
    x = layers.Conv2D(128, (3, 3), strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([x, residual])

    # Classifier Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(len(class_names), activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )

    model.my_class_names = class_names
    model.friendly_name = "resnet_style_v1"
    return model


def load_network(model):
    assert isfile(
        f"{model.friendly_name}.keras"
    ), f"Trained model {model.friendly_name} doesn't exist"
    friendly_name = model.friendly_name
    model = models.load_model(f"{model.friendly_name}.keras")
    model.my_class_names = sorted(os.listdir(DATA_DIR))
    model.friendly_name = friendly_name

    return model
