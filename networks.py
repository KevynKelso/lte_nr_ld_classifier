import os

from tensorflow.keras import layers, models

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

    return model
