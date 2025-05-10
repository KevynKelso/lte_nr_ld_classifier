# %% [markdown]
# ## Project 2: Spectrum Sensing with Deep Learning
# ### ECE 5625
#
# By Caleb Moore, Kevyn Kelso

# %% [markdown]
# # Introduction
#
# This project explores the use of deep learning for spectrum sensing - the task of automatically detecting and classifying different types of radio signals in a given frequency band. We implement and compare several convolutional neural network architectures for classifying spectrograms of various signal types including AM, FM, LTE, 5G NR, and noise.

# %% [markdown]
# ## System Architecture
#
# The system consists of several key components:
#
# 1. **SDR Interface**: Captures raw IQ samples from radio spectrum
# 2. **Signal Processing**: Converts IQ samples to spectrograms
# 3. **Deep Learning Models**: Classify spectrograms into signal types
# 4. **Audio Output**: Optional FM demodulation and audio playback
# 5. **GUI Program**: Used for class demo
#
# Here's the complete implementation:

# %% [markdown]
# ### Constants and Configuration
# %%
# constants.py
IMG_SIZE = (128, 128)
DATA_DIR = "data/lr_training_data"
BATCH_SIZE = 128
EPOCHS = 100
TEST_SPLIT = 0.2  # 20% for testing
VAL_SPLIT = 0.2  # 20% for validation

# SDR parameters
CORRECTION = 1.0  # ppm value
FS_LTE = 30720000.0
SAMPLE_RATE = (FS_LTE / 16) * CORRECTION
SDR_BLOCK_SIZE = 16 * 16384
NFFT = 4096

# Audio parameters
AUDIO_RATE = 48000  # Output audio sample rate

# %% [markdown]
# ### SDR Interface
# The SDR class handles communication with the software-defined radio and IQ sample capture.
# %%
# sdr.py
import os
import threading
from datetime import datetime
from os.path import isdir, join
from queue import Queue

import matplotlib.pyplot as plt
import numpy as np
from rtlsdr import RtlSdr
from scipy import signal
from scipy.fft import fftshift

class MySDR:
    default_center_freq = 99.9e6  # Default FM station frequency

    def __init__(self, buffer: Queue) -> None:
        self.sdr = RtlSdr()
        self.sdr.sample_rate = SAMPLE_RATE
        self.sdr.center_freq = MySDR.default_center_freq
        self.sdr.gain = "auto"
        self.iq_buffer = buffer
        self.enabled = False

    def __del__(self):
        self.sdr.close()

    def enable(self):
        self.enabled = True
        threading.Thread(target=self._sdr_producer, daemon=True).start()

    def disable(self):
        self.enabled = False

    def _sdr_producer(self):
        while self.enabled:
            self.iq_buffer.put(self.sdr.read_samples(SDR_BLOCK_SIZE))

def create_spectrogram(samples, filename):
    """
    Create and save 128x128 spectrogram image
    """
    sdr_dir = join("data", "sdr-captures")
    if not isdir(sdr_dir):
        os.mkdir(sdr_dir)

    f, t, Sxx = signal.spectrogram(samples, fs=SAMPLE_RATE, nperseg=256)
    Sxx_resized = Sxx[:128, :128]

    plt.pcolormesh(t[:128], f[:128], 10 * np.log10(Sxx_resized))
    plt.axis("off")
    plt.savefig(join(sdr_dir, filename), bbox_inches="tight", pad_inches=0)
    plt.close()

# %% [markdown]
# ### Audio Processing
# The FMReceiver class handles demodulation and playback of FM signals.
# %%
# audio.py
import threading
from queue import Queue

import numpy as np
import pyaudio
from scipy.signal import decimate, lfilter, resample_poly

class FMReceiver:
    chunk_size = SDR_BLOCK_SIZE  # Audio buffer size
    audio_cutoff = 15000  # Audio bandwidth (Hz)
    deemphasis_tau = 50e-6  # 50 μs deemphasis (US standard)

    def __init__(self, buffer: Queue) -> None:
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=AUDIO_RATE,
            output=True,
            frames_per_buffer=FMReceiver.chunk_size,
        )
        self.audio_buffer = buffer
        self.enabled = False

    def __del__(self):
        self.stream.stop_stream()
        self.stream.close()
        self.p.terminate()

    def enable(self):
        self.enabled = True
        threading.Thread(target=self._audio_consumer, daemon=True).start()

    def disable(self):
        self.enabled = False

    def is_enabled(self):
        return self.enabled

    def _audio_consumer(self):
        while self.enabled:
            samples = self.audio_buffer.get()
            # FM demodulation processing
            freq_offset = 0
            t = np.arange(len(samples)) / SAMPLE_RATE
            samples = samples * np.exp(-1j * 2 * np.pi * freq_offset * t)
            decimated = decimate(samples, 8, ftype="fir")
            angle = np.unwrap(np.angle(decimated))
            demodulated = np.diff(angle) * (SAMPLE_RATE / 8) / (2 * np.pi)
            audio = resample_poly(demodulated, AUDIO_RATE, int(SAMPLE_RATE / 8))
            alpha = np.exp(-1 / (AUDIO_RATE * FMReceiver.deemphasis_tau))
            audio = lfilter([1 - alpha], [1, -alpha], audio)
            audio /= np.max(np.abs(audio)) * 1.1
            self.stream.write(audio.astype(np.float32).tobytes())

# %% [markdown]
# ### Dataset Handling
# Functions for downloading from matlab and preparing it for training.
# %%
# dataset.py
import os
import shutil
import threading
import time
import zipfile
from glob import glob
from os.path import basename, isdir, isfile, join
from queue import Queue

import matplotlib.pyplot as plt
import numpy as np
import requests
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

DATA_URIS = [
    "https://www.mathworks.com/supportfiles/spc/SpectrumSensing/SpectrumSensingTrainingData128x128.zip",
    "https://www.mathworks.com/supportfiles/spc/SpectrumSensing/SpectrumSensingTrainedCustom_2024.zip",
    "https://www.mathworks.com/supportfiles/spc/SpectrumSensing/SpectrumSensingCapturedData128x128.zip",
]

def download_dataset(clean=False):
    """Download the datasets from matlab if they don't exist."""
    if clean and isdir(DATA_DIR):
        os.remove(DATA_DIR)
    if not isdir(DATA_DIR):
        os.mkdir(DATA_DIR)

    threads = []
    for uri in DATA_URIS:
        x = threading.Thread(target=_get_resource, args=(uri,))
        x.start()
        threads.append(x)
    _ = [t.join() for t in threads]

def load_data(data_dir, img_size, test_split, normalize=True):
    """Load data from training directory"""
    class_names = sorted(os.listdir(data_dir))
    images = []
    labels = []
    
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(data_dir, class_name)
        for img_file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, img_file)
            img = tf.keras.utils.load_img(img_path, target_size=img_size)
            img_array = tf.keras.utils.img_to_array(img)
            images.append(img_array)
            labels.append(class_idx)

    images = np.array(images)
    labels = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(
        images, labels, test_size=test_split, random_state=42, stratify=labels
    )
    
    if normalize:
        X_train = X_train / 255.0
        X_test = X_test / 255.0

    return X_train, X_test, y_train, y_test, class_names

# %% [markdown]
# ### Neural Network Architectures
# We implement and compare three different CNN architectures.
# %%
# networks.py
from tensorflow.keras import Model, layers, models, optimizers

def classification_network():
    """Basic CNN architecture"""
    class_names = sorted(os.listdir(DATA_DIR))
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(len(class_names), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.my_class_names = class_names
    model.friendly_name = "classifier_v2"
    return model

def deeper_cnn_network():
    """Deeper CNN with batch normalization and dropout"""
    class_names = sorted(os.listdir(DATA_DIR))
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        layers.Conv2D(256, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(len(class_names), activation="softmax"),
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.my_class_names = class_names
    model.friendly_name = "deeper_cnn_v1"
    return model

def residual_style_network():
    """Residual network inspired by ResNet"""
    class_names = sorted(os.listdir(DATA_DIR))
    inputs = layers.Input(shape=(128, 128, 3))
    x = layers.Conv2D(32, (7, 7), strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D((3, 3), strides=2, padding="same")(x)
    
    # Residual blocks
    residual = x
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([x, residual])
    
    residual = layers.Conv2D(128, (1, 1), strides=2)(x)
    x = layers.Conv2D(128, (3, 3), strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.add([x, residual])
    
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(len(class_names), activation="softmax")(x)
    
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.my_class_names = class_names
    model.friendly_name = "resnet_style_v1"
    return model

# %% [markdown]
# ### Training and Evaluation
# Functions for training the models and evaluating their performance.
# %%
# training.py
import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

def train_network(model):
    """Train a given model"""
    X_train, X_test, y_train, y_test, class_names = load_data(
        DATA_DIR, IMG_SIZE, TEST_SPLIT
    )
    y_train_onehot = tf.keras.utils.to_categorical(y_train, num_classes=len(class_names))
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

    # Plot training history
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

def evaluate_model_performance(model, sample_size=500):
    """Evaluate model on test data"""
    class_names = sorted(os.listdir(DATA_DIR))
    images = []
    y_true = []

    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(DATA_DIR, class_name)
        all_files = [f for f in os.listdir(class_dir) if f.endswith(".png")]
        sampled_files = random.sample(all_files, min(sample_size, len(all_files)))

        for filename in sampled_files:
            img_path = os.path.join(class_dir, filename)
            img = tf.keras.utils.load_img(img_path, target_size=IMG_SIZE)
            img_array = tf.keras.utils.img_to_array(img) / 255.0
            images.append(img_array)
            y_true.append(class_idx)

    X_test = np.array(images)
    y_true = np.array(y_true)
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print(f"Evaluation for {model.friendly_name}")
    print(classification_report(y_true, y_pred, target_names=class_names))
    plot_confusion_matrix(y_true, y_pred, class_names, model.friendly_name)

def plot_confusion_matrix(y_true, y_pred, class_names, name):
    """Plot labeled confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
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
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{name}-confusion-matrix.png")
    plt.clf()

# %% [markdown]
# ### GUI Application for Class Demo
# We implemented a simple GUI application to demo in the class presentation.
# The GUI combines all the elements of the project together.
# %%
# gui.py
import threading
import tkinter as tk
from queue import Empty, Queue
from tkinter import ttk

import numpy as np
from PIL import ImageTk

from audio import FMReceiver
from constants import IMG_SIZE, NFFT
from networks import classification_network, load_network
from sdr import MySDR
from utils import get_spectrogram_image


class SDRSpectrogramGUI:
    def __init__(self, root):
        self.iq_buffer = Queue()
        self.audio_buffer = Queue()
        self.img_buffer = Queue()

        self.sdr = MySDR(self.iq_buffer)
        self.fm_rx = FMReceiver(self.audio_buffer)

        self.model = classification_network()
        print(f"Loading {self.model.friendly_name}")
        self.model = load_network(self.model)

        self.root = root
        self.root.title("Spectrogram Classifier")
        self.running = False
        self.setup_gui()

    def setup_gui(self):
        # Control Frame
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)

        # Frequency Control
        ttk.Label(control_frame, text="Center Freq (MHz):").grid(row=0, column=0)

        # Frequency Entry
        self.freq_entry_var = tk.StringVar(value=f"{self.sdr.sdr.center_freq/1e6:.2f}")
        self.freq_entry = ttk.Entry(
            control_frame, textvariable=self.freq_entry_var, width=10
        )
        self.freq_entry.grid(row=0, column=1)

        # Frequency Set Button
        self.set_freq_btn = ttk.Button(
            control_frame, text="Set", command=self.set_center_frequency
        )
        self.freq_entry.bind("<Return>", lambda event: self.set_center_frequency())
        self.set_freq_btn.grid(row=0, column=2, padx=5)

        # Frequency Display
        self.freq_var = tk.StringVar(value=f"{self.sdr.sdr.center_freq/1e6:.2f}")
        ttk.Label(control_frame, textvariable=self.freq_var).grid(row=0, column=3)

        # Start/Stop Button
        self.btn_text = tk.StringVar(value="Start")
        self.btn = ttk.Button(
            control_frame, textvariable=self.btn_text, command=self.toggle_sdr
        )
        self.btn.grid(row=0, column=4, padx=10)

        # Audio Button
        self.aud_btn_text = tk.StringVar(value="Enable FM demod")
        self.aud_btn = ttk.Button(
            control_frame, textvariable=self.aud_btn_text, command=self.toggle_audio
        )
        self.aud_btn.grid(row=0, column=5, padx=10)

        # Status Display
        self.status_var = tk.StringVar(value="Status: Ready")
        ttk.Label(control_frame, textvariable=self.status_var).grid(row=1, column=0)

        # Prediction Display
        self.prediction_var = tk.StringVar(value="")
        ttk.Label(control_frame, textvariable=self.prediction_var).grid(row=2, column=2)

        # Spectrogram Display
        self.canvas = tk.Canvas(self.root, width=200, height=200)
        self.canvas.pack()
        self.photo = None

        # Text Info Display
        self.text_display = tk.Text(self.root, height=10, wrap=tk.WORD)
        self.text_display.pack(fill=tk.BOTH, expand=True)
        self.text_display.insert(tk.END, "System initialized\nPress Start to begin\n")

    def set_center_frequency(self):
        try:
            freq_mhz = float(self.freq_entry_var.get())
            freq_hz = int(freq_mhz * 1e6)
            self.sdr.sdr.set_center_freq(freq_hz)
            self.freq_var.set(f"{freq_mhz:.2f}")
            self.text_display.insert(tk.END, f"Frequency set to {freq_mhz:.2f} MHz\n")
        except ValueError:
            self.text_display.insert(tk.END, "Invalid frequency value\n")

    def toggle_audio(self):
        if not self.fm_rx.is_enabled():
            self.fm_rx.enable()
            self.aud_btn_text.set("Disable FM demod")
            self.text_display.insert(tk.END, "Starting audio...\n")
        else:
            self.fm_rx.disable()
            self.aud_btn_text.set("Enable FM demod")
            self.text_display.insert(tk.END, "Stopping audio...\n")

    def toggle_sdr(self):
        if not self.running:
            self.sdr.enable()
            self.running = True
            threading.Thread(target=self.main_thread, daemon=True).start()
            self.btn_text.set("Stop")
            self.status_var.set("Status: Running...")
            self.text_display.insert(tk.END, "Starting SDR...\n")
            self.update_image()
        else:
            self.running = False
            self.sdr.disable()
            self.btn_text.set("Start")
            self.status_var.set("Status: Stopped")
            self.text_display.insert(tk.END, "Stopping SDR...\n")

    def main_thread(self):
        while self.running:
            samples = self.iq_buffer.get()

            if self.fm_rx.is_enabled():
                self.audio_buffer.put(samples)

            if self.img_buffer.empty():
                self.img_buffer.put(samples)

    def update_image(self):
        try:
            if not self.img_buffer.empty():
                samples = self.img_buffer.get_nowait()
                pil_image = get_spectrogram_image(
                    samples, NFFT, self.sdr.sdr.sample_rate, IMG_SIZE
                )

                image_dl = np.array(pil_image) / 255.0
                image_dl = np.expand_dims(image_dl, axis=0)
                y_pred = self.model.predict(image_dl)
                classification = np.argmax(y_pred, axis=1)[0]
                self.text_display.insert(tk.END, str(y_pred) + "\n")
                self.prediction_var.set(
                    f"Prediction: {self.model.my_class_names[classification]}"
                )

                # Update display
                self.photo = ImageTk.PhotoImage(pil_image)
                self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

                # Update frequency display
                current_freq = self.sdr.sdr.get_center_freq()
                self.freq_var.set(f"{current_freq/1e6:.2f}")

        except Empty:
            pass

        # Schedule next update
        if self.running:
            self.root.after(25, self.update_image)

    def on_closing(self):
        self.running = False
        self.toggle_sdr()
        self.root.destroy()


# Uncomment to launch GUI (requires RTL SDR connected)
# if __name__ == "__main__":
#     root = tk.Tk()
#     app = SDRSpectrogramGUI(root)
#     root.protocol("WM_DELETE_WINDOW", app.on_closing)
#     root.mainloop()

# %% [markdown]
# ## Signal Class Descriptions
#
# #### 1. AM (Amplitude Modulation)
# - **Description:** The *amplitude* of the carrier signal varies with the message signal.
# - **Usage:** Common in AM radio broadcasting.
# - **Spectral Features:** Strong carrier peak and symmetrical sidebands.
#
# #### 2. FM (Frequency Modulation)
# - **Description:** The *frequency* of the carrier wave varies with the message signal.
# - **Usage:** Used in FM radio, analog TV audio, and two-way radios.
# - **Spectral Features:** Wider bandwidth than AM with characteristic deviation patterns.
#
# #### 3. LTE (Long-Term Evolution)
# - **Description:** A 4G mobile communication standard that uses OFDM (Orthogonal Frequency-Division Multiplexing).
# - **Usage:** Mobile broadband data and voice (VoLTE).
# - **Spectral Features:** Structured frequency bins with periodic pilot tones.
#
# #### 4. Noise
# - **Description:** Random, unstructured electromagnetic interference.
# - **Usage:** Represents background noise or thermal interference.
# - **Spectral Features:** Flat spectrum (in white noise), no distinct structure.
#
# #### 5. NR (New Radio)
# - **Description:** The 5G air interface, more flexible than LTE but also based on OFDM.
# - **Usage:** 5G mobile and IoT (Internet of Things) communication.
# - **Spectral Features:** Similar to LTE, with variable subcarrier spacing and time/frequency granularity.
#
# #### 6. Unknown
# - **Description:** Signal does not match any known class or is unrecognizable.
# - **Usage:** Represents valid/un-implement signals or corrupted signals
# - **Spectral Features:** Varies significantly; no consistent structure.

# %% [markdown]
# ### Data Exploration
# Below are random samples of each signal type from the dataset:
# %%
import os
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

plt.figure(figsize=(15, 18))
class_names = sorted(os.listdir(DATA_DIR))
for i in range(4):
    img = mpimg.imread(f'final_plots/{i}.png')
    plt.subplot(3,2,i + 1)
    plt.imshow(img)
    plt.axis('off')
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.title(class_names[i], fontsize=11)

img = mpimg.imread(f'final_plots/{i}.png')
plt.subplot(3,2,5)
plt.imshow(img)
plt.axis('off')
plt.subplots_adjust(wspace=0.1, hspace=0.1)
plt.title('Unknown', fontsize=11)

# %% [markdown]
# ### Model Architectures
# We implemented and compared three different CNN architectures:

# %% [markdown]
# #### 1. Basic CNN (classifier_v2)
# | Layer (Type)               | Output Shape        | Param #   |
# |----------------------------|---------------------|-----------|
# | conv2d (Conv2D)            | (None, 126, 126, 32)| 896       |
# | max_pooling2d (MaxPooling2D)| (None, 63, 63, 32)  | 0         |
# | conv2d_1 (Conv2D)          | (None, 61, 61, 64)  | 18,496    |
# | max_pooling2d_1 (MaxPooling2D)| (None, 30, 30, 64)| 0         |
# | conv2d_2 (Conv2D)          | (None, 28, 28, 128) | 73,856    |
# | flatten (Flatten)          | (None, 100352)      | 0         |
# | dense (Dense)              | (None, 128)         | 12,845,184|
# | dense_1 (Dense)            | (None, 5)           | 645       |
#
# **Total params**: 12,939,077 (49.36 MB)  
# **Trainable params**: 12,939,077 (49.36 MB)  

# %% [markdown]
# #### 2. Deeper CNN (deeper_cnn_v1)
# | Layer (Type)               | Output Shape        | Param #   |
# |----------------------------|---------------------|-----------|
# | conv2d_3 (Conv2D)          | (None, 126, 126, 32)| 896       |
# | batch_normalization        | (None, 126, 126, 32)| 128       |
# | max_pooling2d_2 (MaxPooling2D)| (None, 63, 63, 32)| 0         |
# | dropout                    | (None, 63, 63, 32) | 0         |
# | conv2d_4 (Conv2D)          | (None, 61, 61, 64) | 18,496    |
# | batch_normalization_1      | (None, 61, 61, 64) | 256       |
# | max_pooling2d_3 (MaxPooling2D)| (None, 30, 30, 64)| 0         |
# | dropout_1                  | (None, 30, 30, 64) | 0         |
# | conv2d_5 (Conv2D)          | (None, 28, 28, 128)| 73,856    |
# | batch_normalization_2      | (None, 28, 28, 128)| 512       |
# | max_pooling2d_4 (MaxPooling2D)| (None, 14, 14, 128)| 0         |
# | dropout_2                  | (None, 14, 14, 128)| 0         |
# | conv2d_6 (Conv2D)          | (None, 12, 12, 256)| 295,168   |
# | batch_normalization_3      | (None, 12, 12, 256)| 1,024     |
# | global_average_pooling2d   | (None, 256)         | 0         |
# | dense_2 (Dense)            | (None, 256)         | 65,792    |
# | dropout_3                  | (None, 256)         | 0         |
# | dense_3 (Dense)            | (None, 5)           | 1,285     |
#
# **Total params**: 457,413 (1.74 MB)  
# **Trainable params**: 456,453 (1.74 MB)  

# %% [markdown]
# #### 3. ResNet-style (resnet_style_v1)
# | Layer (Type)               | Output Shape        | Param #   |
# |----------------------------|---------------------|-----------|
# | input_1 (InputLayer)       | (None, 128, 128, 3)| 0         |
# | conv2d_7 (Conv2D)          | (None, 64, 64, 32) | 4,736     |
# | batch_normalization_4      | (None, 64, 64, 32) | 128       |
# | activation                 | (None, 64, 64, 32) | 0         |
# | max_pooling2d_5 (MaxPooling2D)| (None, 32, 32, 32)| 0         |
# | conv2d_8 (Conv2D)          | (None, 32, 32, 64) | 18,496    |
# | batch_normalization_5      | (None, 32, 32, 64) | 256       |
# | activation_1               | (None, 32, 32, 64) | 0         |
# | conv2d_9 (Conv2D)          | (None, 32, 32, 32) | 18,464    |
# | batch_normalization_6      | (None, 32, 32, 32) | 128       |
# | add                        | (None, 32, 32, 32) | 0         |
# | conv2d_11 (Conv2D)         | (None, 16, 16, 128)| 36,992    |
# | batch_normalization_7      | (None, 16, 16, 128)| 512       |
# | activation_2               | (None, 16, 16, 128)| 0         |
# | conv2d_12 (Conv2D)         | (None, 16, 16, 128)| 147,584   |
# | batch_normalization_8      | (None, 16, 16, 128)| 512       |
# | conv2d_10 (Conv2D)         | (None, 16, 16, 128)| 4,224     |
# | add_1                      | (None, 16, 16, 128)| 0         |
# | global_average_pooling2d_1 | (None, 128)        | 0         |
# | dense_4 (Dense)            | (None, 256)        | 33,024    |
# | dense_5 (Dense)            | (None, 5)          | 1,285     |
#
# **Total params**: 266,341 (1.02 MB)  
# **Trainable params**: 265,573 (1.01 MB)  

# %% [markdown]
# ### Training Process
# The models were trained using categorical cross-entropy loss:
#
# $$
# \text{Loss} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)
# $$
#
# Where:
# - \( C \) = number of classes  
# - \( y_i \) = true label (one-hot encoded)  
# - \( \hat{y}_i \) = predicted probability for class \( i \)
#
# Below are the training curves for the basic CNN:

# %% [markdown]
# ### Training curves for basic CNN (classifier_v2)

# %%
display(Image(filename='final_plots/classifier_v2-accuracy.png'))
display(Image(filename='final_plots/classifier_v2-loss.png'))

# %% [markdown]
# ### Training curves for deep CNN (deeper_cnn_v1)

# %%
display(Image(filename='final_plots/deeper_cnn_v1-accuracy.png'))
display(Image(filename='final_plots/deeper_cnn_v1-loss.png'))

# %% [markdown]
# ### Training curves for RESNET (resnet_style_v1)

# %%
display(Image(filename='final_plots/resnet_style_v1-accuracy.png'))
display(Image(filename='final_plots/resnet_style_v1-loss.png'))

# %% [markdown]
# ### Results
#
# #### Evaluation for classifier_v2
# | Class   | Precision | Recall | F1-Score | Support |
# |---------|-----------|--------|----------|---------|
# | AM      | 0.99      | 1.00   | 1.00     | 500     |
# | FM      | 1.00      | 1.00   | 1.00     | 500     |
# | LTE     | 0.99      | 0.99   | 0.99     | 500     |
# | NR      | 1.00      | 0.99   | 0.99     | 500     |
# | Noise   | 1.00      | 1.00   | 1.00     | 500     |
# |         |           |        |          |         |
# | **Accuracy** |         |        | **1.00** | **2500** |
# | **Macro Avg** | 1.00   | 1.00   | **1.00** | **2500** |
#
# #### Evaluation for deeper_cnn_v1
# | Class   | Precision | Recall | F1-Score | Support |
# |---------|-----------|--------|----------|---------|
# | AM      | 0.98      | 1.00   | 0.99     | 500     |
# | FM      | 1.00      | 1.00   | 1.00     | 500     |
# | LTE     | 1.00      | 1.00   | 1.00     | 500     |
# | NR      | 1.00      | 1.00   | 1.00     | 500     |
# | Noise   | 1.00      | 0.97   | 0.99     | 500     |
# |         |           |        |          |         |
# | **Accuracy** |         |        | **0.99** | **2500** |
# | **Macro Avg** | 0.99   | 0.99   | **0.99** | **2500** |
#
# #### Evaluation for resnet_style_v1
# | Class   | Precision | Recall | F1-Score | Support |
# |---------|-----------|--------|----------|---------|
# | AM      | 0.99      | 1.00   | 1.00     | 500     |
# | FM      | 1.00      | 1.00   | 1.00     | 500     |
# | LTE     | 1.00      | 1.00   | 1.00     | 500     |
# | NR      | 0.99      | 1.00   | 1.00     | 500     |
# | Noise   | 1.00      | 0.99   | 1.00     | 500     |
# |         |           |        |          |         |
# | **Accuracy** |         |        | **1.00** | **2500** |
# | **Macro Avg** | 1.00   | 1.00   | **1.00** | **2500** |

# %% [markdown]
# ### Confusion Matrices
# %%

# %% [markdown]
# ### Confusion matrix for basic CNN (classifier_v2)

# %%
display(Image(filename='final_plots/classifier_v2-confusion-matrix.png'))

# %% [markdown]
# ### Confusion matrix for deep CNN (deeper_cnn_v1)

# %%
display(Image(filename='final_plots/deeper_cnn_v1-confusion-matrix.png'))

# %% [markdown]
# ### Confusion matrix for RESNET (resnet_style_v1)

# %%
display(Image(filename='final_plots/resnet_style_v1-confusion-matrix.png'))

# %% [markdown]
# ## Conclusion
#
# All three architectures achieved excellent performance (>99% accuracy) on the spectrum sensing task. The basic CNN had the most parameters but similar accuracy to the more sophisticated architectures. The ResNet-style model achieved perfect classification while using only 2% of the parameters of the basic CNN, making it the most efficient choice for deployment.
#
# Future work could explore:
# - Detection of unknown signal types.
# - Multi-label classification for overlapping signals using semantic segmentation.
# - Model architecture optimizations for better performance in resource constrained environments.
