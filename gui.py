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


if __name__ == "__main__":
    root = tk.Tk()
    app = SDRSpectrogramGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    root.mainloop()
