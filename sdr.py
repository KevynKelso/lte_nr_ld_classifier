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

from constants import SAMPLE_RATE, SDR_BLOCK_SIZE

# LTE/NR detection parameters
PREAMBLE_LENGTH = 128  # Length of preamble to detect


class MySDR:
    default_center_freq = 99.9e6  # my 999 local fm station

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
        # self.sdr.close()

    def _sdr_producer(self):
        while self.enabled:
            self.iq_buffer.put(self.sdr.read_samples(SDR_BLOCK_SIZE))


def detect_preamble(samples):
    """
    Detect LTE/NR preamble using correlation with known patterns
    Returns True if preamble detected
    """
    # TODO(kkelso): look at cpp lib for doing this.
    return True
    # # Known preamble patterns (simplified example)
    # lte_preamble = np.exp(1j * np.pi/4 * np.arange(PREAMBLE_LENGTH))
    # nr_preamble = np.exp(1j * np.pi/2 * np.arange(PREAMBLE_LENGTH))
    #
    # # Cross-correlation with known patterns
    # corr_lte = np.abs(signal.correlate(samples, lte_preamble, mode='valid'))
    # corr_nr = np.abs(signal.correlate(samples, nr_preamble, mode='valid'))
    #
    # # Threshold for detection (adjust based on your environment)
    # if np.max(corr_lte) > 0.8*np.max(np.abs(samples)) or \
    #    np.max(corr_nr) > 0.8*np.max(np.abs(samples)):
    #     return True
    # return False


def helper_spec_sense_spectrogram_image(x, Nfft, sr, imgSize):
    # Generate spectrogram using matplotlib
    plt.figure(figsize=(imgSize[0] / 100, imgSize[1] / 100), dpi=100)
    plt.axis("off")
    plt.specgram(
        x, NFFT=Nfft, Fs=sr, window=hann(256), noverlap=10, mode="psd", scale="dB"
    )
    plt.ylim(-sr / 2, sr / 2)  # 'centered' equivalent

    # Save to memory buffer and read back as image
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    img = cv2.imdecode(np.frombuffer(buf.getvalue(), dtype=np.uint8), 1)
    plt.close()

    # Process image to match MATLAB output
    img = cv2.resize(img, imgSize, interpolation=cv2.INTER_NEAREST)
    return img


def create_spectrogram(samples, filename):
    """
    Create and save 128x128 spectrogram image
    """
    sdr_dir = join("data", "sdr-captures")
    if not isdir(sdr_dir):
        os.mkdir(sdr_dir)

    f, t, Sxx = signal.spectrogram(samples, fs=SAMPLE_RATE, nperseg=256)

    Sxx_resized = Sxx[:128, :128]

    # plt.figure(figsize=(1.28, 1.28), dpi=100)
    plt.pcolormesh(t[:128], f[:128], 10 * np.log10(Sxx_resized))
    plt.axis("off")
    plt.savefig(join(sdr_dir, filename), bbox_inches="tight", pad_inches=0)
    plt.close()


def main():
    print("Starting LTE/NR preamble detection...")
    try:
        while True:
            samples = sdr.read_samples(NUM_SAMPLES)

            if detect_preamble(samples):
                print("Preamble detected! Creating spectrogram...")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"spectrogram_{timestamp}.png"
                create_spectrogram(samples, filename)
            break

    except KeyboardInterrupt:
        pass
    finally:
        sdr.close()
        print("SDR closed")


if __name__ == "__main__":
    main()
