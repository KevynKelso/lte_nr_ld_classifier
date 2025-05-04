import threading
from queue import Queue

import numpy as np
import pyaudio
from scipy.signal import decimate, lfilter, resample_poly

from constants import AUDIO_RATE, SAMPLE_RATE, SDR_BLOCK_SIZE


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
            # Shift to exact center frequency (compensate for SDR offset)
            freq_offset = 0  # Sometimes frequency comp is needed, the cpp guys use it in the LTE scanner (+/- 50 kHz)
            t = np.arange(len(samples)) / SAMPLE_RATE
            samples = samples * np.exp(-1j * 2 * np.pi * freq_offset * t)

            # Decimate to ~175 kHz first
            decimated = decimate(samples, 8, ftype="fir")

            # FM demodulation
            angle = np.unwrap(np.angle(decimated))
            demodulated = np.diff(angle) * (SAMPLE_RATE / 8) / (2 * np.pi)

            # Low-pass filter and decimate to audio rate
            audio = resample_poly(demodulated, AUDIO_RATE, int(SAMPLE_RATE / 8))

            # De-emphasis
            alpha = np.exp(-1 / (AUDIO_RATE * FMReceiver.deemphasis_tau))
            audio = lfilter([1 - alpha], [1, -alpha], audio)

            # Normalize and chunk
            audio /= np.max(np.abs(audio)) * 1.1

            # Play audio
            self.stream.write(audio.astype(np.float32).tobytes())
