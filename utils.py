import cv2
import numpy as np
from PIL import Image
from scipy.signal import spectrogram, windows


def get_spectrogram_image(samples, Nfft, sample_rate, img_size):
    """
    Create spectrogram image from IQ samples
    """
    # Generate spectrogram
    window = windows.hann(256)
    nperseg = len(window)
    noverlap = 10  # samples overlap

    _, _, P = spectrogram(
        samples,
        fs=sample_rate,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=Nfft,
        return_onesided=False,
        scaling="density",
        mode="psd",
    )

    # Convert to logarithmic scale and transpose
    P = 10 * np.log10(np.abs(P.T) + np.finfo(float).eps)

    # Rescale to 0-255 and convert to uint8
    P_rescaled = cv2.normalize(P, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # Resize using nearest-neighbor
    im = cv2.resize(P_rescaled, img_size, interpolation=cv2.INTER_NEAREST)

    im_rgb = cv2.applyColorMap(im, cv2.COLORMAP_PARULA)
    # Flip vertically to match MATLAB orientation
    im_rgb = cv2.flip(im_rgb, 0)

    # Convert to PIL format
    image = cv2.cvtColor(im_rgb, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)

    return pil_image
