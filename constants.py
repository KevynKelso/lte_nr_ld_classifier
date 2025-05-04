# Deep learning params
DATA_DIR = "data/lr_training_data"  # Folder with subfolders per class
IMG_SIZE = (128, 128)  # Match spectrogram dimensions
BATCH_SIZE = 256
EPOCHS = 50
TEST_SPLIT = 0.2  # 20% for testing
VAL_SPLIT = 0.2  # 20% of training for validation

# SDR params
CORRECTION = 1.0  # ppm value
FS_LTE = 30720000.0
SDR_BLOCK_SIZE = 16 * 16384
SAMPLE_RATE = (FS_LTE / 16) * CORRECTION
# SAMPLE_RATE=1e6
NFFT = 4096

# Audio params
AUDIO_RATE = 48000  # Output audio sample rate
