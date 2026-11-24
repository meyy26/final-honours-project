import numpy as np
from scipy.signal import butter, filtfilt, iirnotch
import config

def preprocess(data):
    """data: (channels, samples) -> filtered array of the same shape."""
    # 1) band-pass 0.5-40 Hz: removes slow drift and high-frequency noise
    b, a = butter(4, config.BANDPASS, btype="bandpass", fs=config.FS)
    data = filtfilt(b, a, data, axis=1)
    # 2) notch at 60 Hz: removes mains interference
    b, a = iirnotch(config.NOTCH, Q=30, fs=config.FS)
    return filtfilt(b, a, data, axis=1)
