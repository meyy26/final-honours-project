import os, sys, warnings
import numpy as np, mne, pywt
import config
from preprocess import preprocess

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

patient = sys.argv[1]
os.makedirs(config.WAVELET_DIR, exist_ok=True)
CH, win = config.CHANNELS, config.FS * config.WIN_SEC
BANDS = ["A5", "D5", "D4", "D3", "D2", "D1"]      # approx. 0-4, 4-8, 8-16, 16-32, 32-64, 64-128 Hz at 256 Hz

# same files, same window order as the other feature sets
ref = np.load(f"{config.FEATURE_DIR}/{patient}.npz")
files = list(dict.fromkeys(ref["source_file"]))

X_all = []
for fname in files:
    raw = mne.io.read_raw_edf(f"{config.RAW_DIR}/{patient}/{fname}", preload=True)
    picks = [[c for c in raw.ch_names if c == ch or c.startswith(ch + "-")][0] for ch in CH]
    data = preprocess(raw.get_data(picks=picks) * 1e6)                 # (18, samples), microvolts
    n = int((ref["source_file"] == fname).sum())
    w = data[:, :n * win].reshape(len(CH), n, win).transpose(1, 0, 2)  # (windows, 18, 1024)
    coefs = pywt.wavedec(w, config.WAVELET, level=5, mode="periodization", axis=-1)
    E = np.stack([(c ** 2).sum(axis=-1) for c in coefs], axis=1)       # (windows, 6 bands, 18 channels)
    rel = E / (E.sum(axis=1, keepdims=True) + 1e-12)                   # share of total energy per band
    X = np.concatenate([np.log10(E + 1e-12), rel], axis=1)             # (windows, 12, 18)
    X_all.append(X.reshape(n, -1).astype(np.float32))
    print(fname, n)

X = np.vstack(X_all)
assert X.shape[0] == ref["X"].shape[0], "row count differs from the feature file"
names = [f"wav_logE_{b}" for b in BANDS] + [f"wav_relE_{b}" for b in BANDS]
print("features:", X.shape, "non-finite:", int((~np.isfinite(X)).sum()))
np.savez(f"{config.WAVELET_DIR}/{patient}.npz", X=X, family=np.repeat(names, len(CH)),
         source_file=ref["source_file"], win_idx=ref["win_idx"])
print("saved", patient)
