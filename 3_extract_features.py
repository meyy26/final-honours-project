import sys, os, warnings
import numpy as np
import mne
from scipy.signal import welch
from scipy.stats import skew, kurtosis
import config
from preprocess import preprocess

warnings.filterwarnings("ignore")
mne.set_log_level("ERROR")

patient = sys.argv[1]                                   # e.g. chb01
fs, win = config.FS, config.FS * config.WIN_SEC
CH = config.CHANNELS
raw_dir = f"{config.RAW_DIR}/{patient}"
os.makedirs(config.FEATURE_DIR, exist_ok=True)

# left/right mirror channels (used for the spatial features)
PAIRS = [("FP1-F7","FP2-F8"), ("F7-T7","F8-T8"), ("T7-P7","T8-P8"), ("P7-O1","P8-O2"),
         ("FP1-F3","FP2-F4"), ("F3-C3","F4-C4"), ("C3-P3","C4-P4"), ("P3-O1","P4-O2")]
L = [CH.index(a) for a, b in PAIRS]
R = [CH.index(b) for a, b in PAIRS]
PAIR_NAMES = [f"{a}~{b}" for a, b in PAIRS]

def window_features(w):
    """w: (windows, channels, samples) in microvolts -> list of feature blocks."""
    n = w.shape[0]
    w = w - w.mean(axis=2, keepdims=True)               # remove each window's DC offset
    blocks = []
    def add(group, family, labels, arr):
        blocks.append((group, family, labels, np.asarray(arr).reshape(n, -1)))

    # ---- temporal: one value per channel ----
    var = w.var(axis=2)
    d1 = np.diff(w, axis=2)                             # first difference
    d2 = np.diff(d1, axis=2)
    mob = np.sqrt(d1.var(axis=2) / (var + 1e-12))       # Hjorth mobility
    mob_d = np.sqrt(d2.var(axis=2) / (d1.var(axis=2) + 1e-12))
    add("temporal", "variance", CH, np.log10(var + 1e-12))
    add("temporal", "line_length", CH, np.log10(np.abs(d1).sum(axis=2) + 1e-12))
    add("temporal", "hjorth_mobility", CH, mob)
    add("temporal", "hjorth_complexity", CH, mob_d / (mob + 1e-12))
    add("temporal", "skewness", CH, skew(w, axis=2))
    add("temporal", "kurtosis", CH, kurtosis(w, axis=2))
    add("temporal", "zero_crossings", CH,
        (np.diff(np.signbit(w).astype(np.int8), axis=2) != 0).sum(axis=2))

    # ---- spectral: one value per channel ----
    freqs, psd = welch(w, fs=fs, nperseg=win, axis=2)   # (windows, channels, freqs)
    for name, (lo, hi) in config.BANDS.items():
        m = (freqs >= lo) & (freqs < hi)
        add("spectral", f"bandpower_{name}", CH, np.log10(psd[:, :, m].sum(axis=2) + 1e-12))
    m = (freqs >= 0.5) & (freqs < 45)
    p = psd[:, :, m]
    p = p / (p.sum(axis=2, keepdims=True) + 1e-12)      # treat spectrum as a distribution
    add("spectral", "spectral_entropy", CH,
        -(p * np.log(p + 1e-12)).sum(axis=2) / np.log(p.shape[2]))

    # ---- spatial: relationships between channels ----
    z = w / (w.std(axis=2, keepdims=True) + 1e-8)
    corr = np.einsum("nct,ndt->ncd", z, z) / win        # (windows, 18, 18) correlation matrix
    add("spatial", "homologous_corr", PAIR_NAMES, corr[:, L, R])
    add("spatial", "asymmetry", PAIR_NAMES,
        np.log10(var[:, L] + 1e-12) - np.log10(var[:, R] + 1e-12))
    off_diag = np.abs(corr).sum(axis=(1, 2)) - np.abs(np.diagonal(corr, axis1=1, axis2=2)).sum(axis=1)
    add("spatial", "global_corr", ["all"], off_diag / (len(CH) * (len(CH) - 1)))
    add("spatial", "eig_synchrony", ["all"], np.linalg.eigvalsh(corr)[:, -1] / len(CH))
    return blocks

X_all, src, widx = [], [], []
names = groups = fams = None
for fname in sorted(os.listdir(raw_dir)):
    if not fname.endswith(".edf"):
        continue
    raw = mne.io.read_raw_edf(f"{raw_dir}/{fname}", preload=True)
    picks = []
    for ch in CH:
        match = [c for c in raw.ch_names if c == ch or c.startswith(ch + "-")]
        if not match:
            break
        picks.append(match[0])
    if len(picks) < len(CH) or int(raw.info["sfreq"]) != config.FS:
        print("skipping:", fname); continue

    data = preprocess(raw.get_data(picks=picks) * 1e6)  # volts -> microvolts, then filter
    n = data.shape[1] // win
    w = data[:, :n * win].reshape(len(CH), n, win).transpose(1, 0, 2)
    blocks = window_features(w)
    if names is None:
        names = [f"{b[1]}__{l}" for b in blocks for l in b[2]]
        groups = [b[0] for b in blocks for l in b[2]]
        fams = [b[1] for b in blocks for l in b[2]]
    X_all.append(np.concatenate([b[3] for b in blocks], axis=1))
    src += [fname] * n
    widx.append(np.arange(n))
    print(fname, "windows:", n)

X = np.vstack(X_all).astype(np.float32)
print("feature matrix:", X.shape, "| non-finite values:", int((~np.isfinite(X)).sum()))
np.savez(f"{config.FEATURE_DIR}/{patient}.npz", X=X, source_file=np.array(src),
         win_idx=np.concatenate(widx), feature_names=np.array(names),
         group=np.array(groups), family=np.array(fams))
print("saved", patient)
