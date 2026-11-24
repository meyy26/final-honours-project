import os, copy, functools
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
import config

torch.set_num_threads(2)
FEAT, WAV, LAB, WIN = config.FEATURE_DIR, config.WAVELET_DIR, config.LABEL_DIR, config.WIN_SEC

@functools.lru_cache(maxsize=1)
def _families():
    ref = sorted(f[:-4] for f in os.listdir(LAB) if f.endswith(".npz"))[0]
    fo, fw = np.load(f"{FEAT}/{ref}.npz"), np.load(f"{WAV}/{ref}.npz")
    return [str(x) for x in dict.fromkeys(fo["family"])], [str(x) for x in dict.fromkeys(fw["family"])]

def resolve_features(spec):
    fo, fw = _families()
    fams = fo + fw if spec == "all" else spec.split(",")
    unknown = [f for f in fams if f not in fo + fw]
    if unknown:
        raise SystemExit(f"unknown families {unknown}")
    return fams

def usable(pat):
    l = np.load(f"{LAB}/{pat}.npz")
    ok = l["event_eligible"] & (l["event_preictal"] >= config.MIN_WINDOWS_PER_CLASS)
    return int(ok.sum()) >= config.MIN_SEIZURE_EVENTS and int((l["y"] == 0).sum()) >= 1000

def all_usable():
    return [p for p in sorted(f[:-4] for f in os.listdir(LAB) if f.endswith(".npz"))
            if p not in config.EXCLUDE_PATIENTS and usable(p)]

def dev_patients():
    return [p for p in all_usable() if p not in config.LOCKED_PATIENTS]

def load(pat, fams, patient_norm=False):
    fo, _ = _families()
    l = np.load(f"{LAB}/{pat}.npz")
    o, w = np.load(f"{FEAT}/{pat}.npz"), np.load(f"{WAV}/{pat}.npz")
    for f in (o, w):
        assert (f["source_file"] == l["source_file"]).all() and (f["win_idx"] == l["win_idx"]).all()
    arr = {"o": (o["X"], o["family"]), "w": (w["X"], w["family"])}
    cols = [arr["o" if n in fo else "w"][0][:, arr["o" if n in fo else "w"][1] == n] for n in fams]
    X = np.hstack(cols).astype(np.float32)
    flat = (arr["o"][0][:, arr["o"][1] == "variance"] < -6).all(axis=1)      # exactly flat windows
    keep = (l["y"] >= 0) & ~flat
    X = X[keep]
    if patient_norm:                                                          # per-patient z-score, no labels used
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    ok = l["event_eligible"] & (l["event_preictal"] >= config.MIN_WINDOWS_PER_CLASS)
    ids = [int(k + 1) for k in np.where(ok)[0]]
    return X, l["y"][keep], l["t"][keep], l["pre_id"][keep], ids

def event_blocks(y, t, ids):                                                  # same interictal blocks as final_9_eval.py
    inter = np.where(y == 0)[0]
    inter = inter[np.argsort(t[inter])]
    return dict(zip(ids, np.array_split(inter, len(ids))))

def smooth(p, t, k):                                                          # causal moving average; p and t in time order
    if k == 1:
        return p
    run = np.r_[0, np.cumsum(np.diff(t) > 2 * WIN)]
    return pd.Series(p).groupby(run).transform(lambda x: x.rolling(k, min_periods=1).mean()).to_numpy()

def alarms(flag, t):                                                          # alarm indices, with a refractory period
    idx = np.where(flag)[0]
    out, j, ft = [], 0, t[idx]
    while j < len(idx):
        out.append(idx[j])
        j = np.searchsorted(ft, ft[j] + config.ALARM_REFRACTORY_SEC, side="left")
    return np.array(out, dtype=int)

def fit_mlp(X, y, Xv, yv, seed, hidden=16, dropout=0.5, wd=0.01, epochs=40):
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-6                                       # scaling from training windows only
    prep = lambda Z: torch.tensor((Z - mu) / sd, dtype=torch.float32)
    Xt, yt, Xvt = prep(X), torch.tensor(y, dtype=torch.float32), prep(Xv)
    net = nn.Sequential(nn.Linear(X.shape[1], hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=wd)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((y == 0).sum() / (y == 1).sum())))
    best, state = -1.0, None
    for _ in range(epochs):
        net.train()
        for b in torch.randperm(len(yt)).split(256):
            opt.zero_grad(); loss_fn(net(Xt[b]).squeeze(1), yt[b]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            s = roc_auc_score(yv, net(Xvt).squeeze(1).numpy())                # early stopping on the validation data
        if s > best:
            best, state = s, copy.deepcopy(net.state_dict())
    net.load_state_dict(state); net.eval()
    def predict(Z):
        with torch.no_grad():
            return torch.sigmoid(net(prep(Z)).squeeze(1)).numpy()
    return predict
