import os, argparse, copy
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import config

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["mlp", "cnn1d", "rf"], required=True)
ap.add_argument("--features", default="all", help="'all', 'orig', 'wavelet', 'perchannel', or comma-separated families")
ap.add_argument("--drop", default=None, help="comma-separated families to leave out")
ap.add_argument("--patients", default=None, help="comma-separated; default = every usable development patient")
ap.add_argument("--seeds", default=",".join(map(str, config.SEEDS)))
ap.add_argument("--name", default=None, help="label used in the output filename")
ap.add_argument("--include-locked", action="store_true")
ap.add_argument("--hidden", type=int, default=16)
ap.add_argument("--filters", type=int, default=8)
ap.add_argument("--dropout", type=float, default=0.5)
ap.add_argument("--wd", type=float, default=0.01)
ap.add_argument("--epochs", type=int, default=40)
a = ap.parse_args()

torch.set_num_threads(2)
FEAT, WAV, LAB = config.FEATURE_DIR, config.WAVELET_DIR, config.LABEL_DIR
WIN = config.WIN_SEC
seeds = [int(s) for s in a.seeds.split(",")]

# ---- feature families and where each one lives ----
all_pats = sorted(f[:-4] for f in os.listdir(LAB) if f.endswith(".npz"))
ref = all_pats[0]
fo, fw = np.load(f"{FEAT}/{ref}.npz"), np.load(f"{WAV}/{ref}.npz")
fam_orig = [str(x) for x in dict.fromkeys(fo["family"])]
fam_wav = [str(x) for x in dict.fromkeys(fw["family"])]
width = {n: int((fo["family"] == n).sum()) for n in fam_orig}
width.update({n: int((fw["family"] == n).sum()) for n in fam_wav})
if a.features == "all":
    fams = fam_orig + fam_wav
elif a.features == "orig":
    fams = fam_orig
elif a.features == "wavelet":
    fams = fam_wav
elif a.features == "perchannel":
    fams = [n for n in fam_orig + fam_wav if width[n] == 18]
else:
    fams = a.features.split(",")
if a.drop:
    fams = [f for f in fams if f not in a.drop.split(",")]
unknown = [f for f in fams if f not in width]
if unknown:
    raise SystemExit(f"unknown families {unknown}. choose from: {sorted(width)}")
if a.model == "cnn1d" and any(width[f] != 18 for f in fams):
    raise SystemExit("cnn1d needs per-channel families only (18 columns each); use --features perchannel")

# ---- patients ----
def usable(pat):
    l = np.load(f"{LAB}/{pat}.npz")
    ok = l["event_eligible"] & (l["event_preictal"] >= config.MIN_WINDOWS_PER_CLASS)
    return int(ok.sum()) >= config.MIN_SEIZURE_EVENTS and int((l["y"] == 0).sum()) >= 1000

patients = [p for p in all_pats if p not in config.EXCLUDE_PATIENTS and usable(p)]
if not a.include_locked:
    patients = [p for p in patients if p not in config.LOCKED_PATIENTS]
if a.patients:
    patients = [p for p in patients if p in a.patients.split(",")]

def load(pat):
    l = np.load(f"{LAB}/{pat}.npz")
    o, w = np.load(f"{FEAT}/{pat}.npz"), np.load(f"{WAV}/{pat}.npz")
    for f in (o, w):
        assert (f["source_file"] == l["source_file"]).all() and (f["win_idx"] == l["win_idx"]).all()
    arr = {"o": (o["X"], o["family"]), "w": (w["X"], w["family"])}      # read each array from disk once
    cols = [arr["o" if n in fam_orig else "w"][0][:, arr["o" if n in fam_orig else "w"][1] == n] for n in fams]
    X = np.hstack(cols).astype(np.float32)                     # families stay contiguous, in the order given
    flat = (arr["o"][0][:, arr["o"][1] == "variance"] < -6).all(axis=1)      # exactly flat windows (log10 variance = -12)
    keep = (l["y"] >= 0) & ~flat
    ok = l["event_eligible"] & (l["event_preictal"] >= config.MIN_WINDOWS_PER_CLASS)
    ids = [int(k + 1) for k in np.where(ok)[0]]                # usable event numbers (1-based, as in pre_id)
    return X[keep], l["y"][keep], l["t"][keep], l["pre_id"][keep], ids

# ---- helpers ----
def near_mask(t_all, t_ref, buf):                              # True where a window is within buf s of any reference window
    ts = np.sort(t_ref)
    i = np.searchsorted(ts, t_all)
    left = np.abs(t_all - ts[np.clip(i - 1, 0, len(ts) - 1)])
    right = np.abs(t_all - ts[np.clip(i, 0, len(ts) - 1)])
    return np.minimum(left, right) <= buf

def smooth(p, t, k):                                           # causal moving average; restarts after time gaps
    if k == 1:
        return p
    run = np.r_[0, np.cumsum(np.diff(t) > 2 * WIN)]
    return pd.Series(p).groupby(run).transform(lambda x: x.rolling(k, min_periods=1).mean()).to_numpy()

def alarms(flag, t):                                           # alarm indices; refractory period after each alarm
    idx = np.where(flag)[0]
    out, j = [], 0
    ft = t[idx]
    while j < len(idx):
        out.append(idx[j])
        j = np.searchsorted(ft, ft[j] + config.ALARM_REFRACTORY_SEC, side="left")
    return np.array(out, dtype=int)

def pick_thr(sv, tv, yv):                                      # lowest threshold meeting the validation false-alarm target
    hours = (yv == 0).sum() * WIN / 3600
    for c in np.unique(np.quantile(sv, np.linspace(0.5, 1, 101))):
        al = alarms(sv >= c, tv)
        if (yv[al] == 0).sum() / hours <= config.FA_TARGET_PER_HR:
            return float(c)
    return float(sv.max()) + 1e-6

def fit(X, y, Xv, yv, seed):
    if a.model == "rf":
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, max_features="sqrt",
                                    class_weight="balanced", n_jobs=2, random_state=seed).fit(X, y)
        return lambda Z: rf.predict_proba(Z)[:, 1]
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-6                        # scaling from TRAINING windows only
    prep = lambda Z: torch.tensor((Z - mu) / sd, dtype=torch.float32)
    Xt, yt, Xvt = prep(X), torch.tensor(y, dtype=torch.float32), prep(Xv)
    if a.model == "mlp":
        net = nn.Sequential(nn.Linear(X.shape[1], a.hidden), nn.ReLU(), nn.Dropout(a.dropout), nn.Linear(a.hidden, 1))
    else:                                                      # cnn1d: families as input channels, EEG channels as the axis
        F = X.shape[1] // 18
        net = nn.Sequential(nn.Unflatten(1, (F, 18)), nn.Conv1d(F, a.filters, 3, padding=1), nn.ReLU(),
                            nn.Flatten(), nn.Dropout(a.dropout), nn.Linear(a.filters * 18, 1))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=a.wd)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((y == 0).sum() / (y == 1).sum())))
    best, state = -1.0, None
    for ep in range(a.epochs):
        net.train()
        for b in torch.randperm(len(yt)).split(256):
            opt.zero_grad(); loss_fn(net(Xt[b]).squeeze(1), yt[b]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            s = roc_auc_score(yv, net(Xvt).squeeze(1).numpy())           # early stopping on the validation set
        if s > best:
            best, state = s, copy.deepcopy(net.state_dict())
    net.load_state_dict(state); net.eval()
    def predict(Z):
        with torch.no_grad():
            return torch.sigmoid(net(prep(Z)).squeeze(1)).numpy()
    return predict

# ---- run ----
rows, skipped = [], 0
print(f"{a.model} | {len(fams)} families, {sum(width[f] for f in fams)} columns | {len(patients)} patients | seeds {seeds}")
for pat in patients:
    X, y, t, pid, ids = load(pat)
    inter = np.where(y == 0)[0]
    inter = inter[np.argsort(t[inter])]
    blocks = dict(zip(ids, np.array_split(inter, len(ids))))             # one contiguous interictal block per usable event
    for ev in ids:
        vev = [e for e in ids if e != ev][-1]                            # validation event: the latest of the others
        te = np.r_[np.where(pid == ev)[0], blocks[ev]]
        va = np.r_[np.where(pid == vev)[0], blocks[vev]]
        excl = near_mask(t, t[te], config.SPLIT_BUFFER_SEC) | near_mask(t, t[va], config.SPLIT_BUFFER_SEC)
        tr = np.where(~excl)[0]                                          # everything else, with a buffer around test and validation
        cnt = lambda idx, v: int((y[idx] == v).sum())
        if min(cnt(te, 0), cnt(te, 1), cnt(va, 0), cnt(va, 1), cnt(tr, 1)) < config.MIN_WINDOWS_PER_CLASS:
            skipped += 1
            print(pat, "event", ev, "skipped: too few windows (test/val/train preictal+interictal:",
                  cnt(te, 1), cnt(te, 0), cnt(va, 1), cnt(va, 0), cnt(tr, 1), ")"); continue
        te = te[np.argsort(t[te], kind="stable")]
        hours = cnt(te, 0) * WIN / 3600
        for seed in seeds:
            rng = np.random.default_rng(seed)
            tp, tn = tr[y[tr] == 1], tr[y[tr] == 0]
            idx = np.r_[tp, rng.choice(tn, min(len(tn), config.NORMAL_TO_PRE_RATIO * len(tp)), replace=False)]
            predict = fit(X[idx], y[idx], X[va], y[va], seed)
            tr_auc = roc_auc_score(y[idx], predict(X[idx]))
            ps, pv = predict(X[te]), predict(X[va])
            for k in config.SMOOTH_WINDOWS:
                ss, sv = smooth(ps, t[te], k), smooth(pv, t[va], k)
                thr = pick_thr(sv, t[va], y[va])                         # chosen on the validation set only
                al = alarms(ss >= thr, t[te])
                rows.append(dict(patient=pat, event=ev, seed=seed, k=k, auroc=roc_auc_score(y[te], ss),
                                 detected=int((y[te][al] == 1).any()), fa_per_hr=(y[te][al] == 0).sum() / hours,
                                 train_auroc=tr_auc, thr=thr))
        print(pat, "event", ev, "done")

if not rows:
    raise SystemExit("no folds ran")
os.makedirs(config.RESULT_DIR, exist_ok=True)
name = a.name or (a.features if len(a.features) < 50 else f"{len(fams)}fam")
name += f"_no_{a.drop}" if a.drop else ""
pd.DataFrame(rows).to_csv(f"{config.RESULT_DIR}/{a.model}__{name}.csv", index=False)
df = pd.DataFrame(rows)
per = df.groupby(["k", "patient"])[["auroc", "detected", "fa_per_hr", "train_auroc"]].mean()
print(f"\n{skipped} folds skipped; mean over patients by smoothing length k (windows):")
print(per.groupby(level="k").mean().round(3).to_string())
