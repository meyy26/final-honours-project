import os, argparse
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import config
import final_lib as L

ap = argparse.ArgumentParser()
ap.add_argument("--features", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--patients", default=None)
ap.add_argument("--include-locked", action="store_true")
ap.add_argument("--seeds", default=",".join(map(str, config.SEEDS)))
ap.add_argument("--test-frac", type=float, default=0.3)
a = ap.parse_args()

seeds = [int(s) for s in a.seeds.split(",")]
fams = L.resolve_features(a.features)
pats = L.dev_patients() + ([p for p in L.all_usable() if p in config.LOCKED_PATIENTS] if a.include_locked else [])
if a.patients:
    pats = [p for p in pats if p in a.patients.split(",")]

sdir = f"{config.SCORES_DIR}/mlp__{a.name}"
os.makedirs(sdir, exist_ok=True); os.makedirs(config.RESULT_DIR, exist_ok=True)
rows = []
print(f"LEAKY random split | {len(fams)} families | {len(pats)} patients | seeds {seeds}")
for P in pats:
    X, y, t, pid, ids = L.load(P, fams)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    for seed in seeds:
        rng = np.random.default_rng(seed)
        idx = np.r_[pos, rng.choice(neg, min(len(neg), config.NORMAL_TO_PRE_RATIO * len(pos)), replace=False)]
        idx = rng.permutation(idx)                          # the leak: windows from one seizure land on both sides
        n_te = int(a.test_frac * len(idx))
        te, rest = idx[:n_te], idx[n_te:]
        n_va = int(0.2 * len(rest))
        va, tr = rest[:n_va], rest[n_va:]
        predict = L.fit_mlp(X[tr], y[tr], X[va], y[va], seed)
        ps, pv = predict(X[te]), predict(X[va])
        np.savez_compressed(f"{sdir}/{P}_s{seed}.npz", ps=ps.astype(np.float32), pv=pv.astype(np.float32),
                            y_te=y[te].astype(np.int8), y_va=y[va].astype(np.int8))
        rows.append(dict(patient=P, seed=seed, auroc=roc_auc_score(y[te], ps),
                         train_auroc=roc_auc_score(y[tr], predict(X[tr])), n_test=len(te)))
    print(P, "done")

df = pd.DataFrame(rows)
df.to_csv(f"{config.RESULT_DIR}/leaky__mlp__{a.name}.csv", index=False)
per = df.groupby("patient")[["auroc", "train_auroc"]].mean()
print(per.round(3).to_string())
print(per.agg(["mean", "median"]).round(3))
