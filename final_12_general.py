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
ap.add_argument("--patient-norm", action="store_true")
ap.add_argument("--seeds", default=",".join(map(str, config.SEEDS)))
ap.add_argument("--k", type=int, default=config.EVAL_K)
a = ap.parse_args()

seeds = [int(s) for s in a.seeds.split(",")]
SOP_H, WIN = config.SOP_SEC / 3600, config.WIN_SEC
fams = L.resolve_features(a.features)

pool = L.dev_patients()                                    # training pool: development patients only
targets = pool + ([p for p in L.all_usable() if p in config.LOCKED_PATIENTS] if a.include_locked else [])
if a.patients:
    targets = [p for p in targets if p in a.patients.split(",")]
data = {p: L.load(p, fams, a.patient_norm) for p in sorted(set(pool + targets))}
blocks = {p: L.event_blocks(data[p][1], data[p][2], data[p][4]) for p in data}

def segment(p, ev):                                        # an event's preictal windows plus its interictal block, in time order
    t, pid = data[p][2], data[p][3]
    s = np.r_[np.where(pid == ev)[0], blocks[p][ev]]
    return s[np.argsort(t[s], kind="stable")]

def candidates(segs):                                      # false alarms/hour on the pooled validation segments, per threshold
    allv = np.concatenate([s[0] for s in segs])
    levels = np.unique(np.r_[np.linspace(0.5, 0.99, 50), 1 - np.logspace(-2, -4.5, 60), 1.0])
    cands = np.unique(np.quantile(allv, levels))
    hours = sum(s[3] for s in segs)
    fa = np.array([sum((s[2][L.alarms(s[0] >= c, s[1])] == 0).sum() for s in segs) / hours for c in cands])
    return cands, fa

def choose(cands, fa, target):
    ok = np.where(fa <= target)[0]
    return float(cands[ok[0]]) if len(ok) else float(cands[-1]) + 1e-6

sdir = f"{config.SCORES_DIR}/mlp__{a.name}"
os.makedirs(sdir, exist_ok=True); os.makedirs(config.RESULT_DIR, exist_ok=True)
rows = []
print(f"general MLP | {len(fams)} families | {len(targets)} target patients | training pool {len(pool)} | seeds {seeds} | k={a.k}")
for seed in seeds:
    rng = np.random.default_rng(seed)
    parts = {}
    for p in pool:                                         # one balanced sample per training patient
        X, y = data[p][0], data[p][1]
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        idx = np.r_[pos, rng.choice(neg, min(len(neg), config.NORMAL_TO_PRE_RATIO * len(pos)), replace=False)]
        parts[p] = (X[idx], y[idx])
    for P in targets:
        twin = config.SAME_PERSON.get(P)
        cand = [p for p in pool if p != P and p != twin]
        start = pool.index(P) + 1 if P in pool else 0
        order = [pool[(start + j) % len(pool)] for j in range(len(pool))]
        val_p = [p for p in order if p in cand][:3]        # 3 rotating validation patients
        tr_p = [p for p in cand if p not in val_p]
        Xtr, ytr = np.vstack([parts[p][0] for p in tr_p]), np.concatenate([parts[p][1] for p in tr_p])
        Xva, yva = np.vstack([parts[p][0] for p in val_p]), np.concatenate([parts[p][1] for p in val_p])
        predict = L.fit_mlp(Xtr, ytr, Xva, yva, seed)
        tr_auc = roc_auc_score(ytr, predict(Xtr))

        segs, pv, yv_ = [], [], []                         # threshold data: latest event and block of each validation patient
        for v in val_p:
            Xv, yv, tv = data[v][0], data[v][1], data[v][2]
            s = segment(v, data[v][4][-1])
            raw = predict(Xv[s])
            segs.append((L.smooth(raw, tv[s], a.k), tv[s], yv[s], (yv[s] == 0).sum() * WIN / 3600))
            pv.append(raw); yv_.append(yv[s])
        pv, yv_ = np.concatenate(pv), np.concatenate(yv_)
        cands, fa_c = candidates(segs)

        X, y, t, pid, ids = data[P]
        for ev in ids:
            te = segment(P, ev)
            if min((y[te] == 0).sum(), (y[te] == 1).sum()) < config.MIN_WINDOWS_PER_CLASS:
                continue
            ps = predict(X[te]); ss = L.smooth(ps, t[te], a.k)
            hours_te = (y[te] == 0).sum() * WIN / 3600
            auc = roc_auc_score(y[te], ss)
            np.savez_compressed(f"{sdir}/{P}_ev{ev}_s{seed}.npz", ps=ps.astype(np.float32), pv=pv.astype(np.float32),
                                y_te=y[te].astype(np.int8), y_va=yv_.astype(np.int8), t_te=t[te])
            for target in config.FA_TARGETS:
                thr = choose(cands, fa_c, target)           # chosen on other patients only
                flag = ss >= thr
                al = L.alarms(flag, t[te])
                n_fa = int((y[te][al] == 0).sum())
                rows.append(dict(patient=P, event=ev, seed=seed, target=target, auroc=auc, train_auroc=tr_auc,
                                 event_sens=int((y[te][al] == 1).any()), chance_event=min(1.0, n_fa / hours_te * SOP_H),
                                 sens_win=float(flag[y[te] == 1].mean()), spec_win=float((~flag[y[te] == 0]).mean()),
                                 fa_hr=n_fa / hours_te, n_fa=n_fa, hours_te=hours_te, thr=thr))
        print("seed", seed, P, "done")

if not rows:
    raise SystemExit("no folds ran")
pd.DataFrame(rows).to_csv(f"{config.RESULT_DIR}/eval__mlp__{a.name}.csv", index=False)
print(f"\nSaved. Summarise with: python final_10_eval_table.py mlp__{a.name}" + (" --locked" if a.include_locked else ""))
