import os, argparse, copy
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.metrics import roc_auc_score
import config
import final_lib as L

ap = argparse.ArgumentParser()
ap.add_argument("--features", default="bandpower_gamma,variance,line_length")
ap.add_argument("--name", required=True)
ap.add_argument("--patients", default=None)
ap.add_argument("--include-locked", action="store_true")
ap.add_argument("--ft-events", default=",".join(map(str, config.FT_EVENTS)))
ap.add_argument("--ft-lr", type=float, default=config.FT_LR)
ap.add_argument("--ft-epochs", type=int, default=config.FT_EPOCHS)
ap.add_argument("--seeds", default=",".join(map(str, config.SEEDS)))
ap.add_argument("--k", type=int, default=config.EVAL_K)
ap.add_argument("--patient-norm", action="store_true", help="per-patient z-score using the whole recording (upper bound)")
a = ap.parse_args()

seeds = [int(s) for s in a.seeds.split(",")]
ft_list = [int(n) for n in a.ft_events.split(",")]
WIN, SOP_H, MINW = config.WIN_SEC, config.SOP_SEC / 3600, config.MIN_WINDOWS_PER_CLASS
fams = L.resolve_features(a.features)

pool = L.dev_patients()                                        # pretraining pool: development patients only
targets = pool + ([p for p in L.all_usable() if p in config.LOCKED_PATIENTS] if a.include_locked else [])
if a.patients:
    targets = [p for p in targets if p in a.patients.split(",")]
data = {p: L.load(p, fams, a.patient_norm) for p in sorted(set(pool + targets))}
blocks = {p: L.event_blocks(data[p][1], data[p][2], data[p][4]) for p in data}

def segment(P, ev):                                            # an event's preictal windows plus its interictal block, in time order
    t, pid = data[P][2], data[P][3]
    s = np.r_[np.where(pid == ev)[0], blocks[P][ev]]
    return s[np.argsort(t[s], kind="stable")]

def near_mask(t_all, t_ref, buf):
    ts = np.sort(t_ref)
    i = np.searchsorted(ts, t_all)
    left = np.abs(t_all - ts[np.clip(i - 1, 0, len(ts) - 1)])
    right = np.abs(t_all - ts[np.clip(i, 0, len(ts) - 1)])
    return np.minimum(left, right) <= buf

def candidate_table(sv, tv, yv, hours):                        # false alarms/hour on the validation segment, per threshold
    levels = np.unique(np.r_[np.linspace(0.5, 0.99, 50), 1 - np.logspace(-2, -4.5, 60), 1.0])
    cands = np.unique(np.quantile(sv, levels))
    fa = np.array([(yv[L.alarms(sv >= c, tv)] == 0).sum() / hours for c in cands])
    return cands, fa

def choose(cands, fa, target):
    ok = np.where(fa <= target)[0]
    return float(cands[ok[0]]) if len(ok) else float(cands[-1]) + 1e-6

def train_net(X, y, Xv, yv, seed, mu=None, sd=None, init=None, lr=1e-3, wd=0.01, epochs=40,
              freeze_hidden=False, epoch0=False):
    torch.manual_seed(seed)
    if mu is None:
        mu, sd = X.mean(0), X.std(0) + 1e-6                    # pretraining only: scaler from training windows
    prep = lambda Z: torch.tensor((Z - mu) / sd, dtype=torch.float32)
    Xt, yt, Xvt = prep(X), torch.tensor(y, dtype=torch.float32), prep(Xv)
    net = nn.Sequential(nn.Linear(X.shape[1], 16), nn.ReLU(), nn.Dropout(0.5), nn.Linear(16, 1))
    if init is not None:
        net.load_state_dict(init)
    if freeze_hidden:
        for p in net[0].parameters():
            p.requires_grad = False
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((y == 0).sum() / (y == 1).sum())))
    def val_auc():
        net.eval()
        with torch.no_grad():
            return roc_auc_score(yv, net(Xvt).squeeze(1).numpy())
    best, state, best_ep = (val_auc(), copy.deepcopy(net.state_dict()), 0) if epoch0 else (-1.0, None, epochs)
    for ep in range(epochs):
        net.train()
        for b in torch.randperm(len(yt)).split(256):
            opt.zero_grad(); loss_fn(net(Xt[b]).squeeze(1), yt[b]).backward(); opt.step()
        s = val_auc()
        if s > best:
            best, state, best_ep = s, copy.deepcopy(net.state_dict()), ep + 1
    net.load_state_dict(state); net.eval()
    return net, mu, sd, best_ep

def make_predict(net, mu, sd):
    def predict(Z):
        with torch.no_grad():
            return torch.sigmoid(net(torch.tensor((Z - mu) / sd, dtype=torch.float32)).squeeze(1)).numpy()
    return predict

sdir = f"{config.V4_SCORES_DIR}/{a.name}"
os.makedirs(sdir, exist_ok=True); os.makedirs(config.V4_RESULT_DIR, exist_ok=True)
rows = []
print(f"v4 transfer | {len(fams)} families | {len(targets)} target patients | pretraining pool {len(pool)} | N = {ft_list} | seeds {seeds} | k={a.k}")
for seed in seeds:
    rng = np.random.default_rng(seed)
    parts = {}
    for p in pool:                                             # one balanced sample per pretraining patient
        X, y = data[p][0], data[p][1]
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        idx = np.r_[pos, rng.choice(neg, min(len(neg), config.NORMAL_TO_PRE_RATIO * len(pos)), replace=False)]
        parts[p] = (X[idx], y[idx])
    for P in targets:
        X, y, t, pid, ids = data[P]
        if len(ids) < min(ft_list) + 2:
            print(P, "skipped: needs at least", min(ft_list) + 2, "usable events"); continue
        twin = config.SAME_PERSON.get(P)                       # pretrain on others only
        cand = [p for p in pool if p != P and p != twin]
        start = pool.index(P) + 1 if P in pool else 0
        order = [pool[(start + j) % len(pool)] for j in range(len(pool))]
        val_p = [p for p in order if p in cand][:3]            # 3 rotating validation patients (early stopping only)
        tr_p = [p for p in cand if p not in val_p]
        net0, mu0, sd0, _ = train_net(np.vstack([parts[p][0] for p in tr_p]), np.concatenate([parts[p][1] for p in tr_p]),
                                      np.vstack([parts[p][0] for p in val_p]), np.concatenate([parts[p][1] for p in val_p]), seed)
        init = copy.deepcopy(net0.state_dict())
        for N in ft_list:
            if len(ids) < N + 2:
                print(P, f"N={N} skipped: needs {N + 2} usable events"); continue
            ft_ids, va_id, te_ids = ids[:N], ids[N], ids[N + 1:]       # forward chaining: tests are always later
            va = segment(P, va_id)
            tes = {e: segment(P, e) for e in te_ids}
            tes = {e: s for e, s in tes.items() if min((y[s] == 0).sum(), (y[s] == 1).sum()) >= MINW}
            if not tes or min((y[va] == 0).sum(), (y[va] == 1).sum()) < MINW:
                print(P, f"N={N} skipped: too few validation/test windows"); continue
            ft = np.concatenate([segment(P, e) for e in ft_ids])
            ref = np.concatenate([va] + list(tes.values()))
            ft = ft[~near_mask(t[ft], t[ref], config.SPLIT_BUFFER_SEC)]      # buffer around validation and test windows
            pos, neg = ft[y[ft] == 1], ft[y[ft] == 0]
            if min(len(pos), len(neg)) < MINW:
                print(P, f"N={N} skipped: too few fine-tune windows"); continue
            r2 = np.random.default_rng(1000 * seed + 10 * int(P[3:]) + N)
            idx = np.r_[pos, r2.choice(neg, min(len(neg), config.NORMAL_TO_PRE_RATIO * len(pos)), replace=False)]

            preds, ep = {"general": make_predict(net0, mu0, sd0)}, {}
            for name, freeze in (("transfer_all", False), ("transfer_last", True)):
                net, mu, sd, e = train_net(X[idx], y[idx], X[va], y[va], seed, mu=mu0, sd=sd0, init=init,
                                           lr=a.ft_lr, epochs=a.ft_epochs, freeze_hidden=freeze, epoch0=True)
                preds[name], ep[name] = make_predict(net, mu, sd), e
            preds["specific"] = L.fit_mlp(X[idx], y[idx], X[va], y[va], seed)

            hours_va = (y[va] == 0).sum() * WIN / 3600
            saved = {"y_va": y[va].astype(np.int8), "t_va": t[va]}
            for e, s in tes.items():
                saved[f"y_ev{e}"], saved[f"t_ev{e}"] = y[s].astype(np.int8), t[s]
            for cond, predict in preds.items():
                pv = predict(X[va]); sv = L.smooth(pv, t[va], a.k)
                cands, fa_c = candidate_table(sv, t[va], y[va], hours_va)
                saved["pv_" + cond] = pv.astype(np.float32)
                for e, s in tes.items():
                    ps = predict(X[s]); ss = L.smooth(ps, t[s], a.k)
                    saved[f"ps_{cond}_ev{e}"] = ps.astype(np.float32)
                    hours_te = (y[s] == 0).sum() * WIN / 3600
                    auc = roc_auc_score(y[s], ss)
                    for target in config.FA_TARGETS:
                        thr = choose(cands, fa_c, target)                  # chosen on the validation event only
                        flag = ss >= thr
                        al = L.alarms(flag, t[s])
                        n_fa = int((y[s][al] == 0).sum())
                        rows.append(dict(patient=P, n_ft=N, condition=cond, event=e, seed=seed, target=target, auroc=auc,
                                         event_sens=int((y[s][al] == 1).any()), chance_event=min(1.0, n_fa / hours_te * SOP_H),
                                         sens_win=float(flag[y[s] == 1].mean()), spec_win=float((~flag[y[s] == 0]).mean()),
                                         fa_hr=n_fa / hours_te, n_fa=n_fa, hours_te=hours_te, thr=thr,
                                         ft_epoch=ep.get(cond, np.nan)))
            np.savez_compressed(f"{sdir}/{P}_n{N}_s{seed}.npz", **saved)
        print("seed", seed, P, "done")

if not rows:
    raise SystemExit("no folds ran")
df = pd.DataFrame(rows)
df.to_csv(f"{config.V4_RESULT_DIR}/transfer__{a.name}.csv", index=False)
q = df[df.target == 0.5].groupby(["n_ft", "condition"])[["auroc", "event_sens", "chance_event", "fa_hr"]].mean()
print("\nquick look at target 0.5/h (fa_hr here is a plain mean of per-event rates):")
print(q.round(3).to_string())
