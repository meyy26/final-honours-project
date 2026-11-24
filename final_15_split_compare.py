import glob, os
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
import config

RUNS = {"1 specific, whole-event": "mlp__n3",
        "2 general (leave-one-patient-out)": "mlp__n3_general",
        "3 general + per-patient norm": "mlp__n3_general_pnorm",
        "4 specific, random 70/30 (LEAKY)": "mlp__n3_leaky"}

def fold_metrics(f):
    d = np.load(f)
    ps, pv, yt, yv = d["ps"], d["pv"], d["y_te"], d["y_va"]
    out = dict(auroc=roc_auc_score(yt, ps))
    for s in (0.95, 0.99):
        thr = np.quantile(pv[yv == 0], s)                    # threshold from the validation data's interictal scores
        out[f"sens_v{int(s * 100)}"] = float((ps[yt == 1] >= thr).mean())
        out[f"spec_t{int(s * 100)}"] = float((ps[yt == 0] < thr).mean())   # specificity actually reached on test
    return out

rows = []
for label, run in RUNS.items():
    for f in glob.glob(f"{config.SCORES_DIR}/{run}/chb*.npz"):
        pat = os.path.basename(f).split("_")[0]
        if pat not in config.LOCKED_PATIENTS:
            rows.append(dict(protocol=label, patient=pat, **fold_metrics(f)))
if not rows:
    raise SystemExit("no saved scores found")
per = pd.DataFrame(rows).groupby(["protocol", "patient"]).mean(numeric_only=True)    # folds and seeds -> patient first
protos = per.index.get_level_values("protocol").unique()
common = set.intersection(*[set(per.xs(p, level="protocol").index) for p in protos])
per = per[per.index.get_level_values("patient").isin(common)]
print(f"PART B: {len(common)} patients common to all {len(protos)} protocols (unsmoothed scores, development patients)\n")
cols = ["auroc", "sens_v95", "spec_t95", "sens_v99", "spec_t99"]
g = per.groupby(level="protocol")[cols]
print(pd.concat({"mean": g.mean(), "median": g.median()}, axis=1).round(3).to_string())
print("\nAUROC per patient:")
print(per.auroc.unstack(level="protocol").round(3).to_string())

def ev(run):                                                  # PART A: alarm-level results from the eval CSVs
    f = f"{config.RESULT_DIR}/eval__{run}.csv"
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    d = d[~d.patient.isin(config.LOCKED_PATIENTS)]
    g = d.groupby(["target", "patient"])
    p = g[["event_sens", "chance_event", "auroc"]].mean()
    p["fa_hr"] = g.n_fa.sum() / g.hours_te.sum() if "n_fa" in d.columns else g.fa_hr.mean()
    return p

A = ev("mlp__n3")
for run in ("mlp__n3_general", "mlp__n3_general_pnorm"):
    B = ev(run)
    if A is None or B is None:
        continue
    out = []
    for target in config.FA_TARGETS:
        x, y = A.xs(target, level="target"), B.xs(target, level="target")
        idx = x.index.intersection(y.index); x, y = x.loc[idx], y.loc[idx]
        out.append(dict(target=target, patients=len(idx), event_specific=x.event_sens.mean(), event_general=y.event_sens.mean(),
                        chance_s=x.chance_event.mean(), chance_g=y.chance_event.mean(), fa_s=x.fa_hr.mean(), fa_g=y.fa_hr.mean(),
                        auroc_s=x.auroc.mean(), auroc_g=y.auroc.mean(),
                        specific_ahead=int(((x.event_sens - x.chance_event) > (y.event_sens - y.chance_event)).sum())))
    print(f"\nPART A: specific vs {run} (alarm level, k = {config.EVAL_K}; specific_ahead = patients where specific has more skill over chance)")
    print(pd.DataFrame(out).round(3).to_string(index=False))
