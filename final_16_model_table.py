import os, sys
import pandas as pd
import config

runs = [r for r in sys.argv[1:] if not r.startswith("--")] or ["mlp__n3", "rf__n3", "cnn1d__n3"]
lock = "--locked" in sys.argv

def load(run):
    f = f"{config.RESULT_DIR}/eval__{run}.csv"
    if not os.path.exists(f):
        print("missing:", f); return None
    d = pd.read_csv(f)
    d = d[d.patient.isin(config.LOCKED_PATIENTS) == lock]
    g = d.groupby(["target", "patient"])
    p = g[["event_sens", "chance_event", "sens_win", "spec_win", "auroc", "train_auroc"]].mean()
    p["fa_hr"] = g.n_fa.sum() / g.hours_te.sum() if "n_fa" in d.columns else g.fa_hr.mean()
    return p

P = {r: load(r) for r in runs}
P = {r: p for r, p in P.items() if p is not None}
common = set.intersection(*[set(p.index.get_level_values("patient")) for p in P.values()])
print(f"{len(common)} patients common to {list(P)} (k = {config.EVAL_K} windows)\n")
rows = []
for r, p in P.items():
    p = p[p.index.get_level_values("patient").isin(common)]
    for target in config.FA_TARGETS:
        x = p.xs(target, level="target")
        rows.append(dict(run=r, target=target, event_sens=x.event_sens.mean(), chance=x.chance_event.mean(),
                         skill=(x.event_sens - x.chance_event).mean(), beats_chance=int((x.event_sens > x.chance_event).sum()),
                         fa_hr=x.fa_hr.mean(), sens_win=x.sens_win.mean(), spec_win=x.spec_win.mean(),
                         auroc=x.auroc.mean(), train_auroc=x.train_auroc.mean()))
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("\nAUROC per patient (does not depend on the target):")
a = pd.DataFrame({r: p.xs(config.FA_TARGETS[0], level="target").auroc for r, p in P.items()}).loc[sorted(common)]
print(a.round(3).to_string())
