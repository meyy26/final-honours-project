import os
import pandas as pd
import config

STUDY = ["chb01", "chb02", "chb03", "chb05", "chb09", "chb10", "chb13", "chb14", "chb18", "chb19", "chb20", "chb21", "chb23"]

def per_patient(run):
    f = f"{config.RESULT_DIR}/eval__{run}.csv"
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    g = d.groupby(["target", "patient"])
    p = g[["event_sens", "chance_event", "auroc"]].mean()          # events and seeds -> patient first
    p["fa_hr"] = g.n_fa.sum() / g.hours_te.sum() if "n_fa" in d.columns else g.fa_hr.mean()
    return p

dev, lock = per_patient("mlp__n3"), per_patient("mlp__n3_locked")
for label, p in (("development only", dev), ("development + locked", pd.concat([dev, lock]) if lock is not None else dev)):
    p = p[p.index.get_level_values("patient").isin(STUDY)]
    pats = sorted(p.index.get_level_values("patient").unique())
    print(f"\n{label}: {len(pats)} patients {pats}")
    out = []
    for t in config.FA_TARGETS:
        x = p.xs(t, level="target")
        out.append(dict(target=t, event_sens=x.event_sens.mean(), chance=x.chance_event.mean(),
                        fa_hr=x.fa_hr.mean(), auroc_mean=x.auroc.mean(), auroc_median=x.auroc.median()))
    print(pd.DataFrame(out).round(3).to_string(index=False))

sel = pd.read_csv(f"{config.RESULT_DIR}/mlp__sel_n3.csv")
s = sel[(sel.k == 1) & sel.patient.isin(STUDY) & ~sel.patient.isin(config.LOCKED_PATIENTS)].groupby("patient").auroc.mean()
print(f"\nunsmoothed AUROC (k = 1, 3 seeds), {len(s)} development patients: mean {s.mean():.3f}, median {s.median():.3f}")
