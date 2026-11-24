import sys
import pandas as pd
import config

run = sys.argv[1]                                              # e.g. mlp__n3
d = pd.read_csv(f"{config.RESULT_DIR}/eval__{run}.csv")
d = d[d.patient.isin(config.LOCKED_PATIENTS) == ("--locked" in sys.argv)]

cols = ["event_sens", "chance_event", "sens_win", "spec_win", "fa_hr", "auroc"]
per = d.groupby(["target", "patient"])[cols].mean()            # events and seeds -> patient first
g = d.groupby(["target", "patient"])
per["fa_hr"] = g.n_fa.sum() / g.hours_te.sum()      # pooled: total false alarms / total interictal hours

out = []
for target, g in per.groupby(level="target"):
    out.append(dict(target=target, patients=len(g), event_sens=g.event_sens.mean(), chance=g.chance_event.mean(),
                    above_chance=int((g.event_sens > g.chance_event).sum()),
                    sens_win=g.sens_win.mean(), spec_win=g.spec_win.mean(),
                    fa_hr_observed=g.fa_hr.mean(), fa_hr_median=g.fa_hr.median(), auroc=g.auroc.mean()))
print(f"{run} | k = {config.EVAL_K} windows ({config.EVAL_K * config.WIN_SEC} s) | mean over patients")
print(pd.DataFrame(out).round(3).to_string(index=False))
for target in config.FA_TARGETS:
    print(f"\ntarget {target}/h, per patient:")
    print(per.xs(target, level="target").round(3).to_string())
