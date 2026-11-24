import glob, sys
import pandas as pd
import config

lock, common = "--locked" in sys.argv, "--common" in sys.argv
files = sorted(glob.glob(f"{config.V4_RESULT_DIR}/transfer__n3_*.csv"))
if not files:
    raise SystemExit("no transfer result files")
d = pd.concat([pd.read_csv(f) for f in files])
d = d[d.patient.isin(config.LOCKED_PATIENTS) == lock]
if common:                                                    # restrict every N to the events tested at the largest N
    top = d[d.n_ft == d.n_ft.max()][["patient", "event"]].drop_duplicates()
    d = d.merge(top, on=["patient", "event"])
g = d.groupby(["n_ft", "target", "condition", "patient"])
per = g[["event_sens", "chance_event", "sens_win", "spec_win", "auroc"]].mean()
per["fa_hr"] = g.n_fa.sum() / g.hours_te.sum()                # pooled: total false alarms / total interictal hours
per["skill"] = per.event_sens - per.chance_event

conds = ["general", "specific", "transfer_all", "transfer_last"]
rows = []
for n in sorted(per.index.get_level_values("n_ft").unique()):
    for target in config.FA_TARGETS:
        sub = per.xs((n, target), level=["n_ft", "target"])
        for c in conds:
            x = sub.xs(c, level="condition")
            row = dict(n_ft=n, target=target, condition=c, patients=len(x), event_sens=x.event_sens.mean(),
                       chance=x.chance_event.mean(), skill=x.skill.mean(), fa_hr=x.fa_hr.mean(),
                       sens_win=x.sens_win.mean(), spec_win=x.spec_win.mean(), auroc=x.auroc.mean())
            for ref in ("specific", "general"):
                if c != ref:
                    r = sub.xs(ref, level="condition").reindex(x.index)
                    row[f"ahead_of_{ref}"] = int((x.skill > r.skill).sum())
            rows.append(row)
print(f"v4 transfer, {'locked' if lock else 'development'} patients, k = {config.EVAL_K}" + (", common events" if common else ""))
print(pd.DataFrame(rows).round(3).to_string(index=False))
for n in sorted(per.index.get_level_values("n_ft").unique()):
    grid = per.xs((n, config.FA_TARGETS[0]), level=["n_ft", "target"]).auroc.unstack(level="condition")[conds]
    print(f"\nN = {n}: test AUROC per patient (smoothed at k = {config.EVAL_K}; does not depend on the target)")
    print(grid.round(3).to_string())
    ahead = int((grid.transfer_all > grid.specific).sum())
    print(f"transfer_all above specific in {ahead} of {len(grid)} patients | mean AUROC: " + ", ".join(f"{c} {grid[c].mean():.3f}" for c in conds))
