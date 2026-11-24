import glob, os, sys
import pandas as pd
import config

K = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 1      # smoothing length in windows
show_locked = "--locked" in sys.argv

rows = []
for f in sorted(glob.glob(f"{config.RESULT_DIR}/mlp__allfeat_chb*.csv")):
    d = pd.read_csv(f)
    d = d[d.k == K]
    rows.append(dict(patient=d.patient.iloc[0], events=d.event.nunique(), auroc=d.auroc.mean(),
                     seizures_predicted=d.detected.mean(), fa_per_hr=d.fa_per_hr.mean(),
                     train_auroc=d.train_auroc.mean()))
p = pd.DataFrame(rows).set_index("patient")
p["locked"] = p.index.isin(config.LOCKED_PATIENTS)
hidden = int(p.locked.sum())
if not show_locked:
    p = p[~p.locked]
print(f"all-features MLP, smoothing k = {K} windows ({K * config.WIN_SEC} s), 3 seeds")
print(p.drop(columns="locked").round(3).to_string())
s = p.drop(columns=["locked", "events"])
print(f"\n{len(p)} patients | AUROC mean {s.auroc.mean():.3f}, median {s.auroc.median():.3f} | "
      f">=0.7: {(s.auroc >= 0.7).sum()}, >=0.8: {(s.auroc >= 0.8).sum()} | "
      f"train gap {(s.train_auroc - s.auroc).mean():.3f}")
if not show_locked and hidden:
    print(f"({hidden} locked-patient files exist but are hidden; add --locked to show only the locked ones)")
