import glob, os
import pandas as pd
import config

K = 1                                                     # unsmoothed, as in the forward selection
rows, per_pat = [], {}
for f in sorted(glob.glob(f"{config.RESULT_DIR}/mlp__fwd_r1_*.csv")):
    fam = os.path.basename(f)[len("mlp__fwd_r1_"):-4]
    d = pd.read_csv(f)
    d = d[(d.k == K) & ~d.patient.isin(config.LOCKED_PATIENTS)]
    per = d.groupby("patient")[["auroc", "train_auroc"]].mean()          # events -> patient first
    per_pat[fam] = per.auroc
    rows.append(dict(family=fam, patients=len(per), auroc_mean=per.auroc.mean(), auroc_median=per.auroc.median(),
                     ge70=int((per.auroc >= 0.7).sum()), ge80=int((per.auroc >= 0.8).sum()),
                     train_auroc=per.train_auroc.mean()))
t = pd.DataFrame(rows).sort_values("auroc_mean", ascending=False)
t["gap"] = t.train_auroc - t.auroc_mean
print(f"{len(t)} families, MLP, seed 0, k = {K} (unsmoothed), development patients\n")
print(t.round(3).to_string(index=False))
t.to_csv(f"{config.RESULT_DIR}/single_family_table.csv", index=False)
pd.DataFrame(per_pat).T.to_csv(f"{config.RESULT_DIR}/single_family_per_patient.csv")

