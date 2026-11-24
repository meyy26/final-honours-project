import glob, os
import pandas as pd
import config

K = 1
sel = [l.strip() for l in open("fwd_selected.txt") if l.strip()]       # gamma, variance, line_length
rows, best = [], {}
for R in range(1, 8):
    files = sorted(glob.glob(f"{config.RESULT_DIR}/mlp__fwd_r{R}_*.csv"))
    if not files:
        continue
    prefix = f"mlp__fwd_r{R}_"
    base = sel[:R - 1]
    prev = best.get(R - 1, 0.5)                                        # round 1 is measured against chance
    for f in files:
        fam = os.path.basename(f)[len(prefix):-4]
        d = pd.read_csv(f)
        d = d[(d.k == K) & ~d.patient.isin(config.LOCKED_PATIENTS)]
        per = d.groupby("patient")[["auroc", "train_auroc"]].mean()
        rows.append(dict(round=R, base="+".join(base) or "(empty)", candidate=fam, patients=len(per),
                         auroc_mean=per.auroc.mean(), auroc_median=per.auroc.median(),
                         ge70=int((per.auroc >= 0.7).sum()), train_auroc=per.train_auroc.mean()))
    r = pd.DataFrame([x for x in rows if x["round"] == R])
    top = r.sort_values("auroc_mean", ascending=False).iloc[0]
    best[R] = float(r[r.candidate == sel[R - 1]].auroc_mean.iloc[0]) if R <= len(sel) else float(top.auroc_mean)

t = pd.DataFrame(rows)
t["gain_vs_previous_round"] = t.apply(lambda x: x.auroc_mean - (best.get(x["round"] - 1, 0.5)), axis=1)
t["gap"] = t.train_auroc - t.auroc_mean
t = t.sort_values(["round", "auroc_mean"], ascending=[True, False])
os.makedirs(config.RESULT_DIR, exist_ok=True)
t.to_csv(f"{config.RESULT_DIR}/forward_selection_all_rounds.csv", index=False)
for R, g in t.groupby("round"):
    g.to_csv(f"{config.RESULT_DIR}/forward_selection_round{R}.csv", index=False)
    print(f"\nRound {R} | base: {g.base.iloc[0]} | {len(g)} candidates | patients per row: {sorted(g.patients.unique())}")
    print(g.drop(columns=["round", "base"]).head(8).round(3).to_string(index=False))
print("\nSaved forward_selection_all_rounds.csv and one file per round in", config.RESULT_DIR)
