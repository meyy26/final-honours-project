import sys
import pandas as pd
import config

K = int(sys.argv[1]) if len(sys.argv) > 1 else 1            # smoothing length in windows

def per_patient(name):
    d = pd.read_csv(f"{config.RESULT_DIR}/mlp__{name}.csv")
    d = d[(d.k == K) & ~d.patient.isin(config.LOCKED_PATIENTS)]
    g = d.groupby("patient")
    fresh = d[d.seed != 0].groupby("patient").auroc.mean()                # seeds 1 and 2 only
    return pd.DataFrame({"auroc": g.auroc.mean(), "fresh": fresh, "predicted": g.detected.mean(),
                         "fa_hr": g.fa_per_hr.mean(), "train": g.train_auroc.mean()})

n3, al = per_patient("sel_n3"), per_patient("sel_all")
t = pd.concat({"3 families": n3, "all 29": al}, axis=1)
t[("diff", "auroc")] = t[("3 families", "auroc")] - t[("all 29", "auroc")]
print(f"MLP, smoothing k = {K} window(s), mean over events and 3 seeds")
print(t.round(3).to_string())
d = t[("diff", "auroc")]
print(f"\nmean AUROC: 3 families {n3.auroc.mean():.3f} | all 29 {al.auroc.mean():.3f}")
print(f"fresh seeds (1, 2): 3 families {n3.fresh.mean():.3f} | all 29 {al.fresh.mean():.3f}")
print(f"3 families better in {(d > 0).sum()} of {len(d)} patients | mean difference {d.mean():+.3f}")
print("patients at or above 0.7:", int((n3.auroc >= 0.7).sum()), "(3 families),", int((al.auroc >= 0.7).sum()), "(all 29)")
