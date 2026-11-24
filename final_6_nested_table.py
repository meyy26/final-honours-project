import glob, os, sys
import pandas as pd
import config

K = int(sys.argv[1]) if len(sys.argv) > 1 else 1            # smoothing length in windows
order = ["sel_n1", "sel_n2", "sel_n3", "sel_n4", "sel_all"]
rows, per_pat = [], {}
for name in order:
    f = f"{config.RESULT_DIR}/mlp__{name}.csv"
    if not os.path.exists(f):
        continue
    d = pd.read_csv(f)
    d = d[(d.k == K) & ~d.patient.isin(config.LOCKED_PATIENTS)]
    per = d.groupby("patient")[["auroc", "train_auroc"]].mean()           # events and seeds -> patient first
    per_pat[name] = per.auroc
    by_seed = d.groupby(["seed", "patient"]).auroc.mean().groupby(level="seed").mean()
    fresh = d[d.seed != 0].groupby("patient").auroc.mean()                # seeds 1 and 2 only
    row = dict(run=name, patients=len(per), auroc_mean=per.auroc.mean(), median=per.auroc.median(),
               fresh_seeds_mean=fresh.mean(), seed_sd=by_seed.std(),
               ge70=int((per.auroc >= 0.7).sum()), ge80=int((per.auroc >= 0.8).sum()),
               train_auroc=per.train_auroc.mean())
    for s in by_seed.index:
        row[f"seed{s}"] = by_seed[s]
    rows.append(row)

t = pd.DataFrame(rows).set_index("run")
t["gap"] = t.train_auroc - t.auroc_mean
print(f"smoothing length k = {K} windows ({K * config.WIN_SEC} s)")
print(t.round(3).to_string())

print("\nPaired per-patient change from one set to the next:")
names = [n for n in order if n in per_pat]
for a, b in zip(names, names[1:]):
    j = pd.concat([per_pat[a], per_pat[b]], axis=1, keys=["a", "b"]).dropna()
    print(f"{a} -> {b}: mean change {(j.b - j.a).mean():+.3f}, better in {(j.b > j.a).sum()} of {len(j)} patients")
