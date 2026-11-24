import os, sys
import pandas as pd
import config

MODEL, K, MAX_FAMS, MIN_GAIN = "mlp", 1, 10, 0.005       # K = 1: select on unsmoothed AUROC
fams = [l.strip() for l in open("final_families.txt") if l.strip()]
sel = [l.strip() for l in open("fwd_selected.txt") if l.strip()]
rnd = len(sel) + 1
remaining = [f for f in fams if f not in sel]
mode = sys.argv[1]
prev = pd.read_csv("fwd_path.csv").auroc.iloc[-1] if os.path.exists("fwd_path.csv") else 0.5

def summarize(cand):
    f = f"{config.RESULT_DIR}/{MODEL}__fwd_r{rnd}_{cand}.csv"
    if not os.path.exists(f):
        return None
    d = pd.read_csv(f)
    d = d[d.k == K]
    d = d[~d.patient.isin(config.LOCKED_PATIENTS)] 
    per = d.groupby("patient")[["auroc", "detected", "fa_per_hr", "train_auroc"]].mean()   # events -> patient first
    return dict(patients=len(per), auroc=per.auroc.mean(), median=per.auroc.median(),
                ge70=int((per.auroc >= 0.7).sum()), seizures_predicted=per.detected.mean(),
                fa_per_hr=per.fa_per_hr.mean(), train_auroc=per.train_auroc.mean())

if mode == "candidates":
    if len(sel) >= MAX_FAMS:
        raise SystemExit(f"already {len(sel)} families: {'+'.join(sel)}")
    open("fwd_candidates.txt", "w").write("\n".join(remaining) + "\n")
    print(f"round {rnd}; selected so far: {'+'.join(sel) or '(none)'}")
    print(f"{len(remaining)} candidates -> qsub -J 1-{len(remaining)} run_fwd.sh")
else:
    rows = []
    for c in remaining:
        r = summarize(c)
        if r:
            rows.append(dict(add=c, gain=r["auroc"] - prev, **r))
    if not rows:
        raise SystemExit("no result files for this round yet")
    t = pd.DataFrame(rows).sort_values("auroc", ascending=False)
    print(f"round {rnd} | selected: {'+'.join(sel) or '(none)'} | previous AUROC {prev:.3f}" + (" (chance, round 1)" if rnd == 1 else ""))
    print(t.round(3).to_string(index=False))
    print(f"({len(t)} of {len(remaining)} candidates finished)")
    if mode == "add":
        if len(t) < len(remaining) or t.patients.nunique() > 1:
            raise SystemExit("not all candidates finished, or they cover different numbers of patients; nothing added")
        top = t.iloc[0]
        open("fwd_selected.txt", "a").write(top["add"] + "\n")
        pd.DataFrame([dict(round=rnd, added=top["add"], auroc=top["auroc"], gain=top["gain"], patients=int(top["patients"]))]
                     ).to_csv("fwd_path.csv", mode="a", header=not os.path.exists("fwd_path.csv"), index=False)
        print("added:", top["add"], "| gain:", round(top["gain"], 4))
        if rnd > 1 and top["gain"] < MIN_GAIN:
            print(f"gain is below {MIN_GAIN}, about the size of seed noise")
