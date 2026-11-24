import os, re
import numpy as np, pandas as pd
import config

rows = []
for pat in [f"chb{i:02d}" for i in range(1, 25)]:
    path = f"{config.RAW_DIR}/{pat}/{pat}-summary.txt"
    if not os.path.exists(path):
        continue
    text = open(path).read()
    n_raw, n_files, n_files_with = 0, 0, 0
    for block in re.split(r"File Name:\s*", text)[1:]:
        n_files += 1
        starts = re.findall(r"Seizure\s*\d*\s*Start Time:\s*(\d+)", block)
        n_raw += len(starts)
        n_files_with += int(len(starts) > 0)
    row = dict(patient=pat, recordings=n_files, recordings_with_seizure=n_files_with, raw_seizures=n_raw,
               status="excluded (config)" if pat in config.EXCLUDE_PATIENTS else
                      "locked" if pat in config.LOCKED_PATIENTS else "development")
    lf = f"{config.LABEL_DIR}/{pat}.npz"
    if os.path.exists(lf):
        l = np.load(lf)
        ok = l["event_eligible"] & (l["event_preictal"] >= config.MIN_WINDOWS_PER_CLASS)
        row.update(events_after_merge=len(l["event_onset"]), eligible_events=int(l["event_eligible"].sum()),
                   usable_events=int(ok.sum()))
    rows.append(row)

t = pd.DataFrame(rows).set_index("patient")
print(t.to_string())
print(f"\nraw seizures: all 24 patients = {t.raw_seizures.sum()} | usable patients (v3, 19) = "
      f"{t[t.usable_events >= config.MIN_SEIZURE_EVENTS].raw_seizures.sum()}")
print("usable events (v3):", int(t.usable_events.sum()), "| by status:")
print(t.groupby("status")[["raw_seizures", "usable_events"]].sum().to_string())
t.to_csv(f"{config.RESULT_DIR}/seizure_counts.csv")
