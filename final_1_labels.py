import os, re, glob
import numpy as np
import config

os.makedirs(config.LABEL_DIR, exist_ok=True)
WIN = config.WIN_SEC

def hms(s):
    h, m, sec = map(int, s.strip().split(":"))
    return h * 3600 + m * 60 + sec

def parse_summary(patient):
    """file name -> (absolute start time in s, list of absolute (onset, offset))"""
    text = open(f"{config.RAW_DIR}/{patient}/{patient}-summary.txt").read()
    files, day, prev = {}, 0, None
    for block in re.split(r"File Name:\s*", text)[1:]:
        name = block.split()[0]
        m = re.search(r"File Start Time:\s*(\d+:\d+:\d+)", block)
        if not m:
            continue
        s = hms(m.group(1))
        if prev is not None and s < prev:
            day += 1                                   # clock passed midnight
        prev = s
        start = day * 86400 + s
        st = re.findall(r"Seizure\s*\d*\s*Start Time:\s*(\d+)", block)
        en = re.findall(r"Seizure\s*\d*\s*End Time:\s*(\d+)", block)
        files[name] = (start, [(start + int(a), start + int(b)) for a, b in zip(st, en)])
    return files

def merge_events(seizures, gap):
    """seizures closer than gap (end of one to start of next) become one event; chains merge."""
    events = []
    for on, off in sorted(seizures):
        if events and on - events[-1][1] < gap:
            events[-1][1] = max(events[-1][1], off)
        else:
            events.append([on, off])
    return events

print("patient  raw_sz  events  eligible  usable  preictal_win  interictal_win  verdict")
for f in sorted(glob.glob(f"{config.FEATURE_DIR}/chb*.npz")):
    patient = os.path.basename(f)[:-4]
    if patient in config.EXCLUDE_PATIENTS:
        print(f"{patient}   excluded (config)"); continue
    d = np.load(f)
    src, widx = d["source_file"], d["win_idx"]
    files = parse_summary(patient)
    if any(n not in files for n in np.unique(src)):
        print(f"{patient}   skipped: summary has no clock times for some files"); continue

    raw_sz = [sz for _, szs in files.values() for sz in szs]
    events = merge_events(raw_sz, config.MERGE_GAP_SEC)
    on = np.array([e[0] for e in events], dtype=float)
    off = np.array([e[1] for e in events], dtype=float)
    eligible = np.array([i == 0 or on[i] - off[i - 1] >= config.LEAD_MIN_GAP_SEC for i in range(len(events))])

    t0 = np.array([files[s][0] for s in src], dtype=float) + WIN * widx    # absolute start of each window (s)
    t1 = t0 + WIN

    ictal = ((t0[:, None] < off[None, :]) & (t1[:, None] > on[None, :])).any(axis=1)
    since_end = t0[:, None] - off[None, :]
    post = ((since_end >= 0) & (since_end < config.POSTICTAL_SEC)).any(axis=1)
    dist = np.maximum(0, np.maximum(on[None, :] - t1[:, None], t0[:, None] - off[None, :])).min(axis=1)

    y = np.full(len(src), -1)                                              # -1 = excluded
    y[dist >= config.INTERICTAL_MIN_DIST_SEC] = 0                          # interictal: far from every event
    pre_id = np.zeros(len(src), dtype=int)
    for k in range(len(events)):
        m = (t0 >= on[k] - config.PREICTAL_FAR_SEC) & (t1 <= on[k] - config.PREICTAL_NEAR_SEC) & ~ictal & ~post
        y[m] = 1
        pre_id[m] = k + 1                                                  # event number, 1-based
    y[ictal | post] = -1                                                   # ictal and post-ictal never used

    n_pre = np.array([int((pre_id == k + 1).sum()) for k in range(len(events))])
    usable = int((eligible & (n_pre >= config.MIN_WINDOWS_PER_CLASS)).sum())
    n_inter = int((y == 0).sum())
    verdict = "OK" if usable >= config.MIN_SEIZURE_EVENTS and n_inter >= 1000 else "too few"

    np.savez(f"{config.LABEL_DIR}/{patient}.npz", y=y, t=t0, pre_id=pre_id, source_file=src, win_idx=widx,
             event_onset=on, event_end=off, event_eligible=eligible, event_preictal=n_pre)
    print(f"{patient}   {len(raw_sz):>4}   {len(events):>4}   {int(eligible.sum()):>6}   {usable:>5}   "
          f"{int((y == 1).sum()):>10}   {n_inter:>12}   {verdict}")
