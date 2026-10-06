import os, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score

H = os.path.expanduser("~/spill-ml")
d = pd.read_csv(f"{H}/data/dataset3_first.csv")
d = d[(d.spillable == 1) & np.isfinite(d.weight)].reset_index(drop=True)
d["group"] = d.program.str.replace("_unroll", "", regex=False)
d["log_weight"] = np.log10(d.weight.clip(lower=1e-12))
d["log_size"] = np.log10(d.live_size.clip(lower=1))
d["loop_frac"] = d.loop_refs / (d.uses + d.defs).clip(lower=1)
d = pd.get_dummies(d, columns=["reg_class"])
cls = [c for c in d.columns if c.startswith("reg_class_")]
F = ["log_weight", "segments", "log_size", "uses", "defs", "max_depth",
     "loop_frac", "crosses_mask", "total_regs"] + cls
X, y, g = d[F].astype(float), d.spilled.values, d.group.values
hard = (d.free_regs == 0).values
w = -d.log_weight.values
w_pos = w - w.min() + 1                      # positive, so hard intervals outrank non-hard (score 0)
rf = lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                    class_weight="balanced", random_state=0, n_jobs=-1)

print(f"recall ceiling of the gate: {y[hard].sum()}/{y.sum()} = {y[hard].sum()/y.sum():.3f}\n")
res = {"gate + model": [], "gate + weight": [], "weight only (no gate)": []}
for tr, te in GroupKFold(n_splits=5).split(X, y, g):
    if y[te].sum() == 0: continue
    mh = hard[te]; teh = te[mh]; trh = tr[hard[tr]]
    s_m = np.zeros(len(te)); s_b = np.zeros(len(te))
    s_b[mh] = w_pos[teh]
    if mh.any():
        s_m[mh] = 1e-6 + rf().fit(X.iloc[trh], y[trh]).predict_proba(X.iloc[teh])[:, 1]
    res["gate + model"].append(average_precision_score(y[te], s_m))
    res["gate + weight"].append(average_precision_score(y[te], s_b))
    res["weight only (no gate)"].append(average_precision_score(y[te], w[te]))
print(f"End to end PR-AUC over all spillable intervals (random guess {y.mean():.3f})")
for k, v in res.items():
    print(f"  {k:24s} per fold {np.round(v, 3)}  mean {np.mean(v):.3f}")

m = d[(~hard) & (d.spilled == 1)]
print(f"\nThe {len(m)} spills that were not hard at first visit:")
print(f"  examined more than once: {int((m.visits > 1).sum())} of {len(m)}")
print(f"  free registers at first visit (median): {m.free_regs.median():.0f} of {m.total_regs.median():.0f}")
