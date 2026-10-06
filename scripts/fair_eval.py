import os, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, average_precision_score

H = os.path.expanduser("~/spill-ml")
d = pd.read_csv(f"{H}/data/dataset2_dedup.csv")
d = d[(d.spillable == 1) & np.isfinite(d.weight) & (d.free_regs == 0)].reset_index(drop=True)
d["group"] = d.program.str.replace("_unroll", "", regex=False)
d["log_weight"] = np.log10(d.weight.clip(lower=1e-12))
d["log_size"] = np.log10(d.live_size.clip(lower=1))
d["loop_frac"] = d.loop_refs / (d.uses + d.defs).clip(lower=1)
d = pd.get_dummies(d, columns=["reg_class"])
cls = [c for c in d.columns if c.startswith("reg_class_")]
F = ["log_weight", "segments", "log_size", "uses", "defs", "max_depth",
     "loop_frac", "crosses_mask", "total_regs"] + cls
y, g, X = d.spilled.values, d.group.values, d[F].astype(float)
print(f"hard intervals={len(d)} spilled={y.sum()} ({100*y.mean():.1f}%) programs={len(set(g))}")

p, b = np.zeros(len(d)), -d.log_weight.values
pr_m, pr_b = [], []
for tr, te in GroupKFold(n_splits=5).split(d, y, g):
    m = RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                               class_weight="balanced", random_state=0, n_jobs=-1)
    p[te] = m.fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    if 0 < y[te].sum() < len(te):
        pr_m.append(average_precision_score(y[te], p[te]))
        pr_b.append(average_precision_score(y[te], b[te]))
print("per-fold PR-AUC  model:", np.round(pr_m, 3), " baseline:", np.round(pr_b, 3))
print("model better in", int((np.array(pr_m) > np.array(pr_b)).sum()), "of", len(pr_m), "folds\n")

print(f"{'thr':>4} {'flagged':>8} | {'Model P':>7} {'R':>6} {'F1':>6} | {'Base P':>6} {'R':>6} {'F1':>6}")
for t in [0.3, 0.5, 0.7]:
    pred = (p >= t).astype(int); n = int(pred.sum())
    if n == 0: continue
    bp = np.zeros_like(pred); bp[np.argsort(-b)[:n]] = 1
    print(f"{t:4.1f} {n:8d} | {precision_score(y,pred):7.3f} {recall_score(y,pred):6.3f} {f1_score(y,pred):6.3f}"
          f" | {precision_score(y,bp):6.3f} {recall_score(y,bp):6.3f} {f1_score(y,bp):6.3f}")
