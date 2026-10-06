import os, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_score, recall_score, f1_score

H = os.path.expanduser("~/spill-ml")
d = pd.read_csv(f"{H}/data/dataset2_dedup.csv")
d = d[(d.spillable == 1) & np.isfinite(d.weight)].reset_index(drop=True)
d["group"] = d.program.str.replace("_unroll", "", regex=False)
d["log_weight"] = np.log10(d.weight.clip(lower=1e-12))
d["log_size"] = np.log10(d.live_size.clip(lower=1))
d["loop_frac"] = d.loop_refs / (d.uses + d.defs).clip(lower=1)
d["free_frac"] = d.free_regs / d.total_regs.clip(lower=1)
d = pd.get_dummies(d, columns=["reg_class"])
cls = [c for c in d.columns if c.startswith("reg_class_")]
A = ["log_weight", "segments", "log_size"] + cls
NEW = ["uses", "defs", "max_depth", "loop_frac", "crosses_mask", "free_frac", "total_regs"]
sets = {"A (old)": A, "B (all new)": A + NEW,
        "B without free_frac": A + [c for c in NEW if c != "free_frac"]}
y, g = d.spilled.values, d.group.values
hard = (d.free_regs == 0).values
rf = lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                    class_weight="balanced", random_state=0, n_jobs=-1)
folds = list(GroupKFold(n_splits=5).split(d, y, g))

print(f"rows={len(d)} spilled={y.sum()}")
print(f"intervals with NO free register: {hard.sum()}  spilled among them: {y[hard].sum()}"
      f"  (spill rate {100*y[hard].mean():.1f}%)")
print(f"spills that had a free register at decision time: {y[~hard].sum()}\n")

P = {"Baseline (weight)": -d.log_weight.values}
for name, F in sets.items():
    X = d[F].astype(float); p = np.zeros(len(d))
    for tr, te in folds:
        p[te] = rf().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    P[name] = p

def score(mask, name):
    a, b = [], []
    for tr, te in folds:
        t = te[mask[te]]
        if y[t].sum() == 0 or y[t].sum() == len(t): continue
        a.append(roc_auc_score(y[t], P[name][t])); b.append(average_precision_score(y[t], P[name][t]))
    return a, b

for title, mask in [("ALL intervals", np.ones(len(d), bool)),
                    ("HARD intervals only (no free register)", hard)]:
    print(title, f"(random-guess PR-AUC ~ {y[mask].mean():.3f})")
    for name in P:
        a, b = score(mask, name)
        print(f"  {name:22s} ROC-AUC {np.mean(a):.3f} ± {np.std(a):.3f}   PR-AUC {np.mean(b):.3f} ± {np.std(b):.3f}  ({len(a)} folds)")
    print()

p = P["B (all new)"]
print("Precision / recall for B (pooled out-of-fold), baseline flags the same number of intervals")
for t in [0.3, 0.5, 0.7]:
    pred = (p >= t).astype(int); n = pred.sum()
    if n == 0: continue
    bp = np.zeros_like(pred); bp[np.argsort(-P["Baseline (weight)"])[:n]] = 1
    print(f"  thr {t}: flagged {n:4d} | B  P={precision_score(y,pred):.3f} R={recall_score(y,pred):.3f} F1={f1_score(y,pred):.3f}"
          f" | base P={precision_score(y,bp):.3f} R={recall_score(y,bp):.3f} F1={f1_score(y,bp):.3f}")
