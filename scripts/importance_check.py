import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, average_precision_score

H = os.path.expanduser("~/spill-ml")
d = pd.read_csv(f"{H}/data/dataset.csv")
d = d[np.isfinite(d.weight)].copy()
d["group"] = d.program.str.replace("_unroll", "", regex=False)
d["log_weight"] = np.log10(d.weight.clip(lower=1e-12))
d["log_span"] = np.log10(d.span.clip(lower=1))
d["density"] = d.live_size / d.span.clip(lower=1)
d["log_cost"] = d.log_weight + np.log10(d.live_size.clip(lower=1))   # undoes LLVM's size normalization (approx.)
d = pd.get_dummies(d, columns=["reg_class"])
base = ["log_weight", "num_segments", "log_span", "live_size", "density"] + \
       [c for c in d.columns if c.startswith("reg_class_")]
y, g = d.spilled.values, d.group.values
mk = lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                    class_weight="balanced", random_state=0, n_jobs=-1)
folds = [(tr, te) for tr, te in GroupKFold(n_splits=5).split(d, y, g) if y[te].sum() > 0]

# 1) permutation importance on held-out programs (drop in PR-AUC when a feature is shuffled)
X = d[base].astype(float)
imps = []
for tr, te in folds:
    m = mk().fit(X.iloc[tr], y[tr])
    r = permutation_importance(m, X.iloc[te], y[te], scoring="average_precision",
                               n_repeats=10, random_state=0)
    imps.append(r.importances_mean)
imp = pd.Series(np.mean(imps, axis=0), index=base).sort_values()
print("Permutation importance (PR-AUC drop):\n", imp.round(4).to_string())
imp.plot.barh(figsize=(7, 4)); plt.title("Permutation importance (held-out programs)")
plt.xlabel("drop in PR-AUC"); plt.tight_layout()
plt.savefig(f"{H}/results/permutation_importance.png", dpi=150)

# 2) does adding log_cost (weight x size) help?
for name, feats in [("current features", base), ("+ log_cost", base + ["log_cost"])]:
    Xf = d[feats].astype(float); a, b = [], []
    for tr, te in folds:
        p = mk().fit(Xf.iloc[tr], y[tr]).predict_proba(Xf.iloc[te])[:, 1]
        a.append(roc_auc_score(y[te], p)); b.append(average_precision_score(y[te], p))
    print(f"{name:18s} ROC-AUC {np.mean(a):.3f}  PR-AUC {np.mean(b):.3f} ± {np.std(b):.3f}")
