import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, average_precision_score

H = os.path.expanduser("~/spill-ml")
d = pd.read_csv(f"{H}/data/dataset2_dedup.csv")
d = d[(d.spillable == 1) & np.isfinite(d.weight)].copy()
d["group"] = d.program.str.replace("_unroll", "", regex=False)
d["log_weight"] = np.log10(d.weight.clip(lower=1e-12))
d["log_size"] = np.log10(d.live_size.clip(lower=1))
d["loop_frac"] = d.loop_refs / (d.uses + d.defs).clip(lower=1)
d["free_frac"] = d.free_regs / d.total_regs.clip(lower=1)
d = pd.get_dummies(d, columns=["reg_class"])
cls = [c for c in d.columns if c.startswith("reg_class_")]
A = ["log_weight", "segments", "log_size"] + cls
NEW = ["uses", "defs", "max_depth", "loop_frac", "crosses_mask", "free_frac", "total_regs"]
B = A + NEW
y, g = d.spilled.values, d.group.values
print(f"rows={len(d)} spilled={y.sum()} ({100*y.mean():.1f}%) programs={len(set(g))}")
print(f"random-guess PR-AUC ~ {y.mean():.3f}\n")

rf = lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                    class_weight="balanced", random_state=0, n_jobs=-1)
gb = lambda: GradientBoostingClassifier(random_state=0)
setups = {"Baseline (weight only)": None,
          "RF  on A (old features)": (rf, A),
          "RF  on B (+ new features)": (rf, B),
          "GB  on B (+ new features)": (gb, B)}
res = {k: ([], []) for k in setups}
folds = [(tr, te) for tr, te in GroupKFold(n_splits=5).split(d, y, g) if y[te].sum() > 0]
for tr, te in folds:
    for name, s in setups.items():
        if s is None:
            p = -d.log_weight.values[te]
        else:
            mk, F = s
            p = mk().fit(d[F].astype(float).iloc[tr], y[tr]).predict_proba(d[F].astype(float).iloc[te])[:, 1]
        res[name][0].append(roc_auc_score(y[te], p))
        res[name][1].append(average_precision_score(y[te], p))

print(f"{'model':28s} {'ROC-AUC':>14s} {'PR-AUC':>14s}")
for k, (a, b) in res.items():
    print(f"{k:28s} {np.mean(a):.3f} ± {np.std(a):.3f}  {np.mean(b):.3f} ± {np.std(b):.3f}")
diff = np.array(res["RF  on B (+ new features)"][1]) - np.array(res["RF  on A (old features)"][1])
print("\nPer-fold PR-AUC change, B minus A:", np.round(diff, 3),
      f" -> better in {(diff > 0).sum()}/{len(diff)} folds, mean {diff.mean():+.3f}")

# permutation importance of B on held-out programs
imps = []
for tr, te in folds:
    m = rf().fit(d[B].astype(float).iloc[tr], y[tr])
    r = permutation_importance(m, d[B].astype(float).iloc[te], y[te],
                               scoring="average_precision", n_repeats=5, random_state=0)
    imps.append(r.importances_mean)
imp = pd.Series(np.mean(imps, axis=0), index=B).sort_values()
print("\nPermutation importance (drop in PR-AUC), top 10:")
print(imp.tail(10).round(4).to_string())
imp.tail(12).plot.barh(figsize=(7, 4)); plt.xlabel("drop in PR-AUC")
plt.title("Permutation importance, feature set B"); plt.tight_layout()
plt.savefig(f"{H}/results/permutation_importance_v2.png", dpi=150)
