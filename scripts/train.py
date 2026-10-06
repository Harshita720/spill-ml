import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, average_precision_score

H = os.path.expanduser("~/spill-ml")
d = pd.read_csv(f"{H}/data/dataset.csv")
print("spills among non-spillable rows:", d[~np.isfinite(d.weight)].spilled.sum())
d = d[np.isfinite(d.weight)].copy()

d["group"] = d.program.str.replace("_unroll", "", regex=False)
d["log_weight"] = np.log10(d.weight.clip(lower=1e-12))
d["log_span"] = np.log10(d.span.clip(lower=1))
d["density"] = d.live_size / d.span.clip(lower=1)
d = pd.get_dummies(d, columns=["reg_class"])
feats = ["log_weight", "num_segments", "log_span", "live_size", "density"] + \
        [c for c in d.columns if c.startswith("reg_class_")]
X, y, g = d[feats].astype(float), d.spilled.values, d.group.values
print(f"rows={len(d)} spilled={y.sum()} ({100*y.mean():.1f}%) programs={len(set(g))}")

models = {
    "LogReg": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000)),
    "DecisionTree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=0),
    "RandomForest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                           class_weight="balanced", random_state=0, n_jobs=-1),
    "GradBoost": GradientBoostingClassifier(random_state=0),
}

res = {"Baseline (low spill weight)": [[], []]}
for k in models: res[k] = [[], []]

for tr, te in GroupKFold(n_splits=5).split(X, y, g):
    if y[te].sum() == 0: continue
    base = -X.iloc[te].log_weight.values          # lower weight => more likely spilled
    res["Baseline (low spill weight)"][0].append(roc_auc_score(y[te], base))
    res["Baseline (low spill weight)"][1].append(average_precision_score(y[te], base))
    for k, m in models.items():
        m.fit(X.iloc[tr], y[tr])
        p = m.predict_proba(X.iloc[te])[:, 1]
        res[k][0].append(roc_auc_score(y[te], p))
        res[k][1].append(average_precision_score(y[te], p))

rows = []
print(f"\n{'model':30s} {'ROC-AUC':>14s} {'PR-AUC':>14s}")
for k, (a, b) in res.items():
    print(f"{k:30s} {np.mean(a):.3f} ± {np.std(a):.3f}  {np.mean(b):.3f} ± {np.std(b):.3f}")
    rows.append(dict(model=k, roc_auc=np.mean(a), roc_std=np.std(a), pr_auc=np.mean(b), pr_std=np.std(b)))
print(f"(random guessing PR-AUC would be about {y.mean():.3f})")

os.makedirs(f"{H}/results", exist_ok=True)
pd.DataFrame(rows).to_csv(f"{H}/results/model_comparison.csv", index=False)

rf = models["RandomForest"].fit(X, y)
imp = pd.Series(rf.feature_importances_, index=feats).sort_values()
imp.plot.barh(figsize=(7, 4)); plt.title("Random forest feature importance")
plt.tight_layout(); plt.savefig(f"{H}/results/feature_importance.png", dpi=150)
print("\nsaved results/model_comparison.csv and results/feature_importance.png")
