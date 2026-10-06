import os, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

H = os.path.expanduser("~/spill-ml")
d = pd.read_csv(f"{H}/data/dataset.csv")
d = d[np.isfinite(d.weight)].copy()
d["group"] = d.program.str.replace("_unroll", "", regex=False)
d["log_weight"] = np.log10(d.weight.clip(lower=1e-12))
d["log_span"] = np.log10(d.span.clip(lower=1))
d["density"] = d.live_size / d.span.clip(lower=1)
d = pd.get_dummies(d, columns=["reg_class"])
feats = ["log_weight", "num_segments", "log_span", "live_size", "density"] + \
        [c for c in d.columns if c.startswith("reg_class_")]
X, y, g = d[feats].astype(float), d.spilled.values, d.group.values

rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                            class_weight="balanced", random_state=0, n_jobs=-1)
p = cross_val_predict(rf, X, y, groups=g, cv=GroupKFold(n_splits=5),
                      method="predict_proba")[:, 1]
base = -X.log_weight.values            # higher = lower weight = more likely spilled

print(f"{'thr':>5} {'flagged':>8} | {'RF P':>6} {'RF R':>6} {'RF F1':>6} | {'Base P':>6} {'Base R':>6} {'Base F1':>7}")
for t in [0.2, 0.3, 0.4, 0.5, 0.6]:
    pred = (p >= t).astype(int); n = pred.sum()
    if n == 0: continue
    bpred = np.zeros_like(pred); bpred[np.argsort(-base)[:n]] = 1
    print(f"{t:5.1f} {n:8d} | {precision_score(y,pred):6.3f} {recall_score(y,pred):6.3f} {f1_score(y,pred):6.3f}"
          f" | {precision_score(y,bpred):6.3f} {recall_score(y,bpred):6.3f} {f1_score(y,bpred):7.3f}")

pred = (p >= 0.3).astype(int)
print("\nConfusion matrix at threshold 0.3 (rows = actual 0/1, cols = predicted 0/1):")
print(confusion_matrix(y, pred))
