import os, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score

H = os.path.expanduser("~/spill-ml")
d = pd.read_csv(f"{H}/data/dataset2_dedup.csv")
d = d[(d.spillable == 1) & np.isfinite(d.weight) & (d.free_regs == 0)].reset_index(drop=True)
d["project"] = np.where(d.program.str.startswith("zlib_"), "zlib",
               np.where(d.program.str.startswith("lua_"), "lua", "polybench"))
d["log_weight"] = np.log10(d.weight.clip(lower=1e-12))
d["log_size"] = np.log10(d.live_size.clip(lower=1))
d["loop_frac"] = d.loop_refs / (d.uses + d.defs).clip(lower=1)
d = pd.get_dummies(d, columns=["reg_class"])
cls = [c for c in d.columns if c.startswith("reg_class_")]
F = ["log_weight", "segments", "log_size", "uses", "defs", "max_depth",
     "loop_frac", "crosses_mask", "total_regs"] + cls
y, X = d.spilled.values, d[F].astype(float)

print(f"{'test project':12s} {'hard':>5s} {'spilled':>7s} | {'model PR':>8s} {'base PR':>8s} | "
      f"{'model P/R/F1 @0.5':>20s} | {'base P/R/F1 (same #flagged)':>28s}")
for proj in ["polybench", "zlib", "lua"]:
    te = (d.project == proj).values
    if y[te].sum() < 5 or y[~te].sum() < 5:
        print(proj, "too few spills to evaluate"); continue
    m = RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                               class_weight="balanced", random_state=0, n_jobs=-1)
    p = m.fit(X[~te], y[~te]).predict_proba(X[te])[:, 1]
    b = -d.log_weight.values[te]
    pred = (p >= 0.5).astype(int); n = int(pred.sum())
    bp = np.zeros_like(pred); bp[np.argsort(-b)[:n]] = 1
    yt = y[te]
    print(f"{proj:12s} {te.sum():5d} {yt.sum():7d} | {average_precision_score(yt,p):8.3f} "
          f"{average_precision_score(yt,b):8.3f} | "
          f"{precision_score(yt,pred):6.3f}/{recall_score(yt,pred):.3f}/{f1_score(yt,pred):.3f} | "
          f"{precision_score(yt,bp):6.3f}/{recall_score(yt,bp):.3f}/{f1_score(yt,bp):.3f}")
    print(f"{'':12s}  (random-guess PR-AUC {yt.mean():.3f})")
