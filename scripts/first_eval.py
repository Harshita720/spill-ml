import os, numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, precision_score, recall_score, f1_score

H = os.path.expanduser("~/spill-ml")
d = pd.read_csv(f"{H}/data/dataset3_first.csv")
d = d[(d.spillable == 1) & np.isfinite(d.weight)].reset_index(drop=True)
d["group"] = d.program.str.replace("_unroll", "", regex=False)
d["project"] = np.where(d.program.str.startswith("zlib_"), "zlib",
               np.where(d.program.str.startswith("lua_"), "lua", "polybench"))
d["log_weight"] = np.log10(d.weight.clip(lower=1e-12))
d["log_size"] = np.log10(d.live_size.clip(lower=1))
d["loop_frac"] = d.loop_refs / (d.uses + d.defs).clip(lower=1)
d = pd.get_dummies(d, columns=["reg_class"])
cls = [c for c in d.columns if c.startswith("reg_class_")]
F = ["log_weight", "segments", "log_size", "uses", "defs", "max_depth",
     "loop_frac", "crosses_mask", "total_regs"] + cls
X, y, g = d[F].astype(float), d.spilled.values, d.group.values
hard = (d.free_regs == 0).values
base_all = -d.log_weight.values

print(f"spillable intervals={len(d)} spilled={y.sum()} programs={len(set(g))}")
print(f"hard at FIRST visit: {hard.sum()}  spilled among them: {y[hard].sum()} ({100*y[hard].mean():.1f}%)")
print(f"spills that were NOT hard at first visit: {y[~hard].sum()}  <- the gate would miss these\n")

rf = lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                    class_weight="balanced", random_state=0, n_jobs=-1)

# 1) grouped 5-fold CV on first-visit hard intervals
idx = np.where(hard)[0]
Xh, yh, gh, bh = X.iloc[idx], y[idx], g[idx], base_all[idx]
p = np.zeros(len(idx)); pm, pb = [], []
for tr, te in GroupKFold(n_splits=5).split(Xh, yh, gh):
    p[te] = rf().fit(Xh.iloc[tr], yh[tr]).predict_proba(Xh.iloc[te])[:, 1]
    if 0 < yh[te].sum() < len(te):
        pm.append(average_precision_score(yh[te], p[te]))
        pb.append(average_precision_score(yh[te], bh[te]))
print("1) 5-fold CV, hard intervals (random-guess PR-AUC %.3f)" % yh.mean())
print("   per-fold PR-AUC model   :", np.round(pm, 3))
print("   per-fold PR-AUC baseline:", np.round(pb, 3))
print("   model better in", int((np.array(pm) > np.array(pb)).sum()), "of", len(pm), "folds")
print(f"   {'thr':>4} {'flagged':>8} | {'Model P':>7} {'R':>6} {'F1':>6} | {'Base P':>6} {'R':>6} {'F1':>6}")
for t in [0.3, 0.5, 0.7]:
    pred = (p >= t).astype(int); n = int(pred.sum())
    if n == 0: continue
    bp = np.zeros_like(pred); bp[np.argsort(-bh)[:n]] = 1
    print(f"   {t:4.1f} {n:8d} | {precision_score(yh,pred):7.3f} {recall_score(yh,pred):6.3f} {f1_score(yh,pred):6.3f}"
          f" | {precision_score(yh,bp):6.3f} {recall_score(yh,bp):6.3f} {f1_score(yh,bp):6.3f}")

# 2) leave one project out, hard intervals
print("\n2) Leave-one-project-out, hard intervals")
print(f"   {'test':10s} {'hard':>5s} {'spilled':>7s} | {'model PR':>8s} {'base PR':>8s} | {'model F1@0.5':>12s} {'base F1':>8s} | rand")
for proj in ["polybench", "zlib", "lua"]:
    te = (d.project.values == proj) & hard
    tr = (d.project.values != proj) & hard
    if y[te].sum() < 5 or y[tr].sum() < 5:
        print("  ", proj, "too few spills"); continue
    pt = rf().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    bt = base_all[te]; pred = (pt >= 0.5).astype(int); n = int(pred.sum())
    bp = np.zeros_like(pred); bp[np.argsort(-bt)[:n]] = 1
    print(f"   {proj:10s} {te.sum():5d} {y[te].sum():7d} | {average_precision_score(y[te],pt):8.3f} "
          f"{average_precision_score(y[te],bt):8.3f} | {f1_score(y[te],pred):12.3f} {f1_score(y[te],bp):8.3f} | {y[te].mean():.3f}")

# 3) end to end over ALL spillable intervals (non-hard get score 0)
print("\n3) End to end over all intervals (gate + model) vs weight ranking")
pm, pb = [], []
for tr, te in GroupKFold(n_splits=5).split(X, y, g):
    trh = tr[hard[tr]]; teh = te[hard[te]]
    s = np.zeros(len(te))
    if len(teh):
        s[np.isin(te, teh)] = rf().fit(X.iloc[trh], y[trh]).predict_proba(X.iloc[teh])[:, 1]
    if y[te].sum() > 0:
        pm.append(average_precision_score(y[te], s))
        pb.append(average_precision_score(y[te], base_all[te]))
print("   random-guess PR-AUC %.3f" % y.mean())
print("   per-fold PR-AUC end-to-end:", np.round(pm, 3), " mean %.3f" % np.mean(pm))
print("   per-fold PR-AUC baseline  :", np.round(pb, 3), " mean %.3f" % np.mean(pb))
