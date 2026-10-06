import os, pandas as pd
P = os.path.expanduser("~/spill-ml/data/dataset.csv")
d = pd.read_csv(P)
key = ["program", "func", "vreg"]
lab = d.groupby(key).spilled.max().reset_index()
out = d.drop_duplicates(key, keep="first").drop(columns="spilled").merge(lab, on=key)
out.to_csv(P, index=False)
print(len(d), "->", len(out), "rows; spilled", d.spilled.sum(), "->", out.spilled.sum())
