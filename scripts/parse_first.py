import os, subprocess, glob
import pandas as pd

LLC = os.path.expanduser("~/llvm-project/build/bin/llc")
IR = os.path.expanduser("~/spill-ml/data/ir")
OUT = os.path.expanduser("~/spill-ml/data/dataset3_first.csv")
KEEP = set(pd.read_csv(os.path.expanduser("~/spill-ml/data/dataset2_dedup.csv"),
                       usecols=["program"]).program)
NUM = ["segments", "live_size", "weight", "uses", "defs", "max_depth",
       "loop_refs", "crosses_mask", "free_regs", "total_regs", "spillable"]

env = dict(os.environ, SPILL_DUMP="1")
rows = []
for ir in sorted(glob.glob(IR + "/*.ll")):
    prog = os.path.basename(ir)[:-3]
    if prog not in KEEP:
        continue
    log = subprocess.run([LLC, "-regalloc=greedy", ir, "-o", "/dev/null"],
                         capture_output=True, text=True, env=env).stderr
    first, visits, spilled = {}, {}, set()
    for line in log.splitlines():
        if not line.startswith("SPILLDATA,"):
            continue
        p = line.split(",")
        if p[1] == "F":
            if len(p) != 16:
                continue
            key = (p[2], int(p[3]))
            visits[key] = visits.get(key, 0) + 1
            if key not in first:                      # keep the FIRST examination
                r = dict(program=prog, func=p[2], vreg=key[1], reg_class=p[4])
                r.update({n: float(v) for n, v in zip(NUM, p[5:])})
                first[key] = r
        elif p[1] == "S":
            spilled.add((p[2], int(p[3])))
    for key, r in first.items():
        r["visits"] = visits[key]
        r["spilled"] = int(key in spilled)
        rows.append(r)
    got = sum(r["spilled"] for r in first.values())
    flag = "" if got == len(spilled) else "  <-- MISMATCH"
    print(f"{prog:28s} intervals={len(first):5d} spilled={got:3d} S_lines={len(spilled):3d}{flag}")

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print("total intervals:", len(df), " spilled:", int(df.spilled.sum()))
