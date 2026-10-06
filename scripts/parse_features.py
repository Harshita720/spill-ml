import os, subprocess, glob, collections
import pandas as pd

LLC = os.path.expanduser("~/llvm-project/build/bin/llc")
IR = os.path.expanduser("~/spill-ml/data/ir")
OUT = os.path.expanduser("~/spill-ml/data/dataset2.csv")
NUM = ["segments", "live_size", "weight", "uses", "defs", "max_depth",
       "loop_refs", "crosses_mask", "free_regs", "total_regs", "spillable"]

env = dict(os.environ, SPILL_DUMP="1")
rows, bad = [], 0
for ir in sorted(glob.glob(IR + "/*.ll")):
    prog = os.path.basename(ir)[:-3]
    log = subprocess.run([LLC, "-regalloc=greedy", ir, "-o", "/dev/null"],
                         capture_output=True, text=True, env=env).stderr
    cur, visits, s_keys = {}, collections.Counter(), set()
    for line in log.splitlines():
        if not line.startswith("SPILLDATA,"):
            continue
        p = line.split(",")
        if p[1] == "F":
            if len(p) != 16:
                bad += 1
                continue
            key = (p[2], int(p[3]))
            visits[key] += 1
            r = dict(program=prog, func=p[2], vreg=int(p[3]), reg_class=p[4])
            r.update({n: float(v) for n, v in zip(NUM, p[5:])})
            r["spilled"] = cur[key]["spilled"] if key in cur else 0
            cur[key] = r                      # keep the LAST examination
        elif p[1] == "S":
            key = (p[2], int(p[3]))
            s_keys.add(key)
            if key in cur:
                cur[key]["spilled"] = 1
    for key, r in cur.items():
        r["visits"] = visits[key]
    got = sum(r["spilled"] for r in cur.values())
    flag = "" if got == len(s_keys) else "  <-- MISMATCH"
    print(f"{prog:28s} intervals={len(cur):5d} spilled={got:3d} S_lines={len(s_keys):3d}{flag}")
    rows.extend(cur.values())

df = pd.DataFrame(rows)
df.to_csv(OUT, index=False)
print("malformed lines skipped:", bad)
print("total intervals:", len(df), " spilled:", int(df.spilled.sum()),
      f"({100*df.spilled.mean():.1f}%)")
print("non-spillable intervals:", int((df.spillable == 0).sum()))
print("intervals examined more than once:", int((df.visits > 1).sum()))
