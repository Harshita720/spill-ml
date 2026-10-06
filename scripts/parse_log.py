import re, subprocess, glob, os, csv

LLC = os.path.expanduser("~/llvm-project/build/bin/llc")
IR_DIR = os.path.expanduser("~/spill-ml/data/ir")
OUT = os.path.expanduser("~/spill-ml/data/dataset.csv")

sel_re = re.compile(r'^selectOrSplit (\w+):%(\d+) (.*?)\s+weight:(\S+)')
spill_re = re.compile(r'^Inline spilling (\w+):%(\d+) ')
seg_re = re.compile(r'\[(\d+)[A-Za-z],(\d+)[A-Za-z]')
func_re = re.compile(r'^\*+ Function: (\S+)')

rows = []
for ir in sorted(glob.glob(IR_DIR + "/*.ll")):
    prog = os.path.basename(ir)[:-3]
    log = subprocess.run(
        [LLC, "-regalloc=greedy", "-debug-only=regalloc", ir, "-o", "/dev/null"],
        capture_output=True, text=True).stderr
    func, last, spill_lines, first = "?", None, 0, len(rows)
    for line in log.splitlines():
        m = func_re.match(line)
        if m:
            func, last = m.group(1), None
            continue
        m = sel_re.match(line)
        if m:
            cls, vreg, ivl, w = m.groups()
            segs = [(int(a), int(b)) for a, b in seg_re.findall(ivl)]
            if not segs:
                last = None
                continue
            last = dict(program=prog, func=func, vreg=int(vreg), reg_class=cls,
                        num_segments=len(segs),
                        span=segs[-1][1] - segs[0][0],
                        live_size=sum(b - a for a, b in segs),
                        weight=float(w), spilled=0)
            rows.append(last)
            continue
        m = spill_re.match(line)
        if m:
            spill_lines += 1
            if last and last["vreg"] == int(m.group(2)):
                last["spilled"] = 1
    got = sum(r["spilled"] for r in rows[first:])
    flag = "" if got == spill_lines else "  <-- MISMATCH"
    print(f"{prog:28s} rows={len(rows)-first:5d} spilled={got:3d} log_spills={spill_lines:3d}{flag}")

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("total rows:", len(rows), " total spilled:", sum(r["spilled"] for r in rows))
