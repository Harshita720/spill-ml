# ML-guided spill prediction for LLVM register allocation

Predicts whether LLVM's Greedy register allocator (LLVM 19.1.7, x86-64) spills a live interval, using features recorded at the allocator's decision point.

## Pipeline
C source -> LLVM IR -> patched llc (-regalloc=greedy, SPILL_DUMP=1) -> SPILLDATA lines -> CSV -> scikit-learn models -> comparison with a spill-weight baseline

## Layout
- patches/spill_dump.patch : instrumentation for llvm/lib/CodeGen/RegAllocGreedy.cpp (LLVM 19.1.7)
- scripts/ : parsers, training and evaluation scripts
- data/ : generated datasets (CSV)
- results/ : saved outputs and plots

## Run the analysis only (no LLVM needed)
The scripts use the path ~/spill-ml, so clone into your home folder:

    git clone <repo-url> ~/spill-ml
    cd ~/spill-ml
    python3 -m venv venv && source venv/bin/activate
    pip install -r requirements.txt
    python3 scripts/fair_eval.py
    python3 scripts/leave_project_out.py

## Regenerate the data (needs LLVM)
1. Build LLVM 19.1.7 (X86 only, assertions on). From the llvm-project root run: git apply ~/spill-ml/patches/spill_dump.patch, then rebuild llc.
2. Compile PolyBench/C 4.2.1, zlib and lua to LLVM IR (clang -O2 -S -emit-llvm) into data/ir/.
3. Run scripts/parse_features.py, then scripts/dedupe_v2.py, then scripts/parse_first.py.

## Notes
This is an offline prediction study. It does not measure generated-code quality. Evaluation is grouped by program so files from one program are never split between training and test.

## Which scripts matter
Current pipeline: parse_features.py -> dedupe_v2.py -> ablation.py, fair_eval.py, leave_project_out.py (and parse_first.py -> first_eval.py for the first-visit check).
The early scripts (parse_log.py, train.py, dedupe.py, eval_threshold.py, importance_check.py, train_v2.py's first run) were an initial prototype built on a log parser that wrote duplicate rows. They are kept for history, and their numbers should not be used.
Main dataset: data/dataset2_dedup.csv (one row per live interval, identical program copies removed).

## Results summary (first-visit features, 74 programs, 961 spills)
- A first-visit gate (no free register) reaches 746 of 961 spills (77.6%). The other 215 were assigned a register first and spilled after a later re-examination (consistent with eviction).
- Among gated intervals, 5-fold program-grouped CV: model PR-AUC mean 0.91 vs 0.71 for ranking by spill weight; model better in 5/5 folds (results/first_eval.txt).
- Leave-one-project-out: model beats the weight ranking on PolyBench (0.871 vs 0.784), zlib (0.887 vs 0.663) and lua (0.781 vs 0.703). Single runs, no error bars.
- End to end over all spillable intervals: gate+model 0.719 vs gate+weight 0.564 vs weight only 0.112 (random 0.052); model better in 5/5 folds (results/end_to_end_fair.txt). Most of the gain over the plain weight ranking comes from the gate, and the model adds about +0.155 on top.
- These are offline prediction results. They do not measure generated-code quality.
