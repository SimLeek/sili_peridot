"""
scripts/compare_mqar_curriculum_results.py
────────────────────────────────────────────
Reads the stdout logs of one or more train_mqar_curriculum.py runs
(one per precision) and prints a comparison table of the peak vocab/K
each precision actually held for a genuine STREAK_THRESHOLD-in-a-row
streak (the "PEAK" line each run prints at the end) -- answers "show
the top vocab or association fp8 and fp4 achieves vs fp32" directly
from real run output, no re-parsing of raw trajectories needed.

Run: python3 scripts/compare_mqar_curriculum_results.py <log1> [log2] [log3] ...
"""
import re
import sys

PEAK_RE = re.compile(
    r"PEAK precision=(\S+) peak_vocab=(\d+) peak_k=(\d+) peak_phase=(\S+)")
FINAL_RE = re.compile(
    r"FINAL precision=(\S+) final_vocab=(\d+) final_k=(\d+) final_phase=(\S+) "
    r"graduated=(\S+) total_steps=(\d+) \((\d+)s\)")


def parse_log(path: str) -> dict:
    peak = None
    final = None
    with open(path) as f:
        for line in f:
            m = PEAK_RE.search(line)
            if m:
                peak = {"precision": m.group(1), "vocab": int(m.group(2)),
                        "k": int(m.group(3)), "phase": m.group(4)}
            m = FINAL_RE.search(line)
            if m:
                final = {"precision": m.group(1), "vocab": int(m.group(2)),
                         "k": int(m.group(3)), "phase": m.group(4),
                         "graduated": m.group(5) == "True",
                         "steps": int(m.group(6)), "elapsed_s": int(m.group(7))}
    return {"path": path, "peak": peak, "final": final}


def main():
    if len(sys.argv) < 2:
        print("usage: compare_mqar_curriculum_results.py <log1> [log2] ...")
        sys.exit(1)
    rows = [parse_log(p) for p in sys.argv[1:]]
    print(f"{'precision':<10} {'peak_vocab':>10} {'peak_k':>7} {'peak_phase':>10}  "
          f"{'final_vocab':>11} {'final_k':>7} {'graduated':>9} {'steps':>10} {'elapsed_s':>9}")
    for r in rows:
        p, f = r["peak"], r["final"]
        if p is None or f is None:
            print(f"{r['path']}: incomplete/still running (no PEAK or FINAL line yet)")
            continue
        print(f"{p['precision']:<10} {p['vocab']:>10} {p['k']:>7} {p['phase']:>10}  "
              f"{f['vocab']:>11} {f['k']:>7} {str(f['graduated']):>9} {f['steps']:>10} {f['elapsed_s']:>9}")


if __name__ == "__main__":
    main()
