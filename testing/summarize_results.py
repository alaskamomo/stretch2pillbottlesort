#!/usr/bin/env python3
"""
summarize_results.py - success rate per test, per condition, plus metric averages.

    python3 summarize_results.py                  # all tests
    python3 summarize_results.py --test T2        # one test
    python3 summarize_results.py --commit a1b2c3d # only trials run on that code version
    python3 summarize_results.py --session latest # only the most recent run of each test
    python3 summarize_results.py --markdown       # table you can paste into a GitHub issue comment
"""

import argparse
import csv
import glob
import json
import math
import os
from collections import Counter, defaultdict

from test_logger import RESULTS_DIR

# Pass-rate targets from the GitHub tickets. Tweak after talking to Prof. Martinson.
TARGETS = {
    "T1a": 0.95, "T1b": 0.80, "T2": 0.90, "T3": 0.90, "T4": 1.00, "T5": 0.90,
    "T6": 0.90, "T7": 0.90, "T8": 0.90, "T9": 0.90, "T10": 0.90, "T11": 1.00,
    "T12": 0.90, "T13": 0.70, "T14": 0.95, "T15a": 0.95, "T15b": 0.90,
}


def wilson_lower(successes, n, z=1.96):
    """
    95% lower bound on the true success rate. 9/10 looks like 90%, but with only
    10 trials the real rate could be much lower - this shows how much lower.
    """
    if n == 0:
        return 0.0
    p = successes / n
    denom = 1 + z * z / n
    center = p + z * z / (2 * n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (center - spread) / denom


def load_rows(path, commit=None, session=None):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if commit:
        rows = [r for r in rows if r["git_commit"].startswith(commit)]
    if session == "latest" and rows:
        latest = max(r["session_id"] for r in rows)
        rows = [r for r in rows if r["session_id"] == latest]
    elif session:
        rows = [r for r in rows if r["session_id"] == session]
    return rows


def metric_stats(rows):
    """Mean and max of every numeric metric across the trials."""
    values = defaultdict(list)
    for r in rows:
        for k, v in json.loads(r["metrics"] or "{}").items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                values[k].append(v)
    return {k: (sum(v) / len(v), max(v)) for k, v in values.items()}


def summarize(test_id, rows, markdown=False):
    n = len(rows)
    s = sum(int(r["success"]) for r in rows)
    rate = s / n if n else 0.0
    target = TARGETS.get(test_id)
    verdict = "" if target is None else ("PASS" if rate >= target else "FAIL")
    name = rows[0]["test_name"] if rows else ""
    outcomes = Counter(r["outcome"] for r in rows)

    # group by condition (pose, distance, rotation angle...)
    by_cond = defaultdict(list)
    for r in rows:
        by_cond[r["condition"] or "(none)"].append(r)

    if markdown:
        tgt = f"{target:.0%}" if target is not None else "-"
        print(f"### {test_id} {name}\n")
        print(f"**{s}/{n} = {rate:.1%}** (target {tgt}) **{verdict}** "
              f"- 95% lower bound {wilson_lower(s, n):.1%}\n")
        print("| Condition | Success | Rate | Failures |")
        print("|---|---|---|---|")
        for cond, cr in sorted(by_cond.items()):
            cs = sum(int(r["success"]) for r in cr)
            fails = Counter(r["outcome"] for r in cr if r["outcome"] != "SUCCESS")
            fail_txt = ", ".join(f"{k} {v}" for k, v in fails.items()) or "-"
            print(f"| {cond} | {cs}/{len(cr)} | {cs / len(cr):.0%} | {fail_txt} |")
        stats = metric_stats(rows)
        if stats:
            print("\n| Metric | Mean | Max |\n|---|---|---|")
            for k, (mean, mx) in sorted(stats.items()):
                print(f"| {k} | {mean:.3f} | {mx:.3f} |")
        print()
        return

    tgt = f"target {target:.0%}" if target is not None else "no target set"
    print(f"\n{test_id}  {name}")
    print(f"  {s}/{n} = {rate:.1%}   ({tgt})  {verdict}   95% lower bound: {wilson_lower(s, n):.1%}")
    print("  outcomes: " + ", ".join(f"{k} {v}" for k, v in outcomes.most_common()))
    print("  by condition:")
    for cond, cr in sorted(by_cond.items()):
        cs = sum(int(r["success"]) for r in cr)
        print(f"    {cond:<22} {cs:>3}/{len(cr):<3} {cs / len(cr):6.0%}")
    stats = metric_stats(rows)
    if stats:
        print("  metrics (mean / max):")
        for k, (mean, mx) in sorted(stats.items()):
            print(f"    {k:<22} {mean:8.3f} / {mx:8.3f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--test", help="only this test id, e.g. T2")
    p.add_argument("--commit", help="only trials from this git commit (prefix ok)")
    p.add_argument("--session", help="'latest' or a session id like 20261026_141500")
    p.add_argument("--markdown", action="store_true", help="print GitHub-ready markdown")
    p.add_argument("--dir", default=RESULTS_DIR)
    args = p.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "*.csv")))
    if args.test:
        files = [f for f in files if os.path.basename(f) == f"{args.test}.csv"]
    if not files:
        print(f"No results found in {args.dir}")

    for path in files:
        test_id = os.path.splitext(os.path.basename(path))[0]
        rows = load_rows(path, args.commit, args.session)
        if rows:
            summarize(test_id, rows, args.markdown)
