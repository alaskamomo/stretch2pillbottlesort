#!/usr/bin/env python3
"""
run_manual_test.py - walks you through ANY test from test_plan.py and logs each trial.

Use it for tests you judge by eye or tape measure (grasp, T4, T14, ...), or for any
test whose automatic tester isn't written yet. No ROS needed.

    python3 run_manual_test.py --list            # show all tests
    python3 run_manual_test.py T8                # run every condition of T8
    python3 run_manual_test.py T2 --condition 1.0m_center --trials 5
    python3 run_manual_test.py T4 --start 2      # resume from the 2nd condition

Per trial you type one key for the outcome (shown in the prompt), then the metrics
(press Enter to skip any), then optional notes. Type q at the outcome prompt to stop.
Results go to the same CSVs as everything else -> python3 summarize_results.py
"""

import argparse
import sys

from test_logger import TestLogger
from test_plan import TESTS

KEYS = {"SUCCESS": "s", "MISS": "m", "NOT_FOUND": "n", "NO_PLAN": "p", "COLLISION": "c",
        "DROP": "d", "TIMEOUT": "t", "FALSE_POSITIVE": "f", "ERROR": "e"}


def parse_value(text):
    try:
        return float(text)
    except ValueError:
        return text


def run(test_id, conditions, trials):
    t = TESTS[test_id]
    log = TestLogger(test_id, t["name"])
    key_to_outcome = {KEYS[o]: o for o in t["outcomes"]}
    menu = "  ".join(f"[{KEYS[o]}]{o}" for o in t["outcomes"])

    print(f"\n=== {test_id}: {t['name']} ===")
    print(f"Setup:   {t['setup']}")
    print(f"Success: {t['success']}")
    if "auto" in t:
        print(f"(Automatic tester available: {t['auto']})")

    for cond in conditions:
        input(f"\n--- Condition '{cond}' ({trials} trials). Set it up, then press Enter ---")
        for i in range(1, trials + 1):
            while True:
                ans = input(f"[{cond} {i}/{trials}] outcome {menu}  [q]uit: ").strip().lower()
                if ans == "q":
                    print("Stopped. Logged trials are saved.")
                    return
                if ans in key_to_outcome:
                    break
                print("  not a valid key")
            metrics = {}
            for m in t["metrics"]:
                v = input(f"    {m} (Enter to skip): ").strip()
                if v:
                    metrics[m] = parse_value(v)
            notes = input("    notes (Enter to skip): ").strip()
            log.log(key_to_outcome[ans], cond, metrics, notes)
    print(f"\nDone. Summary: python3 summarize_results.py --test {test_id}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("test_id", nargs="?")
    p.add_argument("--list", action="store_true")
    p.add_argument("--condition", help="run just this condition")
    p.add_argument("--trials", type=int, help="override trials per condition")
    p.add_argument("--start", type=int, default=1, help="start at this condition number (1-based)")
    args = p.parse_args()

    if args.list or not args.test_id:
        for tid, t in TESTS.items():
            auto = "  (auto tester)" if "auto" in t else ""
            print(f"{tid:<5} {t['name']}{auto}")
        return
    if args.test_id not in TESTS:
        sys.exit(f"Unknown test {args.test_id}. Use --list.")

    t = TESTS[args.test_id]
    conditions = [args.condition] if args.condition else t["conditions"][args.start - 1:]
    try:
        run(args.test_id, conditions, args.trials or t["trials"])
    except (KeyboardInterrupt, EOFError):
        print("\nStopped. Logged trials are saved.")


if __name__ == "__main__":
    main()
