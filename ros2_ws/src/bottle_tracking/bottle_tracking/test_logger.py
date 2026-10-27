#!/usr/bin/env python3
"""
test_logger.py - records SUCCESS / MISS results for the Stretch 2 component tests.

Every test (T1a, T2, T3, ...) gets its own CSV in ~/stretch_test_results/
(change with the STRETCH_TEST_RESULTS env var). One row = one trial.
Pure standard library, so it runs on the robot with nothing to install.

---------------------------------------------------------------------------
Use 1: inside a tester node / script
---------------------------------------------------------------------------
    from test_logger import TestLogger

    log = TestLogger("T6", "Depth / distance estimation")

    for true_dist in [0.4, 0.6, 0.8, 1.0, 1.5]:
        input(f"Place bottle at {true_dist} m, press Enter...")
        for i in range(10):
            with log.trial(condition=f"{true_dist}m") as t:
                est = get_bottle_distance()            # <- your perception call
                err_cm = abs(est - true_dist) * 100
                t["metrics"] = {"true_m": true_dist, "est_m": est, "error_cm": err_cm}
                t["outcome"] = "SUCCESS" if err_cm <= 2.0 else "MISS"

  - If your code throws, the trial is logged as ERROR and the exception is re-raised.
  - If you forget to set t["outcome"], it is logged as ERROR so it's not silently lost.
  - Ctrl+C mid-trial does NOT log anything for that trial.

---------------------------------------------------------------------------
Use 2: logging by hand from the terminal (for things you judge by eye,
e.g. "did the arm hit the table?", or tape-measured ground truth)
---------------------------------------------------------------------------
    python3 test_logger.py T1b --outcome COLLISION --condition pos2_upright \
        --metric exec_time_s=14.2 --notes "elbow clipped table edge"
"""

import argparse
import csv
import json
import os
import subprocess
import time
from contextlib import contextmanager
from datetime import datetime

# Folder where all the result CSVs live
RESULTS_DIR = os.environ.get(
    "STRETCH_TEST_RESULTS", os.path.expanduser("~/stretch_test_results")
)

# Allowed outcome codes. Only SUCCESS counts as a pass.
OUTCOMES = [
    "SUCCESS",
    "MISS",            # ran fine but didn't meet the criteria (grabbed air, error too big, looked wrong way)
    "NOT_FOUND",       # perception never found the bottle
    "NO_PLAN",         # planner couldn't make a plan when it should have
    "COLLISION",       # robot hit the table / knocked the bottle
    "DROP",            # bottle grabbed, then dropped
    "TIMEOUT",         # went past the time limit
    "FALSE_POSITIVE",  # "found" a bottle that isn't there
    "ERROR",           # code crashed / exception / outcome never set
]

# CSV column order
FIELDS = [
    "timestamp", "session_id", "test_id", "test_name", "trial", "condition",
    "outcome", "success", "duration_s", "metrics", "notes", "git_commit",
]


def _git_commit():
    """Short hash of the current commit, so results can be tied to a code version."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return "unknown"


def _to_jsonable(value):
    """numpy floats/ints etc. aren't JSON serializable, so convert them."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


class TestLogger:
    def __init__(self, test_id, test_name="", results_dir=RESULTS_DIR):
        self.test_id = test_id
        self.test_name = test_name
        os.makedirs(results_dir, exist_ok=True)
        self.path = os.path.join(results_dir, f"{test_id}.csv")
        # session_id groups all trials from one run of the tester
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.git_commit = _git_commit()
        # keep numbering trials from where the CSV left off
        self.trial_count = self._existing_trials()

    def _existing_trials(self):
        if not os.path.exists(self.path):
            return 0
        with open(self.path, newline="") as f:
            return sum(1 for _ in csv.DictReader(f))

    def log(self, outcome, condition="", metrics=None, notes="", duration_s=None):
        """Write one trial row to the CSV."""
        outcome = outcome.upper()
        if outcome not in OUTCOMES:
            raise ValueError(f"Unknown outcome '{outcome}'. Use one of: {OUTCOMES}")

        self.trial_count += 1
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "session_id": self.session_id,
            "test_id": self.test_id,
            "test_name": self.test_name,
            "trial": self.trial_count,
            "condition": condition,
            "outcome": outcome,
            "success": 1 if outcome == "SUCCESS" else 0,
            "duration_s": "" if duration_s is None else round(duration_s, 3),
            "metrics": json.dumps(metrics or {}, default=_to_jsonable),
            "notes": notes,
            "git_commit": self.git_commit,
        }

        new_file = not os.path.exists(self.path)
        with open(self.path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            if new_file:
                writer.writeheader()
            writer.writerow(row)

        # quick console feedback while testing
        print(f"[{self.test_id}] trial {self.trial_count} [{condition}] -> {outcome} {metrics or ''}")
        return row

    @contextmanager
    def trial(self, condition="", notes=""):
        """
        Times one trial and logs it when the block ends.
        Set t["outcome"] (required) and optionally t["metrics"] / t["notes"] / t["condition"] inside the block.
        """
        t = {"outcome": None, "metrics": {}, "notes": notes, "condition": condition}
        start = time.time()
        aborted = False
        try:
            yield t
        except KeyboardInterrupt:
            aborted = True          # don't log a half-finished trial
            raise
        except Exception as e:
            t["outcome"] = "ERROR"
            t["notes"] = f"{t['notes']} | {type(e).__name__}: {e}".strip(" |")
            raise
        finally:
            if not aborted:
                if t["outcome"] is None:
                    t["outcome"] = "ERROR"
                    t["notes"] = f"{t['notes']} | outcome never set".strip(" |")
                self.log(t["outcome"], t["condition"], t["metrics"], t["notes"],
                         duration_s=time.time() - start)


def _parse_metric(text):
    """'error_cm=2.4' -> ('error_cm', 2.4). Leaves non-numbers as strings."""
    key, _, value = text.partition("=")
    try:
        return key, float(value)
    except ValueError:
        return key, value


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Log one component-test trial by hand.")
    p.add_argument("test_id", help="e.g. T1b, T2, T3")
    p.add_argument("--outcome", required=True, choices=OUTCOMES + [o.lower() for o in OUTCOMES])
    p.add_argument("--condition", default="", help="e.g. pos2_upright, 1.0m, rot90")
    p.add_argument("--metric", action="append", default=[], help="key=value, can repeat")
    p.add_argument("--notes", default="")
    p.add_argument("--name", default="", help="test name (optional)")
    args = p.parse_args()

    metrics = dict(_parse_metric(m) for m in args.metric)
    TestLogger(args.test_id, args.name).log(args.outcome, args.condition, metrics, args.notes)
