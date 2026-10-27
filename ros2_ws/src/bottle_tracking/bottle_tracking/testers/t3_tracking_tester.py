#!/usr/bin/env python3
"""
T3 - keep an eye on the pill bottle while the robot moves around, then remember it.

  ros2 run bottle_tracking t3_tracking_tester --ros-args --params-file $P -p trials:=3

Runs Part A, then Part B, in one go.

Part A - watch while moving: the robot finds the bottle, then moves around (turns,
  drives, and a mix) with head tracking ON. The camera should stay on the bottle the
  whole time. Every move pattern ends back where it started.

Part B - remember after moving: first the robot makes sure it sees the bottle (it
  searches and turns the base if needed). Then each trial moves the base with the head
  FROZEN, so the camera loses the bottle. Afterward the head turns straight back to the
  remembered spot (no searching) and should see the bottle there.

Leave about 1 m of clear space around the robot.
Needs: driver, bottle_tracking launch with the YOLO detector.
"""

import math
import threading
import time

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from std_srvs.srv import Trigger

from bottle_tracking import geometry as geo
from bottle_tracking.common import DetectionCollector, MemoryWatcher, SnapshotSaver, call_trigger
from bottle_tracking.stretch_interface import StretchInterface, declare
from bottle_tracking.test_logger import TestLogger
from bottle_tracking.testers.tester_common import ask, run_tester

# Part A move patterns. Each one returns to the start pose.
PATTERNS_A = {
    "turn_45": [("rot", 45), ("rot", -90), ("rot", 45)],
    "turn_90": [("rot", 90), ("rot", -180), ("rot", 90)],
    "drive": [("drive", 0.3), ("drive", -0.3)],
    "mixed": [("rot", 45), ("drive", 0.3), ("drive", -0.3), ("rot", -45)],
}
# Part B moves (head frozen). Each one is undone afterward.
MOVES_B = {
    "rot+90": ("rot", 90),
    "rot-90": ("rot", -90),
    "rot180": ("rot", 180),
    "drive+0.3": ("drive", 0.3),
}


class T3Tester(Node):
    def __init__(self):
        super().__init__("t3_tracking_tester")
        cb = ReentrantCallbackGroup()
        self.robot = StretchInterface(self, cb)
        self.mem = MemoryWatcher(self, cb)
        self.det = DetectionCollector(self, self.robot, cb, memory=self.mem)
        self.snap = SnapshotSaver(self, cb)
        self.search_client = self.create_client(Trigger, "/head_search/search", callback_group=cb)
        self.trials = int(declare(self, "trials", 3))
        self.parts = declare(self, "part", "AB").upper()
        self.log = TestLogger("T3", "Keep an eye on the bottle while moving, then remember it")

    # ---------------------------------------------------------------- helpers
    def centered(self, max_deg):
        err = self.mem.tracker_status().get("angle_err_deg")
        return err is not None and err <= max_deg

    def ensure_seen(self):
        """Make sure the robot sees the bottle and is centered on it. Searches (and turns the base) if needed."""
        self.mem.set_tracking(True)
        if self.mem.wait_for(lambda: self.det.sees() and self.centered(5.0), 4.0):
            return True
        print("Not looking at the bottle; searching (the base may turn)...")
        ok, _ = call_trigger(self.search_client, timeout=180.0)
        return ok and self.mem.wait_for(lambda: self.det.sees() and self.centered(5.0), 8.0)

    def move(self, kind, amount):
        self.robot.switch_to_navigation_mode()
        if kind == "rot":
            return self.robot.rotate_by(math.radians(amount))[0]
        return self.robot.drive_by(amount)[0]

    def run_moves_sampling(self, moves):
        """Do the moves in a thread; every 0.1 s record (sees, angle error, head in range)."""
        result = {"ok": True}

        def go():
            for kind, amount in moves:
                if not self.move(kind, amount):
                    result["ok"] = False
                    return
        th = threading.Thread(target=go)
        th.start()
        samples = []
        while th.is_alive():
            tr = self.mem.tracker_status()
            samples.append((self.det.sees(1.0), tr.get("angle_err_deg"), tr.get("in_range")))
            time.sleep(0.1)
        th.join()
        return result["ok"], samples

    # ---------------------------------------------------------------- Part A
    def trial_a(self, name, moves):
        with self.log.trial(condition=f"A_{name}") as t:
            if not self.ensure_seen():
                t["outcome"], t["notes"] = "NOT_FOUND", "could not find/center the bottle before moving"
                return
            before = self.mem.position()
            print(f"  Part A {name}: moving with tracking ON...")
            ok, samples = self.run_moves_sampling(moves)
            if not ok:
                t["outcome"], t["notes"] = "ERROR", "base motion failed or timed out"
                return
            t_stop = time.time()
            usable = [s for s in samples if s[2] is not False]       # skip time past the pan limit
            pct = 100.0 * sum(1 for s in usable if s[0]) / max(1, len(usable))
            errs = [s[1] for s in usable if s[1] is not None]
            max_err = max(errs) if errs else 180.0
            mean_err = sum(errs) / len(errs) if errs else 180.0
            settled = self.mem.wait_for(lambda: self.centered(5.0), 2.0)
            after = self.mem.position()
            drift = geo.dist(before, after) * 100 if before and after else None
            t["metrics"] = {"percent_time_seen": pct, "max_tracking_error_deg": max_err,
                            "mean_tracking_error_deg": mean_err, "settle_time_s": time.time() - t_stop,
                            "position_drift_cm": drift, "out_of_pan_range_samples": len(samples) - len(usable)}
            good = pct >= 90 and max_err <= 10 and settled and (drift is None or drift <= 3.0)
            t["outcome"] = "SUCCESS" if good else "MISS"
            print(f"    seen {pct:.0f}% of the time, max error {max_err:.1f} deg, drift "
                  f"{drift if drift is None else round(drift, 1)} cm -> {t['outcome']}")

    # ---------------------------------------------------------------- Part B
    def trial_b(self, name, kind, amount):
        with self.log.trial(condition=f"B_{name}") as t:
            if not self.ensure_seen():
                t["outcome"], t["notes"] = "NOT_FOUND", "could not find/center the bottle before moving"
                return
            before = self.mem.position()
            self.mem.set_tracking(False)
            time.sleep(0.3)
            print(f"  Part B {name}: moving with the head FROZEN...")
            ok = self.move(kind, amount)
            if not ok:
                t["outcome"], t["notes"] = "ERROR", "base motion failed or timed out"
                self.undo(kind, amount)
                return
            lost = not self.det.sees(1.0)
            t_look = time.time()
            self.mem.set_tracking(True)            # look back at the REMEMBERED spot, no searching
            found = self.mem.wait_for(lambda: self.centered(5.0) and self.det.sees(1.0), 6.0)
            reacq = time.time() - t_look
            time.sleep(1.0)
            after = self.mem.position()
            drift = geo.dist(before, after) * 100 if before and after else None
            t["metrics"] = {"reacquire_time_s": reacq, "position_drift_cm": drift,
                            "lost_sight_during_move": lost}
            if not found:
                t["outcome"] = "NOT_FOUND"
            elif reacq <= 3.0 and drift is not None and drift <= 3.0:
                t["outcome"] = "SUCCESS"
            else:
                t["outcome"] = "MISS"
            print(f"    looked back in {reacq:.1f} s, drift "
                  f"{drift if drift is None else round(drift, 1)} cm -> {t['outcome']}")
        self.undo(kind, amount)

    def undo(self, kind, amount):
        self.mem.set_tracking(True)
        self.move(kind, -amount)
        time.sleep(0.5)

    def run(self):
        print(__doc__)
        ask("Put the pill bottle on the table near the robot. Press Enter to start.")
        if "A" in self.parts:
            print("\n===== PART A: watch the bottle while moving =====")
            for name, moves in PATTERNS_A.items():
                for _ in range(self.trials):
                    self.trial_a(name, moves)
        if "B" in self.parts:
            print("\n===== PART B: remember the bottle after moving =====")
            print("Making sure the robot sees the bottle first (it will search and turn if needed)...")
            if not self.ensure_seen():
                print("Could not find the bottle. Check the detector and try again.")
                return
            for name, (kind, amount) in MOVES_B.items():
                for _ in range(self.trials):
                    self.trial_b(name, kind, amount)
        print("\nSummary: python3 testing/summarize_results.py --test T3 --session latest")


def main():
    run_tester(T3Tester)


if __name__ == "__main__":
    main()
