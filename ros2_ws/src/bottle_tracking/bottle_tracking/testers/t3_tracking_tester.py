#!/usr/bin/env python3
"""
T3 - remember and track the bottle while the base moves.

Needs running: stretch_driver (broadcast_odom_tf:=True), your detector (or fake_detector),
and `ros2 launch bottle_tracking bottle_tracking.launch.py`.

  ros2 run bottle_tracking t3_tracking_tester                 # Part A then Part B
  ros2 run bottle_tracking t3_tracking_tester --ros-args -p part:=B -p trials:=5

Part A: tracking ON while the base moves -> camera should stay on the bottle.
Part B: tracking OFF (head frozen) while the base turns -> afterwards the head should
        point straight back at the SAVED position (memory test, no re-scan).
After each trial the robot undoes the motion so every trial starts from the same spot.
"""

import math
import threading
import time

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from bottle_tracking import geometry as geo
from bottle_tracking.common import MemoryWatcher
from bottle_tracking.stretch_interface import StretchInterface, declare
from bottle_tracking.test_logger import TestLogger
from bottle_tracking.testers.tester_common import ask, run_tester

MOTIONS_A = [("rot+30", "rot", 30), ("rot-30", "rot", -30), ("rot+90", "rot", 90),
             ("rot-90", "rot", -90), ("rot180", "rot", 180),
             ("fwd0.3", "drive", 0.3), ("back0.3", "drive", -0.3)]
MOTIONS_B = [m for m in MOTIONS_A if m[1] == "rot"]


class T3Tester(Node):
    def __init__(self):
        super().__init__("t3_tracking_tester")
        cb = ReentrantCallbackGroup()
        self.robot = StretchInterface(self, cb)
        self.mem = MemoryWatcher(self, cb)
        self.part = declare(self, "part", "AB")
        self.trials = int(declare(self, "trials", 10))
        self.log = TestLogger("T3", "Remember and track the bottle while the base moves")

    # ---------------------------------------------------------------- helpers
    def lock_on(self, timeout=15.0):
        """Tracking on, wait until the bottle is seen and centered (<5 deg)."""
        self.mem.set_tracking(True)
        return self.mem.wait_for(lambda: self.mem.get().get("status") == "KNOWN"
                                 and self.mem.detection_fresh(0.5) and self.centered(5.0), timeout)

    def centered(self, max_deg):
        err = self.mem.tracker_status().get("angle_err_deg")
        return err is not None and err <= max_deg

    def move(self, kind, amount):
        self.robot.switch_to_navigation_mode()
        if kind == "rot":
            return self.robot.rotate_by(math.radians(amount))[0]
        return self.robot.drive_by(amount)[0]

    def move_sampling(self, kind, amount):
        """Do the motion in a thread; sample detection + tracking error every 0.1 s."""
        result = {}
        th = threading.Thread(target=lambda: result.update(ok=self.move(kind, amount)))
        th.start()
        samples = []
        while th.is_alive():
            t = self.mem.tracker_status()
            samples.append((self.mem.detection_fresh(0.5), t.get("angle_err_deg"), t.get("in_range")))
            time.sleep(0.1)
        th.join()
        return result.get("ok", False), samples

    # ---------------------------------------------------------------- trials
    def trial(self, part, name, kind, amount):
        cond = f"{part}_{name}"
        with self.log.trial(condition=cond) as t:
            if not self.lock_on():
                t["outcome"], t["notes"] = "NOT_FOUND", "could not lock on to the bottle before moving"
                return
            before = self.mem.position()
            if part == "B":
                self.mem.set_tracking(False)
                time.sleep(0.3)

            ok, samples = self.move_sampling(kind, amount)
            if not ok:
                t["outcome"], t["notes"] = "ERROR", "base motion failed or timed out"
                self.undo(kind, amount)
                return
            t_stop = time.time()

            if part == "A":
                usable = [s for s in samples if s[2] is not False]   # skip time past the pan limit
                pct = 100.0 * sum(1 for s in usable if s[0]) / max(1, len(usable))
                errs = [s[1] for s in usable if s[1] is not None]
                max_err = max(errs) if errs else 180.0
                settled = self.mem.wait_for(lambda: self.centered(5.0), 2.0)
                settle_time = time.time() - t_stop
                after = self.mem.position()
                drift = geo.dist(before, after) * 100 if before and after else None
                t["metrics"] = {"percent_frames_in_view": pct, "max_tracking_error_deg": max_err,
                                "reacquire_time_s": settle_time, "position_drift_cm": drift,
                                "out_of_pan_range_samples": len(samples) - len(usable)}
                good = pct >= 90 and max_err <= 10 and settled
                t["outcome"] = "SUCCESS" if good else "MISS"
            else:
                t_look = time.time()
                self.mem.set_tracking(True)
                found = self.mem.wait_for(
                    lambda: self.centered(5.0) and self.mem.detection_fresh(0.5), 6.0)
                reacq = time.time() - t_look
                time.sleep(1.5)   # let a few detections refine the saved position
                after = self.mem.position()
                drift = geo.dist(before, after) * 100 if before and after else None
                t["metrics"] = {"reacquire_time_s": reacq, "position_drift_cm": drift,
                                "rotation_deg": amount}
                if not found:
                    t["outcome"] = "NOT_FOUND"
                elif reacq <= 3.0 and drift is not None and drift <= 3.0:
                    t["outcome"] = "SUCCESS"
                else:
                    t["outcome"] = "MISS"
        self.undo(kind, amount)

    def undo(self, kind, amount):
        self.mem.set_tracking(True)
        self.move(kind, -amount)
        time.sleep(0.5)

    def run(self):
        print(__doc__)
        ask("Place the bottle on the table in front of the robot so the head camera sees it. Press Enter.")
        for part in self.part.upper():
            motions = MOTIONS_A if part == "A" else MOTIONS_B
            for name, kind, amount in motions:
                if ask(f"\nPart {part} {name}: {self.trials} trials. Enter to start, s to skip: ").lower() == "s":
                    continue
                for _ in range(self.trials):
                    self.trial(part, name, kind, amount)
        print("\nSummary: python3 summarize_results.py --test T3")


def main():
    run_tester(T3Tester)


if __name__ == "__main__":
    main()
