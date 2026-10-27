#!/usr/bin/env python3
"""
T12 - bottle moved while the robot is paused.

Needs running: stretch_driver (broadcast_odom_tf:=True), your detector,
and `ros2 launch bottle_tracking bottle_tracking.launch.py`.

  ros2 run bottle_tracking t12_moved_bottle_tester
  ros2 run bottle_tracking t12_moved_bottle_tester --ros-args -p trials:=5 -p only:=not_moved

Each trial:
  1. The robot finds the bottle (search) and saves its position.
  2. PAUSE: tracking turns off and the head looks away, so the robot can't watch you.
     Move the bottle for the condition (or leave it for not_moved) and type the new
     tape-measured position.
  3. RESUME: the head looks at the SAVED spot. The robot should notice the bottle is
     gone/moved, drop the old position and search again (head first). If it wasn't
     moved, it should keep the saved position and not search.
These nodes never move the arm, so "no reach to the old position" is checked in T13
with your full state machine.
"""

import json
import time

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from std_srvs.srv import Trigger

from bottle_tracking import geometry as geo
from bottle_tracking.common import MemoryWatcher, call_trigger
from bottle_tracking.stretch_interface import StretchInterface, declare
from bottle_tracking.test_logger import TestLogger
from bottle_tracking.testers.tester_common import ask, ask_xyz, base_to_fixed, run_tester

CONDITIONS = {
    "moved_small_5_15cm": "slide the bottle 5-15 cm along the table",
    "moved_across_table": "move it across the table (still where the head could see it)",
    "moved_out_of_head_view": "move it behind or beside the robot, out of the head's view",
    "not_moved": "do NOT move it (control)",
}


class T12Tester(Node):
    def __init__(self):
        super().__init__("t12_moved_bottle_tester")
        cb = ReentrantCallbackGroup()
        self.robot = StretchInterface(self, cb)
        self.mem = MemoryWatcher(self, cb)
        self.search_client = self.create_client(Trigger, "/head_search/search", callback_group=cb)
        self.trials = int(declare(self, "trials", 10))
        self.only = declare(self, "only", "")
        self.pause_pan = declare(self, "pause_head_pan", 1.5)     # look away while paused
        self.pause_tilt = declare(self, "pause_head_tilt", 0.0)
        self.notice_timeout = declare(self, "notice_timeout_s", 8.0)
        self.log = TestLogger("T12", "Bottle moved while the robot is paused")

    def search(self):
        ok, msg = call_trigger(self.search_client, timeout=180.0)
        try:
            return ok, json.loads(msg)
        except ValueError:
            return False, {"error": msg}

    def trial(self, cond):
        ask("\nPut the bottle on its START mark where the head can see it, press Enter")
        self.mem.clear()
        time.sleep(0.3)
        ok, _ = self.search()
        if not ok:
            self.log.log("NOT_FOUND", cond, notes="could not find the bottle before the pause")
            return
        time.sleep(1.0)
        old = self.mem.position()
        start_pose = self.robot.base_pose()

        # ---- pause: stop watching ----
        self.mem.set_tracking(False)
        self.robot.move_head_and_wait(self.pause_pan, self.pause_tilt)
        print(f"PAUSED. Now: {CONDITIONS[cond]}.")
        measured = None
        if cond != "not_moved":
            measured = ask_xyz("New tape-measured bottle position in base_link")
        ask("Press Enter to RESUME")
        new_truth = base_to_fixed(measured, start_pose) if (measured and start_pose) else None
        move_cm = geo.dist(old, new_truth) * 100 if (old and new_truth) else None

        with self.log.trial(condition=cond) as t:
            t_resume_wall, t_resume = time.time(), self.mem.now()
            self.mem.set_tracking(True)   # look back at the saved spot

            if cond == "not_moved":
                saw_missing = self.mem.wait_for(
                    lambda: self.mem.get().get("status") == "MISSING", self.notice_timeout)
                pos = self.mem.position()
                drift = geo.dist(old, pos) * 100 if (old and pos) else None
                t["metrics"] = {"moved_detected": saw_missing, "position_drift_cm": drift}
                good = (not saw_missing and self.mem.seen_since(t_resume)
                        and drift is not None and drift <= 3.0)
                t["outcome"] = "SUCCESS" if good else "MISS"
                if saw_missing:
                    t["notes"] = "marked MISSING although the bottle did not move"
                return

            # moved: notice = MISSING, a 'moved' event, or the saved position updated to the new spot
            def noticed():
                if self.mem.get().get("status") == "MISSING":
                    return True
                if any(e.startswith("moved") for e in self.mem.events_since(t_resume_wall)):
                    return True
                pos = self.mem.position()
                return (new_truth is not None and pos is not None
                        and self.mem.seen_since(t_resume) and geo.dist(pos, new_truth) <= 0.03)
            did_notice = self.mem.wait_for(noticed, self.notice_timeout)
            notice_time = time.time() - t_resume_wall

            search_info = {}
            if self.mem.get().get("status") == "MISSING":
                _, search_info = self.search()
            pos = self.mem.position()
            err_cm = geo.dist(pos, new_truth) * 100 if (pos and new_truth) else None
            t["metrics"] = {"move_distance_cm": move_cm, "moved_detected": did_notice,
                            "time_to_notice_s": notice_time,
                            "search_time_s": search_info.get("search_time_s"),
                            "base_rotation_deg": search_info.get("base_rotation_deg"),
                            "found_with": search_info.get("found_with"), "error_cm": err_cm}
            if not did_notice:
                t["outcome"], t["notes"] = "MISS", "never noticed the bottle moved"
            elif search_info and not search_info.get("found"):
                t["outcome"] = "NOT_FOUND"
            elif notice_time > 5.0 or (search_info.get("search_time_s") or 0) > 60:
                t["outcome"] = "TIMEOUT"
            elif err_cm is not None and err_cm > 3.0:
                t["outcome"] = "FALSE_POSITIVE" if err_cm > 30 else "MISS"
                t["notes"] = f"final position error {err_cm:.1f} cm"
            else:
                t["outcome"] = "SUCCESS"

    def run(self):
        print(__doc__)
        conds = [self.only] if self.only else list(CONDITIONS)
        for cond in conds:
            if ask(f"\n=== {cond}: {self.trials} trials. Enter to start, s to skip: ").lower() == "s":
                continue
            for _ in range(self.trials):
                self.trial(cond)
        print("\nSummary: python3 summarize_results.py --test T12")


def main():
    run_tester(T12Tester)


if __name__ == "__main__":
    main()
