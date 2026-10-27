#!/usr/bin/env python3
"""
T15b - approach motion: drive the base so the arm is in front of the bottle.

Uses the TAPE-MEASURED bottle position (not perception) to make the plan, prints it,
then drives it. Head tracking runs during the move so you can see whether the camera
stays on the bottle (needs the detector running for the in-view percentage).

Needs running: stretch_driver (broadcast_odom_tf:=True), bottle_tracking launch,
and your detector (optional, for in-view % and the detection-based alignment check).

  ros2 run bottle_tracking t15b_approach_motion_tester --ros-args \
      --params-file ~/ros2_ws/src/bottle_tracking/config/params.yaml
  ... -p trials:=5 -p only:=ahead_2.0m

Before every trial put the robot on its taped START mark (same heading) and the
bottle on its mark. The bottle position you type is in base_link at that start pose.
"""

import math
import threading
import time

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from bottle_tracking import geometry as geo
from bottle_tracking.approach_planner_node import declare_approach_config
from bottle_tracking.common import MemoryWatcher
from bottle_tracking.stretch_interface import StretchInterface, declare
from bottle_tracking.test_logger import TestLogger
from bottle_tracking.testers.tester_common import ask, ask_float, ask_xyz, ask_yes, base_to_fixed, run_tester

CONDITIONS = {
    "ahead_1.0m": "1.0 m straight ahead",
    "ahead_2.0m": "2.0 m straight ahead",
    "ahead_3.0m": "3.0 m straight ahead",
    "left_1.5m": "1.5 m to the left",
    "right_1.5m": "1.5 m to the right (arm side)",
    "behind_1.5m": "1.5 m behind",
}


class T15bTester(Node):
    def __init__(self):
        super().__init__("t15b_approach_motion_tester")
        cb = ReentrantCallbackGroup()
        self.robot = StretchInterface(self, cb)
        self.mem = MemoryWatcher(self, cb)
        self.cfg = declare_approach_config(self)
        self.trials = int(declare(self, "trials", 10))
        self.only = declare(self, "only", "")
        self.log = TestLogger("T15b", "Approach motion: drive the base to the bottle")

    def drive_plan(self, steps):
        result = {}
        th = threading.Thread(target=lambda: result.update(
            zip(("ok", "failed_step"), self.robot.execute_steps(steps))))
        th.start()
        fresh = []
        while th.is_alive():
            fresh.append(self.mem.detection_fresh(0.5))
            time.sleep(0.1)
        th.join()
        return result.get("ok", False), result.get("failed_step"), fresh

    def trial(self, cond, measured):
        start = self.robot.base_pose()
        if start is None:
            self.log.log("ERROR", cond, notes="no odom TF")
            return
        bottle = base_to_fixed(measured, start)
        plan = geo.plan_approach(bottle, start, self.cfg)
        text = geo.format_plan(plan, self.robot.fixed_frame)
        print("\n" + text)
        if not plan["ok"]:
            self.log.log("NO_PLAN" if not plan["steps"] else "MISS", cond, notes=text)
            return
        if ask("Enter to DRIVE this plan, s to skip: ").lower() == "s":
            return

        with self.log.trial(condition=cond) as t:
            self.mem.set_tracking(True)
            t0 = time.time()
            ok, failed_step, fresh = self.drive_plan(plan["steps"])
            exec_time = time.time() - t0
            pct = 100.0 * sum(fresh) / len(fresh) if fresh and any(fresh) else None

            end = self.robot.base_pose()
            heading_err = math.degrees(abs(geo.wrap_angle(end[2] - plan["final_pose"][2]))) if end else None
            # alignment from odometry + the measured bottle position
            lx, ly = geo.bottle_in_base(bottle[:2], end) if end else (None, None)
            odo_align = (lx - self.cfg.arm_x_offset_m) * 100 if end else None

            print("Measure now (tape): + means the bottle is AHEAD of the arm line.")
            tape_align = ask_float("  bottle center to arm line (cm)")
            clearance = ask_float("  closest base edge to table edge (cm)")
            touched = ask_yes("  Did the robot touch the table or bottle?")

            align = tape_align if tape_align is not None else odo_align
            reach = -ly if ly is not None else None
            t["metrics"] = {"exec_time_s": exec_time, "final_align_error_cm": align,
                            "odometry_align_error_cm": odo_align, "final_reach_m": reach,
                            "final_heading_error_deg": heading_err, "table_clearance_cm": clearance,
                            "percent_frames_in_view": pct, "failed_step": failed_step}
            if touched:
                t["outcome"] = "COLLISION"
            elif not ok:
                t["outcome"], t["notes"] = "ERROR", f"step {failed_step} failed"
            elif exec_time > 45:
                t["outcome"] = "TIMEOUT"
            elif (align is None or abs(align) > 3.0
                  or (reach is not None and not self.cfg.reach_min_m <= reach <= self.cfg.reach_max_m)
                  or (heading_err is not None and heading_err > 3.0)
                  or (clearance is not None and clearance < 10.0)
                  or (pct is not None and pct < 90.0)):
                t["outcome"] = "MISS"
            else:
                t["outcome"] = "SUCCESS"
            if pct is None:
                t["notes"] = (t.get("notes", "") + " no detections during the move (detector off?)").strip()

    def run(self):
        print(__doc__)
        conds = [self.only] if self.only else list(CONDITIONS)
        for cond in conds:
            print(f"\n=== {cond}: bottle {CONDITIONS[cond]} ===")
            if ask("Enter when the robot is on its start mark, s to skip: ").lower() == "s":
                continue
            measured = ask_xyz("Tape-measured bottle position in base_link")
            if measured is None:
                print("A measured position is needed for T15b, skipping this condition.")
                continue
            for i in range(self.trials):
                if i > 0:
                    ask(f"Trial {i + 1}/{self.trials}: robot back on its start mark, press Enter")
                self.trial(cond, measured)
        print("\nSummary: python3 summarize_results.py --test T15b")


def main():
    run_tester(T15bTester)


if __name__ == "__main__":
    main()
