#!/usr/bin/env python3
"""
T5 - search with the head first, base only if needed.

Needs running: stretch_driver (broadcast_odom_tf:=True), your detector (or fake_detector),
and `ros2 launch bottle_tracking bottle_tracking.launch.py`.

  ros2 run bottle_tracking t5_search_tester
  ros2 run bottle_tracking t5_search_tester --ros-args -p trials:=5 -p only:=behind_robot

Each trial: memory is cleared, the head goes to a start pose, then /head_search/search
is called. You can type the tape-measured bottle position (in base_link at the robot's
start pose) once per condition to get the position error. After each trial the robot
turns back to where it started.
"""

import json
import math
import time

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from std_srvs.srv import Trigger

from bottle_tracking import geometry as geo
from bottle_tracking.common import MemoryWatcher, call_trigger
from bottle_tracking.stretch_interface import StretchInterface, declare
from bottle_tracking.test_logger import TestLogger
from bottle_tracking.testers.tester_common import ask, ask_xyz, base_to_fixed, run_tester

# condition: (description, bottle should be found with the head only?)
CONDITIONS = {
    "in_head_range_left": ("on the table to the robot's LEFT, within the head's view", True),
    "in_head_range_right": ("on the table to the robot's RIGHT (arm side), within the head's view", True),
    "edge_of_pan_range": ("near the edge of the head's pan range (far left)", True),
    "behind_robot": ("BEHIND the robot, outside the head's range", False),
    "far_2_to_3m": ("2 to 3 m away, in view", True),
}


class T5Tester(Node):
    def __init__(self):
        super().__init__("t5_search_tester")
        cb = ReentrantCallbackGroup()
        self.robot = StretchInterface(self, cb)
        self.mem = MemoryWatcher(self, cb)
        self.search_client = self.create_client(Trigger, "/head_search/search", callback_group=cb)
        self.trials = int(declare(self, "trials", 10))
        self.only = declare(self, "only", "")
        self.start_pan = declare(self, "start_head_pan", 0.0)
        self.start_tilt = declare(self, "start_head_tilt", -0.45)
        self.log = TestLogger("T5", "Search with the head first, base only if needed")

    def trial(self, cond, head_only, measured_base):
        start_pose = self.robot.base_pose()
        truth = base_to_fixed(measured_base, start_pose) if (measured_base and start_pose) else None
        self.mem.set_tracking(False)
        self.mem.clear()
        self.robot.move_head_and_wait(self.start_pan, self.start_tilt)
        time.sleep(0.5)

        with self.log.trial(condition=cond) as t:
            ok, msg = call_trigger(self.search_client, timeout=180.0)
            try:
                r = json.loads(msg)
            except ValueError:
                r = {"error": msg}
            pos = self.mem.position()
            err_cm = geo.dist(pos, truth) * 100 if (pos and truth) else None
            t["metrics"] = {"search_time_s": r.get("search_time_s"),
                            "base_rotation_deg_during_search": r.get("base_rotation_deg"),
                            "num_base_rotations": r.get("num_base_rotations"),
                            "found_with": r.get("found_with"), "error_cm": err_cm}
            far = cond.startswith("far")
            if not ok:
                t["outcome"] = "TIMEOUT" if r.get("found_with") == "timeout" else "NOT_FOUND"
                t["notes"] = r.get("error", r.get("found_with", ""))
            elif r.get("search_time_s", 999) > 60:
                t["outcome"] = "TIMEOUT"
            elif head_only and r.get("num_base_rotations", 0) > 0:
                t["outcome"], t["notes"] = "MISS", "spun the base although the head could see it"
            elif not head_only and r.get("base_rotation_deg", 0) > 360:
                t["outcome"], t["notes"] = "MISS", "more than one full base rotation"
            elif err_cm is not None and err_cm > (5.0 if far else 3.0):
                t["outcome"] = "FALSE_POSITIVE" if err_cm > 30 else "MISS"
                t["notes"] = f"position error {err_cm:.1f} cm"
            else:
                t["outcome"] = "SUCCESS"

        # turn back to the start heading so the next trial is the same
        if start_pose:
            now = self.robot.base_pose()
            if now and abs(geo.wrap_angle(start_pose[2] - now[2])) > math.radians(2):
                self.robot.switch_to_navigation_mode()
                self.robot.rotate_by(geo.wrap_angle(start_pose[2] - now[2]))

    def run(self):
        print(__doc__)
        conds = [self.only] if self.only else list(CONDITIONS)
        for cond in conds:
            desc, head_only = CONDITIONS[cond]
            print(f"\n=== {cond}: place the bottle {desc} ===")
            if ask("Enter when placed, s to skip: ").lower() == "s":
                continue
            measured = ask_xyz("Tape-measured bottle position in base_link")
            for i in range(self.trials):
                if i > 0:
                    ask(f"Trial {i + 1}/{self.trials}: put the bottle back on its mark, press Enter")
                self.trial(cond, head_only, measured)
        print("\nSummary: python3 summarize_results.py --test T5")


def main():
    run_tester(T5Tester)


if __name__ == "__main__":
    main()
