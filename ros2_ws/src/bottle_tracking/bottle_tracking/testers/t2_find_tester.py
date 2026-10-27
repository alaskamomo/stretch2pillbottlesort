#!/usr/bin/env python3
"""
T2 - find the pill bottle with the head camera and report where it is.

  ros2 run bottle_tracking t2_find_tester --ros-args --params-file $P -p trials:=5
  (change the pause between trials with -p pause_s:=30)

Each trial:
  1. Pauses so you can move the bottle somewhere new (Enter to skip the wait).
  2. Looks around with the head camera only (the base never moves).
  3. Tells you: whether it sees the bottle, where it is in the fixed (odom) frame,
     how far away it is, and whether the robot needs to move to grab it (and how).
  4. Asks you to confirm it found the pill bottle and not something else.

Results are grouped by distance automatically (under 0.6 m, 0.6-1.0 m, ...).
Needs: driver, bottle_tracking launch with the YOLO detector.
"""

import json
import math
import time

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from std_srvs.srv import Trigger

from bottle_tracking import geometry as geo
from bottle_tracking.approach_planner_node import declare_approach_config
from bottle_tracking.common import DetectionCollector, MemoryWatcher, SnapshotSaver, call_trigger
from bottle_tracking.stretch_interface import StretchInterface, declare
from bottle_tracking.test_logger import TestLogger
from bottle_tracking.testers.tester_common import (ask, ask_yes, declare_arm_config, distance_bucket,
                                                   pause, run_tester)


def describe_moves(steps):
    if not steps:
        return "No. The arm can reach it from here."
    parts = []
    for s in steps:
        if s["type"] == "ROTATE":
            parts.append(f"turn {abs(s['deg']):.0f} deg {'left' if s['deg'] > 0 else 'right'}")
        else:
            parts.append(f"drive {abs(s['m']):.2f} m {'forward' if s['m'] > 0 else 'backward'}")
    return "Yes: " + ", then ".join(parts) + "."


class FindTester(Node):
    def __init__(self):
        super().__init__("t2_find_tester")
        cb = ReentrantCallbackGroup()
        self.robot = StretchInterface(self, cb)
        self.mem = MemoryWatcher(self, cb)
        self.det = DetectionCollector(self, self.robot, cb, memory=self.mem)
        self.snap = SnapshotSaver(self, cb)
        self.search_client = self.create_client(Trigger, "/head_search/search_head_only", callback_group=cb)
        self.arm = declare_arm_config(self)
        self.approach_cfg = geo.approach_config_for_arm(self.arm, declare_approach_config(self))
        self.trials = int(declare(self, "trials", 5))
        self.pause_s = declare(self, "pause_s", 20.0)
        self.max_spread_cm = declare(self, "max_spread_cm", 2.0)
        self.log = TestLogger("T2", "Find the pill bottle with the head camera")

    def trial(self, i):
        pause(self.pause_s, f"Trial {i}/{self.trials}: move the pill bottle to a new spot.")
        with self.log.trial() as t:
            self.mem.clear()
            time.sleep(0.3)
            ok, msg = call_trigger(self.search_client, timeout=120.0)
            try:
                info = json.loads(msg)
            except ValueError:
                info = {"error": msg}
            m = self.det.measure(3.0) if ok else None

            name = f"T2_{self.log.session_id}_trial{i}"
            print("\n" + "=" * 60)
            if m is None:
                age = self.det.last_seen_age()
                print("Does NOT see the pill bottle.")
                print(f"  detections received by this tester: {self.det.received}"
                      + (f", last one {age:.1f} s ago" if age is not None else ""))
                pic = self.snap.save(name, [f"T2 trial {i}: does NOT see the pill bottle"])
                print(f"  picture: {pic}" if pic else "  (no camera picture received)")
                print("=" * 60)
                t["condition"] = "not_found"
                t["outcome"] = "NOT_FOUND"
                t["metrics"] = {"search_time_s": info.get("search_time_s")}
                t["notes"] = ("in view of the head? " + ("yes" if ask_yes(
                    "Was the bottle somewhere the head could see?", True) else "no")
                    + (f" | picture: {pic}" if pic else ""))
                return

            fx, fy, fz = m["fixed"]
            bx, by, _ = m["base"]
            bearing = math.degrees(math.atan2(by, bx))
            side = "left" if bearing > 0 else "right"
            plan = geo.plan_approach(m["fixed"], self.robot.base_pose(), self.approach_cfg)
            moves = describe_moves(plan["steps"]) if plan["ok"] else f"Can't plan a way there: {plan['reason']}"
            print("SEES the pill bottle")
            print(f"  Position (fixed {self.robot.fixed_frame} frame): x={fx:.3f}  y={fy:.3f}  z={fz:.3f} m")
            print(f"  Distance from the robot base: {m['base_distance_m']:.2f} m, "
                  f"{abs(bearing):.0f} deg to the {side}")
            if m["camera_range_m"] is not None:
                print(f"  Distance from the head camera: {m['camera_range_m']:.2f} m")
            conf = f", confidence {m['confidence']:.2f}" if m["confidence"] is not None else ""
            print(f"  Seen {m['n']} times in 3 s, position spread {m['spread_cm']:.1f} cm{conf}")
            print(f"  Needs to move to grab it? {moves}")
            if m.get("source") != "detections":
                print(f"  (position from {m['source']})")
            pic = self.snap.save(name, [f"T2 trial {i}: SEES the pill bottle",
                                        f"{m['base_distance_m']:.2f} m away, {abs(bearing):.0f} deg {side}",
                                        f"odom x={fx:.2f} y={fy:.2f} z={fz:.2f}"])
            print(f"  Picture: {pic}" if pic else "  (no camera picture received)")
            print("=" * 60)

            t["condition"] = distance_bucket(m["base_distance_m"])
            t["metrics"] = {"search_time_s": info.get("search_time_s"), "x": fx, "y": fy, "z": fz,
                            "distance_m": m["base_distance_m"], "camera_range_m": m["camera_range_m"],
                            "bearing_deg": bearing, "spread_cm": m["spread_cm"],
                            "confidence": m["confidence"], "needs_to_move": bool(plan["steps"]) or not plan["ok"]}
            t["notes"] = moves + (f" | picture: {pic}" if pic else "")
            if not ask_yes("Is that the pill bottle (not another object)?", True):
                t["outcome"] = "FALSE_POSITIVE"
            elif m["spread_cm"] > self.max_spread_cm:
                t["outcome"], t["notes"] = "MISS", f"position jumped around ({m['spread_cm']:.1f} cm). " + moves
            else:
                t["outcome"] = "SUCCESS"

    def run(self):
        print(__doc__)
        ask("Driver, launch and detector running? Press Enter to begin.")
        for i in range(1, self.trials + 1):
            self.trial(i)
        print("\nSummary: python3 testing/summarize_results.py --test T2 --session latest")


def main():
    run_tester(FindTester)


if __name__ == "__main__":
    main()
