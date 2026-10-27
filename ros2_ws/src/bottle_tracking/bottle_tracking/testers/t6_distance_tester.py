#!/usr/bin/env python3
"""
T6 - distance to the pill bottle at different spots, base never moves.

  ros2 run bottle_tracking t6_distance_tester --ros-args --params-file $P -p trials:=5
  (default pause between spots is 60 s; change with -p pause_s:=30)

Each spot:
  1. Pauses (60 s by default) so you can put the bottle in a new spot. Enter skips the wait.
  2. Looks for it with the head camera only. The base does NOT move.
  3. Averages the detections for 3 s and prints the distance: from the base center
     (flat on the floor) and straight-line from the camera, plus how steady it was.
  4. Optional: type the tape-measured distance from the base center to get the error.

Pass: steady to within 1 cm, and (if you measured) within 2 cm of the tape (3 cm past 1.2 m).
Needs: driver, bottle_tracking launch with the YOLO detector.
"""

import json
import time

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from std_srvs.srv import Trigger

from bottle_tracking import geometry as geo
from bottle_tracking.common import DetectionCollector, MemoryWatcher, call_trigger
from bottle_tracking.stretch_interface import StretchInterface, declare
from bottle_tracking.test_logger import TestLogger
from bottle_tracking.testers.tester_common import ask, ask_float, distance_bucket, pause, run_tester


class DistanceTester(Node):
    def __init__(self):
        super().__init__("t6_distance_tester")
        cb = ReentrantCallbackGroup()
        self.robot = StretchInterface(self, cb)
        self.mem = MemoryWatcher(self, cb)
        self.det = DetectionCollector(self, self.robot, cb)
        self.search_client = self.create_client(Trigger, "/head_search/search_head_only", callback_group=cb)
        self.trials = int(declare(self, "trials", 5))
        self.pause_s = declare(self, "pause_s", 60.0)
        self.measure_s = declare(self, "measure_s", 3.0)
        self.max_spread_cm = declare(self, "max_spread_cm", 1.0)
        self.log = TestLogger("T6", "Distance to the pill bottle (base still)")
        self.start_pose = None

    def trial(self, i):
        pause(self.pause_s, f"Spot {i}/{self.trials}: put the pill bottle in a new spot.")
        with self.log.trial() as t:
            self.mem.clear()
            time.sleep(0.3)
            ok, msg = call_trigger(self.search_client, timeout=120.0)
            try:
                info = json.loads(msg)
            except ValueError:
                info = {"error": msg}
            m = self.det.measure(self.measure_s) if ok else None

            pose = self.robot.base_pose()
            if self.start_pose and pose and geo.dist((*self.start_pose[:2], 0), (*pose[:2], 0)) > 0.02:
                print("WARNING: the base moved since the first spot; distances may not compare.")

            print("\n" + "=" * 60)
            if m is None:
                print("Does NOT see the pill bottle.")
                print("=" * 60)
                t["condition"], t["outcome"] = "not_found", "NOT_FOUND"
                t["metrics"] = {"search_time_s": info.get("search_time_s")}
                return
            print("SEES the pill bottle")
            print(f"  Distance from base center (floor): {m['base_distance_m']:.3f} m")
            if m["camera_range_m"] is not None:
                print(f"  Distance from head camera (straight line): {m['camera_range_m']:.3f} m")
            print(f"  Steadiness: {m['spread_cm']:.2f} cm spread over {m['n']} detections")
            print("=" * 60)

            true_d = ask_float("Tape-measured distance from base center to the bottle (m)")
            t["metrics"] = {"est_m": m["base_distance_m"], "camera_range_m": m["camera_range_m"],
                            "spread_cm": m["spread_cm"], "n": m["n"], "true_m": true_d}
            t["condition"] = distance_bucket(true_d if true_d is not None else m["base_distance_m"])
            notes = []
            good = m["spread_cm"] <= self.max_spread_cm
            if not good:
                notes.append(f"unsteady ({m['spread_cm']:.1f} cm)")
            if true_d is not None:
                signed = (m["base_distance_m"] - true_d) * 100
                tol = 3.0 if true_d > 1.2 else 2.0
                t["metrics"].update(error_cm=abs(signed), signed_error_cm=signed)
                print(f"  Error: {signed:+.1f} cm ({'OK' if abs(signed) <= tol else 'too big'}, limit {tol:.0f} cm)")
                if abs(signed) > tol:
                    good = False
                    notes.append(f"error {signed:+.1f} cm")
            else:
                notes.append("no tape measurement")
            t["outcome"] = "SUCCESS" if good else "MISS"
            t["notes"] = "; ".join(notes)

    def run(self):
        print(__doc__)
        ask("Driver, launch and detector running? Press Enter to begin.")
        self.start_pose = self.robot.base_pose()
        for i in range(1, self.trials + 1):
            self.trial(i)
        print("\nSummary: python3 testing/summarize_results.py --test T6 --session latest")


def main():
    run_tester(DistanceTester)


if __name__ == "__main__":
    main()
