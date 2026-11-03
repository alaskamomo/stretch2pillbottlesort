#!/usr/bin/env python3
"""
T6 - distance to the pill bottle at different spots, then grip it. The base never moves.

  ros2 run bottle_tracking t6_distance_tester --ros-args --params-file $P -p trials:=5
  (default pause between spots is 60 s; change with -p pause_s:=30)

Each spot:
  1. Pauses (60 s by default) so you can put the bottle in a new spot. Enter skips the wait.
  2. Looks for it with the head camera only. The base does NOT move.
  3. Averages the detections for 3 s and prints the distance: from the base center
     (flat on the floor) and straight-line from the camera, plus how steady it was.
  4. Optional: type the tape-measured distance from the base center to get the error.
  5. If the arm can reach the bottle from where the base is, you press Enter and it reaches,
     grips, lifts, holds, and puts the bottle back. If it can't, it says why and skips the grip.

Pass: steady to within 1 cm, within 2 cm of the tape (3 cm past 1.2 m) if measured, and held
in the claw if a grip was tried. The arm reaches out the robot's right side, so put the bottle
beside the arm, within 3 cm of the arm's line, and about 0.50 to 0.80 m from the base center.
Needs: driver, bottle_tracking launch with the YOLO detector.
"""

import json
import time

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from std_srvs.srv import Trigger

from bottle_tracking import geometry as geo
from bottle_tracking.common import DetectionCollector, MemoryWatcher, SnapshotSaver, call_trigger
from bottle_tracking.stretch_interface import StretchInterface, declare
from bottle_tracking.test_logger import TestLogger
from bottle_tracking.testers.tester_common import (ask, ask_float, ask_yes, declare_arm_config,
                                                   distance_bucket, pause, reset_head, run_tester)


class DistanceTester(Node):
    def __init__(self):
        super().__init__("t6_distance_tester")
        cb = ReentrantCallbackGroup()
        self.robot = StretchInterface(self, cb)
        self.mem = MemoryWatcher(self, cb)
        self.det = DetectionCollector(self, self.robot, cb, memory=self.mem)
        self.snap = SnapshotSaver(self, cb)
        self.search_client = self.create_client(Trigger, "/head_search/search_head_only", callback_group=cb)
        self.trials = int(declare(self, "trials", 5))
        self.pause_s = declare(self, "pause_s", 60.0)
        self.measure_s = declare(self, "measure_s", 3.0)
        self.max_spread_cm = declare(self, "max_spread_cm", 1.0)
        self.hold_s = declare(self, "hold_s", 5.0)
        self.arm = declare_arm_config(self)
        self.log = TestLogger("T6", "Distance to the pill bottle, then grip it (base still)")
        self.start_pose = None

    def trial(self, i):
        reset_head(self)
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

            name = f"T6_{self.log.session_id}_spot{i}"
            print("\n" + "=" * 60)
            if m is None:
                print("Does NOT see the pill bottle.")
                print(f"  detections received by this tester: {self.det.received}")
                pic = self.snap.save(name, [f"T6 spot {i}: does NOT see the pill bottle"])
                print(f"  picture: {pic}" if pic else "  (no camera picture received)")
                print("=" * 60)
                t["condition"], t["outcome"] = "not_found", "NOT_FOUND"
                t["notes"] = f"picture: {pic}" if pic else ""
                t["metrics"] = {"search_time_s": info.get("search_time_s")}
                return
            print("SEES the pill bottle")
            print(f"  Distance from base center (floor): {m['base_distance_m']:.3f} m")
            if m["camera_range_m"] is not None:
                print(f"  Distance from head camera (straight line): {m['camera_range_m']:.3f} m")
            print(f"  Steadiness: {m['spread_cm']:.2f} cm spread over {m['n']} detections")
            pic = self.snap.save(name, [f"T6 spot {i}: {m['base_distance_m']:.3f} m from base",
                                        f"camera range {m['camera_range_m']:.3f} m" if m["camera_range_m"] else ""])
            print(f"  Picture: {pic}" if pic else "  (no camera picture received)")
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
            if pic:
                notes.append(f"picture: {pic}")
            t["outcome"] = "SUCCESS" if good else "MISS"
            grip = self.try_grip(m, t)
            if grip:
                notes.append(grip)
            t["notes"] = "; ".join(notes)

    def try_grip(self, m, t):
        """Reach and grip from where the base is. Returns a note, and changes the outcome if the grip failed."""
        plan = geo.plan_arm(m["base"], self.arm)
        print("\n" + geo.format_arm_plan(plan), flush=True)
        t["metrics"]["grip_tried"] = False
        if not plan["ok"]:
            print("  No grip this spot: the base stays still, and the arm can't reach the bottle from here.")
            return f"no grip: {plan['reason']}"
        if ask("Enter to REACH and GRIP, s to skip: ").lower() == "s":
            return "grip skipped by user"
        t["metrics"]["grip_tried"] = True
        lift = min(self.arm.lift_max_m, plan["lift_grasp_m"] + self.arm.approach_height_m)
        if not self.robot.switch_to_position_mode():
            t["outcome"] = "ERROR"
            return "could not switch to position mode"
        ok, msg = self.robot.move_joints({"wrist_extension": 0.0, "joint_lift": lift}, 30.0)
        if ok:
            ok, failed, msg = self.robot.execute_arm_steps(plan["steps"], self.get_logger().info)
        if not ok:
            t["outcome"] = "ERROR"
            return f"arm failed: {msg}"
        print(f"Holding for {self.hold_s:.0f} s...")
        time.sleep(self.hold_s)
        hit = ask_yes("Did it hit anything or knock the bottle over?")
        res = ask("Result: [h]eld deep in the claw, [e]held by the fingertips/cap, "
                  "[d]ropped, [m]issed (air/pushed it): ").lower()[:1]
        t["metrics"]["grip_location"] = {"h": "deep", "e": "tips"}.get(res, "none")
        if hit:
            t["outcome"] = "COLLISION"
        elif res == "d":
            t["outcome"] = "DROP"
        elif res != "h":
            t["outcome"] = "MISS"
        self.put_back(plan)
        return f"grip: {t['metrics']['grip_location']}"

    def put_back(self, plan):
        if not ask_yes("Put the bottle back down and retract?", True):
            return
        lift = plan["lift_grasp_m"]
        self.robot.move_joints({"joint_lift": lift + 0.01})
        self.robot.gripper("open")
        self.robot.move_joints({"wrist_extension": 0.0})
        self.robot.move_joints({"joint_lift": lift + self.arm.approach_height_m})
        if plan.get("floor"):
            self.robot.move_joints({"joint_wrist_pitch": 0.0})

    def run(self):
        print(__doc__)
        ask("Driver, launch and detector running? Press Enter to begin.")
        self.start_pose = self.robot.base_pose()
        for i in range(1, self.trials + 1):
            self.trial(i)
        reset_head(self)
        print("\nSummary: python3 testing/summarize_results.py --test T6 --session latest")


def main():
    run_tester(DistanceTester)


if __name__ == "__main__":
    main()
