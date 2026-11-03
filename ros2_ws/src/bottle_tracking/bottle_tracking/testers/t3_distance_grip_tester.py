#!/usr/bin/env python3
"""
T3a / T3d - how far away the pill bottle is, and whether the robot can grip it.

  T3a (live distance; only the head moves):
    ros2 run bottle_tracking t3a_distance_tester --ros-args --params-file $P
    ros2 run rqt_image_view rqt_image_view /pill_bottle/live_view     (second terminal, ssh -Y)

  T3d (find it, move the base if you approve, then grip it):
    ros2 run bottle_tracking t3d_grip_tester --ros-args --params-file $P -p trials:=3

T3a: runs until you type q. The live view shows the camera with a box on the bottle and its
  distance from the base, and the terminal prints the distance every 2 s. Press Enter to record
  a spot: the head resets and finds the bottle, it averages the detections, saves a picture,
  prints the distance, you can type a tape measurement, and it prints SUCCESS or MISS.
  Pass: steady to within 1 cm, and within 2 cm of the tape (3 cm past 1.2 m) if measured.

T3d, each trial: pauses 20 s (Enter skips) and looks with the head. If the arm can't reach, it
  prints a base plan and ASKS before moving. Once in reach, you press Enter and it reaches,
  grips, lifts, holds, and puts the bottle back. This replaces the old manual T8.
  Pass: the bottle is held deep in the claw, is not crushed, and doesn't slip for 5 s.

Needs: driver, head camera, bottle_tracking launch with the YOLO detector.
"""

import json
import math
import time

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger

from bottle_tracking import geometry as geo
from bottle_tracking.approach_planner_node import declare_approach_config
from bottle_tracking.common import DetectionCollector, MemoryWatcher, SnapshotSaver, call_trigger
from bottle_tracking.stretch_interface import StretchInterface, declare
from bottle_tracking.test_logger import TestLogger
from bottle_tracking.testers.tester_common import (ask, ask_float, ask_yes, declare_arm_config,
                                                   distance_bucket, pause, reset_head, run_tester)


class DistanceTester(Node):
    GRIP = False
    LIVE = True
    TEST_ID, TEST_NAME, DEFAULT_PAUSE = "T3a", "Live distance to the pill bottle", 0.0

    def __init__(self):
        super().__init__(f"{self.TEST_ID.lower()}_tester")
        cb = ReentrantCallbackGroup()
        self.robot = StretchInterface(self, cb)
        self.mem = MemoryWatcher(self, cb)
        self.det = DetectionCollector(self, self.robot, cb, memory=self.mem)
        self.snap = SnapshotSaver(self, cb)
        self.search_client = self.create_client(Trigger, "/head_search/search_head_only", callback_group=cb)
        self.trials = int(declare(self, "trials", 5))
        self.pause_s = declare(self, "pause_s", self.DEFAULT_PAUSE)
        self.measure_s = declare(self, "measure_s", 3.0)
        self.max_spread_cm = declare(self, "max_spread_cm", 1.0)
        self.hold_s = declare(self, "hold_s", 5.0)
        self.arm = declare_arm_config(self)
        self.approach_cfg = geo.approach_config_for_arm(self.arm, declare_approach_config(self))
        self.max_moves = int(declare(self, "max_base_moves", 3))
        self.log = TestLogger(self.TEST_ID, self.TEST_NAME)
        self.start_pose = None
        self.live_printing = False
        self.last_live_print = 0.0
        if self.LIVE:
            self.live_pub = self.create_publisher(Image, "/pill_bottle/live_view", 2)
            self.create_timer(0.5, self.live_step, callback_group=cb)

    # ---------------------------------------------------------------- live view
    def live_distance(self):
        """(base distance, camera distance) from the newest detection, or None if it's old."""
        with self.det._lock:
            newest = self.det.samples[-1] if self.det.samples else None
        if newest is None or time.time() - newest[0] > 1.5:
            return None
        base, _, cam = self.det._convert(newest[1], newest[2])
        if base is None:
            return None
        return math.hypot(base[0], base[1]), math.sqrt(sum(c * c for c in cam)) if cam else None

    def live_step(self):
        d = self.live_distance()
        text = "NOT SEEN" if d is None else f"{d[0]:.2f} m from base"
        msg = self.snap.latest
        if msg is not None:
            try:
                import cv2
                img = self.snap._to_array(msg)
                cv2.rectangle(img, (0, 0), (img.shape[1], 44), (0, 0, 0), -1)
                color = (0, 0, 255) if d is None else (0, 255, 0)
                cv2.putText(img, text, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
                out = Image()
                out.header = msg.header
                out.height, out.width = img.shape[:2]
                out.encoding = "bgr8"
                out.step = out.width * 3
                out.data = img.tobytes()
                self.live_pub.publish(out)
            except Exception as e:      # the live view must never stop the test
                self.get_logger().warn(f"live view: {e}", throttle_duration_sec=10.0)
        if self.live_printing and time.time() - self.last_live_print >= 2.0:
            self.last_live_print = time.time()
            print(f"  LIVE: {text}", flush=True)

    # ---------------------------------------------------------------- one spot
    def trial(self, i):
        reset_head(self)
        if not self.LIVE:
            pause(self.pause_s, f"Spot {i}/{self.trials}: put the pill bottle in a new spot.")
        with self.log.trial() as t:
            ok, msg = call_trigger(self.search_client, timeout=120.0)
            try:
                info = json.loads(msg)
            except ValueError:
                info = {"error": msg}
            m = self.det.measure(self.measure_s) if ok else None

            pose = self.robot.base_pose()
            if not self.GRIP and self.start_pose and pose and geo.dist((*self.start_pose[:2], 0), (*pose[:2], 0)) > 0.02:
                print("WARNING: the base moved since the first spot; distances may not compare.")

            name = f"{self.TEST_ID}_{self.log.session_id}_spot{i}"
            print("\n" + "=" * 60)
            if m is None:
                print("Does NOT see the pill bottle.")
                print(f"  detections received by this tester: {self.det.received}")
                pic = self.snap.save(name, [f"{self.TEST_ID} spot {i}: does NOT see the pill bottle"])
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
            pic = self.snap.save(name, [f"{self.TEST_ID} spot {i}: {m['base_distance_m']:.3f} m from base",
                                        f"camera range {m['camera_range_m']:.3f} m" if m["camera_range_m"] else ""])
            print(f"  Picture: {pic}" if pic else "  (no camera picture received)")
            print("=" * 60)

            if self.GRIP:
                self.grip_trial(m, t, pic)
                return

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
            t["notes"] = "; ".join(notes)
            print(f"RESULT: {t['outcome']}  {m['base_distance_m']:.3f} m from base"
                  + (f"  ({t['notes']})" if t["notes"] else ""), flush=True)

    def grip_trial(self, m, t, pic):
        t["metrics"] = {"start_distance_m": m["base_distance_m"], "spread_cm": m["spread_cm"]}
        t["condition"] = distance_bucket(m["base_distance_m"])
        t["outcome"] = "SUCCESS"
        bottle = self.get_in_reach(m, t)
        note = self.try_grip(bottle, t) if bottle is not None else "no grip: base did not move"
        if t["outcome"] == "SUCCESS" and not t["metrics"].get("grip_tried"):
            t["outcome"] = "NO_PLAN"
        t["notes"] = note + (f"; picture: {pic}" if pic else "")

    def get_in_reach(self, m, t):
        """Move the base, after asking, until the arm can reach. Returns the bottle in base_link, or None."""
        bottle_base, fixed = m["base"], m["fixed"]
        t["metrics"]["base_moves"] = 0
        for _ in range(self.max_moves):
            if geo.plan_arm(bottle_base, self.arm)["ok"]:
                return bottle_base
            pose = self.robot.base_pose()
            if pose is None:
                print("  No base position available, so the base won't move.")
                return bottle_base
            plan = geo.full_pick_plan(fixed, pose, self.arm, self.approach_cfg)
            print("\nThe arm can't reach from here. Base plan:")
            print(geo.format_full_plan(plan, self.robot.fixed_frame), flush=True)
            if not plan["base"]["ok"] or not plan["base"]["steps"]:
                return bottle_base
            if ask("Check the path is clear. Enter to MOVE THE BASE, s to skip: ").lower() == "s":
                return None
            lift = fixed[2] - self.arm.lift_grasp_z_offset_m + 0.10
            lift = min(self.arm.lift_max_m, max(self.arm.lift_min_m, lift))
            if self.robot.switch_to_position_mode():
                self.robot.move_joints({"wrist_extension": 0.0, "joint_lift": lift}, 30.0)
            self.mem.set_tracking(True)
            ok, failed = self.robot.execute_steps(plan["base"]["steps"], None, self.get_logger().info)
            t["metrics"]["base_moves"] += 1
            self.start_pose = self.robot.base_pose()
            if not ok:
                t["outcome"] = "ERROR"
                print(f"  Base step {failed} failed.")
                return None
            self.mem.wait_for(lambda: self.det.sees(1.0), 10.0)
            again = self.det.measure(self.measure_s)
            if again is not None:
                bottle_base, fixed = again["base"], again["fixed"]
            else:
                print("  Didn't see the bottle again; using its saved location.")
                bx, by = geo.bottle_in_base(fixed[:2], self.robot.base_pose() or pose)
                bottle_base = (bx, by, fixed[2])
        return bottle_base

    def try_grip(self, bottle_base, t):
        """Reach and grip from where the base is. Returns a note, and changes the outcome if the grip failed."""
        plan = geo.plan_arm(bottle_base, self.arm)
        print("\n" + geo.format_arm_plan(plan), flush=True)
        t["metrics"]["grip_tried"] = False
        if not plan["ok"]:
            print("  No grip this spot: the arm can't reach the bottle from here.")
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

    def run_live(self):
        print(__doc__)
        ask("Driver, camera and launch running? Press Enter to start the live view.")
        self.start_pose = self.robot.base_pose()
        reset_head(self)
        call_trigger(self.search_client, timeout=120.0)
        i = 0
        while True:
            self.live_printing = True
            answer = input("Enter = record this spot, q = quit\n").strip().lower()
            self.live_printing = False
            if answer == "q":
                break
            i += 1
            self.trial(i)
        reset_head(self)
        print(f"\nSummary: python3 testing/summarize_results.py --test {self.TEST_ID} --session latest")

    def run(self):
        if self.LIVE:
            self.run_live()
            return
        print(__doc__)
        ask("Driver, launch and detector running? Press Enter to begin.")
        self.start_pose = self.robot.base_pose()
        for i in range(1, self.trials + 1):
            self.trial(i)
        reset_head(self)
        print(f"\nSummary: python3 testing/summarize_results.py --test {self.TEST_ID} --session latest")


class GripTester(DistanceTester):
    GRIP = True
    LIVE = False
    TEST_ID, TEST_NAME, DEFAULT_PAUSE = "T3d", "Find the bottle, move if approved, then grip it", 20.0


def main_distance():
    run_tester(DistanceTester)


def main_grip():
    run_tester(GripTester)


if __name__ == "__main__":
    main_distance()
