#!/usr/bin/env python3
"""
T1a / T1b - find the pill bottle with the head camera, plan how to get it, (T1b) do it.

  T1a (plan only, base and arm do NOT move; the head looks around):
    ros2 run bottle_tracking t1a_plan_tester --ros-args --params-file $P -p trials:=3

  T1b (plan, then drive the base and grab the bottle):
    ros2 run bottle_tracking t1b_execute_tester --ros-args --params-file $P -p trials:=3

Each trial:
  1. Pause so you can move the bottle (Enter to skip the wait).
  2. The head searches for the bottle (the base only turns if the head can't find it).
  3. Prints the full plan: base moves to put the arm beside the bottle, then the
     lift / arm / gripper steps to grab the middle of it.
  T1a: you say whether the plan looks right.
  T1b: drives the base, looks again up close, re-plans the arm from the fresh look,
       grabs, lifts, holds 5 s, then you say what happened. Then it puts the bottle back.

Run `ros2 run bottle_tracking calibrate_arm` once first so the arm numbers are real.
Needs: driver, bottle_tracking launch with the YOLO detector.
"""

import json
import time

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from std_srvs.srv import Trigger

from bottle_tracking import geometry as geo
from bottle_tracking.approach_planner_node import declare_approach_config
from bottle_tracking.common import DetectionCollector, MemoryWatcher, call_trigger
from bottle_tracking.stretch_interface import StretchInterface, declare
from bottle_tracking.test_logger import TestLogger
from bottle_tracking.testers.tester_common import (ask, ask_yes, declare_arm_config, distance_bucket,
                                                   pause, run_tester)


class PickTester(Node):
    EXECUTE = False
    TEST_ID, TEST_NAME = "T1a", "Find the bottle and plan the pick (no motion)"

    def __init__(self):
        super().__init__(f"{self.TEST_ID.lower()}_pick_tester")
        cb = ReentrantCallbackGroup()
        self.robot = StretchInterface(self, cb)
        self.mem = MemoryWatcher(self, cb)
        self.det = DetectionCollector(self, self.robot, cb)
        self.search_client = self.create_client(Trigger, "/head_search/search", callback_group=cb)
        self.arm = declare_arm_config(self)
        self.approach_cfg = geo.approach_config_for_arm(self.arm, declare_approach_config(self))
        self.trials = int(declare(self, "trials", 3))
        self.pause_s = declare(self, "pause_s", 15.0)
        self.hold_s = declare(self, "hold_s", 5.0)
        self.max_correction = declare(self, "max_correction_m", 0.15)
        self.log = TestLogger(self.TEST_ID, self.TEST_NAME)

    # ------------------------------------------------------------------ find + plan
    def find_bottle(self):
        self.mem.clear()
        time.sleep(0.3)
        ok, msg = call_trigger(self.search_client, timeout=180.0)
        try:
            info = json.loads(msg)
        except ValueError:
            info = {"error": msg}
        if not ok:
            return None, info
        m = self.det.measure(2.0)
        return m, info

    def plan(self, m):
        pose = self.robot.base_pose()
        t = time.perf_counter()
        full = geo.full_pick_plan(m["fixed"], pose, self.arm, self.approach_cfg)
        plan_time = time.perf_counter() - t
        print("\n" + geo.format_full_plan(full, self.robot.fixed_frame), flush=True)
        return full, plan_time

    @staticmethod
    def plan_metrics(full, plan_time):
        b = full["base"]
        out = {"plan_time_s": plan_time, "base_steps": len(b["steps"]),
               "total_rotation_deg": b.get("total_rotation_deg"), "total_drive_m": b.get("total_drive_m"),
               "predicted_table_clearance_cm": (b["predicted_table_clearance_m"] * 100
                                                if "predicted_table_clearance_m" in b else None)}
        if full["arm"] and full["arm"]["ok"]:
            out.update(arm_extension_m=full["arm"]["extension_m"], lift_m=full["arm"]["lift_grasp_m"])
        return out

    # ------------------------------------------------------------------ trial
    def trial(self, i):
        pause(self.pause_s, f"Trial {i}/{self.trials}: put the pill bottle somewhere on the table.")
        with self.log.trial() as t:
            m, info = self.find_bottle()
            if m is None:
                t["outcome"], t["notes"] = "NOT_FOUND", info.get("found_with", info.get("error", ""))
                t["metrics"] = {"search_time_s": info.get("search_time_s")}
                print("Did NOT find the pill bottle.")
                return
            t["condition"] = distance_bucket(m["base_distance_m"])
            print(f"\nFound the pill bottle {m['base_distance_m']:.2f} m from the base "
                  f"(seen {m['n']} times, spread {m['spread_cm']:.1f} cm).")
            full, plan_time = self.plan(m)
            t["metrics"] = {"search_time_s": info.get("search_time_s"),
                            "search_base_rotation_deg": info.get("base_rotation_deg"),
                            "distance_m": m["base_distance_m"], "spread_cm": m["spread_cm"],
                            **self.plan_metrics(full, plan_time)}
            t["notes"] = geo.format_full_plan(full, self.robot.fixed_frame)

            if not full["ok"]:
                right = ask_yes("Is NO_PLAN correct (the bottle really can't be reached)?")
                t["outcome"] = "SUCCESS" if right else "NO_PLAN"
                return
            if not self.EXECUTE:
                t["outcome"] = "SUCCESS" if ask_yes("Does this plan look right?", True) else "MISS"
                return
            self.execute(full, t)

    # ------------------------------------------------------------------ T1b execution
    def execute(self, full, t):
        if ask("Check the path is clear. Enter to DRIVE + GRAB, s to skip: ").lower() == "s":
            t["outcome"], t["notes"] = "ERROR", "skipped by user"
            return
        t0 = time.time()
        self.mem.set_tracking(True)
        abort = lambda: self.mem.get().get("status") == "MISSING"
        ok, failed = self.robot.execute_steps(full["base"]["steps"], abort, self.get_logger().info)
        if not ok:
            t["outcome"] = "NOT_FOUND" if abort() else "ERROR"
            t["notes"] = f"base step {failed} failed" + (" (bottle missing)" if abort() else "")
            return

        # look again up close and re-plan the arm from the fresh measurement
        m = self.det.measure(2.0)
        if m is None:
            t["outcome"], t["notes"] = "NOT_FOUND", "lost the bottle after driving"
            return
        off = m["base"][0] - self.arm.arm_x_offset_m
        if self.arm.arm_line_tol_m < abs(off) <= self.max_correction:
            print(f"Bottle {off * 100:+.1f} cm off the arm line, correcting with a {off:+.3f} m drive")
            self.robot.switch_to_navigation_mode()
            self.robot.drive_by(off)
            m = self.det.measure(2.0) or m
        arm_plan = geo.plan_arm(m["base"], self.arm)
        print("\n" + geo.format_arm_plan(arm_plan), flush=True)
        t["metrics"].update(final_off_arm_line_cm=(m["base"][0] - self.arm.arm_x_offset_m) * 100,
                            final_reach_m=-m["base"][1])
        if not arm_plan["ok"]:
            t["outcome"], t["notes"] = "NO_PLAN", f"after driving: {arm_plan['reason']}"
            return
        if ask("Enter to run the ARM plan, s to stop here: ").lower() == "s":
            t["outcome"], t["notes"] = "ERROR", "arm skipped by user"
            return
        ok, failed, msg = self.robot.execute_arm_steps(arm_plan["steps"], self.get_logger().info)
        t["metrics"]["exec_time_s"] = time.time() - t0
        if not ok:
            t["outcome"], t["notes"] = "ERROR", f"arm step {failed} failed: {msg}"
            t["metrics"]["failed_step"] = failed
            return

        print(f"Holding for {self.hold_s:.0f} s...")
        time.sleep(self.hold_s)
        hit = ask_yes("Did it hit the table or knock the bottle over?")
        res = ask("Result: [h]eld in the middle of the fingers, [e]held by the end/cap, "
                  "[d]ropped, [m]issed (air/pushed it): ").lower()[:1]
        t["metrics"]["grip_location"] = {"h": "middle", "e": "end"}.get(res, "none")
        if hit:
            t["outcome"] = "COLLISION"
        elif res == "h":
            t["outcome"] = "SUCCESS"
        elif res == "d":
            t["outcome"] = "DROP"
        else:
            t["outcome"] = "MISS"
        self.put_back(arm_plan)

    def put_back(self, arm_plan):
        if not ask_yes("Put the bottle back down and retract?", True):
            return
        lift_grasp = arm_plan["lift_grasp_m"]
        for step in ({"joint_lift": lift_grasp + 0.01},):
            self.robot.move_joints(step)
        self.robot.gripper("open")
        self.robot.move_joints({"wrist_extension": 0.0})
        self.robot.move_joints({"joint_lift": lift_grasp + self.arm.approach_height_m})

    def run(self):
        print(__doc__)
        ask("Driver, launch and detector running? Press Enter to begin.")
        for i in range(1, self.trials + 1):
            self.trial(i)
        print(f"\nSummary: python3 testing/summarize_results.py --test {self.TEST_ID} --session latest")


class ExecuteTester(PickTester):
    EXECUTE = True
    TEST_ID, TEST_NAME = "T1b", "Find the bottle, plan, drive and grab it"


def main_plan():
    run_tester(PickTester)


def main_execute():
    run_tester(ExecuteTester)


if __name__ == "__main__":
    main_plan()
