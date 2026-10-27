#!/usr/bin/env python3
"""
approach_planner_node - moves the base so the ARM ends up in front of the bottle.

The Stretch arm extends out the right side of the base (-y in base_link), so "arm in
front of the bottle" means the bottle is beside the robot, standoff_m out, lined up
with the arm. The planner does at most: rotate, drive straight, rotate.
If the robot is already beside the bottle it only slides forward/back.

Services (std_srvs/Trigger, response message is JSON):
  /approach_planner/plan     make + print the plan only, never moves (dry run)
  /approach_planner/execute  make + print the plan, then drive it (unless dry_run:=true)

Uses the saved bottle position from bottle_memory, so it still works after turning.
Head tracking is switched on during the move so the camera stays on the bottle.
After the move it waits for a fresh detection and fixes any leftover alignment
error with one small forward/back drive (odometry drift correction).
"""

import json
import time

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from std_srvs.srv import Trigger

from bottle_tracking import geometry as geo
from bottle_tracking.common import MemoryWatcher
from bottle_tracking.stretch_interface import StretchInterface, declare


def declare_approach_config(node):
    """Declare every ApproachConfig field as a ROS parameter and build the config."""
    from dataclasses import asdict
    values = {k: declare(node, k, v) for k, v in asdict(geo.ApproachConfig()).items()}
    return geo.config_from_dict(values)


class ApproachPlannerNode(Node):
    def __init__(self):
        super().__init__("approach_planner")
        cb = ReentrantCallbackGroup()
        self.robot = StretchInterface(self, cb)
        self.mem = MemoryWatcher(self, cb)
        self.cfg = declare_approach_config(self)
        self.dry_run = declare(self, "dry_run", False)
        self.correct = declare(self, "correction_enabled", True)
        self.correction_wait = declare(self, "correction_wait_s", 2.0)
        self.max_correction = declare(self, "max_correction_m", 0.15)
        self.max_bottle_age = declare(self, "max_bottle_age_s", 30.0)

        self.busy = False
        self.create_service(Trigger, "/approach_planner/plan", self.on_plan, callback_group=cb)
        self.create_service(Trigger, "/approach_planner/execute", self.on_execute, callback_group=cb)
        self.get_logger().info(f"approach planner ready (dry_run={self.dry_run})")

    # ------------------------------------------------------------------ planning
    def make_plan(self):
        s = self.mem.get()
        if s.get("status") != "KNOWN":
            return None, f"bottle status is {s.get('status')}, run /head_search/search first"
        if s.get("age_s") is None or s["age_s"] > self.max_bottle_age:
            return None, f"saved bottle position is {s.get('age_s')} s old, search again"
        pose = self.robot.base_pose()
        if pose is None:
            return None, "no odom->base_link TF (launch stretch_driver with broadcast_odom_tf:=True)"
        plan = geo.plan_approach((s["x"], s["y"], s["z"]), pose, self.cfg)
        text = geo.format_plan(plan, self.robot.fixed_frame)
        print(text, flush=True)
        self.get_logger().info("\n" + text)
        return plan, text

    @staticmethod
    def summary(plan, text):
        return {"ok": plan["ok"], "reason": plan["reason"], "plan_text": text,
                "steps": plan["steps"], "num_steps": len(plan["steps"]),
                "total_rotation_deg": round(plan.get("total_rotation_deg", 0.0), 1),
                "total_drive_m": round(plan.get("total_drive_m", 0.0), 3),
                "predicted_align_error_cm": round(plan.get("predicted_align_error_m", 0.0) * 100, 2),
                "predicted_reach_m": round(plan.get("predicted_reach_m", 0.0), 3),
                "predicted_table_clearance_cm": round(plan.get("predicted_table_clearance_m", 0.0) * 100, 1),
                "start_distance_m": round(plan["start_distance_m"], 3)}

    def on_plan(self, _req, res):
        t = time.time()
        plan, text = self.make_plan()
        if plan is None:
            res.success, res.message = False, json.dumps({"ok": False, "reason": text})
            return res
        out = self.summary(plan, text)
        out["plan_time_s"] = round(time.time() - t, 3)
        res.success, res.message = plan["ok"], json.dumps(out)
        return res

    # ------------------------------------------------------------------ execution
    def on_execute(self, _req, res):
        if self.busy:
            res.success, res.message = False, json.dumps({"ok": False, "reason": "already executing"})
            return res
        self.busy = True
        try:
            out = self.execute()
        except Exception as e:
            self.robot.stop()
            out = {"ok": False, "reason": f"{type(e).__name__}: {e}"}
        finally:
            self.busy = False
        res.success, res.message = bool(out.get("ok")), json.dumps(out)
        return res

    def execute(self):
        t0 = time.time()
        plan, text = self.make_plan()
        if plan is None:
            return {"ok": False, "reason": text}
        out = self.summary(plan, text)
        if not plan["ok"]:
            return out
        if self.dry_run:
            out["dry_run"] = True
            return out

        self.mem.set_tracking(True)
        abort = lambda: self.mem.get().get("status") == "MISSING"   # bottle gone: stop (T4)
        ok, failed = self.robot.execute_steps(plan["steps"], abort, self.get_logger().info)
        out["failed_step"] = failed
        if not ok:
            out.update(ok=False, reason="bottle missing" if abort() else f"step {failed} failed")
            return out

        # final check / correction using a fresh detection
        out.update(self.check_alignment())
        if (self.correct and out.get("final_align_error_cm") is not None
                and self.cfg.align_tol_m * 100 < abs(out["final_align_error_cm"]) <= self.max_correction * 100):
            # positive error = bottle is ahead of the arm line -> drive forward that much
            fix = out["final_align_error_cm"] / 100.0
            self.get_logger().info(f"correction drive {fix:+.3f} m")
            self.robot.switch_to_navigation_mode()
            self.robot.drive_by(fix)
            out["correction_m"] = round(fix, 3)
            out.update(self.check_alignment())
        out["exec_time_s"] = round(time.time() - t0, 2)
        out["ok"] = (out.get("final_align_error_cm") is not None
                     and abs(out["final_align_error_cm"]) <= 3.0
                     and self.cfg.reach_min_m <= out["final_reach_m"] <= self.cfg.reach_max_m)
        if not out["ok"]:
            out["reason"] = "final alignment outside tolerance"
        return out

    def check_alignment(self):
        """Bottle position relative to the arm line after moving (waits for a fresh detection)."""
        since = self.mem.now()
        fresh = self.mem.wait_for(lambda: self.mem.seen_since(since), self.correction_wait)
        pos = self.mem.position()
        pose = self.robot.base_pose()
        if pos is None or pose is None:
            return {"final_align_error_cm": None, "final_reach_m": None, "fresh_detection": fresh}
        lx, ly = geo.bottle_in_base(pos[:2], pose)
        return {"final_align_error_cm": round((lx - self.cfg.arm_x_offset_m) * 100, 2),
                "final_reach_m": round(-ly, 3), "fresh_detection": fresh}


def main():
    rclpy.init()
    node = ApproachPlannerNode()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.robot.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
