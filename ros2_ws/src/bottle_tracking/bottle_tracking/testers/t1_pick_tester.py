#!/usr/bin/env python3
"""
T1a / T1b - find the pill bottle with the head camera, plan how to get it, (T1b) do it.

  T1a (plan only; ONLY the head moves, the base and arm never move):
    ros2 run bottle_tracking t1a_plan_tester --ros-args --params-file $P -p trials:=3

  T1b (plan, then drive the base and grab the bottle):
    ros2 run bottle_tracking t1b_execute_tester --ros-args --params-file $P -p trials:=3

Each trial:
  1. Pause so you can move the bottle (Enter to skip the wait).
  2. The head looks for the bottle. The base does NOT turn to search.
  3. Prints PLAN and the full plan (base moves, then lift / arm / gripper steps),
     or NO PLAN, why, and what would make a new plan possible.
  T1b: arm in and lift above the table, then drives the base. The bottle's location is
       saved, so losing sight of it while turning is fine. After driving it looks at the
       saved spot for up to 20 s; if it doesn't see it, it searches the area around the
       saved spot with the head; if still nothing, it assumes nothing changed and uses the
       saved location. If the arm can't reach yet, it turns/drives again (up to 2 times)
       to put the arm in front of the bottle. Then it grabs, lifts, holds 5 s, you say
       what happened, and it puts the bottle back.

At the end it saves a picture for every trial (camera view + top-down drawing of the
planned approach) and one overview picture of all trials, in ~/stretch_test_results/images/.

Run `ros2 run bottle_tracking calibrate_arm` once first so the arm numbers are real.
Needs: driver, head camera, bottle_tracking launch with the YOLO detector.
"""

import json
import time

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from std_srvs.srv import Trigger

from bottle_tracking import geometry as geo
from bottle_tracking import plan_drawing
from bottle_tracking.approach_planner_node import declare_approach_config
from bottle_tracking.common import DetectionCollector, MemoryWatcher, SnapshotSaver, call_trigger
from bottle_tracking.stretch_interface import StretchInterface, declare
from bottle_tracking.test_logger import TestLogger
from bottle_tracking.testers.tester_common import (ask, ask_yes, declare_arm_config, distance_bucket,
                                                   pause, run_tester)


def how_to_replan(full, m, arm, cfg):
    """Plain-language list of what would make a plan possible."""
    if m is None:
        return ["The head camera did not see the pill bottle.",
                "To make a new plan: put the bottle where the head can see it (in front of or beside the "
                "robot, on the table), or check the camera picture for why the detector missed it."]
    reason = full["reason"] if full else ""
    d = m["base_distance_m"]
    tips = []
    if "max approach" in reason:
        tips.append(f"Move the bottle (or the robot) to within {cfg.max_approach_m:.1f} m; it is {d:.2f} m away. "
                    f"Or raise max_approach_m in params.yaml.")
    elif "lift range" in reason or "needs lift" in reason:
        tips.append(f"Put the bottle at a height the lift can reach (about {arm.lift_min_m + arm.lift_grasp_z_offset_m:.2f} "
                    f"to {arm.lift_max_m + arm.lift_grasp_z_offset_m - arm.lift_after_grasp_m:.2f} m); "
                    f"it is at {m['fixed'][2]:.2f} m.")
    elif "table clearance" in reason:
        tips.append("Move the bottle closer to the table edge, or lower table_edge_to_bottle_m / "
                    "min_table_clearance_m if the table is really that shallow.")
    elif "arm line" in reason or "too far" in reason or "too close" in reason or "left side" in reason:
        tips.append("The base move doesn't line the arm up well enough. Re-run calibrate_arm, "
                    "then plan again.")
    else:
        tips.append("Move the bottle to a clear spot on the table in front of the robot and plan again.")
    return [f"Reason: {reason}"] + tips


class PickTester(Node):
    EXECUTE = False
    TEST_ID, TEST_NAME = "T1a", "Find the bottle and plan the pick (head only, no base/arm motion)"

    def __init__(self):
        super().__init__(f"{self.TEST_ID.lower()}_pick_tester")
        cb = ReentrantCallbackGroup()
        self.robot = StretchInterface(self, cb)
        self.mem = MemoryWatcher(self, cb)
        self.det = DetectionCollector(self, self.robot, cb, memory=self.mem)
        self.snap = SnapshotSaver(self, cb)
        # head-only search: the base NEVER turns while looking for the bottle
        self.search_client = self.create_client(Trigger, "/head_search/search_head_only", callback_group=cb)
        self.arm = declare_arm_config(self)
        self.approach_cfg = geo.approach_config_for_arm(self.arm, declare_approach_config(self))
        self.trials = int(declare(self, "trials", 3))
        self.pause_s = declare(self, "pause_s", 15.0)
        self.hold_s = declare(self, "hold_s", 5.0)
        self.max_correction = declare(self, "max_correction_m", 0.15)
        self.see_timeout = declare(self, "see_timeout_s", 20.0)       # T1b: look this long before searching
        self.max_replans = int(declare(self, "max_replans", 2))        # T1b: extra base moves to line the arm up
        self.log = TestLogger(self.TEST_ID, self.TEST_NAME)
        self.pictures = []            # one composed picture per trial, for the overview at the end

    # ------------------------------------------------------------------ find + plan
    def find_bottle(self):
        self.mem.clear()
        time.sleep(0.3)
        ok, msg = call_trigger(self.search_client, timeout=120.0)
        try:
            info = json.loads(msg)
        except ValueError:
            info = {"error": msg}
        if not ok:
            return None, info
        return self.det.measure(3.0), info

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

    def picture(self, i, m, full, start_pose, headline):
        """Camera view + top-down plan drawing for this trial. Saved now, and kept for the overview."""
        try:
            cam = self.snap.grab([f"{self.TEST_ID} trial {i}: {headline}"] +
                                 ([f"{m['base_distance_m']:.2f} m from the base"] if m else []))
            title = "PLAN" if full is not None and full["ok"] else "NO PLAN"
            drawing = plan_drawing.draw_plan(full, m["fixed"] if m else None, start_pose, self.arm,
                                             f"Trial {i}: {title}")
            pic = plan_drawing.compose(cam, drawing, f"{self.TEST_ID} trial {i}: {headline}")
            self.pictures.append(pic)
            return self.snap.write(f"{self.TEST_ID}_{self.log.session_id}_trial{i}", pic)
        except Exception as e:      # a picture problem must never stop the test
            print(f"  (could not make the picture: {e})")
            return None

    # ------------------------------------------------------------------ trial
    def trial(self, i):
        pause(self.pause_s, f"Trial {i}/{self.trials}: put the pill bottle somewhere on the table.")
        with self.log.trial() as t:
            start_pose = self.robot.base_pose() or (0.0, 0.0, 0.0)
            m, info = self.find_bottle()
            t["metrics"] = {"search_time_s": info.get("search_time_s")}

            print("\n" + "=" * 70)
            if m is None:
                print(f"NO PLAN  (trial {i})")
                for line in how_to_replan(None, None, self.arm, self.approach_cfg):
                    print(f"  {line}")
                print(f"  (detections received by this tester: {self.det.received})")
                pic = self.picture(i, None, None, start_pose, "NO PLAN - bottle not seen")
                print(f"  Picture: {pic}" if pic else "")
                print("=" * 70)
                t["condition"], t["outcome"] = "not_found", "NOT_FOUND"
                t["notes"] = "bottle not seen" + (f" | picture: {pic}" if pic else "")
                return

            t0 = time.perf_counter()
            full = geo.full_pick_plan(m["fixed"], start_pose, self.arm, self.approach_cfg)
            plan_time = time.perf_counter() - t0
            plan_text = geo.format_full_plan(full, self.robot.fixed_frame)
            t["condition"] = distance_bucket(m["base_distance_m"])
            t["metrics"].update(distance_m=m["base_distance_m"], spread_cm=m["spread_cm"],
                                **self.plan_metrics(full, plan_time))

            if full["ok"]:
                print(f"PLAN  (trial {i}: bottle {m['base_distance_m']:.2f} m from the base)")
                print(plan_text)
                pic = self.picture(i, m, full, start_pose, f"PLAN - bottle {m['base_distance_m']:.2f} m away")
                t["outcome"] = "SUCCESS"
            else:
                print(f"NO PLAN  (trial {i}: bottle {m['base_distance_m']:.2f} m from the base)")
                for line in how_to_replan(full, m, self.arm, self.approach_cfg):
                    print(f"  {line}")
                pic = self.picture(i, m, full, start_pose, f"NO PLAN - {full['reason'][:50]}")
                t["outcome"] = "NO_PLAN"
            print(f"Picture: {pic}" if pic else "")
            print("=" * 70)
            t["notes"] = plan_text + (f"\npicture: {pic}" if pic else "")

            if self.EXECUTE and full["ok"]:
                self.execute(full, t, i)

    def finish(self):
        if self.pictures:
            try:
                overview = plan_drawing.stack(self.pictures)
                path = self.snap.write(f"{self.TEST_ID}_{self.log.session_id}_all_plans", overview)
                print(f"\nPictures of every planned approach: {path}")
            except Exception as e:
                print(f"(could not make the overview picture: {e})")
        print(f"Summary: python3 testing/summarize_results.py --test {self.TEST_ID} --session latest")

    # ------------------------------------------------------------------ T1b execution
    # ------------------------------------------------------------------ T1b helpers
    def prepare_for_driving(self, bottle_z):
        """Before the base moves: arm all the way in, lift above table height (mindful of tables)."""
        st = self.robot.arm_state()
        safe_lift = min(self.arm.lift_max_m, bottle_z - self.arm.lift_grasp_z_offset_m + 0.10)
        target = {"wrist_extension": 0.0}
        if st is None or st[0] < safe_lift:
            target["joint_lift"] = safe_lift
        print(f"  Arm in, lift to {target.get('joint_lift', st[0] if st else 0):.2f} m before driving")
        if not self.robot.switch_to_position_mode():
            return False
        ok, msg = self.robot.move_joints(target, 30.0)
        if not ok:
            print(f"  could not move the arm/lift: {msg}")
        return ok

    def local_search(self, saved_fixed):
        """Sweep the head in a small grid around the saved spot (base does not move)."""
        self.mem.set_tracking(True)
        self.mem.wait_for(lambda: self.mem.tracker_status().get("pan_cmd") is not None, 2.0)
        tr = self.mem.tracker_status()
        if tr.get("pan_cmd") is None:
            return False
        self.mem.set_tracking(False)
        pan0, tilt0 = tr["pan_cmd"], tr["tilt_cmd"]
        for dp, dt in [(0, 0), (0.3, 0), (-0.3, 0), (0, -0.25), (0.3, -0.25), (-0.3, -0.25),
                       (0, 0.2), (0.6, 0), (-0.6, 0)]:
            self.robot.move_head_and_wait(pan0 + dp, tilt0 + dt, timeout=3.0)
            if self.mem.wait_for(lambda: self.det.sees(1.0), 1.5):
                self.mem.set_tracking(True)
                return True
        self.mem.set_tracking(True)
        return False

    def relocate(self, saved_fixed):
        """Find the bottle again after a base move.
        1) look at the saved spot for up to see_timeout_s (20 s)
        2) not seen: search the area around the saved spot with the head
        3) still not seen: assume nothing changed and use the saved location.
        Returns (bottle_fixed, how)."""
        self.mem.set_tracking(True)
        print(f"  Looking at the saved spot (up to {self.see_timeout:.0f} s)...")
        if self.mem.wait_for(lambda: self.det.sees(1.0), self.see_timeout):
            m = self.det.measure(3.0)
            if m:
                return m["fixed"], "seen again"
        print("  Not seen. Searching the area around the saved spot with the head...")
        if self.local_search(saved_fixed):
            m = self.det.measure(3.0)
            if m:
                return m["fixed"], "found by searching nearby"
        print("  Still not seen. Assuming nothing changed: using the saved location.")
        return saved_fixed, "saved location (not re-seen)"

    def execute(self, full, t, i):
        if ask("Check the path is clear. Enter to DRIVE + GRAB, s to skip: ").lower() == "s":
            t["outcome"], t["notes"] = "ERROR", "skipped by user"
            return
        t0 = time.time()
        bottle = tuple(full["base"]["bottle"])          # saved location (odom), kept the whole time
        base_plan = full["base"]
        how, arm_plan, moves = "", None, 0

        for attempt in range(1 + self.max_replans):
            if base_plan["steps"]:
                self.prepare_for_driving(bottle[2])
                self.mem.set_tracking(True)
                # never stop mid-drive: the bottle's location is saved, so losing sight briefly is fine
                ok, failed = self.robot.execute_steps(base_plan["steps"], None, self.get_logger().info)
                moves += 1
                if not ok:
                    t["outcome"], t["notes"] = "ERROR", f"base step {failed} failed or timed out"
                    return
            new, how = self.relocate(bottle)
            if geo.dist(new, bottle) > 0.15:
                print(f"  The bottle is {geo.dist(new, bottle) * 100:.0f} cm from where it was; using the new spot.")
            bottle = new
            pose = self.robot.base_pose()
            bx, by = geo.bottle_in_base(bottle[:2], pose)
            arm_plan = geo.plan_arm((bx, by, bottle[2]), self.arm)
            if arm_plan["ok"]:
                break
            if attempt == self.max_replans:
                break
            # arm can't reach from here: turn/drive again to put the arm in front of the bottle
            print(f"  Arm can't reach yet ({arm_plan['reason']}). Re-planning the base move:")
            base_plan = geo.plan_approach(bottle, pose, self.approach_cfg)
            print(geo.format_plan(base_plan, self.robot.fixed_frame))
            if not base_plan["ok"]:
                break

        close = self.snap.save(f"{self.TEST_ID}_{self.log.session_id}_trial{i}_closeup",
                               [f"{self.TEST_ID} trial {i}: ready to grab", f"bottle: {how}"])
        if close:
            print(f"Close-up picture: {close}")
        pose = self.robot.base_pose()
        bx, by = geo.bottle_in_base(bottle[:2], pose)
        t["metrics"].update(final_off_arm_line_cm=(bx - self.arm.arm_x_offset_m) * 100, final_reach_m=-by,
                            base_moves=moves, bottle_source=how)
        print("\n" + geo.format_arm_plan(arm_plan), flush=True)
        if not arm_plan["ok"]:
            t["outcome"] = "NO_PLAN"
            t["notes"] = f"after {moves} base move(s): {arm_plan['reason']}" + (f" | close-up: {close}" if close else "")
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
        ask("Driver, camera and launch running? Press Enter to begin.")
        try:
            for i in range(1, self.trials + 1):
                self.trial(i)
        finally:
            self.finish()


class ExecuteTester(PickTester):
    EXECUTE = True
    TEST_ID, TEST_NAME = "T1b", "Find the bottle, plan, drive and grab it"


def main_plan():
    run_tester(PickTester)


def main_execute():
    run_tester(ExecuteTester)


if __name__ == "__main__":
    main_plan()
