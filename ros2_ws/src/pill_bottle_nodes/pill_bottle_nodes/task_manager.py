# task_manager: the full run (what full_run.py did), built on the other nodes.
#   search -> plan (turn/drive the base if needed) -> approach -> refine with claw camera -> grasp + grip test
#   -> hold up to the head camera and verify -> place -> retry on a miss (4 tries max)
#
#   ros2 run pill_bottle_nodes task_manager                                  # asks once, then runs
#   ros2 run pill_bottle_nodes task_manager --ros-args -p dry_run:=true      # find + plan only, no motion
#   ros2 run pill_bottle_nodes task_manager --ros-args -p stop_after:=grasp  # stop (and put it back) after grasping
#   ros2 run pill_bottle_nodes test_task                                     # same as dry_run:=true
import json
import math
import os
import time

import rclpy

from . import grasp
from .client import ask, start
from .geometry import look_at

SETTINGS = {
    "dry_run": False,
    "yes": False,                       # skip the start question
    "confirm": False,                   # ask before every move
    "stop_after": "",                   # "", find, plan, grasp, verify
    "search_pans": [-1.6, -0.8, 0.0, -2.4],
    "search_tilt": -0.7,
    "settle_s": 1.0,
    "base_search_turn": math.radians(60),
    "max_base_turns": 6,
    "max_base_moves": 6,
    "max_attempts": 4,
    "hold_seconds": 8.0,
    "verify_conf": 0.25,
    "hold_lift": 0.9,
    "hold_arm": 0.10,
    "wrist_stow": 3.0,
}
SETTINGS.update(grasp.GRASP_SETTINGS)


class TaskManager:
    def __init__(self, overrides=None, name="task_manager"):
        self.c = start(name)
        for k, v in SETTINGS.items():
            self.c.declare_parameter(k, (overrides or {}).get(k, v))
        self.p = {k: self.c.get_parameter(k).value for k in SETTINGS}

    def say(self, text):
        print(text, flush=True)

    def move(self, *label, **joints):
        text = label[0] if label else ""
        if self.p["dry_run"]:
            self.say(f"   (dry run) would {text or 'move'} {joints}")
            return True
        if self.p["confirm"] and not ask(f"Next: {text} {joints}?"):
            raise RuntimeError("stopped at your request")
        ok, msg = self.c.move(**joints)
        if not ok:
            raise RuntimeError(f"Move {joints} failed: {msg}")
        return True

    def stow_for_base(self):
        self.move("retract the arm", wrist_extension=0.0)
        self.move("stow the wrist", joint_wrist_yaw=self.p["wrist_stow"])

    # ---------- stages ----------
    def find(self):
        p = self.p
        for turn in range(p["max_base_turns"]):
            for pan in p["search_pans"]:
                self.move("look", joint_head_pan=pan, joint_head_tilt=p["search_tilt"])
                t = time.time()
                time.sleep(p["settle_s"])
                st = self.c.wait_for("state", 4.0, newer_than=t + p["settle_s"])
                if st:
                    self.say(f"Found it: x {st.position.x:+.2f}, y {st.position.y:+.2f}, z {st.position.z:+.2f} m, "
                             f"{st.orientation}, on a surface at {st.surface_z:.2f} m")
                    return st
            if turn < p["max_base_turns"] - 1:
                self.say(f"Not in view; turning the base {math.degrees(p['base_search_turn']):.0f} deg.")
                self.stow_for_base()
                self.move("turn to search", rotate_mobile_base=p["base_search_turn"])
        return None

    def get_reach_plan(self, state):
        for _ in range(self.p["max_base_moves"] + 1):
            plan = self.c.plan(state)
            if plan is None:
                raise RuntimeError("grasp_planner isn't running")
            self.say(f"Plan [{plan.next_step}]: {plan.message}")
            if not plan.success:
                return None
            if plan.next_step == "reach":
                return plan
            if self.p["dry_run"]:
                self.say("   (dry run) stopping here; a real run would move the base and look again.")
                return None
            self.stow_for_base()
            if plan.next_step == "rotate_base":
                self.move("turn the base", rotate_mobile_base=plan.base_rotate)
            else:
                self.move("drive the base", translate_mobile_base=plan.base_translate)
            state = self.find()
            if state is None:
                self.say("Lost the bottle after moving the base.")
                return None
        self.say("Too many base moves; giving up.")
        return None

    def verify(self):
        p = self.p
        self.move("raise the bottle", joint_lift=p["hold_lift"])
        self.move("bring it in", wrist_extension=p["hold_arm"])
        head = self.c.position("link_head_tilt")
        grip = self.c.position("link_grasp_center")
        pan, tilt = look_at(head, grip)
        self.move("look at the claw", joint_head_pan=pan, joint_head_tilt=tilt)
        start_t, seen = time.time(), 0
        while time.time() - start_t < p["hold_seconds"]:
            det = self.c.wait_for("detections", 2.0)
            if det and any(d.name == "pill_bottle" and d.confidence >= p["verify_conf"] for d in det.detections):
                seen += 1
        self.say(f"Verification: bottle seen in {seen} frame(s) over {p['hold_seconds']:.0f} s")
        return seen >= 2

    def save_log(self, log):
        os.makedirs(grasp.OUT, exist_ok=True)
        with open(os.path.join(grasp.OUT, "grasp_log.jsonl"), "a") as f:
            f.write(json.dumps(log) + "\n")

    # ---------- the run ----------
    def run(self):
        p = self.p
        if not self.c.wait_joints():
            self.say("No /joint_states. Start the Stretch driver first.")
            return
        if not p["dry_run"] and not p["yes"] and not p["confirm"]:
            if not ask("FULL RUN: the arm AND base will move. Clear the area, hand on the runstop. Start?"):
                return
        self.stow_for_base()
        attempt = 0
        while attempt < p["max_attempts"]:
            self.say(f"\n===== Attempt {attempt + 1} of {p['max_attempts']} =====")
            state = self.find()
            if state is None:
                self.say("No pill bottle found.")
                return
            if p["stop_after"] == "find":
                return
            plan = self.get_reach_plan(state)
            if plan is None:
                return
            if p["dry_run"] or p["stop_after"] == "plan":
                self.say(f"Plan: lift {plan.lift:.3f} (travel {plan.lift_clear:.3f}), arm {plan.arm_pregrasp:.3f}"
                         f" -> {plan.arm_grasp:.3f}, base {plan.base_translate:+.3f} m. Nothing else moved.")
                return
            log = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "source": "task_manager", "attempt": attempt + 1,
                   "orientation": state.orientation, "bottle": [round(state.position.x, 3),
                   round(state.position.y, 3), round(state.position.z, 3)],
                   "settings": {k: p[k] for k in grasp.GRASP_SETTINGS}}
            grasp.approach(self.c, plan, p, self.move, self.say, log)
            grasp.refine(self.c, p, self.move, self.say, log)
            outcome = grasp.close_and_check(self.c, plan, p, self.move, self.say, log, tag=f"_try{attempt + 1}")
            log["outcome"] = outcome
            attempt += 1
            if outcome in ("missed", "dropped"):
                self.save_log(log)
                grasp.release(self.c, plan, p, self.move)
                continue
            if p["stop_after"] == "grasp":
                self.save_log(log)
                grasp.place(self.c, plan, p, self.move)
                self.stow_for_base()
                self.say("Stopped after the grasp (stop_after:=grasp); put it back.")
                return
            ok = self.verify()
            log["verified"] = ok
            self.save_log(log)
            self.say("Verified: it's the pill bottle." if ok else "Holding something, but YOLO couldn't confirm it.")
            if p["stop_after"] == "verify":
                self.say("Stopped after verifying (stop_after:=verify). The bottle is still in the claw.")
                return
            grasp.place(self.c, plan, p, self.move)
            self.stow_for_base()
            self.say("Done." if ok else "Done (unverified).")
            return
        self.say(f"Gave up after {p['max_attempts']} attempts.")


def _main(overrides=None, name="task_manager"):
    tm = TaskManager(overrides, name)
    try:
        tm.run()
    except RuntimeError as e:
        print(f"Stopped: {e}")
    except KeyboardInterrupt:
        print("Stopped (Ctrl+C).")
    finally:
        tm.c.close()
        rclpy.try_shutdown()


def main():
    _main()


def test_task():
    """Tester for task_manager: find + plan with no motion (override with -p dry_run:=false -p stop_after:=...)."""
    _main({"dry_run": True}, name="test_task")
