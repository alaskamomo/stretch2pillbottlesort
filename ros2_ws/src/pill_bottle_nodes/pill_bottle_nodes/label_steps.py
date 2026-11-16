# label_steps.py: the robot side of label reading (T16), shared by task_manager and test_label.
#   LabelSteps.set_yaw / set_lift  move the wrist or lift relative to the verify pose, then re-aim the head
#   LabelSteps.read_view           grab a fresh head image + detections, crop/flatten, read
#   make_regrasp                   put the bottle down and grab it again at a different height
# The decisions (which view next, when to give up) are in label_reading.run_label_sequence.
import os
import time
from types import SimpleNamespace

import cv2

from . import label_reading as lr
from .common import image_to_numpy
from .geometry import look_at

OUT = os.path.expanduser("~/pill_tests")


def _stopped(e):
    return "stopped at your request" in str(e)


class LabelSteps:
    def __init__(self, c, s, move, say, reader, rotate="cw", tag=""):
        self.c, self.s, self.move, self.say, self.reader = c, s, move, say, reader
        self.rotate = rotate
        self.tag = tag or time.strftime("%Y%m%d_%H%M%S")
        self.n = 0
        self.reset_reference()

    def reset_reference(self):
        """The verify pose the offsets are measured from (call again after a regrasp)."""
        self.yaw0 = self.c.joints.get("joint_wrist_yaw")
        self.lift0 = self.c.joints.get("joint_lift")
        if self.yaw0 is None or self.lift0 is None:
            raise RuntimeError("no wrist yaw / lift in /joint_states")

    def aim(self):
        try:
            head = self.c.position("link_head_tilt")
            grip = self.c.position("link_grasp_center")
        except Exception as e:
            self.say(f"   (couldn't re-aim the head: {e})")
            return
        pan, tilt = look_at(head, grip)
        self.move("look at the claw", joint_head_pan=pan, joint_head_tilt=tilt)

    def _moved(self, label, **joints):
        try:
            self.move(label, **joints)
            self.aim()
            return True
        except RuntimeError as e:
            if _stopped(e):
                raise
            self.say(f"   {label}: {e}")
            return False

    def set_yaw(self, offset):
        return self._moved(f"turn the wrist to {offset:+.2f} rad from the verify pose",
                           joint_wrist_yaw=self.yaw0 + offset)

    def set_lift(self, offset):
        return self._moved(f"move the lift {offset * 100:+.0f} cm from the verify pose", joint_lift=self.lift0 + offset)

    def read_view(self):
        s = self.s
        t = time.time() + s["label_settle_s"]
        time.sleep(s["label_settle_s"])
        msg = self.c.wait_for("head_color", 3.0, newer_than=t)
        if msg is None:
            raise RuntimeError("no head image on /camera/color/image_raw")
        det = self.c.wait_for("detections", 3.0, newer_than=t)
        dets = [(d.name, d.confidence, (d.x_min, d.y_min, d.x_max, d.y_max)) for d in det.detections] if det else []
        box = lr.pick_box(dets)
        raw = image_to_numpy(msg)
        if raw.ndim == 2:
            raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        v = lr.prepare_view(raw, box[2] if box else None, self.rotate, s["label_unwarp"], s["label_unwarp_max_deg"],
                            s["label_crop_pad"])
        self.n += 1
        os.makedirs(OUT, exist_ok=True)
        path = os.path.join(OUT, f"label_{self.tag}_v{self.n}.jpg")
        cv2.imwrite(path, v["image"])
        if not box:
            self.say("   (no bottle box in this view; reading the whole image)")
        r = self.reader.read(v["image"])
        r.update(_image=path, _axis=v["axis"], _box=box[0] if box else None)
        return r


def copy_plan(plan, **changes):
    """Editable copy of a PlanGrasp response (ROS messages have get_fields_and_field_types())."""
    names = plan.get_fields_and_field_types() if hasattr(plan, "get_fields_and_field_types") else vars(plan)
    d = {k: getattr(plan, k) for k in names}
    d.update(changes)
    return SimpleNamespace(**d)


def regrasp_plan(plan, orientation, s):
    """Where to grab the second time. Upright/tilted: higher on the bottle (the first grasp is low). Lying: the claw
    closes across the bottle with its axis along the arm, so move along the axis instead (reach in less)."""
    if orientation == "lying":
        d = s["regrasp_shift_m"]
        return copy_plan(plan, arm_grasp=max(0.0, plan.arm_grasp - d), arm_pregrasp=max(0.0, plan.arm_pregrasp - d))
    dz = s["regrasp_dz_m"]
    return copy_plan(plan, lift=plan.lift + dz, lift_clear=plan.lift_clear + dz)


REGRASP_SETTINGS = {
    "regrasp_dz_m": 0.025,       # upright: grab this much higher the second time
    "regrasp_shift_m": 0.02,     # lying: reach in this much less the second time
}


def make_regrasp(c, current, orientation, s, move, say, hold_up, steps, log):
    """regrasp() for run_label_sequence: place, grab again with regrasp_plan, back to the verify pose.
    current: {"plan": the plan the bottle is held with}; updated to the new plan, so the caller places the bottle
    with the right one afterwards."""
    from . import grasp

    def regrasp():
        say("   Regrasp: putting the bottle down and grabbing it at a different spot.")
        plan = current["plan"]
        grasp.place(c, plan, s, move)
        p2 = regrasp_plan(plan, orientation, s)
        current["plan"] = p2
        grasp.approach(c, p2, s, move, say, log)
        outcome = grasp.close_and_check(c, p2, s, move, say, log, tag="_regrasp")
        log["regrasp_outcome"] = outcome
        if outcome not in ("held", "slipping"):
            grasp.release(c, p2, s, move)
            return False
        hold_up()
        steps.reset_reference()
        return True
    return regrasp
