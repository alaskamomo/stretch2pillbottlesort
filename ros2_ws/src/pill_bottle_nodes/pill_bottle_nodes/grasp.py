# grasp.py: the grasp itself, shared by task_manager and test_grasp so a fix in one is a fix in both.
#   approach -> refine with claw camera -> close -> check it's holding -> small lift to test the grip
import os
import time

# Defaults; task_manager and test_grasp expose each as a ROS parameter.
GRASP_SETTINGS = {
    "gripper_open": 0.08,         # aperture (m) while approaching
    "gripper_squeeze": -0.01,     # aperture target when closing; below 0 squeezes harder on the bottle
    "extra_reach": 0.01,          # push the arm this much past the planned grasp so the bottle sits deeper in the claw
    "empty_finger_below": 0.03,   # finger (rad) below this after closing = closed on nothing
    "lift_test": 0.05,            # lift this much after closing to test the grip
    "hold_test_s": 3.0,           # watch the finger this long while lifted
    "slip_tol": 0.02,             # finger change (rad) during the hold test that counts as slipping
    "use_gripper_refine": True,
    "gripcam_sign": 1.0,          # flip to -1.0 if refining moves the base the wrong way
    "max_refine": 0.03,           # largest base correction from the claw camera (m)
    "arm_max": 0.50,
    "lift_max": 1.05,
}

OUT = os.path.expanduser("~/pill_tests")


def approach(c, plan, s, move, say, log):
    """Bring the open claw to the bottle's grasp height, 10 cm short of it."""
    move("retract the arm", wrist_extension=0.0)
    move("raise to travel height", joint_lift=plan.lift_clear)
    move("point the wrist out and open the claw", joint_wrist_yaw=plan.wrist_yaw, gripper_aperture=s["gripper_open"])
    if plan.base_translate:
        move(f"line up the base ({plan.base_translate:+.3f} m)", translate_mobile_base=plan.base_translate)
    move("extend to pregrasp", wrist_extension=plan.arm_pregrasp)
    move("lower to grasp height", joint_lift=plan.lift)
    log.update(plan_lift=plan.lift, plan_arm=plan.arm_grasp, plan_base=plan.base_translate)


def refine(c, s, move, say, log):
    """Use the claw camera to correct the base by a few cm before the final reach."""
    gv = c.wait_for("gripper_view", 3.0)
    log["claw_seen"] = bool(gv and gv.seen)
    if gv is None or not gv.seen:
        say("Claw camera: no bottle seen; keeping the head-camera plan.")
        return
    log.update(claw_orientation=gv.orientation, claw_offset_x=round(gv.offset_x_m, 4),
               claw_offset_y=round(gv.offset_y_m, 4), claw_conf=round(gv.confidence, 3))
    say(f"Claw camera: {gv.orientation}, {gv.offset_x_m * 100:+.1f} cm right, {gv.offset_y_m * 100:+.1f} cm down "
        f"(conf {gv.confidence:.2f})")
    if not s["use_gripper_refine"]:
        return
    if abs(gv.offset_x_m) > 3 * s["max_refine"]:
        say("   Offset too large to trust; not correcting.")
        return
    corr = max(-s["max_refine"], min(s["max_refine"], s["gripcam_sign"] * gv.offset_x_m))
    log["refine_base"] = round(corr, 4)
    if abs(corr) > 0.005:
        move(f"correct the base {corr * 100:+.1f} cm", translate_mobile_base=corr)


def close_and_check(c, plan, s, move, say, log, tag=""):
    """Final reach, close, and test the grip with a small lift. Returns 'held', 'slipping', 'dropped' or 'missed'."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    if c.snapshot("claw_image", os.path.join(OUT, f"{stamp}{tag}_claw_before_close.jpg")):
        log["claw_image"] = f"{stamp}{tag}_claw_before_close.jpg"
    arm = min(plan.arm_grasp + s["extra_reach"], s["arm_max"])
    move(f"reach in to {arm:.3f} m", wrist_extension=arm)
    move("close the claw", gripper_aperture=s["gripper_squeeze"])
    time.sleep(0.5)
    f_closed, e_closed = c.finger()
    log.update(finger_closed=_r(f_closed), effort_closed=_r(e_closed))
    say(f"   closed: finger {_fmt(f_closed)} rad, effort {_fmt(e_closed)}")
    if f_closed is not None and f_closed < s["empty_finger_below"]:
        say("   -> MISSED: the claw closed on nothing.")
        return "missed"

    lift_to = min(plan.lift + s["lift_test"], s["lift_max"])
    move(f"lift {s['lift_test'] * 100:.0f} cm to test the grip", joint_lift=lift_to)
    readings = []
    end = time.time() + s["hold_test_s"]
    while time.time() < end:
        readings.append(c.finger()[0])
        time.sleep(0.1)
    readings = [r for r in readings if r is not None]
    f_end = readings[-1] if readings else None
    drift = (f_end - f_closed) if (f_end is not None and f_closed is not None) else None
    log.update(finger_after_lift=_r(f_end), finger_drift=_r(drift), effort_after_lift=_r(c.finger()[1]))
    say(f"   after lifting {s['hold_test_s']:.0f} s: finger {_fmt(f_end)} rad (change {_fmt(drift)})")
    if f_end is not None and f_end < s["empty_finger_below"]:
        say("   -> DROPPED: the fingers closed all the way while lifting.")
        return "dropped"
    if drift is not None and abs(drift) > s["slip_tol"]:
        say("   -> SLIPPING: the finger kept moving while lifted; grip is too weak or too shallow.")
        return "slipping"
    say("   -> HELD")
    return "held"


def release(c, plan, s, move):
    """Open and back away (used after a miss or a drop)."""
    move("open the claw", gripper_aperture=s["gripper_open"])
    move("retract the arm", wrist_extension=0.0)
    move("raise to travel height", joint_lift=plan.lift_clear)


def place(c, plan, s, move):
    """Set the bottle back where it was."""
    move("raise to travel height", joint_lift=plan.lift_clear)
    move("reach back out", wrist_extension=min(plan.arm_grasp + s["extra_reach"], s["arm_max"]))
    move("lower onto the surface", joint_lift=plan.lift + 0.01)
    move("open the claw", gripper_aperture=s["gripper_open"])
    move("retract the arm", wrist_extension=0.0)
    move("raise to travel height", joint_lift=plan.lift_clear)


def _r(v):
    return None if v is None else round(float(v), 4)


def _fmt(v):
    return "?" if v is None else f"{v:+.3f}"
