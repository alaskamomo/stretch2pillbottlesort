# grasp.py: the grasp itself, shared by task_manager and test_grasp so a fix in one is a fix in both.
#   approach -> refine with claw camera -> close -> check it's holding -> small lift to test the grip
import os
import time

# Defaults; task_manager and test_grasp expose each as a ROS parameter.
GRASP_SETTINGS = {
    "gripper_open": 0.10,         # aperture (m) while approaching: wide, so the middle of the claw clears the bottle
    "gripper_squeeze": -0.01,     # aperture target when closing; below 0 squeezes harder on the bottle
    # Stretch's fingers are flexible: the finger POSITION is the same empty or holding (test_gripper showed -0.028
    # both times). Only the motor EFFORT tells them apart (about -0.1 empty, -6.6 on the bottle).
    "grip_effort_min": 3.0,       # |effort| at or above this after closing = holding something (test_gripper prints it)
    "effort_drop_frac": 0.5,      # |effort| falling below this fraction of the closing effort while lifted = slipping
    "lift_test": 0.05,            # lift this much after closing to test the grip
    "hold_test_s": 3.0,           # watch the finger this long while lifted
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
    log.update(plan_lift=plan.lift, plan_arm=plan.arm_grasp, plan_base=plan.base_translate,
               grasp_depth=getattr(plan, "grasp_depth", None))


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
    arm = min(plan.arm_grasp, s["arm_max"])
    depth = getattr(plan, "grasp_depth", 0.0)
    move(f"reach in to {arm:.3f} m (bottle {depth * 100:.0f} cm past the end caps, mid-finger)", wrist_extension=arm)
    move("close the claw", gripper_aperture=s["gripper_squeeze"])
    time.sleep(0.5)
    f_closed, e_closed = c.finger()
    log.update(finger_closed=_r(f_closed), effort_closed=_r(e_closed))
    say(f"   closed: finger {_fmt(f_closed)} rad, effort {_fmt(e_closed)}")
    if e_closed is None:
        say("   (no gripper effort in /joint_states; can't judge the grip, assuming held)")
        return "held"
    grip = abs(e_closed)
    if grip < s["grip_effort_min"]:
        say(f"   -> MISSED: effort {grip:.1f} is below {s['grip_effort_min']:.1f}, so the claw closed on nothing.")
        return "missed"

    lift_to = min(plan.lift + s["lift_test"], s["lift_max"])
    move(f"lift {s['lift_test'] * 100:.0f} cm to test the grip", joint_lift=lift_to)
    efforts = []
    end = time.time() + s["hold_test_s"]
    while time.time() < end:
        e = c.finger()[1]
        if e is not None:
            efforts.append(abs(e))
        time.sleep(0.1)
    tail = efforts[-5:] if efforts else [grip]
    e_end = sum(tail) / len(tail)
    log.update(finger_after_lift=_r(c.finger()[0]), effort_after_lift=_r(e_end),
               effort_min_while_lifted=_r(min(efforts) if efforts else None))
    say(f"   after lifting {s['hold_test_s']:.0f} s: effort {e_end:.1f} (was {grip:.1f} when it closed)")
    if e_end < s["grip_effort_min"]:
        say("   -> DROPPED: the effort fell to the empty level while lifting.")
        return "dropped"
    if e_end < s["effort_drop_frac"] * grip:
        say("   -> SLIPPING: the grip loosened while lifted; the bottle is shifting in the claw.")
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
    move("reach back out", wrist_extension=min(plan.arm_grasp, s["arm_max"]))
    move("lower onto the surface", joint_lift=plan.lift + 0.01)
    move("open the claw", gripper_aperture=s["gripper_open"])
    move("retract the arm", wrist_extension=0.0)
    move("raise to travel height", joint_lift=plan.lift_clear)


def _r(v):
    return None if v is None else round(float(v), 4)


def _fmt(v):
    return "?" if v is None else f"{v:+.3f}"
