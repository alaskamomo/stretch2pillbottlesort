# testers.py: one tester per component. Each prints PASS/CHECK lines so you can tell which part is at fault.
# Run them in order; each assumes the ones above it pass. Start the nodes each needs with
#   ros2 launch pill_bottle_nodes nodes.launch.py stage:=<tester name without test_>
import math
import os
import statistics
import time

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from .client import ask, start
from .common import depth_to_meters, image_to_numpy
from .geometry import rotate_image

OUT = os.path.expanduser("~/pill_tests")


def _out(name):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, name)


def _verdict(ok, text):
    print(f"{'PASS ' if ok else 'CHECK'}  {text}", flush=True)
    return ok


# ---------------- 1. cameras ----------------

class _CamProbe(Node):
    def __init__(self):
        super().__init__("test_cameras")
        self.declare_parameter("rotate", "cw")
        self.topics = {"head color": ("/camera/color/image_raw", Image),
                       "head depth": ("/camera/aligned_depth_to_color/image_raw", Image),
                       "head info": ("/camera/color/camera_info", CameraInfo),
                       "claw camera": ("/gripper_camera/image_raw", Image)}
        self.count = {k: 0 for k in self.topics}
        self.last = {}
        for k, (topic, typ) in self.topics.items():
            self.create_subscription(typ, topic, lambda m, k=k: self._got(k, m), 5)

    def _got(self, k, m):
        self.count[k] += 1
        self.last[k] = m


def test_cameras():
    """Are the head and claw cameras publishing, and is the head image upright after rotation?"""
    rclpy.init()
    n = _CamProbe()
    secs = 5.0
    end = time.time() + secs
    while time.time() < end:
        rclpy.spin_once(n, timeout_sec=0.1)
    rates = {k: c / secs for k, c in n.count.items()}
    need = {"head color": 10, "head depth": 10, "head info": 10, "claw camera": 5}
    allok = True
    for k, hz in rates.items():
        allok &= _verdict(hz >= need[k], f"{k:12s} {hz:5.1f} Hz on {n.topics[k][0]} (want >= {need[k]})")
    if "head color" in n.last:
        img = rotate_image(image_to_numpy(n.last["head color"]), n.get_parameter("rotate").value)
        cv2.imwrite(_out("head_upright.jpg"), img)
        print(f"       saved {_out('head_upright.jpg')}: the floor must be at the BOTTOM. If not, change -p rotate:=")
    if "head depth" in n.last and "head color" in n.last:
        d, c = n.last["head depth"], n.last["head color"]
        allok &= _verdict((d.width, d.height) == (c.width, c.height),
                          f"depth {d.width}x{d.height} matches color {c.width}x{c.height} (aligned)")
        dm = depth_to_meters(d)
        valid = float((dm > 0).mean()) * 100
        allok &= _verdict(valid > 50, f"{valid:.0f}% of depth pixels valid (want > 50%)")
    if "claw camera" in n.last:
        cv2.imwrite(_out("claw.jpg"), image_to_numpy(n.last["claw camera"]))
        print(f"       saved {_out('claw.jpg')}: the fingertips should be visible.")
    print("\nRESULT:", "cameras OK" if allok else "fix the CHECK lines before testing anything else")
    n.destroy_node()
    rclpy.try_shutdown()


# ---------------- 2. detector ----------------

def test_detector():
    """Does YOLO find the bottle in the head camera? Put the bottle in view first."""
    c = start("test_detector")
    frames, hits, confs = 10, 0, []
    print(f"Watching {frames} detector frames...")
    for i in range(frames):
        det = c.wait_for("detections", 5.0)
        if det is None:
            _verdict(False, "no /pill_bottle/detections. Is bottle_detector running?")
            break
        bottles = [d for d in det.detections if d.name == "pill_bottle"]
        best = max((d.confidence for d in bottles), default=0.0)
        others = sorted({d.name for d in det.detections} - {"pill_bottle"})
        print(f"  frame {i + 1:2d}: {'bottle ' + format(best, '.2f') if bottles else 'no bottle'}"
              f"{'   also: ' + ', '.join(others) if others else ''}")
        if best >= 0.4:
            hits += 1
            confs.append(best)
    ok = _verdict(hits >= 9, f"bottle found (conf >= 0.4) in {hits}/{frames} frames (want >= 9)")
    if confs:
        print(f"       confidence: mean {statistics.mean(confs):.2f}, min {min(confs):.2f}")
    print("       See the boxes: ros2 run rqt_image_view rqt_image_view /pill_bottle/debug_image")
    print("\nRESULT:", "detector OK at this distance" if ok else "detector is the problem (lighting, distance, model)")
    c.close()
    rclpy.try_shutdown()


# ---------------- 3. distance ----------------

def test_distance():
    """Where does the robot think the bottle is? Compare with a tape measure from the robot's base."""
    c = start("test_distance")
    xs, ys, zs, ors = [], [], [], []
    print("Reading /pill_bottle/state 10 times (x forward, y left, z up, from base_link)...")
    for i in range(10):
        st = c.wait_for("state", 6.0)
        if st is None:
            _verdict(False, "no /pill_bottle/state. Run test_detector first; then check bottle_localizer's log.")
            break
        xs.append(st.position.x)
        ys.append(st.position.y)
        zs.append(st.position.z)
        ors.append(st.orientation)
        print(f"  {i + 1:2d}: x {st.position.x:+.3f}  y {st.position.y:+.3f}  z {st.position.z:+.3f} m   "
              f"{st.orientation:7s} tilt {st.axis_tilt_deg:4.0f} deg  surface {st.surface_z:+.3f} m  "
              f"({st.num_points} pts)")
    if len(xs) >= 3:
        spread = max(statistics.pstdev(xs), statistics.pstdev(ys), statistics.pstdev(zs))
        _verdict(spread < 0.01, f"steady: largest spread {spread * 100:.1f} cm (want < 1 cm)")
        _verdict(len(set(ors)) == 1, f"orientation consistent: {sorted(set(ors))}")
        print(f"\n       MEAN  x {statistics.mean(xs):+.3f}  y {statistics.mean(ys):+.3f}  z {statistics.mean(zs):+.3f} m")
        print("       Measure from the center of the robot's base (between the wheels) to the bottle's center.")
        print("       PASS if within 2 cm up to 1 m away (4 cm at 1.5 m).")
    c.close()
    rclpy.try_shutdown()


# ---------------- 4. gripper vision ----------------

def test_gripper_vision():
    """Does the claw camera see the bottle, and does the offset follow it? Hold the bottle between the fingertips."""
    c = start("test_gripper_vision")
    print("Printing the claw camera's view every second for 20 s. Move the bottle 3 cm left/right and watch offset.")
    seen = 0
    for _ in range(20):
        gv = c.wait_for("gripper_view", 3.0)
        if gv is None:
            _verdict(False, "no /pill_bottle/gripper_view. Is gripper_vision running (and gripper_camera)?")
            break
        if gv.seen:
            seen += 1
            dist = f"{gv.distance_m * 100:.0f} cm" if gv.distance_m > 0 else "?"
            print(f"  SEEN {gv.orientation:8s} conf {gv.confidence:.2f}  center ({gv.center_u:.0f}, {gv.center_v:.0f})"
                  f"  offset {gv.offset_x_m * 100:+.1f} cm right, {gv.offset_y_m * 100:+.1f} cm down  dist {dist}")
        else:
            print("  no bottle")
        time.sleep(1.0)
    _verdict(seen >= 15, f"seen in {seen}/20 readings (want >= 15 while the bottle is in front of the claw)")
    print("       Offset sign: moving the bottle toward the robot's FRONT should change offset the same way each time.")
    print("       If refine moves the base the wrong way in test_arm, set gripcam_sign:=-1.0 on task_manager.")
    c.close()
    rclpy.try_shutdown()


# ---------------- 5. planner ----------------

def test_planner():
    """What would the robot do? Nothing moves."""
    c = start("test_planner")
    if not c.wait_joints():
        _verdict(False, "no /joint_states. Start the Stretch driver.")
        return
    st = c.wait_for("state", 8.0)
    if st is None:
        _verdict(False, "no /pill_bottle/state. Run test_distance first.")
        return
    plan = c.plan(st)
    if plan is None:
        _verdict(False, "grasp_planner isn't answering /pill_bottle/plan_grasp")
        return
    print(f"Bottle: x {st.position.x:+.3f} y {st.position.y:+.3f} z {st.position.z:+.3f} ({st.orientation})")
    print(f"Robot now: lift {c.joints['joint_lift']:.3f}  arm {c.joints['wrist_extension']:.3f}")
    print(f"Plan: {plan.next_step}  ->  {plan.message}")
    if plan.next_step == "reach":
        ok = (0.0 <= plan.arm_grasp <= 0.50) and (0.05 <= plan.lift <= 1.05) and abs(plan.wrist_yaw) <= 0.6
        print(f"  lift {plan.lift:.3f} (travel at {plan.lift_clear:.3f}), arm {plan.arm_pregrasp:.3f} -> "
              f"{plan.arm_grasp:.3f}, wrist {plan.wrist_yaw:.2f} rad, base {plan.base_translate:+.3f} m")
        _verdict(ok, "targets inside limits (lift 0.05-1.05 m, arm 0-0.50 m, wrist +-0.6 rad)")
    elif plan.next_step == "rotate_base":
        print(f"  turn {math.degrees(plan.base_rotate):+.0f} deg, then look again")
        _verdict(True, "out-of-reach bottle gives a base turn")
    elif plan.next_step == "drive_base":
        _verdict(abs(plan.base_translate) <= 0.5, f"drive {plan.base_translate:+.3f} m (want <= 0.5 m per move)")
    else:
        _verdict(plan.success, plan.message)
    c.close()
    rclpy.try_shutdown()


# ---------------- 6. arm ----------------

def test_arm():
    """Move the open claw to the pregrasp point (10 cm short of the bottle). Asks before every move."""
    c = start("test_arm")
    st = c.wait_for("state", 8.0)
    plan = c.plan(st) if st else None
    if plan is None or plan.next_step != "reach":
        print(f"Can't reach from here: {plan.message if plan else 'no bottle state / planner'}. Run test_planner.")
        c.close()
        rclpy.try_shutdown()
        return
    print(f"Plan: lift {plan.lift:.3f} (travel {plan.lift_clear:.3f}), arm {plan.arm_pregrasp:.3f}, "
          f"base {plan.base_translate:+.3f} m")
    print("Hand on the runstop. Each move is asked first; 'n' stops.")
    steps = [("retract the arm", dict(wrist_extension=0.0)),
             ("raise the lift to travel height", dict(joint_lift=plan.lift_clear)),
             ("point the wrist out and open the claw", dict(joint_wrist_yaw=0.0, gripper_aperture=0.08))]
    if plan.base_translate:
        steps.append((f"drive the base {plan.base_translate:+.3f} m", dict(translate_mobile_base=plan.base_translate)))
    steps += [("extend the arm to pregrasp", dict(wrist_extension=plan.arm_pregrasp)),
              ("lower to grasp height", dict(joint_lift=plan.lift))]
    for text, joints in steps:
        if not ask(f"Next: {text} {joints}?"):
            break
        ok, msg = c.move(**joints)
        print(f"   {'ok' if ok else 'FAILED'}: {msg}")
        if not ok:
            break
    else:
        try:
            g = c.position("link_grasp_center")
            gap = math.hypot(st.position.x - g[0], st.position.y - g[1])
            _verdict(abs(gap - 0.10) <= 0.02, f"claw is {gap * 100:.1f} cm from the bottle center (want 10 +- 2 cm)")
            _verdict(g[2] - st.surface_z >= 0.035, f"claw is {(g[2] - st.surface_z) * 100:.1f} cm above the surface "
                                                  "(want >= 3.5 cm)")
        except Exception as e:
            print(f"Couldn't read tf to check the gap: {e}")
        print("       Also measure the gap with a tape: it should match.")
    c.close()
    rclpy.try_shutdown()


# ---------------- 7. base ----------------

def test_base():
    """Turn in place 60 deg and back, then drive 0.20 m and back. Mark the floor to check."""
    c = start("test_base")
    if not c.wait_joints():
        _verdict(False, "no /joint_states. Start the Stretch driver.")
        return
    print("Clear 1 m around the robot, hand on the runstop. Tape a mark at the front wheel to measure.")
    moves = [("stow the wrist", dict(wrist_extension=0.0)), ("", dict(joint_wrist_yaw=3.0)),
             ("turn +60 deg (left)", dict(rotate_mobile_base=math.radians(60))),
             ("turn back -60 deg", dict(rotate_mobile_base=-math.radians(60))),
             ("drive forward 0.20 m", dict(translate_mobile_base=0.20)),
             ("drive back 0.20 m", dict(translate_mobile_base=-0.20))]
    for text, joints in moves:
        if text and not ask(f"Next: {text}?"):
            break
        t0 = time.time()
        ok, msg = c.move(**joints)
        print(f"   {'ok' if ok else 'FAILED'} in {time.time() - t0:.1f} s: {msg}")
        if not ok:
            break
    print("       PASS if the turn was within +-5 deg and the drive within +-3 cm of the tape marks,")
    print("       and the robot ended where it started.")
    c.close()
    rclpy.try_shutdown()


# ---------------- motion_node ----------------

def test_motion():
    """Small moves of every joint through motion_node; checks each one arrives. The arm and lift move 2 cm."""
    c = start("test_motion")
    if not c.wait_joints():
        _verdict(False, "no /joint_states. Start the Stretch driver.")
        return
    j = c.joints
    print("Hand on the runstop. Each joint moves a little and back.")
    ok, msg = c.move(joint_lift=2.0)
    _verdict(not ok and "outside" in msg, f"unsafe goal refused before moving: {msg}")
    lift, arm = j["joint_lift"], j["wrist_extension"]
    yaw, pan, tilt = j.get("joint_wrist_yaw", 3.0), j.get("joint_head_pan", 0.0), j.get("joint_head_tilt", -0.5)
    tests = [("joint_lift", min(lift + 0.02, 1.05), 0.01), ("joint_lift", lift, 0.01),
             ("wrist_extension", min(arm + 0.02, 0.50), 0.01), ("wrist_extension", arm, 0.01),
             ("joint_wrist_yaw", yaw - 0.2 if yaw > 0 else yaw + 0.2, 0.05), ("joint_wrist_yaw", yaw, 0.05),
             ("joint_head_pan", pan - 0.2, 0.05), ("joint_head_pan", pan, 0.05),
             ("joint_head_tilt", tilt, 0.05),
             ("gripper_aperture", 0.08, None), ("gripper_aperture", 0.0, None)]
    if not ask("Run the joint checks?"):
        c.close()
        rclpy.try_shutdown()
        return
    allok = True
    for name, target, tol in tests:
        t0 = time.time()
        ok, msg = c.move(**{name: target})
        dt = time.time() - t0
        time.sleep(0.3)
        actual = c.joints.get(name)
        if tol is None or actual is None:
            allok &= _verdict(ok, f"{name:18s} -> {target:+.3f} in {dt:4.1f} s: {msg}")
        else:
            err = abs(actual - target)
            allok &= _verdict(ok and err <= tol, f"{name:18s} -> {target:+.3f}, now {actual:+.3f} "
                                                  f"(error {err:.3f}, want <= {tol}) in {dt:4.1f} s")
    print("\nRESULT:", "motion_node and the driver OK" if allok else "check the driver log for the failed joint")
    c.close()
    rclpy.try_shutdown()


# ---------------- gripper (no arm motion) ----------------

def test_gripper():
    """Calibrate and test the claw by itself: empty close, close on the bottle, and a hold test in your hand.

    Run with stage:=gripper (only motion_node is needed).
    Prints the threshold to use for empty_finger_below and whether the grip is strong enough.
    """
    import csv
    c = start("test_gripper")
    c.declare_parameter("gripper_open", 0.08)
    c.declare_parameter("gripper_squeeze", -0.01)
    c.declare_parameter("hold_s", 8.0)
    op, sq, hold_s = (c.get_parameter(n).value for n in ("gripper_open", "gripper_squeeze", "hold_s"))
    if not c.wait_joints():
        _verdict(False, "no /joint_states. Start the Stretch driver.")
        return

    def close_read(label):
        t0 = time.time()
        ok, msg = c.move(gripper_aperture=sq)
        time.sleep(0.5)
        f, e = c.finger()
        if f is None:
            raise RuntimeError("no joint_gripper_finger_left in /joint_states")
        print(f"   {label}: {msg} after {time.time() - t0:.1f} s -> finger {f:+.3f} rad, effort {e if e is not None else '?'}")
        return f, e

    print(f"Settings: open {op} m, squeeze {sq} m. The arm and base do NOT move.")
    c.move(gripper_aperture=op)
    time.sleep(0.5)
    print(f"1) Open: finger {c.finger()[0]} rad")
    if not ask("2) Close with NOTHING in the claw?"):
        return
    f_empty, _ = close_read("empty close")
    c.move(gripper_aperture=op)
    if not ask("3) Hold the bottle upright between the finger pads (pads at its middle). Close?"):
        return
    f_bottle, e_bottle = close_read("close on bottle")
    gap = f_bottle - f_empty
    _verdict(gap > 0.05, f"bottle keeps the fingers {gap:.3f} rad more open than empty (want > 0.05)")
    thr = round((f_empty + f_bottle) / 2, 3)
    print(f"   -> use empty_finger_below:={thr} (halfway between empty {f_empty:+.3f} and bottle {f_bottle:+.3f})")

    if ask(f"4) Hold test: let go of the bottle (keep a hand under it). Watch for {hold_s:.0f} s?"):
        path = _out(f"gripper_hold_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        rows, end = [], time.time() + hold_s
        while time.time() < end:
            f, e = c.finger()
            rows.append((round(time.time() - end + hold_s, 2), f, e))
            time.sleep(0.1)
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["t_s", "finger_rad", "effort"])
            w.writerows(rows)
        fs = [r[1] for r in rows if r[1] is not None]
        drift = fs[-1] - fs[0] if fs else 0.0
        held = fs and fs[-1] > thr
        _verdict(bool(held), f"still holding after {hold_s:.0f} s (finger {fs[-1]:+.3f}, threshold {thr})")
        _verdict(abs(drift) <= 0.02, f"finger changed {drift:+.3f} rad while holding (want within +-0.02)")
        print(f"   saved {path}")
        if not held or abs(drift) > 0.02:
            print("   Grip is weak: try a harder squeeze, e.g. -p gripper_squeeze:=-0.03, and grip lower on the bottle.")
    if ask("5) Open the claw (catch the bottle)?"):
        c.move(gripper_aperture=op)
    c.close()
    rclpy.try_shutdown()


# ---------------- the grasp (the gripper problem) ----------------

def test_grasp():
    """Run ONLY the grasp on a real bottle: approach, claw-camera refine, close, grip test, then put it back.

    Run with stage:=full. Asks before every move unless -p yes:=true.
    Each run adds a line to ~/pill_tests/grasp_log.jsonl and saves the claw image from just before closing,
    so you can compare settings, e.g.  -p gripper_squeeze:=-0.03 -p extra_reach:=0.02
    """
    import json

    from . import grasp
    c = start("test_grasp")
    for k, v in grasp.GRASP_SETTINGS.items():
        c.declare_parameter(k, v)
    c.declare_parameter("yes", False)
    s = {k: c.get_parameter(k).value for k in grasp.GRASP_SETTINGS}
    yes = c.get_parameter("yes").value

    def move(label, **joints):
        if not yes and not ask(f"Next: {label} {joints}?"):
            raise RuntimeError("stopped at your request")
        ok, msg = c.move(**joints)
        print(f"   {'ok' if ok else 'FAILED'}: {msg}")
        if not ok:
            raise RuntimeError(msg)

    try:
        st = c.wait_for("state", 8.0)
        plan = c.plan(st) if st else None
        if plan is None or plan.next_step != "reach":
            print(f"Can't grasp from here: {plan.message if plan else 'no bottle state / planner'}. "
                  "Run test_planner and move the robot or bottle until the plan says reach.")
            return
        print(f"Bottle: {st.orientation}, x {st.position.x:+.3f} y {st.position.y:+.3f} z {st.position.z:+.3f}, "
              f"surface {st.surface_z:+.3f}")
        print(f"Settings: {s}")
        log = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "source": "test_grasp", "orientation": st.orientation,
               "bottle": [round(st.position.x, 3), round(st.position.y, 3), round(st.position.z, 3)], "settings": s}
        grasp.approach(c, plan, s, move, print, log)
        grasp.refine(c, s, move, print, log)
        outcome = grasp.close_and_check(c, plan, s, move, print, log)
        log["outcome"] = outcome
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, "grasp_log.jsonl"), "a") as f:
            f.write(json.dumps(log) + "\n")
        print(f"\nRESULT: {outcome.upper()}  (logged to {os.path.join(OUT, 'grasp_log.jsonl')})")
        print(_GRASP_HINTS[outcome])
        if outcome in ("missed", "dropped"):
            grasp.release(c, plan, s, move)
        else:
            grasp.place(c, plan, s, move)
    except RuntimeError as e:
        print(f"Stopped: {e}")
    finally:
        c.close()
        rclpy.try_shutdown()


_GRASP_HINTS = {
    "held": "Grip is good. Next: run task_manager with -p stop_after:=verify.",
    "slipping": ("The bottle moved in the claw while lifting. Try a harder squeeze (-p gripper_squeeze:=-0.03) "
                 "and a deeper reach (-p extra_reach:=0.02)."),
    "dropped": ("It closed on the bottle but lost it when lifting. Check the claw image: if the bottle is at the "
                "fingertips, add reach (-p extra_reach:=0.02); if it's too high or low, check test_distance's z."),
    "missed": ("The claw closed on nothing. Look at the claw image from just before closing: if the bottle is off to "
               "one side, check gripcam_sign / test_gripper_vision; if it's short, raise extra_reach; if it's not "
               "there at all, re-run test_distance."),
}


def grasp_report():
    """Summarize ~/pill_tests/grasp_log.jsonl: success rate per settings combination. No ROS needed."""
    import json
    from collections import defaultdict
    path = os.path.join(OUT, "grasp_log.jsonl")
    if not os.path.exists(path):
        print(f"No log yet at {path}. Run test_grasp first.")
        return
    groups = defaultdict(list)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            s = r.get("settings", {})
            key = (s.get("gripper_squeeze"), s.get("extra_reach"), r.get("orientation"))
            groups[key].append(r.get("outcome"))
    print(f"{'squeeze':>8} {'reach':>6} {'orientation':>11}  runs  held  slipping  dropped  missed")
    for (sq, er, ori), outs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        n = len(outs)
        print(f"{sq!s:>8} {er!s:>6} {ori!s:>11}  {n:4d}  {outs.count('held'):4d}  {outs.count('slipping'):8d}  "
              f"{outs.count('dropped'):7d}  {outs.count('missed'):6d}")
