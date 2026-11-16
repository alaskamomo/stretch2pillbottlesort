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
    need = {"head color": 10, "head depth": 10, "head info": 10, "claw camera": 2}   # gripper_vision runs at 2 Hz
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
        _verdict(valid > 30, f"{valid:.0f}% of depth pixels valid in the whole view (want > 30%; "
                             "far walls, windows and shiny floors read as 0. test_distance checks the bottle itself)")
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
             ("point the wrist out and open the claw", dict(joint_wrist_yaw=0.0, gripper_aperture=0.10))]
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
            _verdict(abs(gap - 0.10) <= 0.02, f"end caps are {gap * 100:.1f} cm from the bottle center (want 10 +- 2 cm)")
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
    """Calibrate and test the claw by itself: empty close, close on the bottle, and a hold test.

    Run with stage:=gripper (only motion_node is needed). The arm and base don't move.
    Stretch's fingers bend around the bottle, so the finger POSITION reads the same empty or holding;
    this measures the motor EFFORT instead and prints the grip_effort_min to use.
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
        if f is None or e is None:
            raise RuntimeError("no joint_gripper_finger_left position/effort in /joint_states")
        print(f"   {label}: {msg} after {time.time() - t0:.1f} s -> finger {f:+.3f} rad, effort {e:+.2f}")
        return f, abs(e)

    try:
        print(f"Settings: open {op} m, squeeze {sq} m. The arm and base do NOT move.")
        c.move(gripper_aperture=op)
        time.sleep(0.5)
        print(f"1) Open: finger {c.finger()[0]:+.3f} rad")
        if not ask("2) Close with NOTHING in the claw?"):
            return
        _, e_empty = close_read("empty close")
        c.move(gripper_aperture=op)
        if not ask("3) Hold the bottle upright between the finger pads (pads at its middle). Close?"):
            return
        _, e_bottle = close_read("close on bottle")
        _verdict(e_bottle > max(3 * e_empty, e_empty + 2.0),
                 f"effort on the bottle {e_bottle:.2f} vs empty {e_empty:.2f} (want clearly higher)")
        thr = round((e_empty + e_bottle) / 2, 2)
        print(f"   -> use grip_effort_min:={thr} (halfway between empty {e_empty:.2f} and bottle {e_bottle:.2f})")

        if ask(f"4) Hold test: let go of the bottle (keep a hand under it). Watch for {hold_s:.0f} s?"):
            path = _out(f"gripper_hold_{time.strftime('%Y%m%d_%H%M%S')}.csv")
            rows, t_end = [], time.time() + hold_s
            while time.time() < t_end:
                f, e = c.finger()
                rows.append((round(hold_s - (t_end - time.time()), 2), f, e))
                time.sleep(0.1)
            with open(path, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["t_s", "finger_rad", "effort"])
                w.writerows(rows)
            es = [abs(r[2]) for r in rows if r[2] is not None]
            e_end = sum(es[-5:]) / len(es[-5:]) if es else 0.0
            _verdict(e_end >= thr, f"effort after {hold_s:.0f} s {e_end:.2f} (threshold {thr})")
            _verdict(e_end >= 0.5 * e_bottle, f"effort kept {e_end / e_bottle * 100 if e_bottle else 0:.0f}% of "
                                              "the closing effort (want >= 50%)")
            held = ask("   Is the bottle still in the claw right now?")
            print(f"   saved {path}")
            if held and e_end >= thr:
                print("   Grip works on the bottle. Next: test_grasp with -p grip_effort_min:=" + str(thr))
            elif held:
                print("   The bottle is held but the effort sagged. Use a lower grip_effort_min, or a harder squeeze.")
            else:
                print("   It dropped: try a harder squeeze (-p gripper_squeeze:=-0.03) and grip lower on the bottle.")
        if ask("5) Open the claw (catch the bottle)?"):
            c.move(gripper_aperture=op)
    finally:
        c.close()
        rclpy.try_shutdown()


# ---------------- the grasp (the gripper problem) ----------------

def test_grasp():
    """Run ONLY the grasp on a real bottle: approach, claw-camera refine, close, grip test, then put it back.

    Run with stage:=full. Asks before every move unless -p yes:=true.
    Each run adds a line to ~/pill_tests/grasp_log.jsonl and saves the claw image from just before closing,
    so you can compare settings, e.g. -p gripper_squeeze:=-0.03, or change how deep the claw goes while it runs:
        ros2 param set /grasp_planner grasp_depth 0.06
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
                 "and a deeper grasp (ros2 param set /grasp_planner grasp_depth 0.06)."),
    "dropped": ("It closed on the bottle but lost it when lifting. Check the claw image: if the bottle is near the "
                "end caps, grasp deeper (ros2 param set /grasp_planner grasp_depth 0.06); if it's too high or low, "
                "check test_distance's z."),
    "missed": ("The claw closed on nothing. Look at the claw image from just before closing: if the bottle is off to "
               "one side, check gripcam_sign / test_gripper_vision; if the claw stopped short, raise grasp_depth; if it's not "
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
            key = (s.get("gripper_squeeze"), r.get("grasp_depth"), r.get("orientation"))
            groups[key].append(r.get("outcome"))
    print(f"{'squeeze':>8} {'depth':>6} {'orientation':>11}  runs  held  slipping  dropped  missed")
    for (sq, er, ori), outs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        n = len(outs)
        print(f"{sq!s:>8} {er!s:>6} {ori!s:>11}  {n:4d}  {outs.count('held'):4d}  {outs.count('slipping'):8d}  "
              f"{outs.count('dropped'):7d}  {outs.count('missed'):6d}")


# ---------------- T16: label reading at the head camera ----------------

def _results_logger(repo_testing_dir, test_id, name):
    """testing/test_logger.TestLogger from the repo (CSV for summarize_results.py), or None if it isn't there."""
    import sys
    d = os.path.expanduser(repo_testing_dir)
    if d not in sys.path:
        sys.path.insert(0, d)
    try:
        from test_logger import TestLogger
        return TestLogger(test_id, name)
    except Exception as e:
        print(f"(not logging to the results CSV: {e}; set -p repo_testing_dir:=/path/to/stretch2pillbottlesort/testing)")
        return None


def _norm(text):
    import re
    return re.sub(r"[^A-Z0-9.]", "", str(text or "").upper())


def label_matches(fields, expected_drug, expected_strength):
    """Did the read get the drug and strength right? Ignores spaces, case and punctuation; an expected value
    left empty isn't checked."""
    drug, strength = _norm(fields.get("drug_name")), _norm(fields.get("strength"))
    ok_drug = not expected_drug or (bool(drug) and (_norm(expected_drug) in drug or drug in _norm(expected_drug)))
    ok_strength = not expected_strength or strength == _norm(expected_strength)
    return ok_drug and ok_strength


LABEL_CONDITIONS = ["label_facing_camera", "label_facing_away", "label_under_fingers", "lying_in_claw", "glare"]


def test_label():
    """T16: read the label with the bottle held at the head camera (verify pose).

    Runs the whole fallback ladder: wrist turns -> lift change -> regrasp -> give up. Each trial is logged to
    ~/stretch_test_results/T16.csv (summarize_results.py) and ~/pill_tests/label_log.jsonl, and every view's
    flattened image is saved to ~/pill_tests/label_*.jpg.

      ros2 launch pill_bottle_nodes nodes.launch.py stage:=label        (stage:=full for mode:=grasped)
      export GEMINI_API_KEY=...                                         (else EasyOCR only)
      ros2 run pill_bottle_nodes test_label --ros-args -p expected_drug:=FUNKITOL -p expected_strength:=200MG

    mode:=handed (default): you put the bottle in the open claw; the "regrasp" asks you to re-seat it higher.
                            Tests label reading without depending on the grasp.
    mode:=grasped:          the robot finds and grabs the bottle itself and regrasps on its own (needs stage:=full).
    """
    import json

    from . import grasp
    from . import label_reading as lr
    from .geometry import look_at
    from .label_steps import REGRASP_SETTINGS, LabelSteps, make_regrasp

    c = start("test_label")
    settings = dict(grasp.GRASP_SETTINGS, **lr.LABEL_SETTINGS, **REGRASP_SETTINGS)
    extra = {"yes": False, "mode": "handed", "trials": 5, "expected_drug": "", "expected_strength": "",
             "rotate": "cw", "hold_lift": 0.9, "hold_arm": 0.10,
             "repo_testing_dir": "~/stretch2pillbottlesort/testing"}
    for k, v in {**settings, **extra}.items():
        c.declare_parameter(k, v)
    s = {k: c.get_parameter(k).value for k in {**settings, **extra}}

    def move(label, **joints):
        if not s["yes"] and not ask(f"Next: {label} {joints}?"):
            raise RuntimeError("stopped at your request")
        ok, msg = c.move(**joints)
        if not ok:
            raise RuntimeError(f"{label}: {msg}")

    def hold_up():
        move("raise to the verify height", joint_lift=s["hold_lift"])
        move("bring the arm in", wrist_extension=s["hold_arm"])
        pan, tilt = look_at(c.position("link_head_tilt"), c.position("link_grasp_center"))
        move("look at the claw", joint_head_pan=pan, joint_head_tilt=tilt)

    try:
        if not c.wait_joints():
            print("No /joint_states. Start robot.launch.py first.")
            return
        exp_drug = s["expected_drug"] or input("Drug name printed on the test bottle (Enter = don't check): ").strip()
        exp_strength = s["expected_strength"] or input("Strength on the label, e.g. 200MG (Enter = don't check): ").strip()
        reader = lr.build_reader(s, print)
        results = _results_logger(s["repo_testing_dir"], "T16", "Label reading at the head camera")
        print(f"\nConditions to cover: {', '.join(LABEL_CONDITIONS)}")
        for trial in range(1, int(s["trials"]) + 1):
            cond = input(f"\nTrial {trial}/{s['trials']}: condition [{LABEL_CONDITIONS[(trial - 1) % len(LABEL_CONDITIONS)]}]"
                         " (q = quit): ").strip() or LABEL_CONDITIONS[(trial - 1) % len(LABEL_CONDITIONS)]
            if cond.lower() == "q":
                break
            log = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "trial": trial, "condition": cond, "mode": s["mode"],
                   "unwarp": s["label_unwarp"]}
            current, orientation = {"plan": None}, "upright"
            if s["mode"] == "grasped":
                st = c.wait_for("state", 8.0)
                plan = c.plan(st) if st else None
                if plan is None or plan.next_step != "reach":
                    print(f"  Can't grasp from here ({plan.message if plan else 'no bottle'}); skipping (not logged).")
                    continue
                orientation = st.orientation
                grasp.approach(c, plan, s, move, print, log)
                grasp.refine(c, s, move, print, log)
                outcome = grasp.close_and_check(c, plan, s, move, print, log, tag="_label")
                if outcome not in ("held", "slipping"):
                    print(f"  Grasp {outcome}; skipping this trial (not a label result).")
                    grasp.release(c, plan, s, move)
                    continue
                current["plan"] = plan
                hold_up()
            else:
                move("open the claw", gripper_aperture=s["gripper_open"])
                hold_up()
                input(f"  Put the bottle in the claw ({cond}), deep in the fingers, then press Enter...")
                move("close the claw", gripper_aperture=s["gripper_squeeze"])
                time.sleep(0.5)

            steps = LabelSteps(c, s, move, print, reader, rotate=s["rotate"], tag=f"{time.strftime('%Y%m%d_%H%M%S')}_t{trial}")
            if s["mode"] == "grasped":
                regrasp = make_regrasp(c, current, orientation, s, move, print, hold_up, steps, log)
            else:
                def regrasp():
                    move("open the claw", gripper_aperture=s["gripper_open"])
                    input("  REGRASP: re-seat the bottle about 2-3 cm higher in the claw, then press Enter...")
                    move("close the claw", gripper_aperture=s["gripper_squeeze"])
                    time.sleep(0.5)
                    steps.reset_reference()
                    return True
            t0 = time.time()
            try:
                res = lr.run_label_sequence(steps.read_view, steps.set_yaw, steps.set_lift, regrasp, s, print)
                if res["status"] == "regrasp_failed":
                    outcome = "DROP"
                elif res["status"] == "unreadable":
                    outcome = "NOT_FOUND"
                else:
                    outcome = "SUCCESS" if label_matches(res["fields"], exp_drug, exp_strength) else "MISS"
            except RuntimeError as e:
                if "stopped at your request" in str(e):
                    raise
                res, outcome = {"errors": [str(e)], "fields": {}}, "ERROR"
            dt = time.time() - t0
            metrics = {k: res.get(k) for k in ("views", "lift_adjusted", "regrasps", "sources")}
            metrics.update(read_s=round(dt, 1), drug=res["fields"].get("drug_name"),
                           strength=res["fields"].get("strength"), unwarp=s["label_unwarp"])
            print(f"  -> {outcome}: {res['fields']}  {metrics}")
            if res.get("errors"):
                print(f"     errors: {res['errors']}")
            log.update(outcome=outcome, result=res)
            os.makedirs(OUT, exist_ok=True)
            with open(os.path.join(OUT, "label_log.jsonl"), "a") as f:
                f.write(json.dumps(log, default=str) + "\n")
            if results:
                results.log(outcome, cond, metrics, notes="; ".join(res.get("errors") or []), duration_s=dt)
            if s["mode"] == "grasped" and outcome != "DROP":
                grasp.place(c, current["plan"], s, move)
            elif s["mode"] != "grasped":
                move("open the claw", gripper_aperture=s["gripper_open"])
                input("  Take the bottle out of the claw, then press Enter...")
        print("\nSummary: python3 testing/summarize_results.py --session latest   (T16)")
    except RuntimeError as e:
        print(f"Stopped: {e}")
    finally:
        c.close()
        rclpy.try_shutdown()


# ---------------- T17: orientation (depth fit vs vision model vs combined) ----------------

def test_orientation():
    """T17: is the bottle upright or lying? Compares the depth fit, the vision model and the combined answer.

      ros2 launch pill_bottle_nodes nodes.launch.py stage:=orientation orientation_model:=/path/orientation_cls.pt
      ros2 run pill_bottle_nodes test_orientation

    Each trial: you place the bottle and type how it really is (u / l / t); it watches 10 frames.
    SUCCESS = the combined orientation (what the grasp planner gets) matches. Logged to T17.csv.
    """
    import json
    from collections import Counter

    from .orientation import depth_label

    c = start("test_orientation")
    params = {"trials": 10, "frames": 10, "upright_max_deg": 30.0, "lying_min_deg": 60.0,
              "repo_testing_dir": "~/stretch2pillbottlesort/testing"}
    for k, v in params.items():
        c.declare_parameter(k, v)
    p = {k: c.get_parameter(k).value for k in params}
    truth_names = {"u": "upright", "l": "lying", "t": "tilted"}
    results = _results_logger(p["repo_testing_dir"], "T17", "Orientation: depth vs vision vs combined")
    tally = {"depth": [0, 0], "vision": [0, 0], "combined": [0, 0]}
    try:
        for trial in range(1, int(p["trials"]) + 1):
            a = input(f"\nTrial {trial}: place the bottle, then type how it really is "
                      "(u = upright, l = lying/knocked over, t = tilted, q = quit): ").strip().lower()
            if a.startswith("q"):
                break
            truth = truth_names.get(a[:1])
            if truth is None:
                print("  type u, l or t")
                continue
            depth_votes, fused_votes, vision_votes, confs, pts = [], [], [], [], []
            for _ in range(int(p["frames"])):
                st = c.wait_for("state", 3.0)
                if st is None:
                    continue
                depth_votes.append(depth_label(st.axis_tilt_deg, p["upright_max_deg"], p["lying_min_deg"]))
                fused_votes.append(st.orientation)
                pts.append(st.num_points)
                vm = c.wait_for("orientation_vision", 0.6, newer_than=time.time() - 1.0)
                if vm is not None:
                    v = json.loads(vm.data)
                    vision_votes.append(v.get("label"))
                    confs.append(v.get("conf", 0.0))
            if not fused_votes:
                print("  NOT_FOUND: no /pill_bottle/state. Is the bottle in view (test_detector, test_distance)?")
                if results:
                    results.log("NOT_FOUND", truth, {})
                continue
            maj = lambda votes: Counter(votes).most_common(1)[0][0] if votes else None
            got = {"depth": maj(depth_votes), "vision": maj(vision_votes), "combined": maj(fused_votes)}
            for k, v in got.items():
                if v is not None:
                    tally[k][0] += v == truth
                    tally[k][1] += 1
            outcome = "SUCCESS" if got["combined"] == truth else "MISS"
            metrics = {"predicted_orientation": got["combined"], "depth": got["depth"], "vision": got["vision"],
                       "confidence": round(statistics.mean(confs), 3) if confs else None,
                       "depth_points": int(statistics.mean(pts)), "frames": len(fused_votes)}
            print(f"  truth {truth}: depth {got['depth']}, vision {got['vision']}, combined {got['combined']} -> {outcome}")
            if results:
                results.log(outcome, truth, metrics)
        print("\nAccuracy so far (this session):")
        for k, (right, n) in tally.items():
            print(f"  {k:8s} {right}/{n}" + (f"  ({100 * right / n:.0f}%)" if n else ""))
        print("If vision beats depth on lying/elevated bottles, the combined answer should follow it there.")
    finally:
        c.close()
        rclpy.try_shutdown()
