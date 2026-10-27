"""
geometry.py - pure math for head aiming and base approach planning.

No ROS imports in this file on purpose, so it can be unit tested on a laptop
(python3 -m pytest test/) and used by the T15a planning tester without the robot.

Frames used:
  fixed frame (odom): does NOT move when the robot drives. The bottle is saved here.
  base_link: x forward, y left, z up. The Stretch arm extends toward -y (right side).
  camera optical frame: z is the viewing direction.
"""

import math
from dataclasses import dataclass, asdict


# ----------------------------------------------------------------------------
# Small vector / quaternion helpers (quaternions are (x, y, z, w) like ROS)
# ----------------------------------------------------------------------------

def wrap_angle(a):
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def quat_rotate(q, v):
    """Rotate vector v by quaternion q = (x, y, z, w)."""
    x, y, z, w = q
    vx, vy, vz = v
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def transform_point(trans, quat, p):
    """Apply a TF transform (translation, rotation) to point p."""
    r = quat_rotate(quat, p)
    return (r[0] + trans[0], r[1] + trans[1], r[2] + trans[2])


def yaw_from_quat(q):
    x, y, z, w = q
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def norm(v):
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


def dist(a, b):
    return norm(sub(a, b))


def yaw_pitch(v):
    """Yaw (around z) and pitch (up is positive) of a direction vector in a z-up frame."""
    return math.atan2(v[1], v[0]), math.atan2(v[2], math.hypot(v[0], v[1]))


def angle_between(a, b):
    na, nb = norm(a), norm(b)
    if na < 1e-9 or nb < 1e-9:
        return math.pi
    c = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (na * nb)
    return math.acos(max(-1.0, min(1.0, c)))


# ----------------------------------------------------------------------------
# Head aiming
# ----------------------------------------------------------------------------

@dataclass
class HeadLimits:
    # Stretch 2 head joint limits (radians). Check yours with:
    #   ros2 param get /stretch_driver ... or the URDF joint limits
    pan_min: float = -4.04
    pan_max: float = 1.73
    tilt_min: float = -1.53
    tilt_max: float = 0.79


def in_view(cam_pos, cam_dir, target, cone_deg, max_range):
    """True if target is inside a cone around the camera's viewing direction."""
    to_target = sub(target, cam_pos)
    if norm(to_target) > max_range:
        return False
    return math.degrees(angle_between(to_target, cam_dir)) <= cone_deg


def aim_head(cam_pos, cam_dir, target, pan, tilt, limits=None, pan_sign=1.0, tilt_sign=1.0):
    """
    Work out the head pan/tilt that points the camera at target.

    All vectors are in a z-up frame (base_link). cam_dir is the camera's viewing
    direction (optical z axis). Because this compares where the camera IS looking
    against where it SHOULD look, it doesn't care how the camera is mounted
    (the Stretch D435 is rotated 90 deg) and it corrects itself every loop.

    pan_sign / tilt_sign: +1 means a positive joint value turns the view left / up.
    If the head moves AWAY from the bottle on the robot, flip the sign in params.yaml.
    """
    limits = limits or HeadLimits()
    to_target = sub(target, cam_pos)
    t_yaw, t_pitch = yaw_pitch(to_target)
    c_yaw, c_pitch = yaw_pitch(cam_dir)

    yaw_err = wrap_angle(t_yaw - c_yaw)
    pitch_err = t_pitch - c_pitch

    pan_des = pan + pan_sign * yaw_err
    # The pan range is wider than 180 deg on each side, so try the other way around
    if pan_des > limits.pan_max and pan_des - 2 * math.pi >= limits.pan_min:
        pan_des -= 2 * math.pi
    elif pan_des < limits.pan_min and pan_des + 2 * math.pi <= limits.pan_max:
        pan_des += 2 * math.pi
    tilt_des = tilt + tilt_sign * pitch_err

    in_range = (limits.pan_min <= pan_des <= limits.pan_max
                and limits.tilt_min <= tilt_des <= limits.tilt_max)

    return {
        "pan_cmd": max(limits.pan_min, min(limits.pan_max, pan_des)),
        "tilt_cmd": max(limits.tilt_min, min(limits.tilt_max, tilt_des)),
        "in_range": in_range,
        "yaw_err": yaw_err,
        "pitch_err": pitch_err,
        "angle_err": angle_between(to_target, cam_dir),
    }


# ----------------------------------------------------------------------------
# Approach planning (where should the base go so the arm is in front of the bottle)
# ----------------------------------------------------------------------------

@dataclass
class ApproachConfig:
    # How far out the arm side (-y in base_link) the bottle should end up.
    # Calibrate: extend the arm halfway, then run
    #   ros2 run tf2_ros tf2_echo base_link link_grasp_center
    # and use -y as standoff_m and x as arm_x_offset_m.
    standoff_m: float = 0.50
    arm_x_offset_m: float = 0.0      # x position of the arm's line of extension in base_link
    reach_min_m: float = 0.30        # gripper -y with arm fully retracted
    reach_max_m: float = 0.80        # gripper -y with arm fully extended
    max_approach_m: float = 3.5      # refuse to drive farther than this
    lift_min_z: float = 0.05         # lowest bottle height the lift can grasp
    lift_max_z: float = 1.05         # highest bottle height the lift can grasp
    base_half_width_m: float = 0.17  # base center to its side
    base_swept_radius_m: float = 0.25  # base center to its farthest corner (rotating in place)
    table_edge_to_bottle_m: float = 0.15  # how far the bottle sits in from the table edge
    min_table_clearance_m: float = 0.10
    max_simple_drive_m: float = 0.50  # if already beside the bottle, just drive this far max
    max_reverse_m: float = 0.50      # allowed to back up this far instead of turning around
    align_tol_m: float = 0.02
    min_rotate_deg: float = 1.0
    min_drive_m: float = 0.01


def _apply_steps(pose, steps):
    """Simulate ROTATE / DRIVE steps from pose (x, y, yaw)."""
    x, y, yaw = pose
    for s in steps:
        if s["type"] == "ROTATE":
            yaw = wrap_angle(yaw + math.radians(s["deg"]))
        elif s["type"] == "DRIVE":
            x += s["m"] * math.cos(yaw)
            y += s["m"] * math.sin(yaw)
    return (x, y, yaw)


def bottle_in_base(bottle_xy, pose):
    """Bottle (x, y) in the frame of a base at pose (x, y, yaw)."""
    dx, dy = bottle_xy[0] - pose[0], bottle_xy[1] - pose[1]
    c, s = math.cos(pose[2]), math.sin(pose[2])
    return (c * dx + s * dy, -s * dx + c * dy)


def plan_approach(bottle, robot_pose, cfg=None):
    """
    Plan base motion so the arm ends up lined up with the bottle.

    bottle: (x, y, z) in the fixed frame. robot_pose: (x, y, yaw) in the same frame.
    Returns a dict with ok, reason, steps, and predicted results. Never moves anything.
    """
    cfg = cfg or ApproachConfig()
    bx, by, bz = bottle
    rx, ry, ryaw = robot_pose
    plan = {
        "ok": False, "reason": "", "steps": [], "kind": "",
        "bottle": (bx, by, bz), "robot": (rx, ry, ryaw),
        "start_distance_m": math.hypot(bx - rx, by - ry),
    }

    # ---- checks that make the bottle unreachable ----
    if bz < cfg.lift_min_z or bz > cfg.lift_max_z:
        plan["reason"] = (f"bottle height {bz:.2f} m outside lift range "
                          f"{cfg.lift_min_z:.2f}-{cfg.lift_max_z:.2f} m")
        return plan
    if plan["start_distance_m"] > cfg.max_approach_m:
        plan["reason"] = (f"bottle {plan['start_distance_m']:.2f} m away, "
                          f"max approach distance {cfg.max_approach_m:.2f} m")
        return plan

    # Standoff has to leave room to rotate in place without hitting the table
    needed = cfg.base_swept_radius_m + cfg.table_edge_to_bottle_m + cfg.min_table_clearance_m
    standoff = max(cfg.standoff_m, needed)
    if standoff > cfg.reach_max_m:
        plan["reason"] = (f"needs standoff {standoff:.2f} m for table clearance, "
                          f"but max arm reach is {cfg.reach_max_m:.2f} m")
        return plan
    plan["standoff_m"] = standoff

    steps = []
    lx, ly = bottle_in_base((bx, by), robot_pose)
    lateral = -ly                      # distance out the arm side
    along = lx - cfg.arm_x_offset_m    # how far ahead/behind the arm line the bottle is
    simple_clearance = lateral - cfg.base_half_width_m - cfg.table_edge_to_bottle_m

    if (cfg.reach_min_m <= lateral <= cfg.reach_max_m
            and abs(along) <= cfg.max_simple_drive_m
            and simple_clearance >= cfg.min_table_clearance_m):
        # Already beside the bottle: just slide forward/back along the heading
        plan["kind"] = "already beside bottle"
        if abs(along) >= cfg.min_drive_m:
            steps.append({"type": "DRIVE", "m": along, "why": "line the arm up with the bottle"})
        plan["predicted_table_clearance_m"] = simple_clearance
    else:
        # Stop 'standoff' short of the bottle on the line toward it, then turn so the
        # arm side (-y) faces it. Final heading = bearing to bottle + 90 deg.
        plan["kind"] = "rotate, drive, rotate"
        bearing = math.atan2(by - ry, bx - rx)
        final_yaw = wrap_angle(bearing + math.pi / 2)
        tx = bx - standoff * math.cos(bearing) - cfg.arm_x_offset_m * math.cos(final_yaw)
        ty = by - standoff * math.sin(bearing) - cfg.arm_x_offset_m * math.sin(final_yaw)
        d = math.hypot(tx - rx, ty - ry)

        heading = ryaw
        if d >= cfg.min_drive_m:
            to_target = math.atan2(ty - ry, tx - rx)
            rot1 = wrap_angle(to_target - ryaw)
            drive = d
            # Target is behind us and close: back up instead of turning around
            if abs(rot1) > math.pi / 2 and d <= cfg.max_reverse_m:
                rot1 = wrap_angle(to_target + math.pi - ryaw)
                drive = -d
            # skip a tiny turn only if skipping it moves the end point less than half a cm
            if abs(rot1) * abs(drive) > 0.005 or abs(math.degrees(rot1)) >= cfg.min_rotate_deg:
                steps.append({"type": "ROTATE", "deg": math.degrees(rot1),
                              "why": "face the target base pose"})
            heading = wrap_angle(ryaw + rot1)
            steps.append({"type": "DRIVE", "m": drive,
                          "why": "drive straight" if drive > 0 else "back up straight"})
        rot2 = wrap_angle(final_yaw - heading)
        if abs(rot2) * standoff > 0.005 or abs(math.degrees(rot2)) >= cfg.min_rotate_deg:
            steps.append({"type": "ROTATE", "deg": math.degrees(rot2),
                          "why": "turn so the arm side faces the bottle"})
        plan["target_pose"] = (tx, ty, final_yaw)
        plan["predicted_table_clearance_m"] = (
            standoff - cfg.base_swept_radius_m - cfg.table_edge_to_bottle_m)

    # ---- predicted result ----
    final = _apply_steps(robot_pose, steps)
    fx, fy = bottle_in_base((bx, by), final)
    plan["steps"] = steps
    plan["final_pose"] = final
    plan["predicted_align_error_m"] = fx - cfg.arm_x_offset_m
    plan["predicted_reach_m"] = -fy
    plan["total_rotation_deg"] = sum(abs(s["deg"]) for s in steps if s["type"] == "ROTATE")
    plan["total_drive_m"] = sum(abs(s["m"]) for s in steps if s["type"] == "DRIVE")
    plan["num_rotations"] = sum(1 for s in steps if s["type"] == "ROTATE")
    plan["num_drives"] = sum(1 for s in steps if s["type"] == "DRIVE")

    if not (cfg.reach_min_m <= plan["predicted_reach_m"] <= cfg.reach_max_m):
        plan["reason"] = f"predicted reach {plan['predicted_reach_m']:.2f} m outside arm range"
        return plan
    if abs(plan["predicted_align_error_m"]) > cfg.align_tol_m:
        plan["reason"] = f"predicted align error {plan['predicted_align_error_m'] * 100:.1f} cm"
        return plan
    plan["ok"] = True
    return plan


def format_plan(plan, frame="odom"):
    """Readable printout of a plan (same format as the T15a ticket)."""
    bx, by, bz = plan["bottle"]
    rx, ry, ryaw = plan["robot"]
    lines = [f"APPROACH PLAN for bottle at x={bx:.2f} y={by:.2f} z={bz:.2f} ({frame}), "
             f"robot at x={rx:.2f} y={ry:.2f} heading={math.degrees(ryaw):.1f} deg"]
    if not plan["ok"] and not plan["steps"]:
        lines.append(f"NO_PLAN: {plan['reason']}")
        return "\n".join(lines)
    if "target_pose" in plan:
        tx, ty, tyaw = plan["target_pose"]
        lines.append(f"Target base pose: x={tx:.2f} y={ty:.2f} heading={math.degrees(tyaw):.1f} deg "
                     f"(bottle {plan['standoff_m']:.2f} m out the arm side)")
    else:
        lines.append(f"Plan type: {plan['kind']}")
    if not plan["steps"]:
        lines.append("No motion needed: arm is already lined up with the bottle")
    for i, s in enumerate(plan["steps"], 1):
        if s["type"] == "ROTATE":
            lines.append(f"Step {i}  BASE_ROTATE  {s['deg']:+.1f} deg ({s['why']})")
        else:
            lines.append(f"Step {i}  BASE_DRIVE   {s['m']:+.2f} m ({s['why']})")
    if plan["steps"]:
        lines.append(f"Step {len(plan['steps']) + 1}  HEAD_TRACK   keep the bottle centered during the steps above")
    lines.append(
        f"Predicted final alignment: bottle {plan['predicted_reach_m']:.2f} m from base along the arm line, "
        f"{abs(plan['predicted_align_error_m']) * 100:.1f} cm off the arm line, "
        f"{plan['predicted_table_clearance_m'] * 100:.0f} cm table clearance")
    if not plan["ok"]:
        lines.append(f"NO_PLAN: {plan['reason']}")
    return "\n".join(lines)


def config_from_dict(d):
    """Build an ApproachConfig from a dict (e.g. ROS params), ignoring unknown keys."""
    valid = asdict(ApproachConfig()).keys()
    return ApproachConfig(**{k: float(v) for k, v in d.items() if k in valid})
