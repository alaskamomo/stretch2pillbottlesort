# geometry.py: the math the nodes share. No ROS imports, so it can be unit tested anywhere.
import math

import numpy as np


# ---------- image rotation (the head camera is mounted sideways) ----------

def rotate_image(img, rotate):
    """Rotate a raw camera frame so it looks upright, like the YOLO training images."""
    import cv2
    if rotate == "cw":
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if rotate == "ccw":
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotate == "180":
        return cv2.rotate(img, cv2.ROTATE_180)
    return img


def rotated_to_raw(u, v, raw_w, raw_h, rotate):
    """Map a pixel in the rotated (upright) image back to the raw camera image."""
    if rotate == "cw":        # raw (u, v) -> rotated (raw_h - 1 - v, u)
        return v, raw_h - 1 - u
    if rotate == "ccw":       # raw (u, v) -> rotated (v, raw_w - 1 - u)
        return raw_w - 1 - v, u
    if rotate == "180":
        return raw_w - 1 - u, raw_h - 1 - v
    return u, v


def box_rotated_to_raw(box, raw_w, raw_h, rotate):
    x1, y1, x2, y2 = box
    a = rotated_to_raw(x1, y1, raw_w, raw_h, rotate)
    b = rotated_to_raw(x2, y2, raw_w, raw_h, rotate)
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1]))


# ---------- depth -> 3D ----------

def box_points(depth_m, box, K, band=0.06, step=2):
    """3D points (camera optical frame) for the depth pixels inside a box that belong to the bottle.

    Keeps pixels within `band` meters of the box's median depth, which drops the table/wall behind it.
    """
    h, w = depth_m.shape
    x1, y1, x2, y2 = [int(round(c)) for c in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 3))
    vs, us = np.mgrid[y1:y2 + 1:step, x1:x2 + 1:step]
    z = depth_m[vs, us]
    ok = np.isfinite(z) & (z > 0.1) & (z < 6.0)
    if ok.sum() < 10:
        return np.zeros((0, 3))
    med = np.median(z[ok])
    ok &= np.abs(z - med) < band
    us, vs, z = us[ok].astype(float), vs[ok].astype(float), z[ok]
    fx, fy, cx, cy = K[0], K[4], K[2], K[5]
    return np.column_stack([(us - cx) * z / fx, (vs - cy) * z / fy, z])


def transform_points(T, pts):
    """Apply a 4x4 transform to Nx3 points."""
    return pts @ T[:3, :3].T + T[:3, 3]


def quat_to_matrix(tx, ty, tz, qx, qy, qz, qw):
    x, y, z, w = qx, qy, qz, qw
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [tx, ty, tz]
    return T


# ---------- orientation from the point cloud ----------

def classify_bottle(pts_base, radius=0.02, upright_max_deg=30.0, lying_min_deg=60.0):
    """Classify a bottle from its visible points in base_link (z up).

    Returns dict: center (x, y, z), orientation, tilt_deg, axis_yaw, surface_z.
    """
    c = pts_base.mean(axis=0)
    cov = np.cov((pts_base - c).T)
    evals, evecs = np.linalg.eigh(cov)          # ascending
    axis = evecs[:, 2]
    tilt = math.degrees(math.acos(min(1.0, abs(axis[2]))))
    if tilt < upright_max_deg:
        orientation = "upright"
    elif tilt > lying_min_deg:
        orientation = "lying"
    else:
        orientation = "tilted"
    axis_yaw = math.atan2(axis[1], axis[0])
    if orientation == "lying":
        surface_z = float(c[2] - radius)       # the camera sees the top of a lying bottle
    else:
        surface_z = float(np.percentile(pts_base[:, 2], 5))
    return {"center": c, "orientation": orientation, "tilt_deg": tilt,
            "axis_yaw": axis_yaw, "surface_z": surface_z}


# ---------- grasp planning ----------

# Defaults for the grasp_planner node's parameters.
PLANNER_SETTINGS = {
    # arm and lift limits used for planning (m)
    "arm_max": 0.50, "arm_min_grasp": 0.05, "arm_mid": 0.25,
    "lift_min": 0.05, "lift_max": 1.05,
    # your bottle (measure it)
    "bottle_length": 0.08, "bottle_radius": 0.02,
    # grasp geometry (same meaning as in full_run.py)
    "safe_clearance": 0.12, "pregrasp_gap": 0.10, "min_table_clearance": 0.035,
    "lying_min_clear": 0.02, "grasp_frac_upright": 0.45, "lying_align_deg": 35.0,
    # base moves
    "front_band": 0.15, "drive_tol": 0.02, "max_drive": 0.5,
}


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def grasp_center_at_yaw0(wrist, grasp):
    """Where the grasp center would be with the wrist straight out (yaw 0, pointing along -y).

    wrist, grasp: (x, y, z) of link_wrist_yaw and link_grasp_center in base_link right now.
    The grasp center swings around the vertical wrist axis, so only its horizontal reach matters.
    """
    reach = math.hypot(grasp[0] - wrist[0], grasp[1] - wrist[1])
    return (wrist[0], wrist[1] - reach, grasp[2])


def plan_grasp(bottle, orientation, surface_z, axis_yaw, ext, lift, g0, p):
    """Decide the next move toward the bottle.

    bottle: (x, y, z) in base_link. ext/lift: current wrist_extension and joint_lift (m).
    g0: grasp center at yaw 0 (grasp_center_at_yaw0). p: dict of settings (see grasp_planner node).
    Returns a dict matching PlanGrasp.Response fields.
    """
    bx, by, bz = bottle
    out = dict(success=True, next_step="reach", base_rotate=0.0, base_translate=0.0, lift_clear=0.0,
               lift=0.0, arm_pregrasp=0.0, arm_grasp=0.0, wrist_yaw=0.0, message="")
    side_dist = abs(g0[1]) + p["arm_mid"]      # how far to the side a bottle should be for a mid-length reach

    # 1) In front of the robot: drive until it's at the right distance, then turn so the arm faces it.
    if bx > 0 and abs(by) < p["front_band"]:
        d = bx - side_dist
        if abs(d) > p["drive_tol"]:
            d = max(-p["max_drive"], min(p["max_drive"], d))
            out.update(next_step="drive_base", base_translate=d,
                       message=f"Bottle is {bx:.2f} m ahead; driving {d:+.2f} m.")
            return out
        out.update(next_step="rotate_base", base_rotate=wrap(math.atan2(by, bx) + math.pi / 2),
                   message="Bottle is ahead at the right distance; turning so the arm faces it.")
        return out

    # 2) Not on the arm's side (-y): turn so it is.
    if by > -p["front_band"]:
        out.update(next_step="rotate_base", base_rotate=wrap(math.atan2(by, bx) + math.pi / 2),
                   message="Bottle isn't on the arm's side; turning toward it.")
        return out

    # 3) On the arm's side: how far must the arm reach?
    arm_grasp = ext + (g0[1] - by)
    if arm_grasp > p["arm_max"] or arm_grasp < p["arm_min_grasp"]:
        out.update(next_step="rotate_base", base_rotate=wrap(math.atan2(by, bx)),
                   message=(f"Arm would need {arm_grasp:.2f} m (range {p['arm_min_grasp']:.2f}-{p['arm_max']:.2f}); "
                            "turning to face the bottle so the base can drive."))
        return out

    # 4) Lying bottle: the fingers must close across it, so its axis should run along y.
    if orientation == "lying":
        turn = wrap(axis_yaw - math.pi / 2)
        if turn > math.pi / 2:
            turn -= math.pi
        if turn < -math.pi / 2:
            turn += math.pi
        if abs(turn) > math.radians(p["lying_align_deg"]):
            out.update(next_step="rotate_base", base_rotate=turn,
                       message=f"Lying bottle points the wrong way; turning {math.degrees(turn):+.0f} deg to line up.")
            return out

    # 5) Grasp height.
    if orientation == "upright":
        gz = max(surface_z + p["grasp_frac_upright"] * p["bottle_length"], surface_z + p["min_table_clearance"])
    elif orientation == "lying":
        gz = surface_z + max(p["bottle_radius"], p["lying_min_clear"])
    else:
        gz = max(bz, surface_z + p["min_table_clearance"])
    lift_grasp = lift + (gz - g0[2])
    lift_clear = lift_grasp + p["safe_clearance"]
    if not (p["lift_min"] <= lift_grasp <= p["lift_max"]):
        out.update(success=False, message=f"Grasp needs lift {lift_grasp:.2f} m, outside {p['lift_min']}-{p['lift_max']} m.")
        return out
    lift_clear = min(lift_clear, p["lift_max"])

    dx = bx - g0[0]
    if abs(dx) > p["max_drive"]:
        dx = max(-p["max_drive"], min(p["max_drive"], dx))
        out.update(next_step="drive_base", base_translate=dx,
                   message=f"Bottle is {bx - g0[0]:+.2f} m along the base; driving {dx:+.2f} m first.")
        return out

    out.update(base_translate=dx if abs(dx) > p["drive_tol"] else 0.0,
               lift_clear=lift_clear, lift=lift_grasp,
               arm_pregrasp=max(0.0, arm_grasp - p["pregrasp_gap"]), arm_grasp=arm_grasp,
               wrist_yaw=0.0,
               message=f"Reach: lift {lift_grasp:.3f} m, arm {arm_grasp:.3f} m, base {dx:+.3f} m.")
    return out


def look_at(head_xyz, target_xyz):
    """Head pan/tilt (rad) that point the head camera at a target, both in base_link."""
    dx, dy, dz = (target_xyz[i] - head_xyz[i] for i in range(3))
    pan = math.atan2(dy, dx)
    tilt = math.atan2(dz, math.hypot(dx, dy))   # negative = look down
    return pan, tilt
