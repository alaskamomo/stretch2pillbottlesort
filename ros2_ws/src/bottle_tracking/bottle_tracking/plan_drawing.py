"""
plan_drawing.py - pictures of planned approaches for the T1 testers.

draw_plan()     top-down drawing: robot start (gray), path (dashed), robot end pose
                (green), the arm reaching out (orange) and the bottle (blue).
                Drawn in the robot's STARTING frame: forward is up, left is left.
compose()       camera picture + plan drawing side by side with a title bar.
stack()         all trials in one overview image.

Uses only numpy + OpenCV (OpenCV comes with ultralytics).
"""

import math

import numpy as np

from bottle_tracking import geometry as geo

GRAY, GREEN, ORANGE, BLUE, RED, BLACK, WHITE = (
    (150, 150, 150), (60, 170, 60), (0, 140, 255), (220, 120, 30), (40, 40, 220), (30, 30, 30), (255, 255, 255))
FOOT_X, FOOT_Y = 0.34, 0.33          # Stretch base footprint (m)


def _to_start_frame(pt, start):
    """World (x, y) -> robot START frame (x forward, y left)."""
    dx, dy = pt[0] - start[0], pt[1] - start[1]
    c, s = math.cos(start[2]), math.sin(start[2])
    return c * dx + s * dy, -s * dx + c * dy


def draw_plan(full, bottle_fixed, start_pose, arm, title, size=520):
    """Top-down picture of one plan (or of why there is no plan). Returns a BGR numpy image."""
    import cv2
    img = np.full((size, size, 3), 255, np.uint8)

    poses = [start_pose]
    if full is not None and full["base"].get("steps"):
        pose = start_pose
        for st in full["base"]["steps"]:
            pose = geo._apply_steps(pose, [st])
            poses.append(pose)
    pts = [_to_start_frame(p, start_pose) for p in poses]
    if bottle_fixed is not None:
        pts.append(_to_start_frame(bottle_fixed, start_pose))

    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    cx, cy = (max(xs) + min(xs)) / 2, (max(ys) + min(ys)) / 2
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0) + 1.0
    scale = (size - 60) / span

    def px(p):   # start-frame (x fwd, y left) -> pixel (x forward is UP, y left is LEFT)
        return int(size / 2 - (p[1] - cy) * scale), int(size / 2 + 20 - (p[0] - cx) * scale)

    # 0.5 m grid
    step = 0.5
    for k in range(-20, 21):
        v = k * step
        a, b = px((v, cy - span)), px((v, cy + span))
        cv2.line(img, a, b, (235, 235, 235), 1)
        a, b = px((cx - span, v)), px((cx + span, v))
        cv2.line(img, a, b, (235, 235, 235), 1)

    def robot(pose, color, thick):
        lx, ly = _to_start_frame(pose, start_pose)
        yaw = pose[2] - start_pose[2]
        c, s = math.cos(yaw), math.sin(yaw)
        corners = [(FOOT_X / 2, FOOT_Y / 2), (FOOT_X / 2, -FOOT_Y / 2), (-FOOT_X / 2, -FOOT_Y / 2),
                   (-FOOT_X / 2, FOOT_Y / 2)]
        poly = np.array([px((lx + c * a - s * b, ly + s * a + c * b)) for a, b in corners], np.int32)
        cv2.polylines(img, [poly], True, color, thick)
        tip = (lx + c * 0.3, ly + s * 0.3)
        cv2.arrowedLine(img, px((lx, ly)), px(tip), color, thick, tipLength=0.3)
        return lx, ly, yaw

    robot(start_pose, GRAY, 2)
    for a, b in zip(poses[:-1], poses[1:]):        # dashed path
        p0, p1 = _to_start_frame(a, start_pose), _to_start_frame(b, start_pose)
        n = max(1, int(math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / 0.08))
        for k in range(0, n, 2):
            q0 = (p0[0] + (p1[0] - p0[0]) * k / n, p0[1] + (p1[1] - p0[1]) * k / n)
            q1 = (p0[0] + (p1[0] - p0[0]) * (k + 1) / n, p0[1] + (p1[1] - p0[1]) * (k + 1) / n)
            cv2.line(img, px(q0), px(q1), BLACK, 2)

    ok = full is not None and full["ok"]
    if full is not None and full["base"].get("steps") or (full is not None and full["base"].get("ok")):
        end = poses[-1]
        lx, ly, yaw = robot(end, GREEN if ok else RED, 3)
        if bottle_fixed is not None:                   # arm reaching out the right side
            c, s = math.cos(yaw), math.sin(yaw)
            shoulder = (lx + c * arm.arm_x_offset_m + s * FOOT_Y / 2,
                        ly + s * arm.arm_x_offset_m - c * FOOT_Y / 2)
            cv2.line(img, px(shoulder), px(_to_start_frame(bottle_fixed, start_pose)), ORANGE, 4)

    if bottle_fixed is not None:
        b = px(_to_start_frame(bottle_fixed, start_pose))
        cv2.circle(img, b, max(6, int(0.03 * scale)), BLUE if ok else RED, -1)
        cv2.putText(img, "bottle", (b[0] + 9, b[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE if ok else RED, 1)

    cv2.rectangle(img, (0, 0), (size, 34), (245, 245, 245), -1)
    cv2.putText(img, title, (8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN if ok else RED, 2, cv2.LINE_AA)
    bar = min(int(0.5 * scale), size // 3)
    bar_m = 0.5 * bar / (0.5 * scale)
    cv2.line(img, (12, size - 14), (12 + bar, size - 14), BLACK, 2)
    cv2.putText(img, f"{bar_m:.2f} m", (16 + bar, size - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.45, BLACK, 1)
    cv2.putText(img, "gray = start, green = planned end, dashed = path, orange = arm", (12, size - 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (90, 90, 90), 1, cv2.LINE_AA)
    return img


def compose(camera_img, plan_img, header):
    """Camera picture (left) + plan drawing (right) under a header bar."""
    import cv2
    h = plan_img.shape[0]
    if camera_img is None:
        camera_img = np.full((h, int(h * 0.75), 3), 220, np.uint8)
        cv2.putText(camera_img, "no camera picture", (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLACK, 2)
    cam = cv2.resize(camera_img, (int(camera_img.shape[1] * h / camera_img.shape[0]), h))
    body = np.hstack([cam, np.full((h, 6, 3), 255, np.uint8), plan_img])
    bar = np.full((40, body.shape[1], 3), 40, np.uint8)
    cv2.putText(bar, header, (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2, cv2.LINE_AA)
    return np.vstack([bar, body])


def stack(images, max_width=1100):
    """All trial pictures one under another, same width."""
    import cv2
    out = []
    for im in images:
        w = min(max_width, im.shape[1])
        out.append(cv2.resize(im, (w, int(im.shape[0] * w / im.shape[1]))))
        out.append(np.full((10, w, 3), 255, np.uint8))
    width = max(o.shape[1] for o in out)
    out = [np.hstack([o, np.full((o.shape[0], width - o.shape[1], 3), 255, np.uint8)]) if o.shape[1] < width else o
           for o in out]
    return np.vstack(out)
