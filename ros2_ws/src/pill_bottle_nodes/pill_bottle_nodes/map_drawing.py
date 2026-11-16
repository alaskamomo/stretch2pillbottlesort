# map_drawing.py: top-down picture of what the robot remembers (T18 / T15 report, T10 video corner).
import math

import numpy as np

COLORS = {"bottle": (40, 160, 230), "in_bin": (60, 170, 60), "bin": (150, 160, 30), "obstacle": (60, 60, 220),
          "robot": (180, 80, 120), "path": (180, 80, 120), "grid": (215, 215, 215), "text": (40, 40, 40)}


def draw_map(memory, robot_pose=None, path=(), size=600, margin_m=0.8, title=""):
    """BGR image (size x size) of the bottles, bin, obstacles, the robot and the path it drove (odom xy list)."""
    import cv2
    pts = [it["xyz"][:2] for it in memory.items] + [p[:2] for p in path] + ([robot_pose[:2]] if robot_pose else [])
    pts = pts or [(0.0, 0.0)]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0) + 2 * margin_m
    scale = size / span
    img = np.full((size, size, 3), 255, np.uint8)

    def px(x, y):          # odom x to the right, y up
        return int(round((x - cx) * scale + size / 2)), int(round(size / 2 - (y - cy) * scale))

    step = 0.5 if span < 6 else 1.0
    g = math.floor((cx - span / 2) / step) * step
    while g < cx + span / 2:
        cv2.line(img, px(g, cy - span), px(g, cy + span), COLORS["grid"], 1)
        g += step
    g = math.floor((cy - span / 2) / step) * step
    while g < cy + span / 2:
        cv2.line(img, px(cx - span, g), px(cx + span, g), COLORS["grid"], 1)
        g += step
    for a, b in zip(path, path[1:]):
        cv2.line(img, px(*a[:2]), px(*b[:2]), COLORS["path"], 2)
    for it in memory.items:
        x, y = it["xyz"][:2]
        if it["type"] == "obstacle":
            cv2.circle(img, px(x, y), max(3, int(it.get("radius", 0.1) * scale)), COLORS["obstacle"], -1)
            label = it["id"].replace("obstacle_", "obs ")
        elif it["type"] == "bin":
            r = int(memory.s.get("bin_radius_m", 0.15) * scale)
            cv2.rectangle(img, (px(x, y)[0] - r, px(x, y)[1] - r), (px(x, y)[0] + r, px(x, y)[1] + r), COLORS["bin"], -1)
            label = "bin"
        else:
            color = COLORS["in_bin"] if it.get("status") == "in_bin" else COLORS["bottle"]
            cv2.rectangle(img, (px(x, y)[0] - 4, px(x, y)[1] - 7), (px(x, y)[0] + 4, px(x, y)[1] + 7), color, -1)
            label = f"{it['id'].replace('bottle_', 'b')} {it.get('status', '')}"
        cv2.putText(img, label, (px(x, y)[0] + 8, px(x, y)[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["text"], 1,
                    cv2.LINE_AA)
    if robot_pose:
        x, y, yaw = robot_pose
        cv2.circle(img, px(x, y), int(0.17 * scale), COLORS["robot"], 2)
        cv2.line(img, px(x, y), px(x + 0.3 * math.cos(yaw), y + 0.3 * math.sin(yaw)), COLORS["robot"], 2)
    cv2.putText(img, f"{step:g} m grid  {title}", (8, size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COLORS["text"], 1,
                cv2.LINE_AA)
    return img
