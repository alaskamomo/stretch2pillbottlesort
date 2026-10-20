# Unit tests for the math in geometry.py (no robot or ROS needed): python3 -m pytest test/
import math

import numpy as np
import pytest

from pill_bottle_nodes.geometry import (box_points, box_rotated_to_raw, classify_bottle, grasp_center_at_yaw0,
                                        look_at, plan_grasp, rotate_image, rotated_to_raw)
from pill_bottle_nodes.geometry import PLANNER_SETTINGS as SETTINGS


@pytest.mark.parametrize("rot", ["cw", "ccw", "180", "none"])
def test_rotation_maps_pixels_back(rot):
    H, W = 6, 9
    raw = np.arange(H * W, dtype=np.int32).reshape(H, W)
    up = rotate_image(raw, rot)
    for v in range(up.shape[0]):
        for u in range(up.shape[1]):
            ru, rv = rotated_to_raw(u, v, W, H, rot)
            assert raw[rv, ru] == up[v, u]


def test_box_maps_to_raw():
    H, W = 720, 1280
    # a box near the top-left of the rotated (cw) image is near the bottom-left of the raw image
    x1, y1, x2, y2 = box_rotated_to_raw((10, 20, 110, 220), W, H, "cw")
    assert (x1, y1, x2, y2) == (20, H - 1 - 110, 220, H - 1 - 10)


def _cylinder(center, axis, length=0.08, radius=0.02, n=600, seed=0):
    rng = np.random.default_rng(seed)
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    a = np.cross(axis, [0, 0, 1]) if abs(axis[2]) < 0.9 else np.cross(axis, [1, 0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(axis, a)
    t = rng.uniform(-length / 2, length / 2, n)
    th = rng.uniform(0, math.pi, n)          # visible half
    return center + np.outer(t, axis) + radius * (np.outer(np.cos(th), a) + np.outer(np.sin(th), b))


def test_classify_upright_lying_tilted():
    up = classify_bottle(_cylinder([0.5, -0.6, 0.79], [0, 0, 1]))
    assert up["orientation"] == "upright"
    assert abs(up["surface_z"] - 0.75) < 0.01
    lying = classify_bottle(_cylinder([0.5, -0.6, 0.02], [1, 0, 0]))
    assert lying["orientation"] == "lying"
    assert abs(abs(math.cos(lying["axis_yaw"])) - 1) < 0.05
    tilted = classify_bottle(_cylinder([0.5, -0.6, 0.05], [1, 0, 1]))
    assert tilted["orientation"] == "tilted"


def test_box_points_deprojects_flat_patch():
    depth = np.full((480, 640), 1.0, np.float32)
    depth[200:260, 300:340] = 0.6               # bottle in front of a wall
    K = [600, 0, 320, 0, 600, 240, 0, 0, 1]
    pts = box_points(depth, (295, 195, 345, 265), K, band=0.06)
    assert len(pts) > 50
    assert np.allclose(pts[:, 2], 0.6)          # wall pixels dropped
    assert abs(pts[:, 0].mean() - (319.5 - 320) * 0.6 / 600) < 0.005


def test_grasp_center_yaw0_ignores_current_yaw():
    wrist = (0.0, -0.4, 0.8)
    for yaw in (0.0, 1.0, 3.0):                 # grasp center swings on a 0.2 m radius
        grasp = (wrist[0] + 0.2 * math.sin(yaw), wrist[1] - 0.2 * math.cos(yaw), 0.75)
        g0 = grasp_center_at_yaw0(wrist, grasp)
        assert np.allclose(g0, (0.0, -0.6, 0.75))


P = dict(SETTINGS)


def test_plan_reach_on_arm_side():
    g0 = (0.0, -0.6, 0.75)                      # grasp center with arm at 0.1 m, lift 0.8 m
    r = plan_grasp((0.01, -0.9, 0.79), "upright", 0.75, 0.0, ext=0.1, lift=0.8, g0=g0, p=P)
    assert r["next_step"] == "reach" and r["success"]
    assert abs(r["arm_grasp"] - 0.4) < 1e-9          # 0.1 + (-0.6 - -0.9)
    assert abs(r["arm_pregrasp"] - 0.3) < 1e-9
    assert abs(r["lift"] - (0.8 + 0.75 + 0.45 * 0.08 - 0.75)) < 1e-9
    assert r["base_translate"] == 0.0


def test_plan_turns_when_bottle_in_front():
    g0 = (0.0, -0.3, 0.75)
    r = plan_grasp((0.55, 0.0, 0.1), "upright", 0.05, 0.0, ext=0.0, lift=0.8, g0=g0, p=P)
    assert r["next_step"] == "rotate_base"
    assert abs(r["base_rotate"] - math.pi / 2) < 1e-6     # left turn puts it on the -y (arm) side


def test_plan_drives_when_bottle_far_ahead():
    g0 = (0.0, -0.3, 0.75)
    r = plan_grasp((2.0, 0.05, 0.1), "upright", 0.05, 0.0, ext=0.0, lift=0.8, g0=g0, p=P)
    assert r["next_step"] == "drive_base" and r["base_translate"] == pytest.approx(0.5)


def test_plan_turns_when_bottle_on_left():
    g0 = (0.0, -0.3, 0.75)
    r = plan_grasp((0.0, 0.6, 0.1), "upright", 0.05, 0.0, ext=0.0, lift=0.8, g0=g0, p=P)
    assert r["next_step"] == "rotate_base"
    assert abs(abs(r["base_rotate"]) - math.pi) < 1e-6


def test_plan_lines_up_lying_bottle():
    g0 = (0.0, -0.6, 0.3)
    r = plan_grasp((0.0, -0.9, 0.02), "lying", 0.0, 0.0, ext=0.1, lift=0.4, g0=g0, p=P)   # axis along x
    assert r["next_step"] == "rotate_base" and abs(abs(r["base_rotate"]) - math.pi / 2) < 1e-6
    ok = plan_grasp((0.0, -0.9, 0.02), "lying", 0.0, math.pi / 2, ext=0.1, lift=0.4, g0=g0, p=P)
    assert ok["next_step"] == "reach"


def test_plan_rejects_unreachable_height():
    g0 = (0.0, -0.6, 0.75)
    r = plan_grasp((0.0, -0.9, 2.0), "upright", 1.96, 0.0, ext=0.1, lift=0.8, g0=g0, p=P)
    assert not r["success"]


def test_look_at():
    pan, tilt = look_at((0.0, 0.0, 1.3), (0.0, -1.0, 0.3))
    assert pan == pytest.approx(-math.pi / 2) and tilt == pytest.approx(-math.pi / 4)
