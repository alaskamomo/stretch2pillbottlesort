# T19 offline integration test (no robot or ROS): python3 -m pytest test/test_orientation_grasp.py -v
#
# Fake depth points for upright / lying / tilted bottles at different spots and heights go through the same code
# the robot uses: geometry.classify_bottle (depth fit) -> orientation.fuse_orientation -> geometry.plan_grasp,
# with the base moves simulated until the planner says "reach". Then check_plan checks the grip fits the orientation.
import math

import numpy as np
import pytest

from pill_bottle_nodes.geometry import PLANNER_SETTINGS as P
from pill_bottle_nodes.geometry import classify_bottle
from pill_bottle_nodes.orientation import ORIENTATION_SETTINGS, fuse_orientation
from pill_bottle_nodes.orientation_grasp import check_plan, describe, move_base, plan_sequence

LIFT, EXT = 0.6, 0.0
G0 = (0.0, -0.30, LIFT - 0.05)          # grasp center with the arm in and the wrist straight out


def bottle_points(center, axis, length=0.08, radius=0.02, n=700, seed=0, camera=(0.0, 0.0, 1.3)):
    """Points on the half of a cylinder that faces the head camera (what the depth camera sees)."""
    rng = np.random.default_rng(seed)
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    ref = np.array([0, 0, 1.0]) if abs(axis[2]) < 0.9 else np.array([1.0, 0, 0])
    a = np.cross(axis, ref)
    a /= np.linalg.norm(a)
    b = np.cross(axis, a)
    t = rng.uniform(-length / 2, length / 2, n)
    th = rng.uniform(0, 2 * math.pi, n)
    normals = np.outer(np.cos(th), a) + np.outer(np.sin(th), b)
    pts = np.asarray(center) + np.outer(t, axis) + radius * normals
    to_cam = np.asarray(camera) - pts
    return pts[(normals * to_cam).sum(axis=1) > 0]


def bottle(orientation, x, y, surface, yaw=0.0):
    """center, axis and the true surface height for a bottle sitting on a surface at height `surface`."""
    if orientation == "upright":
        return (x, y, surface + 0.04), (0, 0, 1), surface
    if orientation == "lying":
        return (x, y, surface + 0.02), (math.cos(yaw), math.sin(yaw), 0), surface
    tilt = math.radians(45)                    # leaning
    return (x, y, surface + 0.04 * math.cos(tilt) + 0.02), (math.sin(tilt) * math.cos(yaw),
                                                           math.sin(tilt) * math.sin(yaw), math.cos(tilt)), surface


def pipeline(orientation, x, y, surface, yaw=0.0, seed=0):
    center, axis, true_surface = bottle(orientation, x, y, surface, yaw)
    pts = bottle_points(center, axis, seed=seed)
    info = classify_bottle(pts, radius=P["bottle_radius"])
    seen, _, _ = fuse_orientation(info["tilt_deg"], len(pts), None, ORIENTATION_SETTINGS)
    seq = plan_sequence(tuple(info["center"]), seen, info["surface_z"], info["axis_yaw"], EXT, LIFT, G0, P)
    ok, kind, reasons = check_plan(seen, seq, info["surface_z"], LIFT, G0, P, tol=0.015)
    return seen, seq, ok, kind, reasons, info


SPOTS = [(0.05, -0.75), (0.9, -0.1), (0.3, 0.8), (-0.6, -0.4), (1.6, 0.6)]   # arm side, ahead, left, behind, far
HEIGHTS = [0.0, 0.75, 0.9]                                                   # floor, table, raised


@pytest.mark.parametrize("orientation", ["upright", "lying", "tilted"])
@pytest.mark.parametrize("surface", HEIGHTS)
@pytest.mark.parametrize("spot", SPOTS)
def test_every_orientation_gets_a_plan_that_fits(orientation, surface, spot):
    seen, seq, ok, kind, reasons, _ = pipeline(orientation, *spot, surface, yaw=0.7)
    assert seen == orientation, f"depth fit saw {seen}"
    assert seq["ok"], seq["reason"]
    assert ok, f"{kind}: {reasons}\n" + "\n".join(describe(seq, LIFT, G0))
    assert seq["base_moves"] <= 10


@pytest.mark.parametrize("yaw_deg", [0, 30, 60, 90, 135])
def test_lying_bottle_ends_up_along_the_arm(yaw_deg):
    seen, seq, ok, kind, reasons, _ = pipeline("lying", 0.05, -0.75, 0.0, yaw=math.radians(yaw_deg))
    assert seen == "lying" and ok, reasons
    off = abs(((seq["axis_yaw"] - math.pi / 2) + math.pi / 2) % math.pi - math.pi / 2)
    assert math.degrees(off) <= P["lying_align_deg"] + 0.5


def test_upright_grip_is_low_on_the_bottle():
    _, seq, ok, _, _, info = pipeline("upright", 0.05, -0.75, 0.75)
    gz = G0[2] + (seq["final"]["lift"] - LIFT)
    assert ok and gz - info["surface_z"] < P["bottle_length"] / 2


@pytest.mark.parametrize("orientation", ["upright", "lying"])
def test_too_high_is_out_of_range_not_a_bad_grab(orientation):
    seen, seq, ok, kind, reasons, _ = pipeline(orientation, 0.05, -0.75, 1.15)
    assert not seq["ok"] and kind == "no plan"
    assert "outside" in reasons[0]


def test_wrong_orientation_is_caught():
    """If the robot thought a lying bottle was upright, the check flags the plan."""
    center, axis, surface = bottle("lying", 0.05, -0.75, 0.0, yaw=math.pi / 2)
    seq = plan_sequence(center, "upright", surface, math.pi / 2, EXT, LIFT, G0, P)
    ok, _, reasons = check_plan("lying", seq, surface, LIFT, G0, P)
    assert not ok and any("above the surface" in r for r in reasons)


def test_move_base_matches_the_planner_convention():
    # a bottle straight ahead ends up on the arm's side (-y) after a +90 deg (left) turn
    p, yaw = move_base((1.0, 0.0, 0.1), 0.0, "rotate_base", math.pi / 2)
    assert p[0] == pytest.approx(0.0, abs=1e-9) and p[1] == pytest.approx(-1.0) and yaw == pytest.approx(-math.pi / 2)
    p, _ = move_base((1.0, -0.5, 0.1), 0.0, "drive_base", 0.3)
    assert p[0] == pytest.approx(0.7)


def test_describe_lists_every_step():
    _, seq, _, _, _, _ = pipeline("upright", 1.6, 0.6, 0.0)
    lines = describe(seq, LIFT, G0)
    assert len(lines) == len(seq["steps"]) and lines[-1].split(". ")[1].startswith("reach")


def test_random_spots_never_loop():
    """200 random bottles (any orientation, spot up to 2.5 m, heading, height): the plan always finishes within
    task_manager's max_base_moves (10). A 2.5 m bottle can take 7 moves, so 6 wasn't enough."""
    rng = np.random.default_rng(7)
    for i in range(200):
        orientation = ["upright", "lying", "tilted"][i % 3]
        r, a = rng.uniform(0.45, 2.5), rng.uniform(-math.pi, math.pi)
        seen, seq, ok, kind, reasons, _ = pipeline(orientation, r * math.cos(a), r * math.sin(a),
                                                   rng.choice(HEIGHTS), yaw=rng.uniform(0, math.pi), seed=i)
        assert seq["ok"], f"{orientation} at {r:.2f} m, {math.degrees(a):.0f} deg: {seq['reason']}"
        assert ok, reasons
