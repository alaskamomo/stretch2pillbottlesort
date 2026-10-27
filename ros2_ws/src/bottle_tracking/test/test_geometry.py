"""Unit tests for the pure math (no ROS needed):  python3 -m pytest test/ -v"""

import math

import pytest

from bottle_tracking import geometry as geo

CFG = geo.ApproachConfig()


def test_quat_rotate_yaw_90():
    q = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))
    x, y, z = geo.quat_rotate(q, (1.0, 0.0, 0.0))
    assert x == pytest.approx(0.0, abs=1e-9) and y == pytest.approx(1.0)


def test_aim_head_points_left_when_target_left():
    aim = geo.aim_head((0, 0, 1.2), (1, 0, 0), (1, 1, 1.2), pan=0.0, tilt=0.0)
    assert aim["pan_cmd"] == pytest.approx(math.pi / 4)
    assert aim["tilt_cmd"] == pytest.approx(0.0, abs=1e-9)


def test_aim_head_looks_down_at_table():
    aim = geo.aim_head((0, 0, 1.2), (1, 0, 0), (1, 0, 0.2), pan=0.0, tilt=0.0)
    assert aim["tilt_cmd"] == pytest.approx(-math.pi / 4)


def test_aim_head_wraps_to_arm_side_instead_of_hitting_limit():
    # camera looking at pan -3.0 (behind, arm side); target slightly further around
    pan = -3.0
    cam_dir = (math.cos(pan), math.sin(pan), 0)
    target_yaw = -3.3
    target = (math.cos(target_yaw), math.sin(target_yaw), 0)
    aim = geo.aim_head((0, 0, 0), cam_dir, target, pan=pan, tilt=0.0)
    assert aim["in_range"]
    assert aim["pan_cmd"] == pytest.approx(-3.3)


def test_tracking_cancels_base_rotation():
    """Bottle saved in odom; after the base turns 90 deg the head must turn -90 deg."""
    bottle_odom = (1.0, 0.0, 0.8)
    yaw = math.pi / 2                                           # base turned left 90 deg
    # bottle in the new base frame
    c, s = math.cos(yaw), math.sin(yaw)
    bb = (c * bottle_odom[0] + s * bottle_odom[1], -s * bottle_odom[0] + c * bottle_odom[1], 0.8)
    aim = geo.aim_head((0, 0, 0.8), (1, 0, 0), bb, pan=0.0, tilt=0.0)
    assert aim["pan_cmd"] == pytest.approx(-math.pi / 2)


def test_in_view():
    assert geo.in_view((0, 0, 0), (1, 0, 0), (1.0, 0.1, 0), 15, 2.5)
    assert not geo.in_view((0, 0, 0), (1, 0, 0), (0.0, 1.0, 0), 15, 2.5)
    assert not geo.in_view((0, 0, 0), (1, 0, 0), (3.0, 0.0, 0), 15, 2.5)


@pytest.mark.parametrize("bottle,robot", [
    ((1.0, 0.0, 0.76), (0, 0, 0)),
    ((2.0, 0.3, 0.76), (0, 0, 0)),
    ((0.0, 1.5, 0.76), (0, 0, 0)),
    ((0.0, -1.5, 0.76), (0, 0, 0)),
    ((-1.5, 0.0, 0.76), (0, 0, 0)),
    ((2.5, -1.5, 0.76), (0.5, 0.5, 1.0)),
    ((0.2, 0.2, 0.76), (0, 0, 0)),         # closer than standoff: backs up
])
def test_plan_puts_bottle_on_arm_line(bottle, robot):
    plan = geo.plan_approach(bottle, robot, CFG)
    assert plan["ok"], plan["reason"]
    assert abs(plan["predicted_align_error_m"]) <= CFG.align_tol_m
    assert CFG.reach_min_m <= plan["predicted_reach_m"] <= CFG.reach_max_m
    assert plan["num_rotations"] <= 2 and plan["num_drives"] <= 1
    assert plan["predicted_table_clearance_m"] >= CFG.min_table_clearance_m - 1e-9


def test_plan_already_beside_bottle_does_not_move():
    plan = geo.plan_approach((0.0, -0.5, 0.76), (0, 0, 0), CFG)
    assert plan["ok"] and plan["steps"] == []


def test_plan_already_beside_bottle_just_slides():
    plan = geo.plan_approach((0.3, -0.5, 0.76), (0, 0, 0), CFG)
    assert plan["ok"] and [s["type"] for s in plan["steps"]] == ["DRIVE"]
    assert plan["steps"][0]["m"] == pytest.approx(0.3)


@pytest.mark.parametrize("bottle,word", [((5.0, 0, 0.76), "max approach"), ((1.0, 0, 1.4), "lift range")])
def test_unreachable_gives_no_plan(bottle, word):
    plan = geo.plan_approach(bottle, (0, 0, 0), CFG)
    assert not plan["ok"] and not plan["steps"] and word in plan["reason"]
    assert "NO_PLAN" in geo.format_plan(plan)


def test_deep_table_gives_no_plan():
    cfg = geo.ApproachConfig(table_edge_to_bottle_m=0.60)
    plan = geo.plan_approach((1.5, 0, 0.76), (0, 0, 0), cfg)
    assert not plan["ok"] and "table clearance" in plan["reason"]


ARM = geo.ArmConfig(grasp_lateral_offset_m=0.30, lift_grasp_z_offset_m=-0.05, arm_x_offset_m=0.02)


def test_arm_plan_reaches_bottle_middle():
    plan = geo.plan_arm((0.02, -0.55, 0.80), ARM)
    assert plan["ok"], plan["reason"]
    assert plan["extension_m"] == pytest.approx(0.25)
    assert plan["lift_grasp_m"] == pytest.approx(0.85)          # z - offset
    types = [s["type"] for s in plan["steps"]]
    assert types == ["WRIST", "GRIPPER", "LIFT", "ARM", "LIFT", "ARM", "GRIPPER", "LIFT"]
    ext_steps = [s["to"] for s in plan["steps"] if s["type"] == "ARM"]
    assert ext_steps[0] < ext_steps[1]                           # stops short, then goes in


@pytest.mark.parametrize("bottle,word", [
    ((0.10, -0.55, 0.8), "ahead of"),
    ((0.02, -1.00, 0.8), "too far"),
    ((0.02, -0.32, 0.8), "too close"),
    ((0.02, 0.50, 0.8), "left side"),
    ((0.02, -0.55, 1.50), "lift"),
])
def test_arm_plan_rejects(bottle, word):
    plan = geo.plan_arm(bottle, ARM)
    assert not plan["ok"] and word in plan["reason"]


@pytest.mark.parametrize("bottle,robot", [
    ((2.0, 0.0, 0.8), (0, 0, 0)), ((1.0, 1.0, 0.8), (0, 0, 0)), ((-1.5, 0.3, 0.8), (0.2, -0.1, 0.5)),
])
def test_full_pick_plan_base_then_arm(bottle, robot):
    full = geo.full_pick_plan(bottle, robot, ARM)
    assert full["ok"], full["reason"]
    assert full["arm"]["extension_m"] == pytest.approx(ARM.preferred_extension_m, abs=0.02)
    assert "PLAN OK" in geo.format_full_plan(full)


def test_plan_does_not_park_at_the_edge_of_reach():
    """Real case from the robot: bottle 0.79 m out the arm side needed 0.49 of 0.50 m extension.
    The planner must move the base so the arm has room to spare instead."""
    arm = geo.ArmConfig()
    full = geo.full_pick_plan((0.48, -0.79, 0.80), (0.0, 0.0, 0.0), arm)
    assert full["ok"], full["reason"]
    assert full["arm"]["extension_m"] <= arm.ext_max_m - arm.reach_margin_m + 1e-9
    # still valid if the fresh look after driving is 2 cm farther
    fx, fy, fz = full["arm"]["bottle_base"]
    assert geo.plan_arm((fx, fy - 0.02, fz), arm)["ok"]
