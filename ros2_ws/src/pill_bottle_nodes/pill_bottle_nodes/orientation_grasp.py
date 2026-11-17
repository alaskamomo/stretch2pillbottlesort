# orientation_grasp.py: does the grip plan fit how the bottle is sitting? (ticket T19). No ROS, unit tested.
#
#   plan_sequence   runs the real grasp planner (geometry.plan_grasp) the way task_manager does: turn/drive the
#                   base, look again, plan again... until it says "reach". Here the base moves are SIMULATED
#                   (the bottle is moved into the new base frame), so T19a can show the whole plan without moving.
#   check_plan      checks the final plan matches the orientation:
#                     upright -> side grab low on the bottle     lying -> claw across it, bottle along the arm
#                     tilted  -> grab at the bottle's center      too high/low -> "out of range", no grab
import math

from .geometry import plan_grasp, wrap

PLAN_TYPES = {"upright": "side grab, low on the bottle", "lying": "claw across the bottle, bottle along the arm",
              "tilted": "grab at the bottle's center height"}


def move_base(point, axis_yaw, step, amount):
    """Where a point (and a lying bottle's axis) ends up in base_link after the base turns/drives.
    step: 'rotate_base' (amount = rad, counter-clockwise) or 'drive_base' (amount = m forward)."""
    x, y, z = point
    if step == "rotate_base":
        c, s = math.cos(-amount), math.sin(-amount)
        return (c * x - s * y, s * x + c * y, z), wrap(axis_yaw - amount)
    if step == "drive_base":
        return (x - amount, y, z), axis_yaw
    return point, axis_yaw


def plan_sequence(bottle, orientation, surface_z, axis_yaw, ext, lift, g0, p, max_moves=10):
    """Plan, simulate each base move, plan again, until 'reach' (or no plan / too many moves).

    Returns dict(ok, steps=[plan dicts], final=last plan, bottle=(x,y,z) at the end, axis_yaw at the end,
                 base_moves=n, reason)."""
    steps = []
    for _ in range(max_moves + 1):
        plan = plan_grasp(bottle, orientation, surface_z, axis_yaw, ext, lift, g0, p)
        steps.append(plan)
        if not plan["success"]:
            return dict(ok=False, steps=steps, final=plan, bottle=bottle, axis_yaw=axis_yaw,
                        base_moves=len(steps) - 1, reason=plan["message"])
        if plan["next_step"] == "reach":
            # the reach plan's small base_translate lines the arm up; apply it so the checks see the final pose
            bottle, axis_yaw = move_base(bottle, axis_yaw, "drive_base", plan["base_translate"])
            return dict(ok=True, steps=steps, final=plan, bottle=bottle, axis_yaw=axis_yaw,
                        base_moves=len(steps) - 1, reason="")
        amount = plan["base_rotate"] if plan["next_step"] == "rotate_base" else plan["base_translate"]
        bottle, axis_yaw = move_base(bottle, axis_yaw, plan["next_step"], amount)
    return dict(ok=False, steps=steps, final=steps[-1], bottle=bottle, axis_yaw=axis_yaw,
                base_moves=len(steps) - 1, reason=f"still not in reach after {max_moves} base moves")


def grip_height(plan, lift, g0):
    """Height (base_link z) the grasp center will be at, from the plan's lift."""
    return g0[2] + (plan["lift"] - lift)


def check_plan(orientation, seq, surface_z, lift, g0, p, tol=0.01):
    """-> (ok, plan_type, reasons). plan_type is what the plan does; reasons lists anything that doesn't fit."""
    if not seq["ok"]:
        return False, "no plan", [seq["reason"]]
    final = seq["final"]
    gz = grip_height(final, lift, g0)
    reasons = []
    if not (p["arm_min_grasp"] - 1e-9 <= final["arm_grasp"] - final["grasp_depth"] and final["arm_grasp"] <= p["arm_max"] + 1e-9):
        reasons.append(f"arm {final['arm_grasp']:.2f} m is outside {p['arm_min_grasp']}-{p['arm_max']} m")
    if not (p["lift_min"] - 1e-9 <= final["lift"] <= p["lift_max"] + 1e-9):
        reasons.append(f"lift {final['lift']:.2f} m is outside {p['lift_min']}-{p['lift_max']} m")
    if final["lift_clear"] < final["lift"] - 1e-9:
        reasons.append("travel height is below the grasp height")
    above = gz - surface_z
    if orientation == "upright":
        want = max(p["grasp_frac_upright"] * p["bottle_length"], p["min_table_clearance"])
        kind = PLAN_TYPES["upright"]
        if abs(above - want) > tol:
            reasons.append(f"grip {above * 100:.1f} cm above the surface, expected {want * 100:.1f} cm")
        if above > p["bottle_length"] / 2 + tol:
            reasons.append("grip is above the middle of the bottle (would hide the label / slip)")
    elif orientation == "lying":
        want = max(p["bottle_radius"], p["lying_min_clear"])
        kind = PLAN_TYPES["lying"]
        if abs(above - want) > tol:
            reasons.append(f"grip {above * 100:.1f} cm above the surface, expected {want * 100:.1f} cm")
        off = abs(wrap(seq["axis_yaw"] - math.pi / 2))
        off = min(off, math.pi - off)                 # a bottle's axis has no front/back
        if math.degrees(off) > p["lying_align_deg"] + 0.5:
            reasons.append(f"bottle is {math.degrees(off):.0f} deg off the arm line (limit {p['lying_align_deg']:.0f})")
    else:
        kind = PLAN_TYPES["tilted"]
        want = max(seq["bottle"][2], surface_z + p["min_table_clearance"])
        if abs(gz - want) > tol:
            reasons.append(f"grip at {gz:.3f} m, expected the bottle's center {want:.3f} m")
    if abs(seq["bottle"][0] - g0[0]) > p["drive_tol"] + 1e-6:
        reasons.append(f"arm is {abs(seq['bottle'][0] - g0[0]) * 100:.1f} cm off the bottle along the base")
    return not reasons, kind, reasons


def describe(seq, lift, g0):
    """Readable plan, one line per step (for T19a)."""
    lines = []
    for i, st in enumerate(seq["steps"], 1):
        if st["next_step"] == "rotate_base" and st["success"]:
            lines.append(f"{i}. turn the base {math.degrees(st['base_rotate']):+.0f} deg   ({st['message']})")
        elif st["next_step"] == "drive_base" and st["success"]:
            lines.append(f"{i}. drive the base {st['base_translate']:+.2f} m   ({st['message']})")
        elif st["success"]:
            gz = grip_height(st, lift, g0)
            lines.append(f"{i}. reach: lift {st['lift']:.3f} m (grip at {gz:.3f} m), arm {st['arm_pregrasp']:.3f} -> "
                         f"{st['arm_grasp']:.3f} m, line up base {st['base_translate']:+.3f} m, close, lift, hold")
        else:
            lines.append(f"{i}. NO PLAN: {st['message']}")
    return lines
