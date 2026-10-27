#!/usr/bin/env python3
"""
T15a - approach planning test (no motion, no robot needed).

Plans the base motion to typed-in bottle positions, PRINTS every plan, checks it
against the T15a success criteria and logs SUCCESS/MISS per trial.
Each trial adds +-1 cm of noise to the bottle position (like real perception) so the
10 trials aren't identical.

  ros2 run bottle_tracking t15a_approach_plan_tester
  ros2 run bottle_tracking t15a_approach_plan_tester --params ~/ros2_ws/src/bottle_tracking/config/params.yaml
  ros2 run bottle_tracking t15a_approach_plan_tester --condition ahead_2.0m --verbose

Robot starts at x=0 y=0 heading=0 (facing +x). The arm side is -y (right).
"""

import argparse
import random
import time

from bottle_tracking import geometry as geo
from bottle_tracking.test_logger import TestLogger

TABLE_Z = 0.76

# condition: (bottle x, y, z in the robot's starting frame, should it be reachable?)
CONDITIONS = {
    "ahead_1.0m": ((1.0, 0.0, TABLE_Z), True),
    "ahead_2.0m": ((2.0, 0.0, TABLE_Z), True),
    "ahead_3.0m": ((3.0, 0.0, TABLE_Z), True),
    "left_1.5m": ((0.0, 1.5, TABLE_Z), True),
    "right_1.5m": ((0.0, -1.5, TABLE_Z), True),
    "behind_1.5m": ((-1.5, 0.0, TABLE_Z), True),
    "already_in_reach": ((0.0, -0.50, TABLE_Z), True),
    "unreachable_far": ((5.0, 0.0, TABLE_Z), False),
    "unreachable_high": ((1.0, 0.0, 1.35), False),
}


def load_config(path):
    if not path:
        return geo.ApproachConfig()
    import yaml  # comes with ROS
    with open(path) as f:
        data = yaml.safe_load(f)
    params = {}
    for block in data.values():          # merge every node block, e.g. "/**"
        params.update(block.get("ros__parameters", {}))
    return geo.config_from_dict(params)


def judge(plan, reachable, plan_time, cfg):
    """Return (outcome, notes) using the T15a ticket criteria."""
    if not reachable:
        if not plan["ok"] and plan["reason"] and not plan["steps"]:
            return "SUCCESS", f"correct NO_PLAN: {plan['reason']}"
        return "MISS", "made a plan for an unreachable bottle"
    if not plan["ok"]:
        return "NO_PLAN", plan["reason"]
    if plan_time > 2.0:
        return "TIMEOUT", f"plan took {plan_time:.2f} s"
    if plan["predicted_table_clearance_m"] < cfg.min_table_clearance_m - 1e-6:
        return "COLLISION", f"table clearance {plan['predicted_table_clearance_m'] * 100:.1f} cm"
    if abs(plan["predicted_align_error_m"]) > 0.02:
        return "MISS", f"align error {plan['predicted_align_error_m'] * 100:.1f} cm"
    if not cfg.reach_min_m <= plan["predicted_reach_m"] <= cfg.reach_max_m:
        return "MISS", f"reach {plan['predicted_reach_m']:.2f} m out of range"
    if plan["num_rotations"] > 2 or plan["num_drives"] > 2:
        return "MISS", f"too many steps: {plan['num_rotations']} rotations, {plan['num_drives']} drives"
    return "SUCCESS", ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--params", help="params.yaml to read the approach settings from")
    p.add_argument("--condition", choices=list(CONDITIONS))
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--noise", type=float, default=0.01, help="bottle position noise (m)")
    p.add_argument("--verbose", action="store_true", help="print every plan, not just the first per condition")
    args, _ = p.parse_known_args()   # ignore --ros-args if run through ros2 run

    cfg = load_config(args.params)
    log = TestLogger("T15a", "Approach planning to a far-away bottle (no motion)")
    robot = (0.0, 0.0, 0.0)
    conds = [args.condition] if args.condition else list(CONDITIONS)

    for cond in conds:
        (bx, by, bz), reachable = CONDITIONS[cond]
        print(f"\n===== {cond} =====")
        for i in range(args.trials):
            bottle = (bx + random.gauss(0, args.noise), by + random.gauss(0, args.noise), bz)
            t0 = time.perf_counter()
            plan = geo.plan_approach(bottle, robot, cfg)
            plan_time = time.perf_counter() - t0
            text = geo.format_plan(plan)
            if i == 0 or args.verbose:
                print(text)
            outcome, why = judge(plan, reachable, plan_time, cfg)
            metrics = {"plan_time_s": plan_time, "start_distance_m": plan["start_distance_m"],
                       "num_steps": len(plan["steps"])}
            if "predicted_reach_m" in plan:   # only plans that got past the reachability checks
                metrics.update(
                    total_rotation_deg=plan["total_rotation_deg"],
                    total_drive_m=plan["total_drive_m"],
                    predicted_align_error_cm=abs(plan["predicted_align_error_m"]) * 100,
                    predicted_table_clearance_cm=plan["predicted_table_clearance_m"] * 100)
            log.log(outcome, cond, metrics, notes=(why + "\n" + text).strip(), duration_s=plan_time)
    print("\nSummary: python3 summarize_results.py --test T15a   (from the testing/ folder)")


if __name__ == "__main__":
    main()
