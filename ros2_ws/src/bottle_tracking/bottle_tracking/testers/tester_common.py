"""
tester_common.py - shared bits for the ROS testers (prompts, background spinning).
"""

import math
import threading

import rclpy
from rclpy.executors import MultiThreadedExecutor



def run_tester(node_class):
    """Spin the node in the background and run node.run() (which uses input()) in front."""
    rclpy.init()
    node = node_class()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    try:
        node.run()
    except (KeyboardInterrupt, EOFError):
        print("\nStopped. Logged trials are saved.")
    finally:
        try:
            node.robot.stop()
        except Exception:
            pass
        ex.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def ask(prompt):
    return input(prompt).strip()


def ask_yes(prompt, default=False):
    a = input(f"{prompt} [{'Y/n' if default else 'y/N'}]: ").strip().lower()
    return default if not a else a.startswith("y")


def ask_xyz(prompt):
    """Ask for 'x y z' in meters. Returns a tuple or None if skipped."""
    while True:
        a = input(f"{prompt} (x y z in meters, Enter to skip): ").strip()
        if not a:
            return None
        try:
            vals = [float(v) for v in a.replace(",", " ").split()]
            if len(vals) == 3:
                return tuple(vals)
        except ValueError:
            pass
        print("  type three numbers, e.g. 1.20 -0.35 0.76")


def ask_float(prompt):
    a = input(f"{prompt} (Enter to skip): ").strip()
    try:
        return float(a) if a else None
    except ValueError:
        return None


def base_to_fixed(point_base, base_pose):
    """Point measured in base_link (robot at base_pose in odom) -> odom."""
    x, y, yaw = base_pose
    c, s = math.cos(yaw), math.sin(yaw)
    return (x + c * point_base[0] - s * point_base[1],
            y + s * point_base[0] + c * point_base[1],
            point_base[2])


def pause(seconds, message="Move the pill bottle to a new spot."):
    """Count down so you can move the bottle. Press Enter to skip the rest of the wait."""
    import select
    import sys
    import time
    print(f"\n{message}  Starting in {int(seconds)} s (press Enter to start now)", flush=True)
    end = time.time() + seconds
    last_shown = None
    while True:
        left = int(round(end - time.time()))
        if left <= 0:
            break
        if left != last_shown and (left % 10 == 0 or left <= 5):
            print(f"  {left} s...", flush=True)
            last_shown = left
        ready, _, _ = select.select([sys.stdin], [], [], 0.25)
        if ready:
            sys.stdin.readline()
            break
    print("  go", flush=True)


def distance_bucket(d):
    """Condition label from a distance in meters, so results group by range automatically."""
    if d is None:
        return "not_found"
    for hi, label in [(0.6, "under_0.6m"), (1.0, "0.6_to_1.0m"), (1.5, "1.0_to_1.5m"), (2.0, "1.5_to_2.0m")]:
        if d < hi:
            return label
    return "over_2.0m"


def declare_arm_config(node):
    from dataclasses import asdict
    from bottle_tracking import geometry as geo
    from bottle_tracking.stretch_interface import declare
    values = {k: declare(node, k, v) for k, v in asdict(geo.ArmConfig()).items()}
    return geo.arm_config_from_dict(values)
