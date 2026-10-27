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
