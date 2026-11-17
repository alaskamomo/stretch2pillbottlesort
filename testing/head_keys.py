#!/usr/bin/env python3
"""
head_keys.py - aim the head camera from the keyboard (for T6, T2 and taking training pictures).

  python3 testing/head_keys.py

  w / s   tilt up / down          W / S / A / D   bigger steps
  a / d   pan left / right        c               center (level, straight ahead)
  q       quit

Needs the driver running (robot.launch.py or stretch_driver.launch.py). It starts from where the head is now.
"""

import sys
import termios
import threading
import time
import tty

import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

STEP = 0.1
LIMITS = {"joint_head_pan": (-4.0, 1.7), "joint_head_tilt": (-1.5, 0.79)}


def key():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        return sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def main():
    pos = {"joint_head_pan": None, "joint_head_tilt": None}

    def on_joints(m):
        for n, p in zip(m.name, m.position):
            if n in pos and pos[n] is None:
                pos[n] = p

    rclpy.init()
    node = rclpy.create_node("head_keys")
    for topic in ("/stretch/joint_states", "/joint_states"):
        node.create_subscription(JointState, topic, on_joints, 10)
    action = ActionClient(node, FollowJointTrajectory, "/stretch_controller/follow_joint_trajectory")
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    print("Waiting for the driver ...", flush=True)
    if not action.wait_for_server(timeout_sec=10.0):
        print("No /stretch_controller/follow_joint_trajectory: is the driver running?")
        return
    end = time.time() + 5.0
    while None in pos.values() and time.time() < end:
        time.sleep(0.05)
    for j in pos:
        pos[j] = 0.0 if pos[j] is None else pos[j]

    print(__doc__.split("\n\n")[1])
    while True:
        k = key()
        if k in ("q", "Q", "\x03"):
            break
        big = 3 if k.isupper() else 1
        k = k.lower()
        if k == "w":
            pos["joint_head_tilt"] += STEP * big
        elif k == "s":
            pos["joint_head_tilt"] -= STEP * big
        elif k == "a":
            pos["joint_head_pan"] += STEP * big
        elif k == "d":
            pos["joint_head_pan"] -= STEP * big
        elif k == "c":
            pos.update(joint_head_pan=0.0, joint_head_tilt=0.0)
        else:
            continue
        for j, (lo, hi) in LIMITS.items():
            pos[j] = min(hi, max(lo, pos[j]))
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = list(pos)
        goal.trajectory.points = [JointTrajectoryPoint(positions=list(pos.values()), time_from_start=Duration(sec=1))]
        action.send_goal_async(goal)
        print(f"\rpan {pos['joint_head_pan']:+.2f}  tilt {pos['joint_head_tilt']:+.2f}   ", end="", flush=True)
    print()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
