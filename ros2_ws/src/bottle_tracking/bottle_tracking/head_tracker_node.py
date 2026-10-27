#!/usr/bin/env python3
"""
head_tracker_node - keeps the head camera pointed at the saved bottle position.

While the base turns or drives, the saved position (odom frame) stays put, so this
node keeps re-aiming the head at it. That cancels out the base motion and the
camera stays on the bottle. If the bottle is outside the head's pan range, the head
stops at its limit and in_range goes false; it swings back as soon as it can.

In:
  /bottle_memory/pose    geometry_msgs/PointStamped  (from bottle_memory)
  /head_tracker/enable   std_msgs/Bool               turn tracking on/off (latched)
Out:
  /head_tracker/status   std_msgs/String (JSON)      enabled, in_range, angle_err_deg, ...

The head search node turns tracking OFF while it sweeps, then back ON once found.
"""

import json
import math
import time

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy

from geometry_msgs.msg import PointStamped
from std_msgs.msg import Bool, String

from bottle_tracking import geometry as geo
from bottle_tracking.stretch_interface import StretchInterface, declare

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


class HeadTrackerNode(Node):
    def __init__(self):
        super().__init__("head_tracker")
        cb = MutuallyExclusiveCallbackGroup()
        self.robot = StretchInterface(self)

        self.enabled = declare(self, "tracking_enabled_on_start", True)
        rate = declare(self, "tracking_rate_hz", 5.0)
        self.deadband = math.radians(declare(self, "tracking_deadband_deg", 2.0))
        self.min_interval = declare(self, "min_goal_interval_s", 0.3)
        self.pan_sign = declare(self, "pan_sign", 1.0)
        self.tilt_sign = declare(self, "tilt_sign", 1.0)
        self.pose_timeout = declare(self, "pose_timeout_s", 2.0)

        self.target = None        # (x, y, z, frame)
        self.target_time = 0.0
        self.last_goal = 0.0

        self.create_subscription(PointStamped, "/bottle_memory/pose", self.on_pose, 10, callback_group=cb)
        self.create_subscription(Bool, "/head_tracker/enable", self.on_enable, LATCHED, callback_group=cb)
        self.status_pub = self.create_publisher(String, "/head_tracker/status", 10)
        self.create_timer(1.0 / rate, self.step, callback_group=cb)
        self.get_logger().info(f"head tracker ready (enabled={self.enabled})")

    def on_pose(self, msg):
        self.target = (msg.point.x, msg.point.y, msg.point.z, msg.header.frame_id)
        self.target_time = time.time()

    def on_enable(self, msg):
        if msg.data != self.enabled:
            self.get_logger().info(f"tracking {'enabled' if msg.data else 'disabled'}")
        self.enabled = msg.data

    def step(self):
        status = {"enabled": self.enabled, "has_target": False, "in_range": None,
                  "angle_err_deg": None, "pan_cmd": None, "tilt_cmd": None}
        fresh = self.target is not None and time.time() - self.target_time < self.pose_timeout
        if fresh:
            status["has_target"] = True
            x, y, z, frame = self.target
            target = self.robot.point_to((x, y, z), frame, self.robot.base_frame)
            cam_pos, cam_dir = self.robot.camera_in(self.robot.base_frame)
            head = self.robot.head_state()
            if target is not None and cam_pos is not None and head is not None:
                aim = geo.aim_head(cam_pos, cam_dir, target, head[0], head[1],
                                   self.robot.limits, self.pan_sign, self.tilt_sign)
                status.update(in_range=aim["in_range"],
                              angle_err_deg=round(math.degrees(aim["angle_err"]), 2),
                              pan_cmd=round(aim["pan_cmd"], 3), tilt_cmd=round(aim["tilt_cmd"], 3))
                off = max(abs(aim["yaw_err"]), abs(aim["pitch_err"])) > self.deadband
                if (self.enabled and off and time.time() - self.last_goal >= self.min_interval):
                    self.robot.send_head(aim["pan_cmd"], aim["tilt_cmd"])
                    self.last_goal = time.time()
        self.status_pub.publish(String(data=json.dumps(status)))


def main():
    rclpy.init()
    node = HeadTrackerNode()
    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
