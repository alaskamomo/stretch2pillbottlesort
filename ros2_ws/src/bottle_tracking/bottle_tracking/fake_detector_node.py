#!/usr/bin/env python3
"""
fake_detector_node - pretends to be the detector, for debugging without YOLO/SAM3.

You tell it where the bottle really is (tape-measure it, in the odom frame) and it
publishes /pill_bottle/detection only while that point is inside the head camera's
view, like a real detector would. Lets you test memory, tracking, search and the
approach planner on their own (component testing), with perception taken out.

  ros2 run bottle_tracking fake_detector --ros-args -p bottle_x:=1.2 -p bottle_y:=-0.3 -p bottle_z:=0.76

Tip: start the robot, then `ros2 run tf2_ros tf2_echo odom base_link` to see where odom is.
Change the bottle position live with: ros2 param set /fake_detector bottle_x 1.5
"""

import random

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from geometry_msgs.msg import PointStamped

from bottle_tracking import geometry as geo
from bottle_tracking.stretch_interface import StretchInterface, declare


class FakeDetector(Node):
    def __init__(self):
        super().__init__("fake_detector")
        self.robot = StretchInterface(self)
        declare(self, "bottle_x", 1.0)
        declare(self, "bottle_y", 0.0)
        declare(self, "bottle_z", 0.76)
        self.cone = declare(self, "fake_view_cone_deg", 18.0)
        self.max_range = declare(self, "max_detect_range_m", 2.5)
        self.noise = declare(self, "noise_m", 0.005)
        rate = declare(self, "fake_rate_hz", 5.0)
        self.pub = self.create_publisher(PointStamped, "/pill_bottle/detection", 10)
        self.create_timer(1.0 / rate, self.tick)

    def tick(self):
        p = tuple(self.get_parameter(n).value for n in ("bottle_x", "bottle_y", "bottle_z"))
        cam_pos, cam_dir = self.robot.camera_in(self.robot.fixed_frame)
        if cam_pos is None or not geo.in_view(cam_pos, cam_dir, p, self.cone, self.max_range):
            return
        msg = PointStamped()
        msg.header.frame_id = self.robot.fixed_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.point.x, msg.point.y, msg.point.z = (c + random.gauss(0, self.noise) for c in p)
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = FakeDetector()
    ex = MultiThreadedExecutor(num_threads=2)
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
