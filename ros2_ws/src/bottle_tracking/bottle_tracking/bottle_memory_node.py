#!/usr/bin/env python3
"""
bottle_memory_node - remembers where the pill bottle is, even after the robot turns.

THE BUG IT FIXES: the old code kept the bottle position relative to the robot/camera,
so as soon as the base turned, the saved number pointed at the wrong place.
This node saves the bottle in the odom frame, which stays fixed to the room.

In:
  /pill_bottle/detection  geometry_msgs/PointStamped  bottle center from YOUR detector,
                          in any frame (camera frame is fine), stamped with the IMAGE time
  /bottle_memory/clear    std_msgs/Empty              forget the bottle
Out:
  /bottle_memory/pose     geometry_msgs/PointStamped  saved bottle position (odom), 10 Hz
  /bottle_memory/state    std_msgs/String (JSON)      status + position + age, 10 Hz
  /bottle_memory/event    std_msgs/String             "found", "moved ...", "missing ..."

Status values:
  NONE     never seen (or cleared)
  KNOWN    saved position is good
  MISSING  the camera has been looking at the saved spot for missing_timeout_s and
           the bottle isn't there (T4 / T12). The last position is kept until a new detection.

Moved bottle: a detection farther than same_bottle_radius_m from the saved spot only
replaces it after moved_confirm_count detections agree (one false positive can't move it).
"""

import json

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time

from geometry_msgs.msg import PointStamped
from std_msgs.msg import Empty, String

from bottle_tracking import geometry as geo
from bottle_tracking.stretch_interface import StretchInterface, declare


class BottleMemoryNode(Node):
    def __init__(self):
        super().__init__("bottle_memory")
        # One mutually exclusive group for this node's callbacks so the saved state is
        # never changed by two callbacks at once. (They are all fast.)
        cb = MutuallyExclusiveCallbackGroup()
        self.robot = StretchInterface(self)

        detection_topic = declare(self, "detection_topic", "/pill_bottle/detection")
        self.same_radius = declare(self, "same_bottle_radius_m", 0.10)
        self.alpha = declare(self, "smoothing_alpha", 0.3)
        self.confirm_needed = declare(self, "moved_confirm_count", 3)
        self.missing_timeout = declare(self, "missing_timeout_s", 3.0)
        self.view_cone = declare(self, "view_cone_deg", 15.0)
        self.max_range = declare(self, "max_detect_range_m", 2.5)
        rate = declare(self, "publish_rate_hz", 10.0)

        self.pose = None          # (x, y, z) in fixed frame
        self.status = "NONE"
        self.last_seen = None     # ROS time (seconds) of the last detection that matched
        self.n_obs = 0
        self.candidate = []       # detections at a new spot, waiting for confirmation
        self.in_view_since = None
        self.in_view = False

        self.pose_pub = self.create_publisher(PointStamped, "/bottle_memory/pose", 10)
        self.state_pub = self.create_publisher(String, "/bottle_memory/state", 10)
        self.event_pub = self.create_publisher(String, "/bottle_memory/event", 10)
        self.create_subscription(PointStamped, detection_topic, self.on_detection, 10, callback_group=cb)
        self.create_subscription(Empty, "/bottle_memory/clear", self.on_clear, 10, callback_group=cb)
        self.create_timer(1.0 / rate, self.publish, callback_group=cb)
        self.create_timer(0.2, self.check_missing, callback_group=cb)
        self.get_logger().info(f"bottle memory ready, listening on {detection_topic}, "
                               f"saving in '{self.robot.fixed_frame}'")

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def event(self, text):
        self.get_logger().info(text)
        self.event_pub.publish(String(data=text))

    # -------------------------------------------------------------- inputs
    def on_clear(self, _msg):
        self.pose, self.status, self.last_seen, self.n_obs = None, "NONE", None, 0
        self.candidate, self.in_view_since = [], None
        self.event("cleared")

    def on_detection(self, msg):
        p = (msg.point.x, msg.point.y, msg.point.z)
        stamp = Time.from_msg(msg.header.stamp) if msg.header.stamp.sec > 0 else None
        fixed = self.robot.point_to(p, msg.header.frame_id, self.robot.fixed_frame, stamp)
        if fixed is None:
            self.get_logger().warn(f"no TF {msg.header.frame_id} -> {self.robot.fixed_frame}",
                                   throttle_duration_sec=5.0)
            return
        now = self.now()

        if self.pose is None:
            self._accept(fixed, now, "found")
            return

        if geo.dist(fixed, self.pose) <= self.same_radius:
            # same bottle: smooth the position so depth noise doesn't make it jump
            a = self.alpha
            self.pose = tuple(a * n + (1 - a) * o for n, o in zip(fixed, self.pose))
            self.last_seen, self.n_obs = now, self.n_obs + 1
            self.candidate, self.in_view_since = [], None
            if self.status != "KNOWN":
                self.status = "KNOWN"
                self.event("found again at saved position")
            return

        # far from the saved spot: either the bottle moved or it's a false positive
        if self.candidate and geo.dist(fixed, self.candidate[-1]) > self.same_radius:
            self.candidate = []
        self.candidate.append(fixed)
        if len(self.candidate) >= self.confirm_needed:
            new = tuple(sum(c[i] for c in self.candidate) / len(self.candidate) for i in range(3))
            moved = geo.dist(new, self.pose)
            self._accept(new, now, f"moved {moved:.2f} m")

    def _accept(self, p, now, why):
        self.pose, self.status, self.last_seen, self.n_obs = p, "KNOWN", now, 1
        self.candidate, self.in_view_since = [], None
        self.event(f"{why}: x={p[0]:.2f} y={p[1]:.2f} z={p[2]:.2f}")

    # -------------------------------------------------------------- missing check
    def check_missing(self):
        if self.pose is None:
            self.in_view = False
            return
        cam_pos, cam_dir = self.robot.camera_in(self.robot.fixed_frame)
        if cam_pos is None:
            return
        self.in_view = geo.in_view(cam_pos, cam_dir, self.pose, self.view_cone, self.max_range)
        if self.status != "KNOWN" or not self.in_view:
            self.in_view_since = None
            return
        now = self.now()
        if self.in_view_since is None:
            self.in_view_since = now
        looked_long_enough = now - self.in_view_since >= self.missing_timeout
        not_seen = self.last_seen is None or now - self.last_seen >= self.missing_timeout
        if looked_long_enough and not_seen:
            self.status = "MISSING"
            self.event(f"missing: looked at saved spot for {self.missing_timeout:.1f}s, no bottle")

    # -------------------------------------------------------------- outputs
    def publish(self):
        now = self.now()
        state = {"status": self.status, "frame": self.robot.fixed_frame,
                 "n_obs": self.n_obs, "in_view": self.in_view,
                 "last_seen": self.last_seen,
                 "age_s": None if self.last_seen is None else round(now - self.last_seen, 3)}
        if self.pose is not None:
            state.update(x=self.pose[0], y=self.pose[1], z=self.pose[2])
            msg = PointStamped()
            msg.header.frame_id = self.robot.fixed_frame
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.point.x, msg.point.y, msg.point.z = self.pose
            self.pose_pub.publish(msg)
        self.state_pub.publish(String(data=json.dumps(state)))


def main():
    rclpy.init()
    node = BottleMemoryNode()
    ex = MultiThreadedExecutor(num_threads=4)
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
