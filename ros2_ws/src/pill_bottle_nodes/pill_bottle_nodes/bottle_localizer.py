# bottle_localizer node: turns the best bottle box + aligned depth into a 3D position and orientation.
#   subscribes /pill_bottle/detections, /camera/aligned_depth_to_color/image_raw, /camera/color/camera_info, tf,
#              /pill_bottle/orientation_vision (optional, from orientation_classifier; T17)
#   publishes  /pill_bottle/state  (BottleState in base_link)
# The orientation is the depth fit combined with the vision model when it's running (orientation.fuse_orientation).
# axis_tilt_deg stays the raw depth angle, so testers can still see what depth alone said.
import json
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from pill_bottle_interfaces.msg import BottleState, Detections

from .common import depth_to_meters, lookup_matrix
from .geometry import box_points, classify_bottle, transform_points
from .orientation import ORIENTATION_SETTINGS, fuse_orientation


class BottleLocalizerNode(Node):
    def __init__(self):
        super().__init__("bottle_localizer")
        self.declare_parameter("depth_topic", "/camera/aligned_depth_to_color/image_raw")
        self.declare_parameter("info_topic", "/camera/color/camera_info")
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("bottle_class", "pill_bottle")
        self.declare_parameter("min_conf", 0.4)
        self.declare_parameter("min_points", 40)
        self.declare_parameter("bottle_radius", 0.02)
        self.declare_parameter("depth_band", 0.06)      # keep depth within this of the box's median (m)
        self.declare_parameter("upright_max_deg", 30.0)
        self.declare_parameter("lying_min_deg", 60.0)
        self.declare_parameter("min_range", 0.22)       # m from base_link; closer = part of the robot, not the bottle
        for k, v in ORIENTATION_SETTINGS.items():
            self.declare_parameter(k, v)
        g = lambda n: self.get_parameter(n).value
        self.target = g("target_frame")
        self.vision = None                              # (receive time, dict) from orientation_classifier
        self.depth = None
        self.K = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(Image, g("depth_topic"), self.on_depth, 1)
        self.create_subscription(CameraInfo, g("info_topic"), self.on_info, 1)
        self.create_subscription(Detections, "/pill_bottle/detections", self.on_detections, 1)
        self.create_subscription(String, "/pill_bottle/orientation_vision", self.on_vision, 1)
        self.pub = self.create_publisher(BottleState, "/pill_bottle/state", 1)
        self.get_logger().info(f"Localizer ready; publishing /pill_bottle/state in {self.target}")

    def on_depth(self, msg):
        self.depth = msg

    def on_info(self, msg):
        self.K = list(msg.k)

    def on_vision(self, msg):
        try:
            self.vision = (time.time(), json.loads(msg.data))
        except ValueError:
            pass

    def on_detections(self, msg):
        g = lambda n: self.get_parameter(n).value
        if self.depth is None or self.K is None:
            self.get_logger().warn("Waiting for depth image and camera_info...", throttle_duration_sec=5.0)
            return
        bottles = [d for d in msg.detections if d.name == g("bottle_class") and d.confidence >= g("min_conf")]
        if not bottles:
            return
        best = max(bottles, key=lambda d: d.confidence)
        depth = depth_to_meters(self.depth)
        if depth.shape != (msg.image_height, msg.image_width):
            self.get_logger().warn(f"Depth {depth.shape} doesn't match color {msg.image_height}x{msg.image_width}; "
                                   "is align_depth enabled?", throttle_duration_sec=5.0)
            return
        pts_cam = box_points(depth, (best.x_min, best.y_min, best.x_max, best.y_max), self.K, band=g("depth_band"))
        if len(pts_cam) < g("min_points"):
            self.get_logger().warn(f"Only {len(pts_cam)} depth points on the bottle; skipping",
                                   throttle_duration_sec=2.0)
            return
        try:
            T = lookup_matrix(self.tf_buffer, self.target, msg.header.frame_id, Time.from_msg(msg.header.stamp))
        except Exception:
            try:
                T = lookup_matrix(self.tf_buffer, self.target, msg.header.frame_id)
            except Exception as e:
                self.get_logger().warn(f"No tf {msg.header.frame_id} -> {self.target}: {e}", throttle_duration_sec=5.0)
                return
        pts = transform_points(T, pts_cam)
        info = classify_bottle(pts, radius=g("bottle_radius"),
                               upright_max_deg=g("upright_max_deg"), lying_min_deg=g("lying_min_deg"))
        cx, cy = float(info["center"][0]), float(info["center"][1])
        if (cx * cx + cy * cy) ** 0.5 < g("min_range"):
            self.get_logger().warn(f"Ignoring a '{best.name}' at x {cx:+.2f}, y {cy:+.2f} m: within {g('min_range')} m of "
                                   "the robot's center, so it's probably part of the robot.", throttle_duration_sec=3.0)
            return
        out = BottleState()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.target
        out.position.x, out.position.y, out.position.z = (float(v) for v in info["center"])
        vision = None
        if self.vision is not None:
            vision = dict(self.vision[1], age_s=time.time() - self.vision[0])
        s = {k: g(k) for k in ORIENTATION_SETTINGS}
        orientation, source, agree = fuse_orientation(info["tilt_deg"], len(pts), vision, s,
                                                      g("upright_max_deg"), g("lying_min_deg"))
        if agree is False:
            self.get_logger().info(f"Orientation: depth says {info['orientation']} ({info['tilt_deg']:.0f} deg, "
                                   f"{len(pts)} pts), vision says {vision['label']} ({vision['conf']:.2f}); "
                                   f"using {orientation} ({source})", throttle_duration_sec=2.0)
        if orientation != info["orientation"]:           # keep surface_z consistent with the orientation used
            info["surface_z"] = (float(info["center"][2] - g("bottle_radius")) if orientation == "lying"
                                 else float(np.percentile(pts[:, 2], 5)))
        out.orientation = orientation
        out.axis_tilt_deg = float(info["tilt_deg"])
        out.axis_yaw = float(info["axis_yaw"])
        out.surface_z = float(info["surface_z"])
        out.confidence = float(best.confidence)
        out.num_points = int(len(pts))
        self.pub.publish(out)


def main():
    rclpy.init()
    node = BottleLocalizerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
