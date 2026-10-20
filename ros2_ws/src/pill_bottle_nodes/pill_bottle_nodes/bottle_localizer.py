# bottle_localizer node: turns the best bottle box + aligned depth into a 3D position and orientation.
#   subscribes /pill_bottle/detections, /camera/aligned_depth_to_color/image_raw, /camera/color/camera_info, tf
#   publishes  /pill_bottle/state  (BottleState in base_link)
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener

from pill_bottle_interfaces.msg import BottleState, Detections

from .common import depth_to_meters, lookup_matrix
from .geometry import box_points, classify_bottle, transform_points


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
        g = lambda n: self.get_parameter(n).value
        self.target = g("target_frame")
        self.depth = None
        self.K = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(Image, g("depth_topic"), self.on_depth, 1)
        self.create_subscription(CameraInfo, g("info_topic"), self.on_info, 1)
        self.create_subscription(Detections, "/pill_bottle/detections", self.on_detections, 1)
        self.pub = self.create_publisher(BottleState, "/pill_bottle/state", 1)
        self.get_logger().info(f"Localizer ready; publishing /pill_bottle/state in {self.target}")

    def on_depth(self, msg):
        self.depth = msg

    def on_info(self, msg):
        self.K = list(msg.k)

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
        out = BottleState()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.target
        out.position.x, out.position.y, out.position.z = (float(v) for v in info["center"])
        out.orientation = info["orientation"]
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
