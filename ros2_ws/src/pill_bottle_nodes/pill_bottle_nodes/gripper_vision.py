# gripper_vision node: finds the bottle in the claw camera (wraps gripper_cam.analyze).
#   subscribes /gripper_camera/image_raw
#   publishes  /pill_bottle/gripper_view        (GripperView)
#              /pill_bottle/gripper_debug_image (what gripper_cam.save_debug draws, if available)
import os
import tempfile

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from pill_bottle_interfaces.msg import GripperView

from .common import DEFAULT_LEGACY_DIR, image_to_numpy, load_legacy, load_model, numpy_to_image


class GripperVisionNode(Node):
    def __init__(self):
        super().__init__("gripper_vision")
        self.declare_parameter("legacy_dir", DEFAULT_LEGACY_DIR)
        self.declare_parameter("rate_hz", 2.0)
        legacy_dir = self.get_parameter("legacy_dir").value
        self.gc = load_legacy("gripper_cam", legacy_dir)
        db = load_legacy("detect_bottles", legacy_dir)
        self.model = load_model(db, legacy_dir)
        self.latest = None
        self.debug_path = os.path.join(tempfile.gettempdir(), "gripper_view.jpg")
        self.create_subscription(Image, "/gripper_camera/image_raw", self.on_image, 1)
        self.pub = self.create_publisher(GripperView, "/pill_bottle/gripper_view", 1)
        self.dbg = self.create_publisher(Image, "/pill_bottle/gripper_debug_image", 1)
        self.create_timer(1.0 / self.get_parameter("rate_hz").value, self.tick)
        self.get_logger().info("Gripper vision ready")

    def on_image(self, msg):
        self.latest = msg

    def tick(self):
        msg, self.latest = self.latest, None
        if msg is None:
            return
        frame = image_to_numpy(msg)
        info = self.gc.analyze(frame, self.model)
        out = GripperView()
        out.header = msg.header
        out.seen = info is not None
        if info:
            out.orientation = str(info.get("orientation", ""))
            out.confidence = float(info.get("conf", 0.0))
            cu, cv = info.get("center", (0.0, 0.0))
            out.center_u, out.center_v = float(cu), float(cv)
            ox, oy = info.get("offset_m", (0.0, 0.0))
            out.offset_x_m, out.offset_y_m = float(ox), float(oy)
            dist = info.get("dist")
            out.distance_m = float(dist) if dist else -1.0
        else:
            out.distance_m = -1.0
        self.pub.publish(out)
        if hasattr(self.gc, "save_debug"):
            try:
                self.gc.save_debug(frame, info, self.debug_path)
                img = cv2.imread(self.debug_path)
                if img is not None:
                    self.dbg.publish(numpy_to_image(img, msg.header.stamp, msg.header.frame_id))
            except Exception as e:
                self.get_logger().warn(f"save_debug failed: {e}", throttle_duration_sec=10.0)


def main():
    rclpy.init()
    node = GripperVisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
