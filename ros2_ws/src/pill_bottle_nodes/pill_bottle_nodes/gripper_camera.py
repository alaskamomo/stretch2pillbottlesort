# gripper_camera node: publishes the black-and-white claw camera (wraps gripper_cam.GripperCamera).
#   publishes /gripper_camera/image_raw (mono8, already rotated the way gripper_cam.py sets ROTATE)
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from .common import DEFAULT_LEGACY_DIR, load_legacy, numpy_to_image


class GripperCameraNode(Node):
    def __init__(self):
        super().__init__("gripper_camera")
        self.declare_parameter("legacy_dir", DEFAULT_LEGACY_DIR)
        self.declare_parameter("device", "")          # empty = gripper_cam.DEVICE (/dev/video6)
        self.declare_parameter("rate_hz", 10.0)
        self.declare_parameter("frame_id", "gripper_camera")
        legacy_dir = self.get_parameter("legacy_dir").value
        self.gc = load_legacy("gripper_cam", legacy_dir)
        device = self.get_parameter("device").value or self.gc.DEVICE
        self.frame_id = self.get_parameter("frame_id").value
        self.cam = self.gc.GripperCamera(device)
        self.pub = self.create_publisher(Image, "/gripper_camera/image_raw", 1)
        self.timer = self.create_timer(1.0 / self.get_parameter("rate_hz").value, self.tick)
        self.fails = 0
        self.get_logger().info(f"Gripper camera {device} open, publishing /gripper_camera/image_raw")

    def tick(self):
        try:
            frame = self.cam.grab()
        except Exception as e:  # camera unplugged or busy
            self.fails += 1
            if self.fails % 20 == 1:
                self.get_logger().warn(f"Gripper camera read failed: {e}")
            return
        if frame is None:
            return
        if frame.ndim == 3 and frame.shape[2] == 1:
            frame = frame[:, :, 0]
        self.pub.publish(numpy_to_image(frame, self.get_clock().now().to_msg(), self.frame_id))

    def destroy_node(self):
        try:
            self.cam.stop()
        except Exception:
            pass
        super().destroy_node()


def main():
    rclpy.init()
    node = GripperCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
