# bottle_detector node: runs your YOLO model (detect_bottles.detect_all) on the head camera.
#   subscribes /camera/color/image_raw
#   publishes  /pill_bottle/detections   (boxes in RAW image pixels, so they line up with depth)
#              /pill_bottle/debug_image  (upright image with boxes drawn)
import time

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image

from pill_bottle_interfaces.msg import Detection, Detections

from .common import DEFAULT_LEGACY_DIR, image_to_numpy, load_legacy, load_model, numpy_to_image
from .geometry import box_rotated_to_raw, rotate_image


class BottleDetectorNode(Node):
    def __init__(self):
        super().__init__("bottle_detector")
        self.declare_parameter("legacy_dir", DEFAULT_LEGACY_DIR)
        self.declare_parameter("image_topic", "/camera/color/image_raw")
        self.declare_parameter("rotate", "cw")          # same as ROTATE in full_run.py
        self.declare_parameter("conf", 0.25)            # low on purpose; the localizer and verifier set their own bars
        self.declare_parameter("rate_hz", 2.0)
        self.declare_parameter("save_debug", "")        # e.g. /tmp/last_detection.jpg
        legacy_dir = self.get_parameter("legacy_dir").value
        self.db = load_legacy("detect_bottles", legacy_dir)
        self.model = load_model(self.db, legacy_dir)
        self.rotate = self.get_parameter("rotate").value
        self.conf = self.get_parameter("conf").value
        self.save_debug = self.get_parameter("save_debug").value
        self.latest = None
        self.create_subscription(Image, self.get_parameter("image_topic").value, self.on_image, 1)
        self.pub = self.create_publisher(Detections, "/pill_bottle/detections", 1)
        self.dbg = self.create_publisher(Image, "/pill_bottle/debug_image", 1)
        self.create_timer(1.0 / self.get_parameter("rate_hz").value, self.tick)
        self.get_logger().info(f"Detector ready (rotate={self.rotate}, conf={self.conf})")

    def on_image(self, msg):
        self.latest = msg

    def tick(self):
        msg, self.latest = self.latest, None
        if msg is None:
            return
        raw = image_to_numpy(msg)
        if raw.ndim == 2:
            raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        H, W = raw.shape[:2]
        up = rotate_image(raw, self.rotate)
        t0 = time.time()
        result, dets = self.db.detect_all(self.model, up, conf=self.conf)
        out = Detections()
        out.header = msg.header
        out.image_width, out.image_height = W, H
        for d in dets:
            x1, y1, x2, y2 = box_rotated_to_raw(d["box"], W, H, self.rotate)
            out.detections.append(Detection(name=str(d["name"]), confidence=float(d["conf"]),
                                            x_min=float(x1), y_min=float(y1), x_max=float(x2), y_max=float(y2)))
        self.pub.publish(out)
        annotated = result.plot() if result is not None else up
        self.dbg.publish(numpy_to_image(annotated, msg.header.stamp, msg.header.frame_id))
        if self.save_debug:
            cv2.imwrite(self.save_debug, annotated)
        self.get_logger().debug(f"{len(dets)} detections in {time.time() - t0:.2f} s")


def main():
    rclpy.init()
    node = BottleDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
