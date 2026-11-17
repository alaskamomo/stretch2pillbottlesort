# orientation_classifier node (T17): your YOLOv8 CLASSIFICATION model says whether the bottle is upright or lying.
#   subscribes /camera/color/image_raw, /pill_bottle/detections
#   publishes  /pill_bottle/orientation_vision  (std_msgs/String, JSON: label, class, conf, box, stamp)
# bottle_localizer combines this with its depth fit (orientation.fuse_orientation) before publishing
# /pill_bottle/state. Train the model with training/orientation_classifier_colab.ipynb.
#
#   ros2 run pill_bottle_nodes orientation_classifier --ros-args -p model_path:=/home/hello-robot/orientation_cls.pt
import json
import os

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from pill_bottle_interfaces.msg import Detections

from .common import image_to_numpy
from .geometry import rotate_image
from .orientation import label_from_class


def crop_box(raw, box, pad=0.10):
    """Padded crop of a RAW-pixel box (clipped to the image)."""
    H, W = raw.shape[:2]
    x1, y1, x2, y2 = box
    px, py = pad * (x2 - x1), pad * (y2 - y1)
    x1, y1 = int(max(0, x1 - px)), int(max(0, y1 - py))
    x2, y2 = int(min(W, x2 + px)), int(min(H, y2 + py))
    return raw[y1:y2, x1:x2]


class OrientationClassifierNode(Node):
    def __init__(self):
        super().__init__("orientation_classifier")
        p = self.declare_parameter
        self.model_path = os.path.expanduser(p("model_path", "~/pill_bottle_robot/orientation_cls.pt").value)
        self.rotate = p("rotate", "cw").value                  # same as bottle_detector: crops must look upright
        self.min_box_conf = p("min_box_conf", 0.4).value      # same bar as bottle_localizer's min_conf
        self.pad = p("pad", 0.10).value                       # same padding as the Colab notebook's crops
        rate = p("rate_hz", 2.0).value
        image_topic = p("image_topic", "/camera/color/image_raw").value
        if not os.path.isfile(self.model_path):
            raise SystemExit(f"orientation_classifier: model not found: {self.model_path}  (-p model_path:=...)")
        try:
            from ultralytics import YOLO
        except ImportError:
            raise SystemExit('orientation_classifier: run  pip3 install ultralytics "numpy<2"')
        self.model = YOLO(self.model_path)
        if getattr(self.model, "task", "classify") != "classify":
            raise SystemExit(f"orientation_classifier: {self.model_path} is a '{self.model.task}' model; it needs the "
                             "CLASSIFICATION model from the orientation notebook (yolov8n-cls)")
        mapped = {n: label_from_class(n) for n in self.model.names.values()}
        if None in mapped.values():
            self.get_logger().warn(f"Some classes don't map to upright/lying/tilted and will be ignored: {mapped}")
        self.image, self.dets = None, None
        self.create_subscription(Image, image_topic, lambda m: setattr(self, "image", m), 1)
        self.create_subscription(Detections, "/pill_bottle/detections", lambda m: setattr(self, "dets", m), 1)
        self.pub = self.create_publisher(String, "/pill_bottle/orientation_vision", 1)
        self.create_timer(1.0 / rate, self.tick)
        self.get_logger().info(f"Orientation classifier ready: {self.model_path}, classes {mapped}")

    def tick(self):
        img, dets = self.image, self.dets
        if img is None or dets is None:
            return
        self.dets = None                                       # one answer per detector frame
        bottles = [d for d in dets.detections if d.name == "pill_bottle" and d.confidence >= self.min_box_conf]
        if not bottles:
            return
        best = max(bottles, key=lambda d: d.confidence)         # the same box bottle_localizer uses
        raw = image_to_numpy(img)
        if raw.ndim == 2:
            raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        if raw.shape[:2] != (dets.image_height, dets.image_width):
            self.get_logger().warn("color image size doesn't match the detections", throttle_duration_sec=5.0)
            return
        box = (best.x_min, best.y_min, best.x_max, best.y_max)
        crop = rotate_image(crop_box(raw, box, self.pad), self.rotate)
        if crop.size == 0:
            return
        r = self.model(crop, verbose=False)[0]
        name = self.model.names[int(r.probs.top1)]
        out = {"label": label_from_class(name), "class": name, "conf": round(float(r.probs.top1conf), 3),
               "box": [round(v) for v in box],
               "stamp": dets.header.stamp.sec + dets.header.stamp.nanosec * 1e-9}
        self.pub.publish(String(data=json.dumps(out)))


def main():
    rclpy.init()
    node = OrientationClassifierNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
