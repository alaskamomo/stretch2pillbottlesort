#!/usr/bin/env python3
"""
yolo_detector_node - finds the pill bottle with YOUR trained YOLOv8 model on the head (top) camera.

Runs the model on the head camera color image, takes the depth at the bottle from the
aligned depth image, and publishes the bottle's 3D position:

  /pill_bottle/detection       geometry_msgs/PointStamped  bottle center, camera frame, image timestamp
  /pill_bottle/detection_info  std_msgs/String (JSON)      confidence, pixel, depth, box
  /pill_bottle/debug_image     sensor_msgs/Image           camera image with the box drawn (view with rqt_image_view)

Nothing is published when no pill bottle is seen.

  ros2 launch bottle_tracking bottle_tracking.launch.py model_path:=/home/hello-robot/best.pt

Settings (config/params.yaml): model_path, class_name, conf_threshold, rotate_image.
The Stretch head camera is mounted sideways. If you trained on UPRIGHT images, keep
rotate_image: cw (the image is turned upright before YOLO). If you trained on the raw
sideways images, set rotate_image: none. Check with rqt_image_view on /pill_bottle/debug_image.

Needs: pip3 install ultralytics "numpy<2"   (numpy 2 breaks ROS Humble)
"""

import json
import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

import message_filters
from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from bottle_tracking.detection_publisher import deproject


def image_to_numpy(msg):
    """sensor_msgs/Image -> numpy, without cv_bridge."""
    if msg.encoding in ("rgb8", "bgr8"):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
        arr = arr[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
        return arr[:, :, ::-1].copy() if msg.encoding == "rgb8" else arr.copy()   # -> BGR
    if msg.encoding in ("16UC1", "mono16"):
        arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.step // 2)
        return arr[:, :msg.width]
    if msg.encoding == "32FC1":
        arr = np.frombuffer(msg.data, dtype=np.float32).reshape(msg.height, msg.step // 4)
        return arr[:, :msg.width]
    raise ValueError(f"unsupported image encoding {msg.encoding}")


def numpy_to_image(arr_bgr, header):
    msg = Image()
    msg.header = header
    msg.height, msg.width = arr_bgr.shape[:2]
    msg.encoding = "bgr8"
    msg.step = msg.width * 3
    msg.data = np.ascontiguousarray(arr_bgr).tobytes()
    return msg


def rotate(img, mode):
    if mode == "cw":
        return np.rot90(img, k=-1)
    if mode == "ccw":
        return np.rot90(img, k=1)
    return img


def box_to_raw(box, mode, raw_w, raw_h):
    """Map a box (x1, y1, x2, y2) found in the rotated image back to raw image pixels."""
    x1, y1, x2, y2 = box
    if mode == "cw":        # rotated (u', v') <- raw (u = v', v = H-1-u')
        return y1, raw_h - 1 - x2, y2, raw_h - 1 - x1
    if mode == "ccw":       # rotated (u', v') <- raw (u = W-1-v', v = u')
        return raw_w - 1 - y2, x1, raw_w - 1 - y1, x2
    return x1, y1, x2, y2


def draw_box(img, box, color=(0, 255, 0), t=3):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, x2 = max(0, x1), min(w - 1, x2)
    y1, y2 = max(0, y1), min(h - 1, y2)
    img[y1:y1 + t, x1:x2] = color
    img[max(y2 - t, 0):y2, x1:x2] = color
    img[y1:y2, x1:x1 + t] = color
    img[y1:y2, max(x2 - t, 0):x2] = color


class YoloDetector(Node):
    def __init__(self):
        super().__init__("yolo_detector")
        p = self.declare_parameter
        override = p("model_path_override", "").value     # from the launch file's model_path:=...
        self.model_path = os.path.expanduser(override or p("model_path", "").value)
        self.class_name = p("class_name", "pill_bottle").value
        self.conf_thr = p("conf_threshold", 0.5).value
        self.rotate_mode = p("rotate_image", "cw").value
        self.max_rate = p("max_rate_hz", 5.0).value
        self.depth_scale = p("depth_scale", 0.001).value          # 16UC1 depth is millimeters
        self.depth_offset = p("depth_offset_m", 0.02).value       # front surface -> bottle center
        self.publish_debug = p("publish_debug_image", True).value
        color_topic = p("color_topic", "/camera/color/image_raw").value
        depth_topic = p("depth_topic", "/camera/aligned_depth_to_color/image_raw").value
        info_topic = p("camera_info_topic", "/camera/color/camera_info").value

        if not self.model_path:
            raise SystemExit("yolo_detector: set model_path in params.yaml or launch with model_path:=/path/to/best.pt")
        if not os.path.isfile(self.model_path):
            raise SystemExit(f"yolo_detector: model file not found: {self.model_path}")
        try:
            from ultralytics import YOLO
        except ImportError:
            raise SystemExit('yolo_detector: run  pip3 install ultralytics "numpy<2"')
        self.model = YOLO(self.model_path)
        names = self.model.names
        self.class_ids = [i for i, n in names.items() if n == self.class_name]
        if not self.class_ids:
            raise SystemExit(f"yolo_detector: class '{self.class_name}' not in model classes {names}")

        self.K = None
        self.last_run = 0.0
        self.det_pub = self.create_publisher(PointStamped, "/pill_bottle/detection", 10)
        self.info_pub = self.create_publisher(String, "/pill_bottle/detection_info", 10)
        self.dbg_pub = self.create_publisher(Image, "/pill_bottle/debug_image", 2)
        self.create_subscription(CameraInfo, info_topic, self.on_info, qos_profile_sensor_data)
        color_sub = message_filters.Subscriber(self, Image, color_topic, qos_profile=qos_profile_sensor_data)
        depth_sub = message_filters.Subscriber(self, Image, depth_topic, qos_profile=qos_profile_sensor_data)
        self.sync = message_filters.ApproximateTimeSynchronizer([color_sub, depth_sub], 5, 0.05)
        self.sync.registerCallback(self.on_images)
        self.get_logger().info(f"YOLO detector ready: model={self.model_path}, class='{self.class_name}', "
                               f"rotate_image={self.rotate_mode}, waiting for {color_topic} + {depth_topic}")

    def on_info(self, msg):
        self.K = list(msg.k)

    def on_images(self, color_msg, depth_msg):
        now = time.time()
        if self.K is None or now - self.last_run < 1.0 / self.max_rate:
            return
        self.last_run = now
        try:
            color = image_to_numpy(color_msg)
            depth = image_to_numpy(depth_msg)
        except ValueError as e:
            self.get_logger().error(str(e), throttle_duration_sec=10.0)
            return
        raw_h, raw_w = color.shape[:2]
        view = np.ascontiguousarray(rotate(color, self.rotate_mode))

        result = self.model(view, conf=self.conf_thr, classes=self.class_ids, verbose=False)[0]
        best = None
        for b in result.boxes:
            c = float(b.conf[0])
            if best is None or c > best[0]:
                best = (c, [float(v) for v in b.xyxy[0]])

        published = False
        if best is not None:
            conf, box = best
            u1, v1, u2, v2 = box_to_raw(box, self.rotate_mode, raw_w, raw_h)
            # median depth over the middle half of the box (edges hit the table/background)
            cu, cv = (u1 + u2) / 2, (v1 + v2) / 2
            hw, hh = max(2, (u2 - u1) / 4), max(2, (v2 - v1) / 4)
            region = depth[int(max(0, cv - hh)):int(min(raw_h, cv + hh)),
                           int(max(0, cu - hw)):int(min(raw_w, cu + hw))].astype(np.float64)
            if depth.dtype == np.uint16:
                region = region * self.depth_scale
            valid = region[np.isfinite(region) & (region > 0.1) & (region < 5.0)]
            if valid.size >= 10:
                d = float(np.median(valid)) + self.depth_offset
                x, y, z = deproject(cu, cv, d, self.K)
                msg = PointStamped()
                msg.header = color_msg.header
                msg.point.x, msg.point.y, msg.point.z = x, y, z
                self.det_pub.publish(msg)
                self.info_pub.publish(String(data=json.dumps({
                    "confidence": round(conf, 3), "pixel": [round(cu), round(cv)],
                    "depth_m": round(d, 3), "valid_depth_pixels": int(valid.size),
                    "box_rotated_image": [round(v) for v in box]})))
                published = True
            else:
                self.get_logger().warn("bottle seen but no valid depth at it (too close/far or shiny?)",
                                       throttle_duration_sec=3.0)
        if self.publish_debug and self.dbg_pub.get_subscription_count() > 0:
            if best is not None:
                draw_box(view, best[1], (0, 255, 0) if published else (0, 0, 255))
            self.dbg_pub.publish(numpy_to_image(view, color_msg.header))


def main():
    rclpy.init()
    node = YoloDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
