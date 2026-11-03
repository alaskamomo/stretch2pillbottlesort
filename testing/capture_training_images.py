#!/usr/bin/env python3
"""
capture_training_images.py - save head camera pictures for YOLO training.

Needs the camera running (terminal 2). Saves full-size color images, turned upright the
same way the detector sees them, with no boxes drawn on them.

  python3 testing/capture_training_images.py                 press Enter for each picture
  python3 testing/capture_training_images.py --name box      start file names with "box"
  python3 testing/capture_training_images.py --name empty    pictures with NO bottle in them

Type q and press Enter to stop. Pictures go to ~/stretch_training_images/.
"""

import argparse
import os
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


def to_bgr(msg):
    """sensor_msgs/Image (rgb8 or bgr8) -> BGR numpy array."""
    if msg.encoding not in ("rgb8", "bgr8"):
        raise ValueError(f"unsupported image encoding {msg.encoding}")
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)[:, :msg.width * 3]
    img = img.reshape(msg.height, msg.width, 3)
    return img[:, :, ::-1] if msg.encoding == "rgb8" else img


def upright(img, mode):
    if mode == "cw":
        return np.rot90(img, k=-1)
    if mode == "ccw":
        return np.rot90(img, k=1)
    return img


class Capture(Node):
    def __init__(self, topic):
        super().__init__("capture_training_images")
        self.latest = None
        self.stamp = 0.0
        self.create_subscription(Image, topic, self.on_image, qos_profile_sensor_data)

    def on_image(self, msg):
        self.latest = msg
        self.stamp = time.time()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="head", help="start of each file name")
    p.add_argument("--topic", default="/camera/color/image_raw")
    p.add_argument("--rotate", default="cw", choices=["cw", "ccw", "none"], help="same as rotate_image")
    p.add_argument("--dir", default=os.path.expanduser("~/stretch_training_images"))
    args = p.parse_args()

    os.makedirs(args.dir, exist_ok=True)
    rclpy.init()
    node = Capture(args.topic)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    count = 0
    print(f"Saving to {args.dir}. Press Enter for a picture, q then Enter to stop.")
    try:
        while True:
            if input(f"[{count} saved] ").strip().lower() == "q":
                break
            if node.latest is None or time.time() - node.stamp > 2.0:
                print("  No fresh camera image. Is terminal 2 running?")
                continue
            img = upright(to_bgr(node.latest), args.rotate)
            path = os.path.join(args.dir, f"{args.name}_{time.strftime('%Y%m%d_%H%M%S')}_{count:03d}.jpg")
            if cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                count += 1
                print(f"  {path}")
            else:
                print("  Could not write the picture.")
    except (KeyboardInterrupt, EOFError):
        pass
    print(f"\nSaved {count} pictures in {args.dir}")
    if rclpy.ok():
        rclpy.shutdown()


if __name__ == "__main__":
    main()
