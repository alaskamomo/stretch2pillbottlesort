"""
common.py - small helpers shared by the nodes and testers.
"""

import json
import os
import threading
import time

from rclpy.qos import QoSProfile, DurabilityPolicy
from std_msgs.msg import Bool, Empty, String
from std_srvs.srv import Trigger

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


class MemoryWatcher:
    """Keeps the latest /bottle_memory/state and /head_tracker/status, and lets you
    turn tracking on/off or clear the memory."""

    def __init__(self, node, cb_group=None):
        self.node = node
        self.state = {"status": "NONE"}
        self.tracker = {}
        self.events = []
        self._lock = threading.Lock()
        node.create_subscription(String, "/bottle_memory/state", self._on_state, 10, callback_group=cb_group)
        node.create_subscription(String, "/head_tracker/status", self._on_tracker, 10, callback_group=cb_group)
        node.create_subscription(String, "/bottle_memory/event", self._on_event, 10, callback_group=cb_group)
        self.enable_pub = node.create_publisher(Bool, "/head_tracker/enable", LATCHED)
        self.clear_pub = node.create_publisher(Empty, "/bottle_memory/clear", 10)

    def _on_state(self, msg):
        with self._lock:
            self.state = json.loads(msg.data)

    def _on_tracker(self, msg):
        with self._lock:
            self.tracker = json.loads(msg.data)

    def _on_event(self, msg):
        with self._lock:
            self.events.append((time.time(), msg.data))

    def get(self):
        with self._lock:
            return dict(self.state)

    def tracker_status(self):
        with self._lock:
            return dict(self.tracker)

    def events_since(self, t):
        with self._lock:
            return [e for ts, e in self.events if ts >= t]

    def now(self):
        return self.node.get_clock().now().nanoseconds * 1e-9

    def seen_since(self, t_ros):
        """True if memory got a matching detection after ROS time t_ros."""
        s = self.get()
        return s.get("last_seen") is not None and s["last_seen"] > t_ros

    def detection_fresh(self, max_age=0.5):
        s = self.get()
        return s.get("age_s") is not None and s["age_s"] <= max_age

    def position(self):
        s = self.get()
        if "x" not in s:
            return None
        return (s["x"], s["y"], s["z"])

    def set_tracking(self, on):
        self.enable_pub.publish(Bool(data=bool(on)))

    def clear(self):
        self.clear_pub.publish(Empty())

    def wait_for(self, condition, timeout, period=0.05):
        end = time.time() + timeout
        while time.time() < end:
            if condition():
                return True
            time.sleep(period)
        return False


def call_trigger(client, timeout=120.0):
    """Call a std_srvs/Trigger service and wait. Returns (success, message) or (False, reason)."""
    if not client.wait_for_service(timeout_sec=5.0):
        return False, f"service {client.srv_name} not available"
    fut = client.call_async(Trigger.Request())
    end = time.time() + timeout
    while not fut.done() and time.time() < end:
        time.sleep(0.05)
    if not fut.done() or fut.result() is None:
        return False, "service call timed out"
    return fut.result().success, fut.result().message


class DetectionCollector:
    """Collects raw detections from /pill_bottle/detection for the testers.

    Positions are converted with the LATEST transform when you call measure(), not
    per message (that was slow and could drop detections). If too few detections
    arrive, it falls back to bottle_memory's saved position so a slow detector
    (about 2 Hz on the robot's CPU) still works."""

    def __init__(self, node, robot, cb_group=None, memory=None):
        from geometry_msgs.msg import PointStamped
        self.robot = robot
        self.memory = memory          # optional MemoryWatcher for the fallback
        self.samples = []             # (wall_time, (x, y, z), frame_id, confidence)
        self.received = 0
        self.confidence = None
        self._lock = threading.Lock()
        node.create_subscription(PointStamped, "/pill_bottle/detection", self._on_det, 10,
                                 callback_group=cb_group)
        node.create_subscription(String, "/pill_bottle/detection_info", self._on_info, 10,
                                 callback_group=cb_group)

    def _on_info(self, msg):
        try:
            self.confidence = json.loads(msg.data).get("confidence")
        except ValueError:
            pass

    def _on_det(self, msg):
        with self._lock:
            self.received += 1
            self.samples.append((time.time(), (msg.point.x, msg.point.y, msg.point.z),
                                 msg.header.frame_id, self.confidence))
            if len(self.samples) > 2000:
                self.samples = self.samples[-1000:]

    def last_seen_age(self):
        with self._lock:
            return time.time() - self.samples[-1][0] if self.samples else None

    def sees(self, max_age=1.0):
        age = self.last_seen_age()
        return age is not None and age <= max_age

    def _convert(self, p, frame):
        base = self.robot.point_to(p, frame, self.robot.base_frame)
        fixed = self.robot.point_to(p, frame, self.robot.fixed_frame)
        cam = self.robot.point_to(p, frame, self.robot.camera_frame)
        return base, fixed, cam

    def measure(self, seconds=3.0, min_samples=2):
        """Average the detections over the next `seconds` (robot and bottle still).
        Returns a dict, or None if the bottle wasn't seen. dict["source"] says where it came from."""
        t0 = time.time()
        time.sleep(seconds)
        with self._lock:
            got = [s for s in self.samples if s[0] >= t0 - 0.5]
        pts = []
        for _, p, frame, conf in got:
            base, fixed, cam = self._convert(p, frame)
            if base is not None and fixed is not None:
                pts.append((base, fixed, cam, conf))
        source = "detections"
        if len(pts) < min_samples:
            if self.memory is None or not self.memory.detection_fresh(max_age=seconds + 1.0):
                return None
            pos = self.memory.position()
            frame = self.memory.get().get("frame", self.robot.fixed_frame)
            base = self.robot.point_to(pos, frame, self.robot.base_frame)
            cam = self.robot.point_to(pos, frame, self.robot.camera_frame)
            if base is None:
                return None
            pts, source = [(base, pos, cam, self.confidence)], "bottle_memory (few detections)"

        def mean(vals):
            return sum(vals) / len(vals)

        base = tuple(mean([q[0][i] for q in pts]) for i in range(3))
        fixed = tuple(mean([q[1][i] for q in pts]) for i in range(3))
        spread = (mean([(q[1][0] - fixed[0]) ** 2 + (q[1][1] - fixed[1]) ** 2 + (q[1][2] - fixed[2]) ** 2
                        for q in pts])) ** 0.5
        rngs = [(q[2][0] ** 2 + q[2][1] ** 2 + q[2][2] ** 2) ** 0.5 for q in pts if q[2] is not None]
        confs = [q[3] for q in pts if q[3] is not None]
        return {"n": len(pts), "base": base, "fixed": fixed, "spread_cm": spread * 100,
                "camera_range_m": mean(rngs) if rngs else None,
                "base_distance_m": (base[0] ** 2 + base[1] ** 2) ** 0.5,
                "confidence": mean(confs) if confs else None, "source": source}


class SnapshotSaver:
    """Keeps the latest /pill_bottle/debug_image (camera view with the bottle boxed) and
    saves it as a picture for each trial. Subscribing is what makes the detector publish it."""

    def __init__(self, node, cb_group=None, folder=None):
        from sensor_msgs.msg import Image
        self.folder = folder or os.path.join(
            os.environ.get("STRETCH_TEST_RESULTS", os.path.expanduser("~/stretch_test_results")), "images")
        self.latest = None
        self._lock = threading.Lock()
        node.create_subscription(Image, "/pill_bottle/debug_image", self._on_img, 2, callback_group=cb_group)

    def _on_img(self, msg):
        with self._lock:
            self.latest = msg

    def grab(self, lines=()):
        """Newest camera picture as a BGR numpy array with text lines on it, or None."""
        import numpy as np
        with self._lock:
            msg = self.latest
        if msg is None:
            return None
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)[:, :msg.width * 3]
        img = img.reshape(msg.height, msg.width, 3).copy()
        try:
            import cv2
            lines = [ln for ln in lines if ln]
            if lines:   # dark band behind the text so it's readable on any background
                cv2.rectangle(img, (0, 0), (img.shape[1], 12 + 24 * len(lines)), (0, 0, 0), -1)
            for i, line in enumerate(lines):
                cv2.putText(img, line, (8, 26 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                            cv2.LINE_AA)
        except ImportError:
            pass
        return img

    def write(self, name, img):
        """Save a BGR numpy image as <folder>/<name>.png. Returns the path."""
        os.makedirs(self.folder, exist_ok=True)
        path = os.path.join(self.folder, f"{name}.png")
        try:
            import cv2
            cv2.imwrite(path, img)
        except ImportError:
            from PIL import Image as PILImage
            PILImage.fromarray(img[:, :, ::-1]).save(path)
        return path

    def save(self, name, lines=()):
        """Save the newest camera picture as <folder>/<name>.png with text lines on it. Returns the path or None."""
        img = self.grab(lines)
        return None if img is None else self.write(name, img)
