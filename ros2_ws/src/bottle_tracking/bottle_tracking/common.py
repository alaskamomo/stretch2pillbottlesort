"""
common.py - small helpers shared by the nodes and testers.
"""

import json
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
    """Collects raw detections from /pill_bottle/detection, converted to base_link and
    the fixed (odom) frame as they arrive. Used by testers to measure where the bottle is."""

    def __init__(self, node, robot, cb_group=None):
        from geometry_msgs.msg import PointStamped
        from rclpy.time import Time
        self._Time = Time
        self.robot = robot
        self.samples = []          # (wall_time, base_xyz, fixed_xyz, camera_range_m, confidence)
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
        p = (msg.point.x, msg.point.y, msg.point.z)
        stamp = self._Time.from_msg(msg.header.stamp) if msg.header.stamp.sec > 0 else None
        base = self.robot.point_to(p, msg.header.frame_id, self.robot.base_frame, stamp)
        fixed = self.robot.point_to(p, msg.header.frame_id, self.robot.fixed_frame, stamp)
        if base is None or fixed is None:
            return
        cam = self.robot.point_to(p, msg.header.frame_id, self.robot.camera_frame, stamp)
        rng = (cam[0] ** 2 + cam[1] ** 2 + cam[2] ** 2) ** 0.5 if cam else None
        with self._lock:
            self.samples.append((time.time(), base, fixed, rng, self.confidence))
            if len(self.samples) > 2000:
                self.samples = self.samples[-1000:]

    def last_seen_age(self):
        with self._lock:
            return time.time() - self.samples[-1][0] if self.samples else None

    def sees(self, max_age=0.5):
        age = self.last_seen_age()
        return age is not None and age <= max_age

    def measure(self, seconds=2.0, min_samples=3):
        """Average the detections over the next `seconds`. Returns a dict, or None if too few."""
        t0 = time.time()
        time.sleep(seconds)
        with self._lock:
            got = [s for s in self.samples if s[0] >= t0]
        if len(got) < min_samples:
            return None

        def mean(vals):
            return sum(vals) / len(vals)

        def std3(pts, m):
            return (sum((p[0] - m[0]) ** 2 + (p[1] - m[1]) ** 2 + (p[2] - m[2]) ** 2 for p in pts)
                    / len(pts)) ** 0.5

        base = tuple(mean([s[1][i] for s in got]) for i in range(3))
        fixed = tuple(mean([s[2][i] for s in got]) for i in range(3))
        rngs = [s[3] for s in got if s[3] is not None]
        confs = [s[4] for s in got if s[4] is not None]
        return {"n": len(got), "base": base, "fixed": fixed,
                "spread_cm": std3([s[2] for s in got], fixed) * 100,
                "camera_range_m": mean(rngs) if rngs else None,
                "base_distance_m": (base[0] ** 2 + base[1] ** 2) ** 0.5,
                "confidence": mean(confs) if confs else None}
