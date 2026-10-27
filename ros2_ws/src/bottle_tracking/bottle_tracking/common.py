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
