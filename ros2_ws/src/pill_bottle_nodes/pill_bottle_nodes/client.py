# client.py: a helper node the testers and the task manager use to talk to the other nodes.
# It spins in a background thread, so callers can write plain step-by-step code.
import os
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from tf2_ros import Buffer, TransformListener

from pill_bottle_interfaces.msg import BottleState, Detections, GripperView
from pill_bottle_interfaces.srv import MoveJoints, PlanGrasp

from .common import frame_position, image_to_numpy, wait_future


class PillClient(Node):
    def __init__(self, name):
        super().__init__(name)
        self.joints = {}
        self.efforts = {}
        self._latest = {}
        self._events = {}
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(JointState, "/joint_states", self._on_joints, 10)
        for key, typ, topic in (("state", BottleState, "/pill_bottle/state"),
                                ("detections", Detections, "/pill_bottle/detections"),
                                ("gripper_view", GripperView, "/pill_bottle/gripper_view"),
                                ("claw_image", Image, "/gripper_camera/image_raw"),
                                ("head_debug", Image, "/pill_bottle/debug_image")):
            self._events[key] = threading.Condition()
            self.create_subscription(typ, topic, lambda m, k=key: self._store(k, m), 1)
        self.move_cli = self.create_client(MoveJoints, "/pill_bottle/move_joints")
        self.plan_cli = self.create_client(PlanGrasp, "/pill_bottle/plan_grasp")
        self._executor = MultiThreadedExecutor(num_threads=2)
        self._executor.add_node(self)
        self._thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._thread.start()

    # ---- topics ----
    def _on_joints(self, msg):
        for n, p in zip(msg.name, msg.position):
            self.joints[n] = p
        for n, e in zip(msg.name, msg.effort):
            self.efforts[n] = e

    def _store(self, key, msg):
        with self._events[key]:
            self._latest[key] = (time.time(), msg)
            self._events[key].notify_all()

    def wait_for(self, key, timeout_s=5.0, newer_than=None):
        """Latest message on a topic ('state', 'detections', 'gripper_view') received after `newer_than`."""
        newer_than = time.time() if newer_than is None else newer_than
        end = time.time() + timeout_s
        with self._events[key]:
            while True:
                got = self._latest.get(key)
                if got and got[0] > newer_than:
                    return got[1]
                left = end - time.time()
                if left <= 0:
                    return None
                self._events[key].wait(left)

    def snapshot(self, key, path, timeout_s=2.0):
        """Save the next image on 'claw_image' or 'head_debug' to path. Returns True if saved."""
        import cv2
        msg = self.wait_for(key, timeout_s)
        if msg is None:
            return False
        os.makedirs(os.path.dirname(path), exist_ok=True)
        return bool(cv2.imwrite(path, image_to_numpy(msg)))

    def finger(self):
        """(position rad, effort) of joint_gripper_finger_left, or (None, None)."""
        return self.joints.get("joint_gripper_finger_left"), self.efforts.get("joint_gripper_finger_left")

    # ---- services ----
    def move(self, **joints):
        """move(joint_lift=0.6, wrist_extension=0.1) -> (ok, message)"""
        if not self.move_cli.wait_for_service(timeout_sec=3.0):
            return False, "motion_node isn't running"
        req = MoveJoints.Request()
        req.names = list(joints)
        req.positions = [float(v) for v in joints.values()]
        res = wait_future(self.move_cli.call_async(req), 60.0)
        if res is None:
            return False, "move_joints timed out"
        return res.success, res.message

    def plan(self, state):
        if not self.plan_cli.wait_for_service(timeout_sec=3.0):
            return None
        req = PlanGrasp.Request()
        req.bottle = state
        return wait_future(self.plan_cli.call_async(req), 10.0)

    # ---- tf ----
    def position(self, frame, base="base_link"):
        return frame_position(self.tf_buffer, base, frame)

    def wait_joints(self, timeout_s=5.0):
        end = time.time() + timeout_s
        while time.time() < end and "joint_lift" not in self.joints:
            time.sleep(0.1)
        return "joint_lift" in self.joints

    def close(self):
        self._executor.shutdown()
        self.destroy_node()


def start(name):
    rclpy.init()
    return PillClient(name)


def ask(question):
    """Ask before moving. Returns False on 'n' or Ctrl+D."""
    try:
        return input(f"{question} [Enter = yes, n = no] ").strip().lower() not in ("n", "no")
    except EOFError:
        return False
