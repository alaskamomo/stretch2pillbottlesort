# motion_node: the only node that moves the robot. Sends goals to the Stretch driver and waits for them.
#   service  /pill_bottle/move_joints (MoveJoints)
#   uses     /stretch_controller/follow_joint_trajectory (driver in position mode), /joint_states
import threading
import time

import rclpy
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint

from pill_bottle_interfaces.srv import MoveJoints

from .common import update_joints, wait_future

# Hard limits checked before anything is sent (driver limits are wider; these keep tests safe).
LIMITS = {
    "joint_lift": (0.0, 1.1),
    "wrist_extension": (0.0, 0.52),
    "joint_wrist_yaw": (-1.75, 4.0),
    "joint_head_pan": (-4.0, 1.7),
    "joint_head_tilt": (-1.5, 0.5),
    "gripper_aperture": (-0.03, 0.10),
    "translate_mobile_base": (-0.5, 0.5),
    "rotate_mobile_base": (-3.15, 3.15),
}
BASE = {"translate_mobile_base", "rotate_mobile_base"}


class MotionNode(Node):
    def __init__(self):
        super().__init__("motion_node")
        self.declare_parameter("goal_timeout_s", 30.0)
        self.declare_parameter("gripper_stall_s", 0.6)   # finger still this long = closed on something
        self.group = ReentrantCallbackGroup()
        self.lock = threading.Lock()                  # one motion at a time
        self.client = ActionClient(self, FollowJointTrajectory, "/stretch_controller/follow_joint_trajectory",
                                   callback_group=self.group)
        self.joints = {}
        # own group so finger readings keep arriving while a service call waits
        self.create_subscription(JointState, "/joint_states", self.on_joints, 10, callback_group=self.group)
        self.create_service(MoveJoints, "/pill_bottle/move_joints", self.on_move, callback_group=self.group)
        self.get_logger().info("Motion node ready: /pill_bottle/move_joints")

    def on_joints(self, msg):
        update_joints(self.joints, None, msg)

    def check(self, names, positions):
        if len(names) != len(positions) or not names:
            return "names and positions must be the same, non-zero length"
        unknown = [n for n in names if n not in LIMITS]
        if unknown:
            return f"unknown joints {unknown}; allowed: {list(LIMITS)}"
        if set(names) & BASE and len(names) > 1:
            return "base moves must be sent alone (one of translate_mobile_base / rotate_mobile_base per call)"
        for n, v in zip(names, positions):
            lo, hi = LIMITS[n]
            if not lo <= v <= hi:
                return f"{n} = {v:.3f} is outside the safe range {lo} to {hi}"
        return None

    def wait_gripper(self, result_future, target_aperture):
        """While the gripper moves, watch the finger. Returns the finger position if it stalled, else None."""
        stall_s = self.get_parameter("gripper_stall_s").value
        start, last_pos, still_since = time.time(), None, None
        while not result_future.done() and time.time() - start < 15.0:
            time.sleep(0.05)
            pos = self.joints.get("joint_gripper_finger_left")
            if pos is None:
                continue
            if last_pos is not None and abs(pos - last_pos) < 0.002:
                still_since = still_since or time.time()
                if time.time() - start > 0.5 and time.time() - still_since > stall_s:
                    return pos
            else:
                still_since = None
            last_pos = pos
        return None

    def on_move(self, req, res):
        names, positions = list(req.names), [float(v) for v in req.positions]
        err = self.check(names, positions)
        if err:
            res.success, res.message = False, err
            return res
        if not self.client.wait_for_server(timeout_sec=3.0):
            res.success, res.message = False, "Driver action server not found. Start the Stretch driver in position mode."
            return res
        with self.lock:
            goal = FollowJointTrajectory.Goal()
            goal.trajectory.joint_names = names
            point = JointTrajectoryPoint()
            point.positions = positions
            goal.trajectory.points = [point]
            timeout = self.get_parameter("goal_timeout_s").value
            handle = wait_future(self.client.send_goal_async(goal), 5.0)
            if handle is None or not handle.accepted:
                res.success, res.message = False, "Driver rejected the goal"
                return res
            result_future = handle.get_result_async()
            if names == ["gripper_aperture"]:
                stalled = self.wait_gripper(result_future, positions[0])
                if stalled is not None:
                    # Fingers stopped on something before reaching the target. The driver keeps squeezing
                    # until its own timeout, which is what holds the bottle; we just don't wait for it.
                    res.success, res.message = True, f"gripper stopped at finger {stalled:.3f} rad (holding something?)"
                    self.get_logger().info(f"move gripper_aperture={positions[0]:.3f}: {res.message}")
                    return res
            result = wait_future(result_future, timeout)
            if result is None:
                handle.cancel_goal_async()
                res.success, res.message = False, f"Motion didn't finish within {timeout:.0f} s"
                return res
            code = result.result.error_code
            text = result.result.error_string
            if code == FollowJointTrajectory.Result.SUCCESSFUL:
                res.success, res.message = True, "done"
            elif names == ["gripper_aperture"] and code == FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED:
                # Closing on a bottle: the fingers stop before the target, so the driver times out. That's a grip.
                res.success, res.message = True, "gripper stopped early (likely holding something)"
            else:
                res.success, res.message = False, f"driver error {code}: {text}"
        self.get_logger().info(f"move {dict(zip(names, [round(v, 3) for v in positions]))}: {res.message}")
        return res


def main():
    rclpy.init()
    node = MotionNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
