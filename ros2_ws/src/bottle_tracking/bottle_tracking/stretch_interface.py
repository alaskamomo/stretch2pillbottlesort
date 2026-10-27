"""
stretch_interface.py - shared helpers for talking to the Stretch 2 driver.

Used by the search node, approach node and testers so they all move the robot
the same way.

  Base:  /stretch/cmd_vel (navigation mode), closed loop on odom TF
  Head:  /stretch_controller/follow_joint_trajectory (joint_head_pan, joint_head_tilt)
  Modes: /switch_to_navigation_mode (std_srvs/Trigger)

Base moves BLOCK, so only call them from a service callback / thread running in a
MultiThreadedExecutor (every node in this package uses one).
"""

import math
import time

from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.time import Time

from builtin_interfaces.msg import Duration as DurationMsg
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener
from trajectory_msgs.msg import JointTrajectoryPoint

from bottle_tracking import geometry as geo


def declare(node, name, default):
    """Declare a parameter once (several helpers share a node) and return its value."""
    if not node.has_parameter(name):
        node.declare_parameter(name, default)
    return node.get_parameter(name).value


class StretchInterface:
    def __init__(self, node, cb_group=None):
        self.node = node
        self.cb = cb_group or ReentrantCallbackGroup()

        self.fixed_frame = declare(node, "fixed_frame", "odom")
        self.base_frame = declare(node, "base_frame", "base_link")
        self.camera_frame = declare(node, "camera_frame", "camera_color_optical_frame")
        cmd_vel_topic = declare(node, "cmd_vel_topic", "/stretch/cmd_vel")
        joint_states_topic = declare(node, "joint_states_topic", "/stretch/joint_states")
        traj_action = declare(node, "trajectory_action", "/stretch_controller/follow_joint_trajectory")
        nav_srv = declare(node, "nav_mode_service", "/switch_to_navigation_mode")

        self.max_lin = declare(node, "max_linear_speed", 0.10)        # m/s
        self.min_lin = declare(node, "min_linear_speed", 0.02)
        self.max_ang = declare(node, "max_angular_speed", 0.40)       # rad/s
        self.min_ang = declare(node, "min_angular_speed", 0.08)
        self.rot_tol = math.radians(declare(node, "rotate_tolerance_deg", 1.0))
        self.drive_tol = declare(node, "drive_tolerance_m", 0.01)
        self.move_timeout = declare(node, "base_motion_timeout_s", 30.0)

        self.limits = geo.HeadLimits(
            pan_min=declare(node, "head_pan_min", -4.04),
            pan_max=declare(node, "head_pan_max", 1.73),
            tilt_min=declare(node, "head_tilt_min", -1.53),
            tilt_max=declare(node, "head_tilt_max", 0.79),
        )

        self.tf_buffer = Buffer()
        # TF callbacks run in the node's default group; all of our own callbacks use
        # the reentrant group, so a blocking base move never starves TF updates.
        self.tf_listener = TransformListener(self.tf_buffer, node, spin_thread=False)

        self.cmd_pub = node.create_publisher(Twist, cmd_vel_topic, 10)
        self.joints = {}
        node.create_subscription(JointState, joint_states_topic, self._on_joints, 10,
                                 callback_group=self.cb)
        self.traj_client = ActionClient(node, FollowJointTrajectory, traj_action,
                                        callback_group=self.cb)
        self.nav_client = node.create_client(Trigger, nav_srv, callback_group=self.cb)

    # ------------------------------------------------------------------ TF
    def lookup(self, target, source, stamp=None, timeout=0.1):
        """(translation, quaternion) mapping points from source frame into target frame, or None."""
        try:
            t = self.tf_buffer.lookup_transform(
                target, source, stamp if stamp is not None else Time(),
                timeout=Duration(seconds=timeout))
        except Exception:
            if stamp is None:
                return None
            # image timestamp too old/new for the buffer: fall back to latest
            try:
                t = self.tf_buffer.lookup_transform(target, source, Time())
            except Exception:
                return None
        tr, q = t.transform.translation, t.transform.rotation
        return (tr.x, tr.y, tr.z), (q.x, q.y, q.z, q.w)

    def point_to(self, point, from_frame, to_frame, stamp=None):
        tf = self.lookup(to_frame, from_frame, stamp)
        if tf is None:
            return None
        return geo.transform_point(tf[0], tf[1], point)

    def base_pose(self):
        """(x, y, yaw) of the base in the fixed (odom) frame."""
        tf = self.lookup(self.fixed_frame, self.base_frame)
        if tf is None:
            return None
        return tf[0][0], tf[0][1], geo.yaw_from_quat(tf[1])

    def camera_in(self, frame):
        """Camera position and viewing direction (optical z axis) in frame."""
        tf = self.lookup(frame, self.camera_frame)
        if tf is None:
            return None, None
        return tf[0], geo.quat_rotate(tf[1], (0.0, 0.0, 1.0))

    # ------------------------------------------------------------------ head
    def _on_joints(self, msg):
        for name, pos in zip(msg.name, msg.position):
            self.joints[name] = pos

    def head_state(self):
        if "joint_head_pan" not in self.joints or "joint_head_tilt" not in self.joints:
            return None
        return self.joints["joint_head_pan"], self.joints["joint_head_tilt"]

    def send_head(self, pan, tilt):
        """Send a head goal and return right away (a newer goal replaces it)."""
        if not self.traj_client.server_is_ready():
            self.node.get_logger().warn("trajectory action server not ready", throttle_duration_sec=5.0)
            return False
        pan = max(self.limits.pan_min, min(self.limits.pan_max, pan))
        tilt = max(self.limits.tilt_min, min(self.limits.tilt_max, tilt))
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ["joint_head_pan", "joint_head_tilt"]
        point = JointTrajectoryPoint()
        point.positions = [float(pan), float(tilt)]
        point.time_from_start = DurationMsg(sec=0, nanosec=0)
        goal.trajectory.points = [point]
        self.traj_client.send_goal_async(goal)
        return True

    def head_settled(self, pan, tilt, tol=0.05):
        hs = self.head_state()
        return hs is not None and abs(hs[0] - pan) < tol and abs(hs[1] - tilt) < tol

    def move_head_and_wait(self, pan, tilt, timeout=4.0):
        self.send_head(pan, tilt)
        end = time.time() + timeout
        while time.time() < end:
            if self.head_settled(pan, tilt):
                return True
            time.sleep(0.05)
        return False

    # ------------------------------------------------------------------ base
    def switch_to_navigation_mode(self, timeout=5.0):
        if not self.nav_client.wait_for_service(timeout_sec=timeout):
            self.node.get_logger().error("navigation mode service not available")
            return False
        fut = self.nav_client.call_async(Trigger.Request())
        end = time.time() + timeout
        while not fut.done() and time.time() < end:
            time.sleep(0.02)
        return fut.done() and fut.result() is not None and fut.result().success

    def stop(self):
        for _ in range(3):
            self.cmd_pub.publish(Twist())
            time.sleep(0.02)

    def _send(self, lin, ang):
        msg = Twist()
        msg.linear.x = float(lin)
        msg.angular.z = float(ang)
        self.cmd_pub.publish(msg)

    @staticmethod
    def _speed(err, gain, vmin, vmax):
        v = gain * err
        v = math.copysign(max(abs(v), vmin), v)
        return max(-vmax, min(vmax, v))

    def rotate_by(self, angle_rad, should_abort=None):
        """Rotate in place by angle_rad (closed loop on odom). Returns (ok, achieved_rad)."""
        pose = self.base_pose()
        if pose is None:
            self.node.get_logger().error("no odom->base_link TF; is broadcast_odom_tf on?")
            return False, 0.0
        last_yaw, turned = pose[2], 0.0
        end = time.time() + self.move_timeout
        ok = False
        while time.time() < end:
            if should_abort and should_abort():
                break
            pose = self.base_pose()
            if pose is not None:
                turned += geo.wrap_angle(pose[2] - last_yaw)   # unwrapped, works past 180 deg
                last_yaw = pose[2]
            err = angle_rad - turned
            if abs(err) < self.rot_tol:
                ok = True
                break
            self._send(0.0, self._speed(err, 1.5, self.min_ang, self.max_ang))
            time.sleep(0.05)
        self.stop()
        return ok, turned

    def drive_by(self, dist_m, should_abort=None):
        """Drive straight dist_m (negative = backward), holding heading. Returns (ok, achieved_m)."""
        start = self.base_pose()
        if start is None:
            self.node.get_logger().error("no odom->base_link TF; is broadcast_odom_tf on?")
            return False, 0.0
        sx, sy, syaw = start
        end = time.time() + self.move_timeout
        done, ok = 0.0, False
        while time.time() < end:
            if should_abort and should_abort():
                break
            pose = self.base_pose()
            if pose is not None:
                done = (pose[0] - sx) * math.cos(syaw) + (pose[1] - sy) * math.sin(syaw)
                heading_fix = max(-0.15, min(0.15, 1.0 * geo.wrap_angle(syaw - pose[2])))
            else:
                heading_fix = 0.0
            err = dist_m - done
            if abs(err) < self.drive_tol:
                ok = True
                break
            self._send(self._speed(err, 1.0, self.min_lin, self.max_lin), heading_fix)
            time.sleep(0.05)
        self.stop()
        return ok, done

    def execute_steps(self, steps, should_abort=None, log=None):
        """Run ROTATE / DRIVE steps from geometry.plan_approach. Returns (ok, failed_step_number)."""
        if steps and not self.switch_to_navigation_mode():
            return False, 0
        for i, s in enumerate(steps, 1):
            if log:
                log(f"step {i}: {s['type']} {s.get('deg', s.get('m'))}")
            if s["type"] == "ROTATE":
                ok, _ = self.rotate_by(math.radians(s["deg"]), should_abort)
            else:
                ok, _ = self.drive_by(s["m"], should_abort)
            if not ok:
                return False, i
        return True, None
