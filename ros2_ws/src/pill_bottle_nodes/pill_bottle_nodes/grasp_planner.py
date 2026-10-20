# grasp_planner node: decides the next move toward the bottle. Never moves the robot itself.
#   service  /pill_bottle/plan_grasp (PlanGrasp)
#   uses     /joint_states, tf (base_link, link_wrist_yaw, link_grasp_center)
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import Buffer, TransformListener

from pill_bottle_interfaces.srv import PlanGrasp

from .common import frame_position
from .geometry import PLANNER_SETTINGS as SETTINGS, grasp_center_at_yaw0, plan_grasp


class GraspPlannerNode(Node):
    def __init__(self):
        super().__init__("grasp_planner")
        for k, v in SETTINGS.items():
            self.declare_parameter(k, v)
        self.joints = {}
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(JointState, "/joint_states", self.on_joints, 10)
        self.create_service(PlanGrasp, "/pill_bottle/plan_grasp", self.on_plan)
        self.get_logger().info("Grasp planner ready: /pill_bottle/plan_grasp")

    def on_joints(self, msg):
        for n, p in zip(msg.name, msg.position):
            self.joints[n] = p

    def on_plan(self, req, res):
        p = {k: self.get_parameter(k).value for k in SETTINGS}
        if "wrist_extension" not in self.joints or "joint_lift" not in self.joints:
            res.success, res.message = False, "No /joint_states yet. Is the Stretch driver running?"
            return res
        try:
            wrist = frame_position(self.tf_buffer, "base_link", "link_wrist_yaw")
            grasp = frame_position(self.tf_buffer, "base_link", "link_grasp_center")
        except Exception as e:
            res.success, res.message = False, f"tf lookup failed: {e}"
            return res
        b = req.bottle
        if b.header.frame_id and b.header.frame_id != "base_link":
            res.success, res.message = False, f"Bottle must be in base_link, got {b.header.frame_id}"
            return res
        g0 = grasp_center_at_yaw0(wrist, grasp)
        plan = plan_grasp((b.position.x, b.position.y, b.position.z), b.orientation, b.surface_z, b.axis_yaw,
                          self.joints["wrist_extension"], self.joints["joint_lift"], g0, p)
        for k, v in plan.items():
            setattr(res, k, v)
        self.get_logger().info(f"[{plan['next_step']}] {plan['message']}")
        return res


def main():
    rclpy.init()
    node = GraspPlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
