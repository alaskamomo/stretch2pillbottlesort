#!/usr/bin/env python3
"""
calibrate_arm - measures where the gripper really is, so T1 plans reach the bottle.

Moves the wrist straight out (yaw/pitch/roll 0), puts the lift at 0.80 m, then reads
link_grasp_center from TF at two arm extensions. Prints the three numbers the arm
planner needs and (with -p write:=true) saves them into config/params.yaml.

  ros2 run bottle_tracking calibrate_arm
  ros2 run bottle_tracking calibrate_arm --ros-args -p write:=true

THE ARM AND WRIST WILL MOVE. Clear the space beside the robot (right side) first.
Needs the driver running and the robot homed.
"""

import os
import re
import time

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node

from bottle_tracking.stretch_interface import StretchInterface, declare
from bottle_tracking.testers.tester_common import ask, run_tester


class CalibrateArm(Node):
    def __init__(self):
        super().__init__("calibrate_arm")
        self.robot = StretchInterface(self, ReentrantCallbackGroup())
        self.write = declare(self, "write", False)
        self.lift_height = declare(self, "calibration_lift_m", 0.80)

    def measure(self, ext):
        ok, msg = self.robot.move_joints({"joint_lift": self.lift_height, "wrist_extension": ext}, 30.0)
        if not ok:
            raise RuntimeError(msg)
        time.sleep(1.0)
        tf = self.robot.lookup(self.robot.base_frame, self.robot.grasp_frame)
        st = self.robot.arm_state()
        if tf is None or st is None:
            raise RuntimeError(f"no TF for {self.robot.grasp_frame} or no joint states")
        (x, y, z), _ = tf
        lift, real_ext = st
        print(f"  extension {real_ext:.3f} m -> gripper center x={x:.3f} y={y:.3f} z={z:.3f}, lift {lift:.3f}")
        return x, -y - real_ext, z - lift

    def run(self):
        print(__doc__)
        if ask("Arm and wrist will move. Type GO to continue: ") != "GO":
            print("Cancelled.")
            return
        if not self.robot.switch_to_position_mode():
            print("Could not switch to position mode.")
            return
        ok, msg = self.robot.move_joints({"joint_wrist_yaw": 0.0, "joint_wrist_pitch": 0.0,
                                          "joint_wrist_roll": 0.0}, 20.0)
        if not ok:
            print(f"Wrist move failed: {msg}")
            return
        a = self.measure(0.05)
        b = self.measure(0.30)
        self.robot.move_joints({"wrist_extension": 0.0}, 30.0)

        arm_x = (a[0] + b[0]) / 2
        lateral = (a[1] + b[1]) / 2
        z_off = (a[2] + b[2]) / 2
        print("\nResults (put these in config/params.yaml):")
        print(f"    arm_x_offset_m: {arm_x:.3f}")
        print(f"    grasp_lateral_offset_m: {lateral:.3f}")
        print(f"    lift_grasp_z_offset_m: {z_off:.3f}")
        if abs(a[1] - b[1]) > 0.02 or abs(a[0] - b[0]) > 0.02:
            print("  WARNING: the two measurements disagree by more than 2 cm. Check the wrist is straight.")

        if self.write:
            self.write_params({"arm_x_offset_m": arm_x, "grasp_lateral_offset_m": lateral,
                               "lift_grasp_z_offset_m": z_off})

    def write_params(self, values):
        from ament_index_python.packages import get_package_share_directory
        path = os.path.realpath(os.path.join(get_package_share_directory("bottle_tracking"),
                                             "config", "params.yaml"))
        text = open(path).read()
        for k, v in values.items():
            text, n = re.subn(rf"^(\s*{k}:\s*)[-0-9.]+", rf"\g<1>{v:.3f}", text, flags=re.M)
            if n != 1:
                print(f"  could not find '{k}' in {path}; add it by hand")
        open(path, "w").write(text)
        print(f"\nSaved to {path}. Restart the bottle_tracking launch to use them.")


def main():
    run_tester(CalibrateArm)


if __name__ == "__main__":
    main()
