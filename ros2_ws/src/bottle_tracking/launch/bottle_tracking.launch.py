"""
Starts the bottle memory, head tracker, head search and approach planner nodes.

  ros2 launch bottle_tracking bottle_tracking.launch.py
  ros2 launch bottle_tracking bottle_tracking.launch.py fake_detector:=true   # no YOLO needed
  ros2 launch bottle_tracking bottle_tracking.launch.py dry_run:=true         # planner never drives

Start the Stretch driver first, with odom TF on:
  ros2 launch stretch_core stretch_driver.launch.py broadcast_odom_tf:=True
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    params = os.path.join(get_package_share_directory("bottle_tracking"), "config", "params.yaml")
    dry_run = LaunchConfiguration("dry_run")

    def node(executable, extra=None):
        return Node(package="bottle_tracking", executable=executable, output="screen",
                    parameters=[params] + (extra or []))

    return LaunchDescription([
        DeclareLaunchArgument("fake_detector", default_value="false"),
        DeclareLaunchArgument("dry_run", default_value="false"),
        node("bottle_memory"),
        node("head_tracker"),
        node("head_search"),
        node("approach_planner", [{"dry_run": ParameterValue(dry_run, value_type=bool)}]),
        Node(package="bottle_tracking", executable="fake_detector", output="screen",
             parameters=[params], condition=IfCondition(LaunchConfiguration("fake_detector"))),
    ])
