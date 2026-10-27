"""
Starts the detector, bottle memory, head tracker, head search and approach planner.

  ros2 launch bottle_tracking bottle_tracking.launch.py model_path:=/home/hello-robot/best.pt
  ros2 launch bottle_tracking bottle_tracking.launch.py detector:=fake   # fake bottle, no YOLO
  ros2 launch bottle_tracking bottle_tracking.launch.py detector:=none   # your own detector publishes

Start the Stretch driver and the head camera first:
  ros2 launch stretch_core stretch_driver.launch.py broadcast_odom_tf:=True mode:=navigation
  ros2 launch stretch_core d435i_high_resolution.launch.py
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    params = os.path.join(get_package_share_directory("bottle_tracking"), "config", "params.yaml")
    detector = LaunchConfiguration("detector")
    model_path = LaunchConfiguration("model_path")
    dry_run = LaunchConfiguration("dry_run")

    def node(executable, extra=None, condition=None):
        return Node(package="bottle_tracking", executable=executable, output="screen",
                    parameters=[params] + (extra or []), condition=condition)

    def detector_is(name):
        return IfCondition(PythonExpression(["'", detector, "' == '", name, "'"]))

    return LaunchDescription([
        DeclareLaunchArgument("detector", default_value="yolo", description="yolo, fake or none"),
        DeclareLaunchArgument("model_path", default_value="",
                              description="path to your trained best.pt (overrides model_path in params.yaml)"),
        DeclareLaunchArgument("dry_run", default_value="false"),
        node("yolo_detector", [{"model_path_override": model_path}], detector_is("yolo")),
        node("fake_detector", condition=detector_is("fake")),
        node("bottle_memory"),
        node("head_tracker"),
        node("head_search"),
        node("approach_planner", [{"dry_run": ParameterValue(dry_run, value_type=bool)}]),
    ])
