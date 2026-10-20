# Starts the Stretch driver (position mode) and the head D435i camera with aligned depth.
#   ros2 launch pill_bottle_nodes robot.launch.py
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource


def generate_launch_description():
    core = os.path.join(get_package_share_directory("stretch_core"), "launch")
    return LaunchDescription([
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(core, "stretch_driver.launch.py")),
                                 launch_arguments={"mode": "position"}.items()),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(core, "d435i_high_resolution.launch.py"))),
    ])
