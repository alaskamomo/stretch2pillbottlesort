# Starts the Stretch driver (position mode, with the odom -> base_link tf), the head D435i camera with aligned
# depth, and the lidar (/scan) for driving around obstacles (T12, T13, T15, T18).
#   ros2 launch pill_bottle_nodes robot.launch.py              # lidar on
#   ros2 launch pill_bottle_nodes robot.launch.py lidar:=false
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def _setup(context):
    core = os.path.join(get_package_share_directory("stretch_core"), "launch")
    out = [
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(core, "stretch_driver.launch.py")),
                                 launch_arguments={"mode": "position", "broadcast_odom_tf": "True"}.items()),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(core, "d435i_high_resolution.launch.py"))),
    ]
    if LaunchConfiguration("lidar").perform(context).lower() in ("true", "1", "yes"):
        lidar = os.path.join(core, "rplidar.launch.py")
        if os.path.isfile(lidar):
            out.append(IncludeLaunchDescription(PythonLaunchDescriptionSource(lidar)))
        else:
            out.append(LogInfo(msg=f"No {lidar}; start the lidar yourself so /scan exists (T12/T13/T15/T18 need it)."))
    return out


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("lidar", default_value="true", description="start the RPLidar (/scan)"),
        OpaqueFunction(function=_setup),
    ])
