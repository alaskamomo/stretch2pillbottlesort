# Starts only the nodes a tester needs.
#   ros2 launch pill_bottle_nodes nodes.launch.py stage:=distance
# stages: cameras | detector | distance | gripper_vision | planner | motion | gripper | arm | base | grasp | full
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

STAGES = {
    "cameras": ["gripper_camera"],
    "detector": ["bottle_detector"],
    "distance": ["bottle_detector", "bottle_localizer"],
    "gripper_vision": ["gripper_camera", "gripper_vision"],
    "planner": ["bottle_detector", "bottle_localizer", "grasp_planner"],
    "arm": ["bottle_detector", "bottle_localizer", "grasp_planner", "motion_node"],
    "base": ["motion_node"],
    "motion": ["motion_node"],
    "gripper": ["motion_node"],
    "grasp": ["gripper_camera", "bottle_detector", "bottle_localizer", "gripper_vision", "grasp_planner",
              "motion_node"],
    "full": ["gripper_camera", "bottle_detector", "bottle_localizer", "gripper_vision", "grasp_planner",
             "motion_node"],
}
USES_LEGACY = {"gripper_camera", "bottle_detector", "gripper_vision"}


def make_nodes(context):
    stage = LaunchConfiguration("stage").perform(context)
    legacy_dir = LaunchConfiguration("legacy_dir").perform(context)
    rotate = LaunchConfiguration("rotate").perform(context)
    if stage not in STAGES:
        raise RuntimeError(f"stage must be one of {list(STAGES)}, got '{stage}'")
    nodes = []
    for name in STAGES[stage]:
        params = {}
        if name in USES_LEGACY:
            params["legacy_dir"] = legacy_dir
        if name == "bottle_detector":
            params["rotate"] = rotate
        nodes.append(Node(package="pill_bottle_nodes", executable=name, name=name, output="screen",
                          parameters=[params]))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("stage", default_value="full"),
        DeclareLaunchArgument("legacy_dir", default_value="/home/hello-robot/pill_bottle_robot",
                              description="folder with detect_bottles.py, gripper_cam.py and the model"),
        DeclareLaunchArgument("rotate", default_value="cw", description="head image rotation: cw | ccw | 180 | none"),
        OpaqueFunction(function=make_nodes),
    ])
