# Starts only the nodes a tester needs.
#   ros2 launch pill_bottle_nodes nodes.launch.py stage:=distance
# stages: cameras | detector | distance | gripper_vision | planner | motion | gripper | arm | base | grasp | full
#         | label | orientation | range | bin
#         (T12/T13/T15/T18/T19b testers use stage:=full; T19a uses stage:=arm)
# orientation_model:=/path/orientation_cls.pt also starts orientation_classifier (T17) wherever the localizer runs.
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
    "label": ["bottle_detector", "motion_node"],                                       # T16 test_label
    "range": ["motion_node"],                                                          # T6 test_range / live_view
    "bin": ["bottle_detector", "motion_node"],                                         # T11 test_bin_detection
    "orientation": ["bottle_detector", "bottle_localizer", "orientation_classifier"],  # T17 test_orientation
}
USES_LEGACY = {"gripper_camera", "bottle_detector", "gripper_vision"}


def make_nodes(context):
    stage = LaunchConfiguration("stage").perform(context)
    legacy_dir = LaunchConfiguration("legacy_dir").perform(context)
    rotate = LaunchConfiguration("rotate").perform(context)
    orientation_model = LaunchConfiguration("orientation_model").perform(context)
    if stage not in STAGES:
        raise RuntimeError(f"stage must be one of {list(STAGES)}, got '{stage}'")
    names = list(STAGES[stage])
    if orientation_model and "bottle_localizer" in names and "orientation_classifier" not in names:
        names.append("orientation_classifier")
    if "orientation_classifier" in names and not orientation_model:
        raise RuntimeError("stage:=orientation needs orientation_model:=/path/to/orientation_cls.pt")
    nodes = []
    for name in names:
        params = {}
        if name in USES_LEGACY:
            params["legacy_dir"] = legacy_dir
        if name in ("bottle_detector", "orientation_classifier"):
            params["rotate"] = rotate
        if name == "orientation_classifier":
            params["model_path"] = orientation_model
        nodes.append(Node(package="pill_bottle_nodes", executable=name, name=name, output="screen",
                          parameters=[params]))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("stage", default_value="full"),
        DeclareLaunchArgument("legacy_dir", default_value="/home/hello-robot/pill_bottle_robot",
                              description="folder with detect_bottles.py, gripper_cam.py and the model"),
        DeclareLaunchArgument("rotate", default_value="cw", description="head image rotation: cw | ccw | 180 | none"),
        DeclareLaunchArgument("orientation_model", default_value="",
                              description="orientation classification model (T17); empty = depth-only orientation"),
        OpaqueFunction(function=make_nodes),
    ])
