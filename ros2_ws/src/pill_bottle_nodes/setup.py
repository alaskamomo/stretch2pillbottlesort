from glob import glob

from setuptools import setup

package_name = "pill_bottle_nodes"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Lauren Aughton",
    maintainer_email="laughton257@gmail.com",
    description="ROS 2 nodes and per-component testers for the Stretch 2 pill bottle task.",
    license="MIT",
    entry_points={
        "console_scripts": [
            "gripper_camera = pill_bottle_nodes.gripper_camera:main",
            "bottle_detector = pill_bottle_nodes.bottle_detector:main",
            "bottle_localizer = pill_bottle_nodes.bottle_localizer:main",
            "gripper_vision = pill_bottle_nodes.gripper_vision:main",
            "grasp_planner = pill_bottle_nodes.grasp_planner:main",
            "motion_node = pill_bottle_nodes.motion_node:main",
            "task_manager = pill_bottle_nodes.task_manager:main",
            "test_cameras = pill_bottle_nodes.testers:test_cameras",
            "test_detector = pill_bottle_nodes.testers:test_detector",
            "test_distance = pill_bottle_nodes.testers:test_distance",
            "test_gripper_vision = pill_bottle_nodes.testers:test_gripper_vision",
            "test_planner = pill_bottle_nodes.testers:test_planner",
            "test_arm = pill_bottle_nodes.testers:test_arm",
            "test_base = pill_bottle_nodes.testers:test_base",
            "test_motion = pill_bottle_nodes.testers:test_motion",
            "test_gripper = pill_bottle_nodes.testers:test_gripper",
            "test_grasp = pill_bottle_nodes.testers:test_grasp",
            "grasp_report = pill_bottle_nodes.testers:grasp_report",
            "test_task = pill_bottle_nodes.task_manager:test_task",
        ],
    },
)
