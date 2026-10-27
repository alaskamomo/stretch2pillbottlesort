from glob import glob
from setuptools import setup

package_name = "bottle_tracking"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name, f"{package_name}.testers"],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Lauren Aughton",
    maintainer_email="laughton257@gmail.com",
    description="Pill bottle memory, head tracking, search, approach planning and testers for Stretch 2",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            # nodes
            "bottle_memory = bottle_tracking.bottle_memory_node:main",
            "head_tracker = bottle_tracking.head_tracker_node:main",
            "head_search = bottle_tracking.head_search_node:main",
            "approach_planner = bottle_tracking.approach_planner_node:main",
            "fake_detector = bottle_tracking.fake_detector_node:main",
            "yolo_detector = bottle_tracking.yolo_detector_node:main",
            "calibrate_arm = bottle_tracking.calibrate_arm:main",
            # component testers
            "t1a_plan_tester = bottle_tracking.testers.t1_pick_tester:main_plan",
            "t1b_execute_tester = bottle_tracking.testers.t1_pick_tester:main_execute",
            "t2_find_tester = bottle_tracking.testers.t2_find_tester:main",
            "t3_tracking_tester = bottle_tracking.testers.t3_tracking_tester:main",
            "t6_distance_tester = bottle_tracking.testers.t6_distance_tester:main",
        ],
    },
)
