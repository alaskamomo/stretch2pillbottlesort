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
            # component testers
            "t3_tracking_tester = bottle_tracking.testers.t3_tracking_tester:main",
            "t5_search_tester = bottle_tracking.testers.t5_search_tester:main",
            "t12_moved_bottle_tester = bottle_tracking.testers.t12_moved_bottle_tester:main",
            "t15a_approach_plan_tester = bottle_tracking.testers.t15a_approach_plan_tester:main",
            "t15b_approach_motion_tester = bottle_tracking.testers.t15b_approach_motion_tester:main",
        ],
    },
)
