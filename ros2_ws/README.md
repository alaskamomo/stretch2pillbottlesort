# Pill bottle nodes (ROS 2 Jazzy, Stretch 2)

The pill bottle demo is split into ROS 2 nodes. Each component has its own tester, so a problem with
perception or with the distance calculation can be checked on its own.

The nodes **wrap the existing scripts** (`detect_bottles.py`, `gripper_cam.py`) and the YOLO model where they
already live on the robot, `~/pill_bottle_robot`. Nothing there needs to change. If they live elsewhere, pass
`legacy_dir:=/path`.

## Packages

| Package | What it holds |
| --- | --- |
| `pill_bottle_interfaces` | Messages `Detection(s)`, `BottleState`, `GripperView`; services `PlanGrasp`, `MoveJoints` |
| `pill_bottle_nodes` | The 7 nodes, a tester for each, grasp testers, and the launch files |

## Nodes

| Node | Wraps | In | Out |
| --- | --- | --- | --- |
| `gripper_camera` | `gripper_cam.GripperCamera` | claw camera `/dev/video6` | `/gripper_camera/image_raw` |
| `bottle_detector` | `detect_bottles.detect_all` | `/camera/color/image_raw` | `/pill_bottle/detections`, `/pill_bottle/debug_image` |
| `bottle_localizer` | depth + tf | detections, aligned depth, camera_info | `/pill_bottle/state` (x, y, z in `base_link`, upright/tilted/lying) |
| `gripper_vision` | `gripper_cam.analyze` | `/gripper_camera/image_raw` | `/pill_bottle/gripper_view` |
| `grasp_planner` | grasp geometry from `full_run.py` | `/joint_states`, tf | service `/pill_bottle/plan_grasp` (never moves) |
| `motion_node` | `stretch_body` moves, now via the driver | service `/pill_bottle/move_joints` | `/stretch_controller/follow_joint_trajectory` |
| `task_manager` | `full_run.py` main loop | all of the above | the full search, grasp, verify and place run |

Only `motion_node` moves the robot. It checks every goal against safe limits, and it allows only one base
joint per goal, as the driver requires.

## Build (on the robot)

```bash
mkdir -p ~/ament_ws/src && cp -r ros2_ws/src/* ~/ament_ws/src/
cd ~/ament_ws && colcon build --packages-select pill_bottle_interfaces pill_bottle_nodes
source install/setup.bash
```

## Run a tester

Use three terminals, each with `source ~/ament_ws/install/setup.bash`. Free the robot first with
`stretch_free_robot_process.py` if a script was using it.

```bash
# 1. robot: driver in position mode + head camera with aligned depth
ros2 launch pill_bottle_nodes robot.launch.py

# 2. only the nodes this tester needs
ros2 launch pill_bottle_nodes nodes.launch.py stage:=distance

# 3. the tester
ros2 run pill_bottle_nodes test_distance
```

Run them in this order. Each one assumes the ones above it pass, so the first one that fails points to the
component at fault. There is one tester per node, plus two for the grasp.

| Tester | Tests node | `stage:=` | Moves? | Pass when |
| --- | --- | --- | --- | --- |
| `test_cameras` | `gripper_camera` (+ head camera) | `cameras` | no | Head color/depth at 10 Hz or more, claw at 2 Hz or more, depth aligned, `~/pill_tests/head_upright.jpg` upright |
| `test_detector` | `bottle_detector` | `detector` | no | Bottle found in 9 of 10 frames at confidence 0.4 or more |
| `test_distance` | `bottle_localizer` | `distance` | no | x, y, z steady within 1 cm and within 2 cm of a tape measure (up to 1 m) |
| `test_gripper_vision` | `gripper_vision` | `gripper_vision` | no | Bottle seen in 15 of 20 readings; offset follows the bottle |
| `test_planner` | `grasp_planner` | `planner` | no | Reach targets inside limits, or a sensible base turn or drive |
| `test_motion` | `motion_node` | `motion` | a little | Unsafe goal refused; every joint arrives within tolerance (lift/arm 1 cm, wrist/head 0.05 rad) |
| `test_gripper` | the claw | `gripper` | claw only | Effort on the bottle clearly above empty; still holding after 8 s with at least half the closing effort |
| `test_arm` | approach only | `arm` | arm | Claw stops 10 cm (±2 cm) from the bottle, at least 3.5 cm above the table |
| `test_base` | base moves | `base` | base | 60° turn within ±5°, 0.20 m drive within ±3 cm |
| `test_grasp` | the grasp | `grasp` | arm | Result is HELD; logged to `~/pill_tests/grasp_log.jsonl` |
| `test_task` | `task_manager` | `full` | no | Finds the bottle and prints a reach plan (dry run) |

## Fixing the gripper (this week)

The grasp now has its own code path, `grasp.py`, shared by `test_grasp` and `task_manager`, so a fix found
in testing carries over to the full run. After closing, it lifts 5 cm and watches the finger for 3 s. Every
attempt ends as one of four results:

| Result | Meaning | What to change first |
| --- | --- | --- |
| HELD | Grip is good | Nothing; move on to `task_manager -p stop_after:=verify` |
| SLIPPING | Grip effort sagged below half while lifted | Harder squeeze `-p gripper_squeeze:=-0.03`, deeper grasp `ros2 param set /grasp_planner grasp_depth 0.06` |
| DROPPED | Grip effort fell to the empty level while lifting | Look at the claw image: bottle near the end caps means a bigger `grasp_depth`; wrong height means check `test_distance` z |
| MISSED | Closed on nothing | Look at the claw image: off to one side means `gripcam_sign` or `test_gripper_vision`; stopped short means a bigger `grasp_depth` |

The claw grabs with the **middle of the fingers**, not the end caps: the arm reaches `grasp_depth` past
`link_grasp_center` (the point between the end caps), with the claw opened wide so it clears the bottle.

The grip is judged by the gripper motor's **effort**, not the finger position. Stretch's fingers are flexible
and bend around the bottle, so the finger position reads the same empty or holding (-0.028 rad both times on
our robot). The effort is what changes: about 0.1 empty and 6.6 on the bottle.

Workflow:

1. `test_gripper` (stage `gripper`) first. It prints the right `grip_effort_min` for your claw and shows
   whether the squeeze can hold the bottle at all, with no arm motion.
2. `test_grasp` (stage `grasp`) with the bottle upright on the table, 5 or more times. Each run saves the claw
   image from just before closing to `~/pill_tests/`.
3. Change one setting at a time, for example `ros2 run pill_bottle_nodes test_grasp --ros-args -p gripper_squeeze:=-0.03`.
4. `ros2 run pill_bottle_nodes grasp_report` shows the HELD, SLIPPING, DROPPED and MISSED counts for each setting.
5. When HELD is consistent, run `task_manager --ros-args -p stop_after:=grasp`, then `stop_after:=verify`, then
   the full run.

`motion_node` now returns as soon as the fingers stop on the bottle (`gripper_stall_s` 0.6 s), instead of
waiting out the driver's 10 s timeout. The driver keeps squeezing after that.

Full run: `stage:=full`, then `ros2 run pill_bottle_nodes task_manager`.
Add `--ros-args -p dry_run:=true` to find and plan without moving.

To see what the cameras see: `ros2 run rqt_image_view rqt_image_view /pill_bottle/debug_image`
(or `/pill_bottle/gripper_debug_image`).

## Settings worth knowing

- `bottle_detector`: `rotate` (`cw`, the same as `ROTATE` in `full_run.py`) and `conf`.
- `bottle_localizer`: `min_conf` 0.4, `bottle_radius`, and the upright/lying angle thresholds.
- `grasp_planner`: `bottle_length`, `pregrasp_gap` 0.10, `safe_clearance` 0.12, `max_drive` 0.5.
- `grasp_planner`: `grasp_depth` 0.05, how far past the end caps the claw reaches so the bottle sits in the
  middle of the fingers. Measure it on your claw: from the end caps to the middle of the finger pads.
  Change it while running with `ros2 param set /grasp_planner grasp_depth 0.06`.
- `task_manager` and `test_grasp`: `gripper_open` 0.10, `gripper_squeeze` -0.01, `grip_effort_min` 3.0,
  `effort_drop_frac` 0.5, `lift_test` 0.05, `gripcam_sign` 1.0. `task_manager` also takes `search_pans`, `max_attempts` 4,
  `hold_seconds` 8, `stop_after` (find, plan, grasp, verify) and `confirm` (ask before every move).

Change any of these with `--ros-args -p name:=value`.

## Unit tests (no robot needed)

```bash
cd ros2_ws/src/pill_bottle_nodes && python3 -m pytest test/
```

These cover the head-image rotation mapping, depth-to-3D conversion, upright/tilted/lying classification,
grasp planning, head aiming, and how a grasp is judged held, slipping, dropped or missed.
