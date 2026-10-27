# Stretch 2 pill bottle: search/memory fix + component testing

```
stretch_pill_bottle_update/
├── README.md                      <- this file
├── docs/github_tickets.txt        <- every ticket (BUG + T0 to T15b), plain text, with labels
├── testing/                       <- logging + summary + manual tester (plain Python, no ROS)
│   ├── test_logger.py
│   ├── summarize_results.py
│   ├── run_manual_test.py
│   └── test_plan.py               <- conditions/criteria for every test
└── ros2_ws/src/bottle_tracking/   <- new ROS 2 package (the bug fix + testers)
    ├── bottle_tracking/
    │   ├── bottle_memory_node.py      remembers the bottle in odom (THE fix)
    │   ├── head_tracker_node.py       keeps the camera on the bottle while the base moves
    │   ├── head_search_node.py        head sweep first, base rotation only if needed
    │   ├── approach_planner_node.py   moves the base so the arm is in front of the bottle
    │   ├── fake_detector_node.py      fake detections for testing without YOLO
    │   ├── detection_publisher.py     3 lines to add to YOUR detector
    │   ├── geometry.py                math (no ROS, unit tested)
    │   ├── stretch_interface.py       head / base / TF helpers
    │   ├── common.py, test_logger.py
    │   └── testers/                   T3, T5, T12, T15a, T15b
    ├── config/params.yaml
    ├── launch/bottle_tracking.launch.py
    └── test/test_geometry.py
```

## Why the robot forgot the bottle

The bottle position was saved relative to the robot (camera or base_link). When the base turns, that frame turns with it, so the saved numbers point at the wrong place. `bottle_memory` saves the position in **odom**, which stays fixed to the room. After any turn or drive, TF converts it back to where the bottle is relative to the robot now. `head_tracker` uses that to keep the camera on the bottle during the move. If the head can't turn far enough, it looks straight back at the saved spot once the move is done.

## New flow

```
your detector ──/pill_bottle/detection──> bottle_memory ──/bottle_memory/pose──> head_tracker ──> head joints
                                              │
state machine ──/head_search/search──> head_search   (head sweep, base only if needed)
state machine ──/approach_planner/execute──> approach_planner   (rotate, drive, rotate; arm ends beside bottle)
state machine ──> your grasp code (T1a/T1b)
```

| Node | Topics / services |
|---|---|
| bottle_memory | in: `/pill_bottle/detection` (PointStamped), `/bottle_memory/clear` (Empty). out: `/bottle_memory/pose`, `/bottle_memory/state` (JSON: status NONE / KNOWN / MISSING), `/bottle_memory/event` |
| head_tracker | in: `/bottle_memory/pose`, `/head_tracker/enable` (Bool). out: `/head_tracker/status` (JSON) |
| head_search | service `/head_search/search` (Trigger, JSON reply) |
| approach_planner | services `/approach_planner/plan` (print only) and `/approach_planner/execute` (Trigger, JSON reply) |

## Setup on the robot

```bash
# 1. copy the package into your workspace and build
cp -r ros2_ws/src/bottle_tracking ~/ament_ws/src/       # or whatever your workspace is
cd ~/ament_ws && colcon build --packages-select bottle_tracking && source install/setup.bash

# 2. start the driver WITH odom TF (the memory needs it)
ros2 launch stretch_core stretch_driver.launch.py broadcast_odom_tf:=True mode:=navigation
ros2 launch stretch_core d435i_high_resolution.launch.py      # head camera, if not already running

# 3. start the new nodes
ros2 launch bottle_tracking bottle_tracking.launch.py
#    test without your detector:  ... fake_detector:=true   (set bottle_x/y/z in params)
#    plan without driving:        ... dry_run:=true
```

Check that `ros2 run tf2_ros tf2_echo odom base_link` prints a transform. If it doesn't, odom TF is off.

## Hooking up your code (the only changes to your existing code)

**1. Detector.** Publish every detection. See `detection_publisher.py`:
```python
from bottle_tracking.detection_publisher import BottleDetectionPublisher, deproject
self.bottle_pub = BottleDetectionPublisher(self)
x, y, z = deproject(u, v, depth_m, camera_info.k)
self.bottle_pub.publish(x, y, z, color_msg.header)   # header of the image it came from
```
Keep the detector running continuously (a few Hz is enough). That's how the memory notices the bottle is gone (MISSING).

**2. State machine.** Replace the old rotate-to-search code:
```python
ok, msg = call_trigger(self.search_client)      # /head_search/search
ok, msg = call_trigger(self.approach_client)    # /approach_planner/execute
# then your grasp. The saved bottle is on /bottle_memory/pose (odom frame).
```
`call_trigger` is in `bottle_tracking/common.py`. Both replies are JSON. If `/bottle_memory/state` says `MISSING`, go back to search (T4 / T12).

## Calibrate before testing (10 minutes)

1. **Head direction.** Launch with `fake_detector:=true` and a bottle position to the robot's left. If the head turns away from it, set `pan_sign: -1.0` (same for `tilt_sign` up/down) in `config/params.yaml`.
2. **Arm position.** Extend the arm about halfway, then run `ros2 run tf2_ros tf2_echo base_link link_grasp_center`. Use `-y` as `standoff_m` and `x` as `arm_x_offset_m`. With the arm retracted and then fully extended, `-y` gives `reach_min_m` and `reach_max_m`.
3. **Table.** Measure how far the bottle sits in from the table edge and set `table_edge_to_bottle_m`.
4. **Plan check.** Run `ros2 service call /approach_planner/plan std_srvs/srv/Trigger` and read the printed plan before letting it drive.

## Running the tests

| Test | How |
|---|---|
| T3 tracking/memory | `ros2 run bottle_tracking t3_tracking_tester` |
| T5 head-first search | `ros2 run bottle_tracking t5_search_tester` |
| T12 moved bottle | `ros2 run bottle_tracking t12_moved_bottle_tester` |
| T15a approach plan (no robot) | `ros2 run bottle_tracking t15a_approach_plan_tester` |
| T15b approach drive | `ros2 run bottle_tracking t15b_approach_motion_tester --ros-args --params-file <ws>/src/bottle_tracking/config/params.yaml` |
| Everything else (T1a, T1b, T2, T4, T6 to T11, T13, T14) | `python3 testing/run_manual_test.py T8` (keys: s=SUCCESS, m=MISS, ...) |
| Results | `python3 testing/summarize_results.py` (add `--markdown` to paste into the issue) |

Testers take ROS params, e.g. `--ros-args -p trials:=5 -p only:=behind_robot`. All results go to `~/stretch_test_results/<TEST_ID>.csv`, with the git commit on every row.

Unit tests for the math (no robot needed): `cd ros2_ws/src/bottle_tracking && python3 -m pytest test/ -v`

## GitHub labels to create

perception, planning, search, safety, integration test. Each ticket in `docs/github_tickets.txt` lists its labels.

## Not tested on the real robot yet

The math is unit tested (18 tests) and every node and tester was smoke-tested. They have not run on the Stretch yet. Things to check the first time:
- Head tracking sends a new head goal every 0.3 s or more. If the driver rejects goals while one is running (warnings in the log), raise `min_goal_interval_s`.
- Base moves use `/stretch/cmd_vel` in navigation mode. If your driver uses `/cmd_vel`, change `cmd_vel_topic`.
- If your joint states come on `/joint_states` instead of `/stretch/joint_states`, change `joint_states_topic`.
