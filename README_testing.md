# Stretch 2 pill bottle: perception, bottle memory and component testing

```
testing/                         logging + summary + manual tester + bottle watcher (plain Python)
docs/github_tickets.txt          current tickets (T0, T1a, T1b, T2, T3, T4, T6, T8, T9, T10, T11)
ros2_ws/src/bottle_tracking/     ROS 2 package
  bottle_tracking/
    yolo_detector_node.py        YOUR trained YOLOv8 model on the head camera -> /pill_bottle/detection
    bottle_memory_node.py        remembers the bottle in odom, so turning doesn't lose it
    head_tracker_node.py         keeps the head on the bottle while the base moves
    head_search_node.py          head sweep first, base turn only if needed
    approach_planner_node.py     moves the base so the arm is beside the bottle
    calibrate_arm.py             measures where the gripper really is (run once)
    fake_detector_node.py        fake bottle for testing without YOLO
    geometry.py                  base + arm planning math (unit tested)
    testers/                     T1a, T1b, T2, T3, T6
  config/params.yaml             all settings
  launch/bottle_tracking.launch.py
```

## One-time setup on the robot

```bash
pip3 install ultralytics "numpy<2"
find ~ -name "*.pt" 2>/dev/null
```

The second command finds your trained model (`best.pt`). Put its full path in `config/params.yaml` as `model_path`, or pass `model_path:=` to the launch file. Keep `numpy<2`, because NumPy 2 breaks ROS 2 Humble.

```bash
cd ~/ament_ws
colcon build --packages-select bottle_tracking --symlink-install
source install/setup.bash
```

Rebuild after pulling this update, because it adds new programs.

## Every session

Open 4 terminals and run `source ~/ament_ws/install/setup.bash` in each:

```bash
ros2 launch stretch_core stretch_driver.launch.py broadcast_odom_tf:=True mode:=navigation
ros2 launch stretch_core d435i_high_resolution.launch.py
ros2 launch bottle_tracking bottle_tracking.launch.py model_path:=/home/hello-robot/PATH/TO/best.pt
python3 ~/stretch2pillbottlesort/testing/watch_bottle.py
```

Then, in a 5th terminal:

```bash
cd ~/stretch2pillbottlesort
P=$(ros2 pkg prefix bottle_tracking)/share/bottle_tracking/config/params.yaml
```

## Check the detector (first time)

```bash
ros2 run rqt_image_view rqt_image_view /pill_bottle/debug_image
```

Use `ssh -Y` and XQuartz to see the window. The image should be **upright**, with a green box on the bottle (red means no depth there).

If the image is sideways, change `rotate_image` in params.yaml: `cw`, `ccw` or `none`. Use whichever matches how your training images looked. Restart the launch after changing it.

## Calibrate the arm (once, before T1)

```bash
ros2 run bottle_tracking calibrate_arm --ros-args -p write:=true
```

The arm and wrist will move, so clear the right side of the robot first. This saves the real gripper position into params.yaml. Also copy the gripper open/close values your current pick code uses into `gripper_open` and `gripper_closed`.

## Tests

| Test | Command |
|---|---|
| T1a find + plan (no motion) | `ros2 run bottle_tracking t1a_plan_tester --ros-args --params-file $P -p trials:=3` |
| T1b find + plan + drive + grab | `ros2 run bottle_tracking t1b_execute_tester --ros-args --params-file $P -p trials:=3` |
| T2 find + report position | `ros2 run bottle_tracking t2_find_tester --ros-args --params-file $P -p trials:=5` |
| T3 watch while moving, then remember | `ros2 run bottle_tracking t3_tracking_tester --ros-args --params-file $P -p trials:=3` |
| T6 distance, base still | `ros2 run bottle_tracking t6_distance_tester --ros-args --params-file $P -p trials:=5` |
| T4, T8, T9, T10, T11 | `python3 testing/run_manual_test.py T8` |
| Results | `python3 testing/summarize_results.py --session latest` (add `--markdown` for GitHub) |

Change the pause between trials with `-p pause_s:=30`.

Unit tests (no robot needed): `cd ros2_ws/src/bottle_tracking && python3 -m pytest test/ -v`
