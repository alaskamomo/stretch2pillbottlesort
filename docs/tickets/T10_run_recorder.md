# T10: Run recorder (video of runs)

## What this test does
Records every run as an .mp4: the head camera view with the YOLO box, a mini map, stage progress, an event log,
and an empty slot for outside footage that you add after the run. Starts/stops from the testers or by hand.
At the start the robot shows "START" and beeps so outside footage can be lined up later
(`testing/add_footage.py run.mp4 phone.mp4 --start <seconds>`).

Log each trial: `python3 testing/run_manual_test.py T10` (conditions: robot only, with outside footage).

## Accepted results
- [ ] Video plays and matches the real run length
- [ ] File name matches the trial row in the results CSV (new `video` column)
- [ ] Overlay shows test ID, trial, time, stage and events (found, moved, missing, grip result)
- [ ] A map PNG is saved at the end of each run
- [ ] add_footage puts outside footage in the slot, in sync within 0.2 s

## Possible issues
- Recording slows the robot computer down (lower the FPS or resolution if it does)
- Disk fills up; delete old videos or copy them off the robot
- No beep in the outside footage means syncing by eye
