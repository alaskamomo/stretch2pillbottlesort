# T7: Starting positions and replanning

## What this test does
Starts the robot from different spots in the room and checks it still lines the arm up with the bottle,
and replans if the bottle moves or disappears mid-drive.
Start spot → search → plan → drive → (bottle moved? stop + replan) → arm lined up

Log each trial: `python3 testing/run_manual_test.py T7` (conditions: front, side, far corner, facing away, moved mid drive, removed mid drive).

## Accepted results
- [ ] Arm lined up within 3 cm from 4+ start spots (front, side, far corner, facing away) in 4 of 5 trials
- [ ] Replans after the bottle is moved mid-drive in 4 of 5 trials
- [ ] If the bottle goes missing, it stops and searches instead of driving to the old spot

## Possible issues
- Odometry drift on long drives, so the saved bottle spot is slightly off
- Starting "facing away" depends on the base-turn search, which is slow
- Bottle moved while the camera isn't looking won't be noticed until it looks back
