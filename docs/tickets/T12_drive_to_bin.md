# T12: Search for and drive to the bin

## What this test does
With a bottle in the claw: looks around for the bin (head sweep; drives to lookout spots if it isn't in view),
drives beside it around obstacles, turns so the arm faces it, lifts the bottle's bottom 8 cm over the rim,
reaches to the bin's center, lets go, retracts. It remembers the bin, so the next bottle skips the search.
Look around → (no bin? lookout spots) → drive beside the bin → drop plan → lift → reach → open → retract

Run: `ros2 launch pill_bottle_nodes robot.launch.py` (lidar on) + `nodes.launch.py stage:=full`, then
`ros2 run pill_bottle_nodes test_bin_drop` (you put the bottle in the claw; it asks before every move).
Conditions: bin in view, bin behind the robot, remembered (second bottle). Saves a map of each trial.

## Accepted results
- [ ] Bottle ends up in the bin in 4 of 5 trials, no collision
- [ ] Second bottle goes to the remembered bin spot without a full search

## Possible issues
- Holding the bottle may block the camera's view of the bin
- Bin opening height vs. lift range; the arm must clear the rim
- Bottle bounces out if dropped from too high
