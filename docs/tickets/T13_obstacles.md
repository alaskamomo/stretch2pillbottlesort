# T13: Navigate around obstacles

## What this test does
The lidar sees obstacles; before every drive it checks the strip in front of the base; the path planner goes
around obstacles (up to 3 detour points), and bottles still on the floor and the bin count as obstacles too.
Drive → anything in the way? → stop, add it, replan → continue (no way around → reports it, doesn't move)

Run: `ros2 run pill_bottle_nodes test_obstacles` (target: nearest bottle, or `-p target:=bin`).
Conditions: no obstacle, one, two, no way around. Saves a map of each trial.

## Accepted results
- [ ] Reaches the bottle/bin around 1 obstacle in 4 of 5 trials
- [ ] Zero collisions
- [ ] No way around → stops safely and says so

## Possible issues
- Lidar only sees at its own height; low or overhanging objects are missed
- Detour loses sight of the bottle, so it depends on T3c memory
- Narrow gaps: the robot's footprint plus the arm sticking out
