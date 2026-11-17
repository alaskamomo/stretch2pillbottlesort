# T19: Orientation-aware grasp (T19a plan, T19b execute)

## What this test does
**T19a: plan only (no base/arm motion, like T1a).** Head-only search finds the bottle, works out its orientation
(depth fit, plus the T17 model if loaded), and prints the whole grip plan (every base turn/drive, then the reach).
You type the real orientation; it checks the orientation and that the plan matches it.
**T19b: plan, then carry it out (like T1b).** Same as T19a, then it asks, turns/drives the base, grabs
(up to 3 tries), lifts, holds 5 s and puts the bottle back.

| Orientation | Expected plan |
|---|---|
| Upright | Side grab, low on the bottle, deep in the claw |
| Lying | Drive to the spot beside it, turn so the bottle runs along the arm, claw closes across it |
| Tilted | Grab at the bottle's center height |
| Out of reach | "out of range", no grab |

No new YOLO model needed (current detector + depth fit; T17 model optional).
Run: `nodes.launch.py stage:=arm` + `ros2 run pill_bottle_nodes t19a_orientation_plan`;
`stage:=full` + `ros2 run pill_bottle_nodes t19b_orientation_grasp`.
Offline: `pytest test/test_orientation_grasp.py` (upright/lying/tilted at 5 spots × floor/table/raised + 200 random).
Found and fixed: a lying bottle could loop forever (turning to line it up swung it off the arm line); far bottles
need up to 7 base moves, so max_base_moves is now 10.

## Accepted results
- [ ] Offline: correct plan for every orientation and height
- [ ] T19a: orientation right in ≥ 90 % of trials; plan matches the orientation every time it's right
- [ ] T19b: HELD upright 4 of 5, lying 3 of 5, tilted 3 of 5, no collisions

## Possible issues
- Wrong orientation → wrong plan, so check T19a before running T19b
- Lying bottle pointing straight at the robot needs a big base turn to line up
- No wrist pitch on Stretch 2, so tilted bottles are the hardest
- Raised bottles: YOLO misses at some heights (T3 fix)
