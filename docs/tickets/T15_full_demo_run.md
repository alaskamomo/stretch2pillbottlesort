# T15: Full demo run

## What this test does
The complete run (`ros2 run pill_bottle_nodes demo_run`):
1. You type how many bottles are out
2. Looks around (head + lidar) and remembers the bottles, the bin and obstacles (T18); finds the bin first
3. Nearest bottle: drives beside it around obstacles (T13), turns/drives until it's in reach, grabs it
   for its orientation (T19, 3 tries), verifies it at the head camera, reads the label (T16)
4. Drives to the bin and drops it in (T12); looks around again; repeats
5. Fewer found than you said: drives to lookout spots and looks again
6. Checks the bin with the head camera, prints and saves the report (bottles found / in the bin / missed,
   each label, orientation, obstacles, blocked / out of range), the list (JSON) and a map, in ~/pill_tests/run_<time>/

Depends on: T2, T3, T6, T9, T11, T12, T13, T16, T17, T18, T19

## Accepted results
- [ ] Clears 3 bottles into the bin in one run, then stops on its own, in 3 of 5 runs
- [ ] Report matches what really happened (count, labels, orientation, misses)

## Possible issues
- Any single stage failing stops the loop, so test the parts first
- Already-binned bottles seen again and picked back up
- Battery runs low over a long run
