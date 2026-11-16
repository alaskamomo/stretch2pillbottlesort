# T3: Distance, tracking, memory and grip (T3a–T3d)

## What this test does
Checks that the robot can measure the bottle's distance (T3a), keep the camera on it while moving (T3b),
remember where it was after turning away (T3c), and reach, grip and lift it (T3d).
Detect bottle → measure distance → move base → look back at saved spot → reach → grip → lift → put back

## Accepted results
- [ ] T3a: distance steady within 1 cm, within 2 cm of a tape measure, at floor, table and raised heights
- [ ] T3b: bottle in view ≥ 90 % of the time while turning/driving
- [ ] T3c: finds the bottle again in ≤ 3 s, drift ≤ 3 cm
- [ ] T3d: held deep in the claw, lifted and held 5 s, no collision
- [ ] T2 finds the bottle at every height in 4 of 5 trials

## Possible issues
- **YOLO misses the bottle at different heights (main issue):** add head-camera training photos at floor, table and shelf height, at both head tilts (-0.25, -0.85), at 0.5 / 1 / 1.5 m, plus "empty" photos, then retrain.
- `min_bottle_height_m` / `max_bottle_height_m` in params.yaml may filter out a raised bottle
- Shiny or far bottles give few depth points, so the distance jumps around
