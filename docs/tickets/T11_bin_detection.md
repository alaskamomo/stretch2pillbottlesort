# T11: Bin detection (YOLO)

## What this test does
Adds a `bin` class to the detector and checks it finds the bin from the head camera.
1. Label the bin photos (last meeting + head-camera photos at 0.5–2 m, different angles/light) as class `bin` in AnyLabeling.
   Keep the old class order (pill_bottle, cap, label) and add `bin` last, then retrain.
2. `ros2 launch pill_bottle_nodes nodes.launch.py stage:=bin` then `ros2 run pill_bottle_nodes test_bin_detection`
3. Each trial: bin at 0.5 / 1 / 1.5 / 2 m (or no bin): counts 10 frames, the confidence, the distance, and any
   pill_bottle box on top of the bin box (= labeled as a bottle).

## Accepted results
- [ ] Bin found in 9 of 10 frames at 0.5–2 m
- [ ] Bin never labeled as a pill bottle (and the other way round)

## Possible issues
- Not enough bin photos from the robot's own camera (photos from the last meeting alone may not be enough)
- Bin partly out of frame up close
- Other containers or boxes in the room look like the bin
