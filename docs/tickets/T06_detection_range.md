# T6: Detection range: how far away can it see the pill bottle?

## What this test does
Finds the real maximum distance the head camera + YOLO can see the bottle, and why it misses past that.
No tape measure and no questions: it measures each spot itself.

1. Prints a **live camera link** (`http://<robot ip>:8080`, open on a Mac or phone), then every second:
   `pill bottle 1/2 detected at x=1.20 y=-0.31 z=0.02 m | 1.24 m away | conf 0.83`
   (base frame: x forward, y left, z up). Low-confidence boxes are listed too, with the reason.
2. Move the bottle anywhere (floor, table, raised). Aim the head with `python3 testing/head_keys.py`.
3. **Enter** records the spot: 10 frames at YOLO **640** and 10 at **1280** → seen %, confidence, box size (px),
   depth pixels, fps, and the reason for every miss (not detected / low confidence / no depth / too far),
   plus a picture. Distance and height (floor < 0.3 m < table < 0.9 m < raised) come from the depth camera.
4. **q + Enter** prints the max reliable distance per height and YOLO size, and the settings to change.

Run: `ros2 launch pill_bottle_nodes nodes.launch.py stage:=range` then
`ros2 run pill_bottle_nodes test_range --ros-args -p model_path:=/home/hello-robot/pill_bottle_robot/final_pill_bottle.pt`
Live view only (no recording): `ros2 run pill_bottle_nodes live_view` (same link).

Why far bottles get missed (expected): YOLO shrinks the 1280 px image to 640 before looking, so an 8 × 4 cm bottle
is 19 × 9 px at 2 m and 12 × 6 px at 3 m; the detector's 0.5 confidence cutoff; too few depth pixels.

## Accepted results
- [ ] Live lines and link work; every bottle in view is listed (low-confidence ones marked)
- [ ] Prints the max reliable distance per height and YOLO size (seen in ≥ 80 % of frames)
- [ ] Reliable to at least 2.5 m on the floor after the suggested settings / retraining
- [ ] Settings changed from its suggestions (`imgsz`, `conf_threshold`, `max_bottle_distance_m` in params.yaml)

## Possible issues
- 1280 runs about 3–4× slower on the robot's CPU; check the fps it prints
- Raised / cluttered scenes score low (0.12 seen on Nov 17): needs those in the training pictures
- A spot with no depth at all has no distance, so it's listed separately (picture saved)
- Shiny orange plastic gives little depth far away
