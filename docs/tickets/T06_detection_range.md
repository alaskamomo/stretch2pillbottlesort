# T6: Detection range: how far away can it see the pill bottle?

## What this test does
Finds the real maximum distance the head camera + YOLO can see the bottle, and why it misses past that.
Tape marks straight out from the base (1, 1.5, 2 … 4 m) → bottle on a mark → head aims at it → 20 frames with
YOLO at **640** and 20 at **1280** → seen %, confidence, box size (px), distance vs tape, fps, and the reason
for every miss (not detected / low confidence / no depth / filtered as too far).
Repeat on the floor, a table and a raised spot.

Run: `ros2 launch pill_bottle_nodes nodes.launch.py stage:=range` then
`ros2 run pill_bottle_nodes test_range --ros-args -p model_path:=/home/hello-robot/pill_bottle_robot/final_pill_bottle.pt`
Live view (boxes with why each was or wasn't used, distance, orientation, distance to the bin):
`ros2 run pill_bottle_nodes live_view` + `rqt_image_view /pill_bottle/range_view`

Why far bottles get missed (expected): YOLO shrinks the 1280 px image to 640 before looking, so an 8 × 4 cm bottle
is 19 × 9 px at 2 m and 12 × 6 px at 3 m; the detector's 0.5 confidence cutoff; too few depth pixels; the 3 m filter.

## Accepted results
- [ ] Prints the max reliable distance per height and YOLO size (seen ≥ 80 %, error ≤ 3 cm, ≤ 5 cm past 2.5 m)
- [ ] Reliable to at least 2.5 m on the floor after the suggested settings / retraining
- [ ] Settings changed from its suggestions (`imgsz`, `conf_threshold`, `max_bottle_distance_m` in params.yaml)

## Possible issues
- 1280 runs about 3–4× slower on the robot's CPU; check the fps it prints
- Far bottles need far training pictures (same retrain as the T3 elevation fix)
- Shiny orange plastic gives little depth far away
