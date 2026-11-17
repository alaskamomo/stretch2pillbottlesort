#!/usr/bin/env python3
"""
T2 - find EVERY pill bottle with the head camera and report where each one is.

  ros2 run bottle_tracking t2_find_tester --ros-args --params-file $P -p trials:=5
  (-p bottles:=2 to skip the "how many are out" question, -p pause_s:=30 for a longer pause)

It prints a link to the live camera (open it on your Mac or phone):  http://<robot ip>:8080

Each trial:
  1. Pauses so you can move the bottle(s) (Enter to skip the wait), asks how many are out.
  2. Sweeps the head through every position (head_pan_positions x head_tilt_positions). The base
     never moves. At each stop it collects EVERY pill bottle box, also low-confidence ones.
  3. Saves a picture of every view where it saw something (boxes, confidence and distance drawn on
     it: green = pill bottle, yellow = low confidence, red = no depth) plus one picture of them all.
  4. Merges the sightings into one entry per bottle (odom frame) and prints:
       Pill bottle 1/2 at x=.. y=.. z=.. m (odom) | 1.24 m away, 18 deg right | conf 0.83 | seen 4x in 2 views
       Possible pill bottle 1/1 at ... conf 0.12 (below 0.50)
  5. Scores it: SUCCESS = found exactly the bottles you put out; MISS = fewer (says if the missing ones
     were seen with low confidence); FALSE_POSITIVE = more, or you say one isn't a pill bottle.

Needs: driver, camera, bottle_tracking launch with the YOLO detector.
"""

import json
import math
import threading
import time
from collections import deque

from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from bottle_tracking import geometry as geo
from bottle_tracking.approach_planner_node import declare_approach_config
from bottle_tracking.common import MemoryWatcher, SnapshotSaver
from bottle_tracking.multi_find import bottle_lines, cluster, score
from bottle_tracking.stretch_interface import StretchInterface, declare
from bottle_tracking.test_logger import TestLogger
from bottle_tracking.testers.tester_common import ask, ask_yes, declare_arm_config, pause, reset_head, run_tester


def describe_moves(steps):
    if not steps:
        return "no, the arm can reach it from here"
    parts = []
    for s in steps:
        if s["type"] == "ROTATE":
            parts.append(f"turn {abs(s['deg']):.0f} deg {'left' if s['deg'] > 0 else 'right'}")
        else:
            parts.append(f"drive {abs(s['m']):.2f} m {'forward' if s['m'] > 0 else 'backward'}")
    return "yes: " + ", then ".join(parts)


def stamp_s(header):
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def overlay(img, lines):
    import cv2
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 10 + 22 * len(lines)), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (8, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def mosaic(images, height=480, per_row=4):
    """Side-by-side picture of several views (all scaled to the same height)."""
    import cv2
    import numpy as np
    tiles = [cv2.resize(im, (int(im.shape[1] * height / im.shape[0]), height)) for im in images]
    w = max(t.shape[1] for t in tiles)
    tiles = [np.pad(t, ((0, 0), (0, w - t.shape[1]), (0, 0))) for t in tiles]
    rows = []
    for i in range(0, len(tiles), per_row):
        row = tiles[i:i + per_row]
        row += [np.zeros_like(tiles[0])] * (per_row - len(row)) if len(tiles) > per_row else []
        rows.append(np.hstack(row))
    return np.vstack(rows)


class FindTester(Node):
    def __init__(self):
        super().__init__("t2_find_tester")
        cb = ReentrantCallbackGroup()
        self.robot = StretchInterface(self, cb)
        self.mem = MemoryWatcher(self, cb)
        self.snap = SnapshotSaver(self, cb)
        self.arm = declare_arm_config(self)
        self.approach_cfg = geo.approach_config_for_arm(self.arm, declare_approach_config(self))
        self.trials = int(declare(self, "trials", 5))
        self.pause_s = declare(self, "pause_s", 20.0)
        self.bottles = int(declare(self, "bottles", 0))            # 0 = ask every trial
        self.pans = list(declare(self, "head_pan_positions", [1.4, 0.7, 0.0, -0.7, -1.4, -2.1, -2.8, -3.5]))
        self.tilts = list(declare(self, "head_tilt_positions", [-0.45, -0.85]))
        self.dwell = declare(self, "dwell_s", 1.2)
        self.settle = declare(self, "settle_s", 0.3)
        self.radius = declare(self, "same_bottle_radius_m", 0.10)
        self.min_z = declare(self, "min_bottle_height_m", -0.05)
        self.max_z = declare(self, "max_bottle_height_m", 1.40)
        self.max_dist = declare(self, "max_bottle_distance_m", 4.0)
        self.conf_thr = declare(self, "conf_threshold", 0.5)
        port = int(declare(self, "stream_port", 8080))
        self.log = TestLogger("T2", "Find every pill bottle with the head camera")

        self._lock = threading.Lock()
        self.frames = deque(maxlen=400)          # detections_all: (stamp, frame_id, boxes)
        self.debug = deque(maxlen=60)            # debug pictures: (stamp, msg)
        self.n_frames = 0
        self.create_subscription(String, "/pill_bottle/detections_all", self._on_all, 20, callback_group=cb)
        self.create_subscription(Image, "/pill_bottle/debug_image", self._on_debug, 2, callback_group=cb)
        self.stream = None
        if port > 0:
            try:
                from bottle_tracking.web_stream import WebStream
                self.stream = WebStream(port, "T2: find every pill bottle (green = bottle, yellow = low confidence)")
            except Exception as e:
                print(f"(no live link: {e})")

    # ------------------------------------------------------------------ inputs
    def _on_all(self, msg):
        try:
            d = json.loads(msg.data)
        except ValueError:
            return
        with self._lock:
            self.frames.append((self._when(d["stamp"]), d["frame"], d["boxes"]))
            self.n_frames += 1

    def _on_debug(self, msg):
        with self._lock:
            self.debug.append((self._when(stamp_s(msg.header)), msg))
        if self.stream is not None:
            self.stream.update(SnapshotSaver._to_array(msg))

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def _when(self, stamp):
        """The picture's time, or the time it arrived if the camera clock is off (then stamps can't be compared)."""
        now = self.now()
        return stamp if abs(now - stamp) < 2.0 else now

    def frames_since(self, t):
        with self._lock:
            return [f for f in self.frames if f[0] >= t]

    def best_picture_since(self, t):
        """The debug picture (taken after t) with the most boxes on it."""
        with self._lock:
            counts = {round(f[0], 3): len(f[2]) for f in self.frames if f[0] >= t}   # same picture = same time
            pics = [(s, m) for s, m in self.debug if s >= t]
        if not pics:
            return None
        s, m = max(pics, key=lambda p: (counts.get(round(p[0], 3), 0), p[0]))
        return SnapshotSaver._to_array(m)

    # ------------------------------------------------------------------ the sweep
    def sweep(self, name):
        self.mem.set_tracking(False)
        sightings, no_pos, pics = [], [], []
        frames_before = self.n_frames
        total = len(self.pans) * len(self.tilts)
        view = 0
        t0 = time.time()
        for tilt in self.tilts:
            for pan in self.pans:
                view += 1
                if not self.robot.move_head_and_wait(pan, tilt):
                    print(f"  view {view}/{total}: the head did not reach pan {pan:.2f} tilt {tilt:.2f}")
                time.sleep(self.settle)
                start = self.now()
                time.sleep(self.dwell)
                frames = self.frames_since(start)
                n_box, best = 0, 0.0
                for stamp, frame, boxes in frames:
                    for e in boxes:
                        n_box += 1
                        best = max(best, e["conf"])
                        if e["xyz"] is None or e["status"] == "too_far":
                            no_pos.append((view, e["conf"]))
                            continue
                        fixed = self.robot.point_to(tuple(e["xyz"]), frame, self.robot.fixed_frame)
                        base = self.robot.point_to(tuple(e["xyz"]), frame, self.robot.base_frame)
                        if fixed is None or base is None:
                            continue
                        if not (self.min_z <= fixed[2] <= self.max_z) or math.hypot(base[0], base[1]) > self.max_dist:
                            no_pos.append((view, e["conf"]))
                            continue
                        sightings.append({"fixed": fixed, "base": base, "conf": e["conf"], "view": view,
                                          "status": e["status"]})
                status = (f"{n_box} box(es), best conf {best:.2f}" if n_box else "nothing") + \
                    ("" if frames else "  (NO detector frames!)")
                print(f"  view {view}/{total}  pan {pan:+.2f} tilt {tilt:+.2f}: {status}", flush=True)
                if n_box:
                    img = self.best_picture_since(start)
                    if img is not None:
                        img = overlay(img, [f"T2 {name}  view {view}/{total}  pan {pan:+.2f} tilt {tilt:+.2f}",
                                            f"{n_box} box(es) over {len(frames)} frames, best conf {best:.2f}  "
                                            f"(threshold {self.conf_thr:.2f})"])
                        pics.append((view, img, self.snap.write(f"{name}_view{view}", img)))
        return sightings, no_pos, pics, self.n_frames - frames_before, time.time() - t0

    # ------------------------------------------------------------------ one trial
    def trial(self, i, n_expected):
        reset_head(self)
        pause(self.pause_s, f"Trial {i}/{self.trials}: put out the pill bottle(s).")
        if not self.bottles:
            a = ask(f"How many pill bottles are out? [{n_expected or 'skip'}] ")
            n_expected = int(a) if a.isdigit() else n_expected
        name = f"T2_{self.log.session_id}_trial{i}"
        with self.log.trial() as t:
            sightings, no_pos, pics, n_frames, secs = self.sweep(name)
            bottles = cluster(sightings, self.radius)
            print("\n" + "=" * 70)
            if n_frames == 0:
                print("The detector sent NO frames during the sweep: a camera/detector problem, not confidence.\n"
                      "Is bottle_tracking.launch.py running?  ros2 topic hz /pill_bottle/detections_all")
            for line in bottle_lines(bottles, self.conf_thr, no_pos, self.robot.fixed_frame):
                print(line)
            pose = self.robot.base_pose()
            for k, b in enumerate([b for b in bottles if b["confident"]], 1):
                plan = geo.plan_approach(b["fixed"], pose, self.approach_cfg) if pose else None
                moves = (describe_moves(plan["steps"]) if plan and plan["ok"]
                         else f"can't plan: {plan['reason'] if plan else 'no robot pose'}")
                print(f"  Bottle {k}: needs to move to grab it? {moves}")
            all_pic = None
            if pics:
                all_pic = self.snap.write(f"{name}_all", mosaic([p[1] for p in pics]))
                print(f"Pictures: {len(pics)} views with boxes, all of them in {all_pic}")
                for view, _, path in pics:
                    print(f"  view {view}: {path}")
            else:
                print("Pictures: none (nothing was seen in any view)")
            print(f"Sweep: {secs:.0f} s, {n_frames} detector frames")
            print("=" * 70)

            outcome, note = score(n_expected, bottles)
            if outcome == "SUCCESS" and not ask_yes("Are those all real pill bottles (nothing else)?", True):
                outcome, note = "FALSE_POSITIVE", note + "; you said one isn't a pill bottle"
            sure = [b for b in bottles if b["confident"]]
            maybe = [b for b in bottles if not b["confident"]]
            t["condition"] = f"{n_expected}_bottles" if n_expected else "unknown_count"
            t["outcome"] = outcome
            t["metrics"] = {
                "search_time_s": round(secs, 1), "expected": n_expected, "found": len(sure),
                "possible": len(maybe), "no_position": len(no_pos), "detector_frames": n_frames,
                "confidence": round(min(b["conf"] for b in sure), 2) if sure else None,
                "best_low_conf": round(max(b["conf"] for b in maybe), 2) if maybe else None,
                "farthest_m": round(max(b["distance_m"] for b in sure), 2) if sure else None,
                "bottles": [{"x": round(b["fixed"][0], 3), "y": round(b["fixed"][1], 3), "z": round(b["fixed"][2], 3),
                             "dist_m": round(b["distance_m"], 2), "conf": round(b["conf"], 2),
                             "confident": b["confident"]} for b in bottles]}
            t["notes"] = note + (f" | picture: {all_pic}" if all_pic else "")
            print(f"-> {outcome}: {note}")
        return n_expected

    def run(self):
        print(__doc__)
        if self.stream is not None:
            print(f"Live camera: {self.stream.url}   (open on your Mac or phone, same Wi-Fi)\n")
        ask("Driver, camera and detector running? Press Enter to begin.")
        n = self.bottles or None
        for i in range(1, self.trials + 1):
            n = self.trial(i, n)
        reset_head(self)
        print("\nSummary: python3 testing/summarize_results.py --test T2 --session latest")


def main():
    run_tester(FindTester)


if __name__ == "__main__":
    main()
