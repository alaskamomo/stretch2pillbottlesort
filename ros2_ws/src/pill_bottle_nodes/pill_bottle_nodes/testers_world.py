# testers_world.py: robot testers for T6 (range + live view), T11 (bin detection), T12 (drop in the bin),
# T13 (obstacles), T18 (memory + obstacles), T19a/b (orientation-aware grasp) and T15 (the full demo run).
# Each one logs to ~/stretch_test_results/<ID>.csv (testing/summarize_results.py) and saves pictures / maps /
# JSON to ~/pill_tests/. They ask before moving the base unless you pass -p confirm:=false.
import json
import math
import os
import time

import rclpy

from . import grasp
from .geometry import PLANNER_SETTINGS, box_points, box_rotated_to_raw, grasp_center_at_yaw0, rotate_image
from .mission import MISSION_SETTINGS, Mission
from .orientation import label_from_class
from .orientation_grasp import check_plan, describe, grip_height, plan_sequence
from .range_test import (RANGE_SETTINGS, detection_lines, expected_box_px, height_bucket, max_reliable,
                         orientation_guess, spot_distance, spot_height, suggestions, summarize_spot)
from .testers import _results_logger
from .world import (WORLD_SETTINGS, WorldMemory, plan_path, report_lines, stand_spot, to_robot)
from .world_robot import ROBOT_WORLD_SETTINGS, RosIO

OUT = os.path.expanduser("~/pill_tests")
COMMON = {"trials": 5, "repo_testing_dir": "~/stretch2pillbottlesort/testing"}


def make_tm(name, extra, overrides=None):
    """A TaskManager (moves, grasp, verify, label) with extra parameters declared on its node."""
    from .task_manager import TaskManager
    tm = TaskManager(overrides or {}, name)
    for k, v in extra.items():
        if k not in tm.p:
            tm.c.declare_parameter(k, v)
            tm.p[k] = tm.c.get_parameter(k).value
    return tm


def ask_line(prompt, default=""):
    try:
        a = input(f"{prompt} " + (f"[{default}] " if default != "" else "")).strip()
    except EOFError:
        return "q"
    return a or str(default)


def yes(prompt, default=False):
    a = ask_line(f"{prompt} (y/n)", "y" if default else "n").lower()
    return a.startswith("y")


def stamp():
    return time.strftime("%Y%m%d_%H%M%S")


def save_json(obj, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=str)
    return path


def _finish(tm):
    tm.c.close()
    rclpy.try_shutdown()


def _run(fn, tm):
    try:
        fn()
    except RuntimeError as e:
        print(f"Stopped: {e}")
    except KeyboardInterrupt:
        print("\nStopped (Ctrl+C). Logged trials are saved.")
    finally:
        _finish(tm)


# =====================================================================================================
# T6: how far can it see the bottle? (+ live view)
# =====================================================================================================

class RangeView:
    """Runs YOUR YOLO model on the head camera at a chosen input size and explains every pill_bottle box."""

    def __init__(self, tm):
        from sensor_msgs.msg import Image
        from ultralytics import YOLO
        self.tm, self.c, self.p = tm, tm.c, tm.p
        path = os.path.expanduser(self.p["model_path"])
        if not os.path.isfile(path):
            raise RuntimeError(f"model not found: {path}  (-p model_path:=/path/to/your_detector.pt)")
        self.model = YOLO(path)
        self.names = self.model.names
        self.omodel = None
        if self.p["orientation_model"]:
            self.omodel = YOLO(os.path.expanduser(self.p["orientation_model"]))
        self.pub = self.c.create_publisher(Image, "/pill_bottle/range_view", 1)
        self.last_view = None
        self.stream = None
        if int(self.p.get("stream_port", 0)) > 0:
            try:
                from .web_stream import WebStream
                self.stream = WebStream(int(self.p["stream_port"]),
                                        "Head camera: green = used, yellow = low confidence, red = rejected")
            except Exception as e:
                print(f"(no live link: {e})")

    def link(self):
        return self.stream.url if self.stream is not None else "(off: -p stream_port:=8080)"

    def analyze(self, size):
        import cv2
        from .common import depth_to_meters, image_to_numpy, numpy_to_image
        from .world import bin_from_points
        p, c = self.p, self.c
        msg = c.wait_for("head_color", 3.0, newer_than=time.time())
        depth_msg, info = c.latest("depth"), c.latest("camera_info")
        if msg is None or depth_msg is None or info is None:
            raise RuntimeError("no head camera color/depth/camera_info (is robot.launch.py running?)")
        raw = image_to_numpy(msg)
        if raw.ndim == 2:
            raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        H, W = raw.shape[:2]
        depth = depth_to_meters(depth_msg)
        K = list(info.k)
        up = rotate_image(raw, p["rotate"])
        t0 = time.time()
        res = self.model(up, imgsz=int(size), conf=p["show_conf"], verbose=False)[0]
        ms = (time.time() - t0) * 1000
        T = c.matrix("base_link", msg.header.frame_id, msg.header.stamp)
        bottles, bins = [], []
        for b in res.boxes:
            name = self.names[int(b.cls)]
            ux1, uy1, ux2, uy2 = [float(v) for v in b.xyxy[0]]
            rbox = box_rotated_to_raw((ux1, uy1, ux2, uy2), W, H, p["rotate"])
            pts = box_points(depth, rbox, K, band=0.06 if name == p["bottle_class"] else 0.3) \
                if depth.shape == (H, W) else []
            xyz = None
            if len(pts):
                import numpy as np
                q = pts @ np.asarray(T)[:3, :3].T + np.asarray(T)[:3, 3]
                xyz = tuple(float(v) for v in np.median(q, axis=0))
            entry = dict(conf=float(b.conf), w=ux2 - ux1, h=uy2 - uy1, depth_px=len(pts),
                         dist=math.hypot(xyz[0], xyz[1]) if xyz else None, xyz=xyz, up=(ux1, uy1, ux2, uy2))
            if name == p["bottle_class"]:
                bottles.append(entry)
            elif name == p["bin_class"] and len(pts):
                bins.append(entry)
                entry["center"] = bin_from_points(q, (T[0][3], T[1][3]))[0]
        bin_xy = max(bins, key=lambda e: e["conf"])["center"][:2] if bins else None
        view = up.copy()
        for e in bottles:
            from .range_test import box_reasons
            why = box_reasons(e["conf"], e["depth_px"], e["dist"], p)
            if self.omodel is not None:
                x1, y1, x2, y2 = [int(v) for v in e["up"]]
                crop = up[max(0, y1):y2, max(0, x1):x2]
                r = self.omodel(crop, verbose=False)[0] if crop.size else None
                e["orientation"] = label_from_class(self.omodel.names[int(r.probs.top1)]) if r is not None else "?"
            else:
                e["orientation"] = orientation_guess(e["w"], e["h"]) + "?"
            e["bin_m"] = math.hypot(e["xyz"][0] - bin_xy[0], e["xyz"][1] - bin_xy[1]) if (bin_xy and e["xyz"]) else None
            color = (0, 200, 0) if not why else ((0, 200, 255) if why == ["low conf"] else (0, 0, 255))
            x1, y1, x2, y2 = [int(v) for v in e["up"]]
            cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
            text = (f"{e['conf']:.2f} {e['w']:.0f}x{e['h']:.0f}px "
                    + (f"{e['dist']:.2f}m " if e["dist"] else "no depth ")
                    + f"{e['orientation']} " + (f"bin {e['bin_m']:.2f}m " if e["bin_m"] is not None else "")
                    + ("" if not why else "(" + ", ".join(why) + ")"))
            cv2.putText(view, text, (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        for e in bins:
            x1, y1, x2, y2 = [int(v) for v in e["up"]]
            cv2.rectangle(view, (x1, y1), (x2, y2), (200, 160, 0), 2)
            cv2.putText(view, f"bin {e['conf']:.2f} {e['dist']:.2f}m" if e["dist"] else "bin",
                        (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 160, 0), 1, cv2.LINE_AA)
        cv2.putText(view, f"YOLO {size}  {1000 / ms:.1f} fps   green=used  yellow=low conf  red=rejected",
                    (6, view.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        if not bins:
            cv2.putText(view, "bin: not detected" + ("" if p["bin_class"] in self.names.values() else " (no bin class yet, T11)"),
                        (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 160, 0), 1, cv2.LINE_AA)
        self.last_view = view
        self.pub.publish(numpy_to_image(view, msg.header.stamp, msg.header.frame_id))
        if self.stream is not None:
            self.stream.update(view)
        return {"size": int(size), "boxes": bottles, "ms": ms}


RANGE_EXTRA = dict(RANGE_SETTINGS, model_path="~/pill_bottle_robot/final_pill_bottle.pt", orientation_model="",
                   bottle_class="pill_bottle", bin_class="bin", live_size=1280, print_s=1.0, stream_port=8080,
                   **COMMON)


def live_view():
    """Live view only: every pill bottle box (with why each was or wasn't used), x, y, z, distance, orientation,
    distance to the bin. Prints a link to open in a browser.

      ros2 launch pill_bottle_nodes robot.launch.py
      ros2 run pill_bottle_nodes live_view --ros-args -p live_size:=1280
    """
    tm = make_tm("live_view", RANGE_EXTRA)

    def go():
        rv = RangeView(tm)
        print(f"\nLive camera: {rv.link()}   (open on your Mac or phone)\nCtrl+C to stop.\n")
        last = 0.0
        while True:
            f = rv.analyze(tm.p["live_size"])
            if time.time() - last >= max(tm.p["print_s"], 0.2):
                last = time.time()
                for line in detection_lines(f["boxes"], tm.p):
                    print("  " + line, flush=True)
                if f["boxes"]:
                    print("  ---", flush=True)
    _run(go, tm)


def _keyboard():
    """(time, line) typed in the terminal, read in the background so the live view keeps running."""
    import queue
    import threading
    q = queue.Queue()

    def read():
        while True:
            try:
                q.put((time.time(), input()))
            except (EOFError, KeyboardInterrupt):
                q.put((time.time(), "q"))
                return
    threading.Thread(target=read, daemon=True).start()
    return q


def test_range():
    """T6: how far away can it see the pill bottle? No questions: it measures the distance itself.

      ros2 launch pill_bottle_nodes robot.launch.py
      ros2 launch pill_bottle_nodes nodes.launch.py stage:=range
      ros2 run pill_bottle_nodes test_range --ros-args -p model_path:=/home/hello-robot/pill_bottle_robot/final_pill_bottle.pt

    It prints a link to the live camera, then every second:
      pill bottle 1/2 detected at x=1.20 y=-0.31 z=0.02 m | 1.24 m away | conf 0.83
      pill bottle 2/2 detected at x=2.61 y=0.40 z=0.75 m | 2.64 m away | conf 0.21 (low conf)
    (base frame: x forward, y left, z up from the floor). Low-confidence boxes are listed too.
    Press Enter to RECORD the spot: it compares YOLO sizes (640 vs 1280), saves a picture and logs it,
    using the measured distance and height (floor / table / raised). q + Enter to finish with the summary.
    The head doesn't move; aim it with  python3 testing/head_keys.py  first.
    """
    tm = make_tm("test_range", RANGE_EXTRA)

    def go():
        import cv2
        p = tm.p
        rv = RangeView(tm)
        logger = _results_logger(p["repo_testing_dir"], "T6", "Detection range (YOLO size vs distance)")
        print(f"\nT6 range test.  Live camera: {rv.link()}")
        print("Expected bottle size in the image: " + ", ".join(
            f"{d:g} m {expected_box_px(d):.0f}px" for d in p["distances"]))
        print("Move the bottle around. Enter = record this spot, q + Enter = finish.\n")
        keys = _keyboard()
        spots, last = [], 0.0
        while True:
            f = rv.analyze(p["live_size"])
            if time.time() - last >= max(p["print_s"], 0.2):
                last = time.time()
                for line in detection_lines(f["boxes"], p):
                    print("  " + line, flush=True)
                if f["boxes"]:
                    print("  ---", flush=True)
            if keys.empty():
                continue
            cmd = keys.get()[1].strip().lower()
            if cmd.startswith("q"):
                break
            print(f"\n  Recording spot {len(spots) + 1}: {int(p['frames'])} frames at each of {list(p['sizes'])} ...",
                  flush=True)
            frames = []
            for _ in range(int(p["frames"])):
                for size in p["sizes"]:
                    frames.append(rv.analyze(size))
            done, keep = time.time(), []
            while not keys.empty():          # extra Enter presses while recording don't record twice; q still counts
                ts, c = keys.get()
                if ts > done or c.strip().lower().startswith("q"):
                    keep.append((ts, c))
            for k in keep:
                keys.put(k)
            dist, z = spot_distance(frames), spot_height(frames)
            h = height_bucket(z)
            summ = summarize_spot(frames, None, p)
            where = f"{dist:.2f} m" if dist is not None else "no distance (not seen with depth)"
            os.makedirs(OUT, exist_ok=True)
            pic = os.path.join(OUT, f"T6_spot{len(spots) + 1}_{h}_{dist or 0:g}m_{stamp()}.jpg")
            if rv.last_view is not None:
                cv2.imwrite(pic, rv.last_view)
            print(f"  Spot {len(spots) + 1}: {where}, height {h}" + (f" (z={z:.2f} m)" if z is not None else ""))
            for size in sorted(summ):
                s = summ[size]
                print(f"    {size}: seen {s['seen']}/{s['n']} (conf {s['conf']}, box {s['box']} px, depth px "
                      f"{s['depth_px']})  fps {s['fps']}  "
                      f"{'' if not s['reasons'] else 'misses: ' + str(s['reasons'])}  "
                      f"-> {'SUCCESS' if s['passed'] else 'MISS'}")
                if logger:
                    logger.log("SUCCESS" if s["passed"] else ("NOT_FOUND" if s["seen"] == 0 else "MISS"),
                               f"{h}_{dist:g}m_imgsz{size}" if dist is not None else f"no_distance_imgsz{size}",
                               {"seen": s["seen"], "frames": s["n"], "conf": s["conf"], "box_px": s["box"],
                                "depth_px": s["depth_px"], "est_m": dist, "z_m": z, "fps": s["fps"],
                                "reasons": s["reasons"]},
                               notes=f"picture: {pic}")
            print(f"    picture: {pic}\n", flush=True)
            spots.append({"dist": dist, "z": z, "height": h, "summary": summ, "picture": pic})
        if not spots:
            print("No spots recorded.")
            return
        print("\n" + "=" * 70)
        for h in sorted({s["height"] for s in spots if s["dist"] is not None}):
            results = [(s["dist"], s["summary"]) for s in spots if s["height"] == h and s["dist"] is not None]
            sizes = sorted({z for _, sm in results for z in sm})
            print(f"Max reliable distance [{h}]: " + "   ".join(
                f"{z} -> {max_reliable(results, z) or 'none'} m" for z in sizes))
            for line in suggestions(results, p):
                print("  Suggest:", line)
        lost = [s for s in spots if s["dist"] is None]
        if lost:
            print(f"{len(lost)} spot(s) not seen with depth at all (no distance): "
                  + ", ".join(s["picture"] for s in lost))
        path = save_json(spots, f"T6_range_{stamp()}.json")
        print(f"Saved {path}. Summary: python3 testing/summarize_results.py --session latest")
    _run(go, tm)


# =====================================================================================================
# T11: bin detection
# =====================================================================================================

def test_bin_detection():
    """T11: does the detector find the bin (and never call it a pill bottle)?

      ros2 launch pill_bottle_nodes nodes.launch.py stage:=bin
      ros2 run pill_bottle_nodes test_bin_detection
    """
    extra = dict(ROBOT_WORLD_SETTINGS, frames=10, **COMMON)
    tm = make_tm("test_bin_detection", extra)

    def go():
        p = tm.p
        io = RosIO(tm, p)
        logger = _results_logger(p["repo_testing_dir"], "T11", "Bin detection (YOLO bin class)")
        conds = ["0.5m", "1m", "1.5m", "2m", "no_bin_in_view"]
        for trial in range(1, int(p["trials"]) * len(conds) + 1):
            cond = ask_line(f"\nTrial {trial}: condition ({', '.join(conds)}; q = quit)", conds[(trial - 1) % len(conds)])
            if cond.startswith("q"):
                break
            d = float(cond.rstrip("m")) if cond.endswith("m") else 1.0
            io.look_at_base((d, 0.0, 0.15))
            input("  Set it up (bin straight ahead at that distance, or no bin), then press Enter...")
            found = mislabeled = 0
            dists, confs = [], []
            for _ in range(int(p["frames"])):
                det, depth, info = io.frame(settle=False)
                if det is None:
                    continue
                bins = [x for x in det.detections if x.name == p["bin_class"] and x.confidence >= p["det_conf"]]
                if bins:
                    found += 1
                    b = max(bins, key=lambda x: x.confidence)
                    confs.append(b.confidence)
                    for x in det.detections:
                        if x.name == p["bottle_class"] and _iou(x, b) > 0.5:
                            mislabeled += 1
                    loc = [l for l in io.locate(det, depth, info, kinds=("bin",))]
                    if loc:
                        dists.append(math.hypot(loc[0][1][0], loc[0][1][1]))
            n = int(p["frames"])
            if cond == "no_bin_in_view":
                outcome = "FALSE_POSITIVE" if found else "SUCCESS"
            else:
                outcome = "FALSE_POSITIVE" if mislabeled else ("SUCCESS" if found >= 0.9 * n else
                                                               ("NOT_FOUND" if found == 0 else "MISS"))
            m = {"frames_found": f"{found}/{n}", "confidence": round(sum(confs) / len(confs), 2) if confs else None,
                 "distance_m": round(sorted(dists)[len(dists) // 2], 2) if dists else None,
                 "labeled_as_bottle": mislabeled}
            print(f"  -> {outcome} {m}")
            if logger:
                logger.log(outcome, cond, m)
    _run(go, tm)


def _iou(a, b):
    ix = max(0.0, min(a.x_max, b.x_max) - max(a.x_min, b.x_min))
    iy = max(0.0, min(a.y_max, b.y_max) - max(a.y_min, b.y_min))
    inter = ix * iy
    ua = (a.x_max - a.x_min) * (a.y_max - a.y_min) + (b.x_max - b.x_min) * (b.y_max - b.y_min) - inter
    return inter / ua if ua > 0 else 0.0


# =====================================================================================================
# shared: set up a mission on the robot
# =====================================================================================================

WORLD_EXTRA = dict(ROBOT_WORLD_SETTINGS, **WORLD_SETTINGS, **MISSION_SETTINGS, **COMMON)


def _mission(tm, memory):
    io = RosIO(tm, tm.p)
    return io, Mission(io, memory, tm.p, tm.say)


def _hand_bottle(tm):
    tm.move("open the claw", gripper_aperture=tm.p["gripper_open"])
    tm.hold_up()
    input("  Put the bottle deep in the claw, then press Enter...")
    tm.move("close the claw", gripper_aperture=tm.p["gripper_squeeze"])
    time.sleep(0.5)


def _save_trial(io, memory, tid, trial):
    base = f"{tid}_trial{trial}_{stamp()}"
    os.makedirs(OUT, exist_ok=True)
    j = memory.save(os.path.join(OUT, base + ".json"))
    png = os.path.join(OUT, base + "_map.png")
    io.save_map(memory, png, title=f"{tid} trial {trial}")
    return j, png


# =====================================================================================================
# T12: find the bin, drive to it, drop the bottle in
# =====================================================================================================

def test_bin_drop():
    """T12: with a bottle in the claw, find the bin, drive beside it around obstacles, and drop the bottle in.

      ros2 launch pill_bottle_nodes robot.launch.py lidar:=true
      ros2 launch pill_bottle_nodes nodes.launch.py stage:=full
      ros2 run pill_bottle_nodes test_bin_drop                 (asks before every move; -p confirm:=false to stop asking)
    """
    tm = make_tm("test_bin_drop", WORLD_EXTRA, {"confirm": True})

    def go():
        p = tm.p
        memory = WorldMemory()
        io, mission = _mission(tm, memory)
        logger = _results_logger(p["repo_testing_dir"], "T12", "Search for and drive to the bin, drop the bottle in")
        conds = ["bin_in_view", "bin_behind", "remembered"]
        for trial in range(1, int(p["trials"]) + 1):
            cond = ask_line(f"\nTrial {trial}: condition ({', '.join(conds)}; q = quit)", conds[(trial - 1) % 3])
            if cond.startswith("q"):
                break
            _hand_bottle(tm)
            t0 = time.time()
            if cond != "remembered" or memory.bin() is None:
                for it in list(memory.items):
                    if it["type"] == "bin":
                        memory.items.remove(it)
                mission.look_around()
                if memory.bin() is None and not mission.find_bin():
                    print("  -> NOT_FOUND: no bin anywhere")
                    if logger:
                        logger.log("NOT_FOUND", cond, {"search_time_s": round(time.time() - t0, 1)})
                    continue
            search_s = time.time() - t0
            binn = memory.bin()
            start_d = math.hypot(*to_robot(io.pose(), binn["xyz"][:2]))
            ok, why = mission.drive_to(binn["xyz"][:2], what="bin")
            dropped = ok and io.drop(binn)
            landed = dropped and yes("  Did the bottle land IN the bin?")
            hit = yes("  Did the robot touch anything?")
            outcome = "COLLISION" if hit else ("SUCCESS" if landed else ("NO_PLAN" if not ok else "MISS"))
            m = {"search_time_s": round(search_s, 1), "bin_distance_m": round(start_d, 2),
                 "drop_result": "in" if landed else ("missed" if dropped else why or "drop failed"),
                 "detours": mission.stats["detours"], "replans": mission.stats["replans"]}
            j, png = _save_trial(io, memory, "T12", trial)
            print(f"  -> {outcome} {m}\n  map: {png}")
            if logger:
                logger.log(outcome, cond, m, notes=f"map: {png}")
    _run(go, tm)


# =====================================================================================================
# T13: drive around obstacles
# =====================================================================================================

def test_obstacles():
    """T13: look around, then drive to the nearest bottle (or the bin, -p target:=bin) around obstacles.

      ros2 launch pill_bottle_nodes robot.launch.py lidar:=true
      ros2 launch pill_bottle_nodes nodes.launch.py stage:=full
      ros2 run pill_bottle_nodes test_obstacles
    """
    tm = make_tm("test_obstacles", dict(WORLD_EXTRA, target="bottle"), {"confirm": True})

    def go():
        p = tm.p
        logger = _results_logger(p["repo_testing_dir"], "T13", "Navigate around obstacles")
        conds = ["no_obstacle", "one_obstacle", "two_obstacles", "no_way_around"]
        for trial in range(1, int(p["trials"]) * len(conds) + 1):
            cond = ask_line(f"\nTrial {trial}: condition ({', '.join(conds)}; q = quit)", conds[(trial - 1) % 4])
            if cond.startswith("q"):
                break
            input("  Set up the obstacle(s) between the robot and the target, then press Enter...")
            memory = WorldMemory()
            io, mission = _mission(tm, memory)
            mission.look_around()
            target = memory.bin() if p["target"] == "bin" else memory.nearest("bottle", io.pose()[:2])
            if target is None:
                print("  -> NOT_FOUND: no target seen")
                if logger:
                    logger.log("ERROR", cond, {"obstacles_seen": len(memory.of_type("obstacle"))}, notes="no target")
                continue
            ok, why = mission.drive_to(target["xyz"][:2], what=target["id"])
            hit = yes("  Did the robot touch anything?")
            if hit:
                outcome = "COLLISION"
            elif cond == "no_way_around":
                outcome = "SUCCESS" if not ok else "MISS"     # it should say there's no way, not squeeze through
            else:
                outcome = "SUCCESS" if ok else "NO_PLAN"
            m = {"obstacles_seen": len(memory.of_type("obstacle")), "detours": mission.stats["detours"],
                 "replans": mission.stats["replans"], "stopped_by_lidar": mission.stats["stopped_by_lidar"],
                 "result": "arrived" if ok else why}
            j, png = _save_trial(io, memory, "T13", trial)
            print(f"  -> {outcome} {m}\n  map: {png}")
            if logger:
                logger.log(outcome, cond, m, notes=f"map: {png}")
    _run(go, tm)


# =====================================================================================================
# T18: remember bottles, bin and obstacles
# =====================================================================================================

def test_world():
    """T18: look around and remember every bottle, the bin and obstacles; check the list holds after driving;
    drive around an obstacle to a bottle; report a bottle there's no way to reach.

      ros2 launch pill_bottle_nodes robot.launch.py lidar:=true
      ros2 launch pill_bottle_nodes nodes.launch.py stage:=full
      ros2 run pill_bottle_nodes test_world
    """
    tm = make_tm("test_world", WORLD_EXTRA, {"confirm": True})

    def go():
        p = tm.p
        logger = _results_logger(p["repo_testing_dir"], "T18", "Remember bottles, bin and obstacles")
        expected = int(ask_line("How many pill bottles are out?", "3"))
        conds = ["look_around", "after_drive_2m_turn_180", "bottle_behind_obstacle", "no_way_around"]
        memory = WorldMemory(expected=expected)
        io, mission = _mission(tm, memory)
        for trial in range(1, int(p["trials"]) * len(conds) + 1):
            cond = ask_line(f"\nTrial {trial}: condition ({', '.join(conds)}; q = quit)", conds[(trial - 1) % 4])
            if cond.startswith("q"):
                break
            if cond in ("look_around", "bottle_behind_obstacle", "no_way_around"):
                input("  Set up the room, then press Enter...")
                memory = WorldMemory(expected=expected)
                io, mission = _mission(tm, memory)
                mission.look_around()
            elif not memory.items:
                mission.look_around()
            print("\n".join("  " + _item_line(it) for it in memory.items))
            bottles = memory.of_type("bottle")
            dups = sum(1 for i, a in enumerate(bottles) for b in bottles[i + 1:]
                       if math.hypot(a["xyz"][0] - b["xyz"][0], a["xyz"][1] - b["xyz"][1]) < 0.3)
            m = {"bottles_saved": len(bottles), "duplicates": dups, "bin_found": memory.bin() is not None,
                 "obstacles": len(memory.of_type("obstacle"))}
            if cond == "look_around":
                right = yes("  Is every bottle and the bin in the list, in about the right place?", True)
                outcome = "SUCCESS" if (len(bottles) == expected and dups == 0 and m["bin_found"] and right) else "MISS"
            elif cond == "after_drive_2m_turn_180":
                before = {b["id"]: (b["xyz"], b["n_seen"]) for b in bottles}
                # 1 m out, turn around, 1 m back, turn around (2 m and two 180s); the lidar checks every drive
                moved = mission._execute([("translate", 0.5), ("translate", 0.5), ("rotate", math.pi),
                                          ("translate", 0.5), ("translate", 0.5), ("rotate", math.pi)])
                if not moved:
                    print("  (something was in the way, so it stopped early; comparing from here)")
                mission.look_around()
                shifts = [math.hypot(memory.get(i)["xyz"][0] - xyz[0], memory.get(i)["xyz"][1] - xyz[1]) * 100
                          for i, (xyz, n) in before.items() if memory.get(i)["n_seen"] > n]
                new = len(memory.of_type("bottle")) - len(before)
                m.update(max_position_error_cm=round(max(shifts), 1) if shifts else None, new_entries=new)
                outcome = "SUCCESS" if (shifts and max(shifts) <= 10 and new == 0) else "MISS"
            elif cond == "bottle_behind_obstacle":
                b = memory.nearest("bottle", io.pose()[:2])
                if b is None:
                    outcome = "NOT_FOUND"
                else:
                    ok, why = mission.drive_to(b["xyz"][:2], what=b["id"])
                    st = io.reacquire(b["xyz"]) if ok else None
                    if st is not None:
                        now = io.to_odom(st)
                        m["max_position_error_cm"] = round(math.hypot(now[0] - b["xyz"][0], now[1] - b["xyz"][1]) * 100, 1)
                    hit = yes("  Did the robot touch anything?")
                    m.update(detours=mission.stats["detours"], collisions=int(hit), result="arrived" if ok else why)
                    outcome = "COLLISION" if hit else ("SUCCESS" if ok and st is not None and
                                                       m["max_position_error_cm"] <= 10 else ("NO_PLAN" if not ok else "MISS"))
            else:
                b = memory.nearest("bottle", io.pose()[:2])
                if b is None:
                    outcome = "NOT_FOUND"
                else:
                    pose = io.pose()
                    spot = stand_spot(b["xyz"][:2], pose[:2], memory.obstacles(), p)
                    plan = plan_path(pose[:2], spot, memory.obstacles(), p) if spot else {"ok": False, "blocked_by": "?"}
                    if not plan["ok"]:
                        memory.set_status(b["id"], "blocked", why=f"no way around {plan.get('blocked_by')}")
                        print(f"  {b['id']} marked BLOCKED (no way around {plan.get('blocked_by')}); nothing moved.")
                        outcome = "SUCCESS"
                    else:
                        print(f"  It found a way: {plan['waypoints']} (not driving). Is there really no way around?")
                        outcome = "MISS"
            j, png = _save_trial(io, memory, "T18", trial)
            print(f"  -> {outcome} {m}\n  list: {j}\n  map:  {png}")
            if logger:
                logger.log(outcome, cond, m, notes=f"map: {png}")
    _run(go, tm)


def _item_line(it):
    x, y, _ = it["xyz"]
    extra = it.get("status", "") + (f" {it['orientation']}" if it.get("orientation") else "") + \
        (f" r {it['radius'] * 100:.0f} cm" if it["type"] == "obstacle" else "")
    return f"{it['id']:12s} ({x:+.2f}, {y:+.2f}) m  seen {it['n_seen']}x  {extra}"


# =====================================================================================================
# T19: orientation-aware grasp (a = plan only, b = plan and do it)
# =====================================================================================================

T19_CONDS = ["upright_floor", "upright_table", "upright_raised", "lying_toward", "lying_across", "lying_away", "tilted"]


def _t19(execute):
    name = "t19b_orientation_grasp" if execute else "t19a_orientation_plan"
    over = {"max_base_turns": 1} if not execute else {"confirm": True}
    tm = make_tm(name, dict(ROBOT_WORLD_SETTINGS, **COMMON), over)

    def go():
        p = tm.p
        tid = "T19b" if execute else "T19a"
        logger = _results_logger(p["repo_testing_dir"], tid, "Orientation-aware grasp: " +
                                 ("plan, then carry it out" if execute else "plan only (no base/arm motion)"))
        io = RosIO(tm, p) if execute else None
        tm.stow_for_base()
        for trial in range(1, int(p["trials"]) * len(T19_CONDS) + 1):
            cond = ask_line(f"\nTrial {trial}: condition ({', '.join(T19_CONDS)}; q = quit)",
                            T19_CONDS[(trial - 1) % len(T19_CONDS)])
            if cond.startswith("q"):
                break
            truth = cond.split("_")[0]
            input(f"  Place the bottle ({cond}), then press Enter...")
            st = tm.find()
            if st is None:
                print("  -> NOT_FOUND")
                if logger:
                    logger.log("NOT_FOUND", cond, {})
                continue
            wrist, grasp_c = tm.c.position("link_wrist_yaw"), tm.c.position("link_grasp_center")
            g0 = grasp_center_at_yaw0(wrist, grasp_c)
            ext, lift = tm.c.joints["wrist_extension"], tm.c.joints["joint_lift"]
            seq = plan_sequence((st.position.x, st.position.y, st.position.z), st.orientation, st.surface_z,
                                st.axis_yaw, ext, lift, g0, PLANNER_SETTINGS, max_moves=p["max_base_moves"])
            ok, kind, reasons = check_plan(st.orientation, seq, st.surface_z, lift, g0, PLANNER_SETTINGS, tol=0.015)
            print(f"  Sees: {st.orientation} (tilt {st.axis_tilt_deg:.0f} deg, {st.num_points} depth points); "
                  f"really: {truth}")
            print("  Plan:\n" + "\n".join("    " + line for line in describe(seq, lift, g0)))
            print(f"  Plan type: {kind}; " + ("fits the orientation" if ok else "PROBLEMS: " + "; ".join(reasons)))
            m = {"seen_orientation": st.orientation, "plan_type": kind, "base_moves": seq["base_moves"],
                 "grip_height_m": round(grip_height(seq["final"], lift, g0), 3) if seq["ok"] else None}
            if st.orientation != truth:
                outcome = "MISS"
            elif not seq["ok"]:
                outcome = "NO_PLAN"
            else:
                outcome = "SUCCESS" if ok else "MISS"
            if execute and outcome == "SUCCESS":
                outcome, more = _t19_execute(tm, io, st)
                m.update(more)
            elif execute:
                print("  Not grabbing: fix the orientation/plan first (run T19a).")
            print(f"  -> {outcome} {m}")
            if logger:
                logger.log(outcome, cond, m, notes="; ".join(reasons))
            tm.stow_for_base()
    _run(go, tm)


def _t19_execute(tm, io, st):
    if not yes("  Carry out the plan (base and arm will move)?", True):
        return "NO_PLAN", {"attempts": 0}
    p = tm.p
    for attempt in range(1, 4):
        plan, st2, why = io.reach_plan(st)
        if plan is None:
            return "NO_PLAN", {"attempts": attempt, "why": why}
        st = st2
        log = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "source": "t19b", "orientation": st.orientation,
               "attempt": attempt}
        grasp.approach(tm.c, plan, p, tm.move, tm.say, log)
        grasp.refine(tm.c, p, tm.move, tm.say, log)
        outcome = grasp.close_and_check(tm.c, plan, p, tm.move, tm.say, log, tag=f"_t19b_try{attempt}")
        tm.save_log(log)
        if outcome in ("held", "slipping"):
            tm.move("lift it", joint_lift=min(plan.lift + 0.10, p["lift_max"]))
            time.sleep(5.0)
            _, effort = tm.c.finger()
            still = effort is not None and abs(effort) >= p["grip_effort_min"]
            grasp.place(tm.c, plan, p, tm.move)
            hit = yes("  Did the robot touch anything?")
            res = "COLLISION" if hit else ("SUCCESS" if still and outcome == "held" else "DROP")
            return res, {"attempts": attempt, "grip_effort": round(abs(effort), 2) if effort is not None else None}
        grasp.release(tm.c, plan, p, tm.move)
        st = io.reacquire(io.to_odom(st))
        if st is None:
            return "NOT_FOUND", {"attempts": attempt}
    return "MISS", {"attempts": 3}


def t19a_orientation_plan():
    """T19a: see the bottle, work out its orientation and print the grip plan for it. Only the head moves.

      ros2 launch pill_bottle_nodes nodes.launch.py stage:=arm        (detector, localizer, planner, motion for the head)
      ros2 run pill_bottle_nodes t19a_orientation_plan
    """
    _t19(False)


def t19b_orientation_grasp():
    """T19b: same as T19a, then (after you say yes) turn/drive the base, grab, lift, hold 5 s, put it back.

      ros2 launch pill_bottle_nodes nodes.launch.py stage:=full
      ros2 run pill_bottle_nodes t19b_orientation_grasp
    """
    _t19(True)


# =====================================================================================================
# T15: the full demo run
# =====================================================================================================

def demo_run():
    """T15: the full run. You say how many bottles are out; it looks around, finds the bin, fetches every bottle
    around obstacles (reading each label), drops them in the bin, checks the bin, and prints the report.

      ros2 launch pill_bottle_nodes robot.launch.py lidar:=true
      ros2 launch pill_bottle_nodes nodes.launch.py stage:=full
      export GEMINI_API_KEY=...            (else EasyOCR reads the labels)
      ros2 run pill_bottle_nodes demo_run                      (-p confirm:=true to approve every move)
    """
    tm = make_tm("demo_run", WORLD_EXTRA)

    def go():
        p = tm.p
        if not tm.c.wait_joints():
            print("No /joint_states. Start robot.launch.py first.")
            return
        expected = int(ask_line("How many pill bottles are out?", "3"))
        cond = ask_line("Condition (3_bottles_open_floor / 3_bottles_with_obstacle / mixed_orientations)",
                        "3_bottles_open_floor")
        if not p["confirm"] and not yes("FULL RUN: the base and arm will move on their own. Area clear, hand on "
                                        "the runstop. Start?", False):
            return
        memory = WorldMemory(expected=expected)
        io, mission = _mission(tm, memory)
        t0 = time.time()
        folder = os.path.join(OUT, f"run_{stamp()}")
        os.makedirs(folder, exist_ok=True)
        try:
            tm.stow_for_base()
            mission.run()
        finally:
            memory.save(os.path.join(folder, "world.json"))
            io.save_map(memory, os.path.join(folder, "map.png"), title="T15 run")
            lines = report_lines(memory)
            with open(os.path.join(folder, "report.txt"), "w") as f:
                f.write("\n".join(lines) + "\n\nLog:\n" + "\n".join(f"{t} {x}" for t, x in mission.log))
            print(f"\nSaved the report, the list and the map in {folder}")
        s = memory.summary()
        hit = yes("Did the robot touch anything during the run?")
        logger = _results_logger(p["repo_testing_dir"], "T15", "Full demo run")
        labels = sum(1 for b in memory.of_type("bottle") if (b.get("label") or {}).get("status") == "read")
        outcome = "COLLISION" if hit else ("SUCCESS" if s["in_bin"] == expected else "MISS")
        m = {"bottles_expected": expected, "bottles_found": s["bottles_found"], "bottles_binned": s["in_bin"],
             "labels_read": labels, "bin_camera_count": memory.bin_camera_count,
             "total_time_s": round(time.time() - t0)}
        print(f"-> {outcome} {m}")
        if logger:
            logger.log(outcome, cond, m, notes=f"report: {folder}")
    _run(go, tm)
