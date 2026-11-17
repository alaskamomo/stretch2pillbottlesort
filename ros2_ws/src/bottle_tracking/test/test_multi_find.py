# T2 several bottles (no robot): python3 -m pytest test/test_multi_find.py
# The sweep itself is run with a fake head/camera; ROS modules are stubbed when ROS isn't installed.
import importlib.util
import math
import sys
import types

import numpy as np

from bottle_tracking.multi_find import GREEN, RED, YELLOW, bottle_lines, box_label, box_status, cluster, score


def sight(x, y, z=0.0, conf=0.8, view=1, status="ok"):
    return {"fixed": (x + 10.0, y, z), "base": (x, y, z), "conf": conf, "view": view, "status": status}


def test_box_status_and_colors():
    assert box_status(0.8, 1.2, 0.5, 4.0) == "ok"
    assert box_status(0.12, 0.78, 0.5, 4.0) == "low_conf"
    assert box_status(0.9, None, 0.5, 4.0) == "no_depth"
    assert box_status(0.9, 4.5, 0.5, 4.0) == "too_far"
    e = {"conf": 0.12, "depth_m": 0.78, "status": "low_conf"}
    text, color = box_label(e)
    assert color == YELLOW and "0.12" in text and "0.78 m" in text
    assert box_label({"conf": 0.9, "depth_m": 1.0, "status": "tracked"})[1] == GREEN
    assert box_label({"conf": 0.9, "depth_m": None, "status": "no_depth"})[1] == RED


def test_cluster_merges_views_and_keeps_bottles_apart():
    s = [sight(1.0, 0.0, view=1), sight(1.02, 0.01, view=2, conf=0.9), sight(1.01, -0.01, view=2),
         sight(2.0, 0.5, z=0.75, conf=0.12, view=3, status="low_conf"),
         sight(1.0, 0.25, conf=0.7, view=1)]                        # 25 cm away = a different bottle
    b = cluster(s, 0.10)
    assert len(b) == 3
    assert [round(x["distance_m"], 1) for x in b] == [1.0, 1.0, 2.1]
    first = max(b[:2], key=lambda x: x["n"])
    assert first["n"] == 3 and first["views"] == [1, 2] and first["conf"] == 0.9 and first["confident"]
    assert not b[2]["confident"] and b[2]["fixed"][2] == 0.75


def test_low_confidence_bottle_joins_a_confident_one():
    b = cluster([sight(1.0, 0.0, conf=0.9), sight(1.03, 0.0, conf=0.2, status="low_conf")], 0.10)
    assert len(b) == 1 and b[0]["confident"] and b[0]["n"] == 2


def test_report_lines():
    b = cluster([sight(1.2, -0.3, conf=0.83), sight(2.0, 0.5, z=0.75, conf=0.12, status="low_conf")], 0.10)
    lines = bottle_lines(b, 0.5, no_depth=[(4, 0.3)])
    assert lines[0] == "Found 1 pill bottle (+1 possible, low confidence)"
    assert lines[1].startswith("  Pill bottle 1/1 at x=11.20 y=-0.30") and "deg right" in lines[1]
    assert "Possible pill bottle 1/1" in lines[2] and "below 0.50" in lines[2]
    assert "no depth" in lines[3] and "view 4" in lines[3]
    assert bottle_lines([], 0.5)[0].startswith("Does NOT see")


def test_score():
    two = cluster([sight(1, 0), sight(2, 0)], 0.1)
    one_plus_maybe = cluster([sight(1, 0), sight(2, 0, conf=0.1, status="low_conf")], 0.1)
    assert score(2, two)[0] == "SUCCESS"
    assert score(1, two)[0] == "FALSE_POSITIVE"
    out, note = score(2, one_plus_maybe)
    assert out == "MISS" and "low confidence" in note
    assert score(2, [])[0] == "NOT_FOUND"
    assert score(None, two)[0] == "SUCCESS"


# ---------------------------------------------------------------- the sweep, with a fake head and camera

def _stub_ros():
    if importlib.util.find_spec("rclpy") is not None:
        return

    class _Any:
        def __init__(self, *a, **k):
            pass

        def __getattr__(self, name):
            return _Any()

        def __call__(self, *a, **k):
            return _Any()

    for name in ("rclpy", "rclpy.node", "rclpy.executors", "rclpy.qos", "rclpy.time", "rclpy.duration",
                 "rclpy.action", "rclpy.callback_groups", "sensor_msgs", "sensor_msgs.msg", "std_msgs",
                 "std_msgs.msg", "std_srvs", "std_srvs.srv", "tf2_ros", "control_msgs", "control_msgs.action",
                 "trajectory_msgs", "trajectory_msgs.msg", "geometry_msgs", "geometry_msgs.msg",
                 "builtin_interfaces", "builtin_interfaces.msg"):
        m = types.ModuleType(name)
        m.__getattr__ = lambda n: type(n, (_Any,), {})
        sys.modules[name] = m
    sys.modules["rclpy.qos"].QoSProfile = _Any
    sys.modules["rclpy.qos"].DurabilityPolicy = _Any()


def test_sweep_finds_two_bottles_and_a_low_confidence_one(tmp_path, capsys):
    _stub_ros()
    from bottle_tracking.testers import t2_find_tester as t2

    # bottles in the camera frame (= base frame here); which head pan sees which
    world = {0.0: [(1.0, -0.2, 0.0, 0.85)], -0.7: [(1.0, -0.2, 0.0, 0.40), (1.5, -1.2, 0.75, 0.15)],
             0.7: [(2.0, 1.5, 0.75, 0.9)], 1.4: [(0.5, 0.5, 2.5, 0.9)]}     # last one: on a wall, ignored
    head = {"pan": 0.0}

    class Robot:
        fixed_frame, base_frame = "odom", "base_link"

        def move_head_and_wait(self, pan, tilt):
            head["pan"] = pan
            return True

        def point_to(self, p, frm, to):
            return (p[0] + 5.0, p[1], p[2]) if to == "odom" else p

    class Snap:
        def write(self, name, img):
            return str(tmp_path / f"{name}.png")

    tr = object.__new__(t2.FindTester)
    tr.robot, tr.snap = Robot(), Snap()
    tr.mem = types.SimpleNamespace(set_tracking=lambda on: None)
    tr.pans, tr.tilts = [0.7, 0.0, -0.7, 1.4], [-0.45, -0.85]
    tr.dwell = tr.settle = 0.0
    tr.radius, tr.min_z, tr.max_z, tr.max_dist, tr.conf_thr = 0.10, -0.05, 1.40, 4.0, 0.5
    tr.n_frames = 0
    tr.now = lambda: 0.0

    def frames_since(t):
        boxes = []
        for x, y, z, c in world.get(head["pan"], []):
            boxes.append({"conf": c, "xyz": [x, y, z], "status": "ok" if c >= 0.5 else "low_conf"})
        if head["pan"] == 0.7:
            boxes.append({"conf": 0.3, "xyz": None, "status": "no_depth"})
        return [(1.0, "camera", boxes)] * 3
    tr.frames_since = frames_since
    tr.best_picture_since = lambda t: np.zeros((640, 360, 3), np.uint8)

    sightings, no_pos, pics, n, secs = tr.sweep("T2_test")
    bottles = cluster(sightings, tr.radius)
    sure = [b for b in bottles if b["confident"]]
    maybe = [b for b in bottles if not b["confident"]]
    assert len(sure) == 2 and len(maybe) == 1
    near = sure[0]
    assert math.isclose(near["fixed"][0], 6.0) and near["views"] == [2, 3, 6, 7]   # seen at pan 0 and -0.7
    assert maybe[0]["conf"] == 0.15 and math.isclose(maybe[0]["fixed"][2], 0.75)
    assert {v for v, _ in no_pos} == {1, 4, 5, 8}             # no-depth box at pan 0.7, wall bottle at pan 1.4
    assert len(pics) == 8                                      # every view saw something
    assert score(2, bottles)[0] == "SUCCESS"
    big = t2.mosaic([p[1] for p in pics])
    assert big.shape[0] == 2 * 480
