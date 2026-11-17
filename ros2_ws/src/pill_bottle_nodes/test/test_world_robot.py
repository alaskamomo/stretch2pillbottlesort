# Robot-side code (world_robot.RosIO and the testers) checked without a robot: python3 -m pytest test/
# If ROS isn't installed (laptop / CI), tiny stand-ins for the ROS modules are loaded so the files can be imported;
# on the robot the real ones are used. The camera, depth, lidar, tf and moves are faked with known geometry.
import importlib.util
import math
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest


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

    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__getattr__ = lambda n: type(n, (_Any,), {})
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m
        return m

    for name in ("rclpy", "rclpy.node", "rclpy.executors", "rclpy.qos", "rclpy.time", "rclpy.duration",
                 "rclpy.action", "rclpy.callback_groups", "sensor_msgs", "sensor_msgs.msg", "std_msgs",
                 "std_msgs.msg", "tf2_ros", "pill_bottle_interfaces", "pill_bottle_interfaces.msg",
                 "pill_bottle_interfaces.srv", "control_msgs", "control_msgs.action", "trajectory_msgs",
                 "trajectory_msgs.msg", "geometry_msgs", "geometry_msgs.msg", "builtin_interfaces",
                 "builtin_interfaces.msg"):
        mod(name)
    sys.modules["rclpy"].init = lambda *a, **k: None
    sys.modules["rclpy"].try_shutdown = lambda *a, **k: None
    sys.modules["rclpy.qos"].qos_profile_sensor_data = object()


_stub_ros()

from pill_bottle_nodes import world_robot as wr                         # noqa: E402
from pill_bottle_nodes.geometry import PLANNER_SETTINGS as P, quat_to_matrix   # noqa: E402
from pill_bottle_nodes.world import WorldMemory                          # noqa: E402

K = [600.0, 0, 640.0, 0, 600.0, 360.0, 0, 0, 1]
W, H = 1280, 720


def cam_matrix(height=1.2, pitch_deg=30.0):
    """base_link <- camera optical frame: camera at (0, 0, height) looking forward (+x), tilted down."""
    p = math.radians(pitch_deg)
    # optical: z forward, x right, y down. Columns = optical axes expressed in base.
    z = np.array([math.cos(p), 0, -math.sin(p)])
    x = np.array([0, -1.0, 0])
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, (0, 0, height)
    return T


def render(points_base, T_base_cam):
    """Depth image (m) and the pixel box of points (base_link) seen by the camera."""
    T_cam = np.linalg.inv(T_base_cam)
    pc = points_base @ T_cam[:3, :3].T + T_cam[:3, 3]
    pc = pc[pc[:, 2] > 0.1]
    u = np.round(K[0] * pc[:, 0] / pc[:, 2] + K[2]).astype(int)
    v = np.round(K[4] * pc[:, 1] / pc[:, 2] + K[5]).astype(int)
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    depth = np.full((H, W), np.inf)
    for uu, vv, zz in zip(u[ok], v[ok], pc[ok, 2]):
        depth[vv, uu] = min(depth[vv, uu], zz)
    depth[~np.isfinite(depth)] = 0
    return depth.astype(np.float32), (u[ok].min(), v[ok].min(), u[ok].max(), v[ok].max())


def cylinder(center, axis=(0, 0, 1), length=0.08, radius=0.02, n=40000, seed=0):
    rng = np.random.default_rng(seed)
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    a = np.cross(axis, [0, 0, 1.0]) if abs(axis[2]) < 0.9 else np.cross(axis, [1.0, 0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(axis, a)
    t, th = rng.uniform(-length / 2, length / 2, n), rng.uniform(0, 2 * math.pi, n)
    return np.asarray(center) + np.outer(t, axis) + radius * (np.outer(np.cos(th), a) + np.outer(np.sin(th), b))


class FakeClient:
    def __init__(self, T_base_cam, odom_pose=(0.0, 0.0, 0.0)):
        self.T_base_cam = T_base_cam
        self.odom_pose = odom_pose
        self.joints = {"wrist_extension": 0.0, "joint_lift": 0.6}
        self.msgs = {}

    def matrix(self, target, source, stamp=None):
        x, y, yaw = self.odom_pose
        T_odom_base = quat_to_matrix(x, y, 0.0, 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))
        if (target, source) == ("base_link", "camera"):
            return self.T_base_cam
        if (target, source) == ("odom", "base_link"):
            return T_odom_base
        if (target, source) == ("odom", "laser"):
            return T_odom_base
        if (target, source) == ("base_link", "laser"):
            return np.eye(4)
        raise KeyError((target, source))

    def position(self, frame):
        return {"link_head_tilt": (0.0, 0.0, 1.2), "link_wrist_yaw": (0.0, -0.25, 0.55),
                "link_grasp_center": (0.0, -0.30, 0.55)}[frame]

    def wait_for(self, key, timeout_s=5.0, newer_than=None):
        return self.msgs.get(key)

    def latest(self, key):
        return self.msgs.get(key)


class FakeTM:
    def __init__(self, c):
        self.c = c
        self.p = dict(P, gripper_open=0.10, max_base_moves=10, read_label=False)
        self.moves = []

    def say(self, text):
        pass

    def move(self, label, **joints):
        self.moves.append((label, joints))

    def stow_for_base(self):
        self.moves.append(("stow", {}))


def depth_msg(depth_m):
    mm = (depth_m * 1000).astype(np.uint16)
    return SimpleNamespace(encoding="16UC1", height=H, width=W, step=W * 2, data=mm.tobytes())


def det(name, box, conf=0.8):
    return SimpleNamespace(name=name, confidence=conf, x_min=float(box[0]), y_min=float(box[1]),
                           x_max=float(box[2]), y_max=float(box[3]))


def frame_msg(dets):
    return SimpleNamespace(detections=dets, image_width=W, image_height=H,
                           header=SimpleNamespace(frame_id="camera", stamp=None))


def make_io(odom_pose=(0.0, 0.0, 0.0)):
    c = FakeClient(cam_matrix(), odom_pose)
    tm = FakeTM(c)
    return wr.RosIO(tm, {"look_settle_s": 0.0}), c, tm


@pytest.mark.parametrize("orientation,axis", [("upright", (0, 0, 1)), ("lying", (0, 1, 0))])
def test_locate_finds_the_bottle_and_its_orientation(orientation, axis):
    io, c, _ = make_io(odom_pose=(1.0, 2.0, math.pi / 2))
    true = (1.4, -0.1, 0.04 if orientation == "upright" else 0.02)
    depth, box = render(cylinder(true, axis), c.T_base_cam)
    found = io.locate(frame_msg([det("pill_bottle", box)]), depth_msg(depth), SimpleNamespace(k=K))
    assert len(found) == 1
    kind, xyz_base, xyz_odom, extra = found[0]
    assert kind == "bottle" and extra["orientation"] == orientation
    assert math.hypot(xyz_base[0] - true[0], xyz_base[1] - true[1]) < 0.03
    # robot at (1, 2) facing +y: base (1.4, -0.1) is odom (1.1, 3.4)
    assert xyz_odom[0] == pytest.approx(1.1, abs=0.03) and xyz_odom[1] == pytest.approx(3.4, abs=0.03)


def test_locate_ignores_low_confidence_and_far_and_robot_parts():
    io, c, _ = make_io()
    depth, box = render(cylinder((1.4, 0.0, 0.04)), c.T_base_cam)
    assert io.locate(frame_msg([det("pill_bottle", box, conf=0.2)]), depth_msg(depth), SimpleNamespace(k=K)) == []
    depth2, box2 = render(cylinder((0.15, 0.0, 0.5)), cam_matrix(height=1.2, pitch_deg=75))
    io.c.T_base_cam = cam_matrix(height=1.2, pitch_deg=75)
    assert io.locate(frame_msg([det("pill_bottle", box2)]), depth_msg(depth2), SimpleNamespace(k=K)) == []


def test_locate_bin_center_and_rim():
    io, c, _ = make_io()
    rng = np.random.default_rng(1)
    # a 30 cm wide, 25 cm tall bin 1.5 m ahead: the camera sees its near wall and some of the inside
    n = 30000
    wall = np.column_stack([np.full(n, 1.35), rng.uniform(-0.15, 0.15, n), rng.uniform(0, 0.25, n)])
    back = np.column_stack([np.full(n, 1.65), rng.uniform(-0.15, 0.15, n), rng.uniform(0.15, 0.25, n)])
    depth, box = render(np.vstack([wall, back]), c.T_base_cam)
    found = io.locate(frame_msg([det("bin", box)]), depth_msg(depth), SimpleNamespace(k=K), kinds=("bin",))
    assert len(found) == 1
    kind, xyz, odom, extra = found[0]
    assert kind == "bin" and 1.35 < xyz[0] < 1.65 and abs(xyz[1]) < 0.05 and extra["rim_z"] == pytest.approx(0.25, abs=0.03)


def test_scan_circles_and_robot_points():
    io, c, _ = make_io(odom_pose=(0.5, 0.0, 0.0))
    n = 720
    ranges = []
    for i in range(n):                      # one 20 cm post 1.2 m straight ahead
        a = -math.pi + i * 2 * math.pi / n
        dx, dy = math.cos(a), math.sin(a)
        fx, fy = -1.2, 0.0
        b = fx * dx + fy * dy
        disc = b * b - (fx * fx + fy * fy - 0.04)
        ranges.append(-b - math.sqrt(disc) if disc >= 0 and -b - math.sqrt(disc) > 0 else float("inf"))
    c.msgs["scan"] = SimpleNamespace(ranges=ranges, angle_min=-math.pi, angle_increment=2 * math.pi / n,
                                     header=SimpleNamespace(frame_id="laser", stamp=None))
    circles = io.scan_circles()
    assert len(circles) == 1 and circles[0][0] == pytest.approx(1.7, abs=0.06) and abs(circles[0][1]) < 0.03
    from pill_bottle_nodes.world import corridor_blocked
    assert corridor_blocked(io.scan_robot(), 0.5) is None                    # 0.5 m drive + 0.3 m stop < 1.0 m
    assert corridor_blocked(io.scan_robot(), 0.8) == pytest.approx(1.0, abs=0.03)


def test_no_lidar_refuses_to_drive():
    io, c, _ = make_io()
    with pytest.raises(RuntimeError, match="lidar"):
        io.scan_robot()


def test_drop_over_the_bin_uses_the_drop_plan():
    io, c, tm = make_io()
    c.msgs["detections"] = None                         # bin not in view: uses the remembered spot and rim
    bin_item = {"id": "bin", "xyz": (0.0, -0.62, 0.0), "rim_z": 0.30}
    assert io.drop(bin_item)
    labels = [m[0] for m in tm.moves]
    assert labels.index("lift over the bin rim") < labels.index("reach over the bin") < labels.index("let go")
    lift = dict(tm.moves[labels.index("lower toward the rim")][1])["joint_lift"]
    # grasp center sits at 0.55 with lift 0.6: bottle bottom = rim + 8 cm
    assert 0.55 + (lift - 0.6) - P["grasp_frac_upright"] * P["bottle_length"] == pytest.approx(0.38, abs=1e-6)


def test_drop_drives_and_turns_when_the_bin_is_ahead():
    io, c, tm = make_io()
    c.msgs["detections"] = None
    steps = []

    def rotate(rad):
        x, y, yaw = c.odom_pose
        steps.append(("turn", round(math.degrees(rad))))
        c.odom_pose = (x, y, yaw + rad)

    def translate(m):
        x, y, yaw = c.odom_pose
        steps.append(("drive", round(m, 2)))
        c.odom_pose = (x + m * math.cos(yaw), y + m * math.sin(yaw), yaw)
    io.rotate, io.translate = rotate, translate
    assert io.drop({"id": "bin", "xyz": (1.5, 0.0, 0.0), "rim_z": 0.3})
    assert steps[0][0] == "drive" and ("turn", 90) in steps          # drive up to it, then turn the arm toward it
    assert [m[0] for m in tm.moves][-4] == "let go"


def test_testers_import():
    import pill_bottle_nodes.testers_world as tw
    for fn in ("test_range", "live_view", "test_bin_detection", "test_bin_drop", "test_obstacles", "test_world",
               "t19a_orientation_plan", "t19b_orientation_grasp", "demo_run"):
        assert callable(getattr(tw, fn))


def test_world_memory_works_with_rosio_observations():
    io, c, _ = make_io()
    depth, box = render(cylinder((1.4, -0.1, 0.04)), c.T_base_cam)
    c.msgs.update(detections=frame_msg([det("pill_bottle", box)]), depth=depth_msg(depth),
                  camera_info=SimpleNamespace(k=K))
    io.head = lambda pan, tilt: None
    mem = WorldMemory(expected=1)
    for pan in (0.0, 0.3):
        for kind, xyz, extra in io.look(pan):
            mem.observe(kind, xyz, **extra)
    b = mem.of_type("bottle")
    assert len(b) == 1 and b[0]["n_seen"] == 2 and b[0]["orientation"] == "upright"


def test_reacquire_picks_the_bottle_at_the_saved_spot_not_the_most_confident(monkeypatch):
    io, c, _ = make_io()
    monkeypatch.setattr(wr, "make_state", lambda xyz, extra: SimpleNamespace(xyz=xyz, **extra))
    io.head = lambda pan, tilt: None
    target, other = (1.4, -0.3, 0.04), (1.5, 0.35, 0.04)
    pts = np.vstack([cylinder(target), cylinder(other, seed=1)])
    depth, _ = render(pts, c.T_base_cam)
    _, box_t = render(cylinder(target), c.T_base_cam)
    _, box_o = render(cylinder(other, seed=1), c.T_base_cam)
    c.msgs.update(detections=frame_msg([det("pill_bottle", box_o, 0.95), det("pill_bottle", box_t, 0.6)]),
                  depth=depth_msg(depth), camera_info=SimpleNamespace(k=K))
    st = io.reacquire((1.42, -0.28, 0.04))
    assert st is not None and st.xyz[1] == pytest.approx(-0.3, abs=0.03) and st.orientation == "upright"
    assert st.num_points > 30 and "axis_yaw" in st.__dict__
    assert io.reacquire((0.5, 1.5, 0.0)) is None                        # nothing there


def test_t6_runs_without_questions(monkeypatch, tmp_path, capsys):
    """T6 with a fake camera and keyboard: live lines, Enter records two spots (distance measured), q finishes."""
    import queue
    import pill_bottle_nodes.testers_world as tw
    from pill_bottle_nodes.range_test import RANGE_SETTINGS
    spots = [[(1.2, -0.3, 0.02, 0.83)], [(2.6, 0.4, 0.75, 0.21)]]   # spot 2: far + raised + low confidence
    calls = {"640": 0}

    class FakeView:
        def __init__(self, tm):
            self.last_view = np.zeros((64, 36, 3), np.uint8)

        def link(self):
            return "http://10.0.11.20:8080"

        def analyze(self, size):
            if size == 640:                                           # only recording uses 640
                calls["640"] += 1
            spot = spots[min(max(0, calls["640"] - 1) // RANGE_SETTINGS["frames"], len(spots) - 1)]
            boxes = [dict(conf=c, xyz=(x, y, z), dist=math.hypot(x, y), depth_px=40, w=10, h=20)
                     for x, y, z, c in spot if not (size == 640 and x > 2)]   # 640 misses the far bottle
            return {"size": size, "boxes": boxes, "ms": 100.0}

    keys = queue.Queue()
    for c in ("", "", "q"):
        keys.put((math.inf, c))
    p = dict(RANGE_SETTINGS, live_size=1280, print_s=0.0, stream_port=0, repo_testing_dir="")
    tm = SimpleNamespace(p=p, c=SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(tw, "make_tm", lambda name, extra: tm)
    monkeypatch.setattr(tw, "RangeView", FakeView)
    monkeypatch.setattr(tw, "_keyboard", lambda: keys)
    monkeypatch.setattr(tw, "_results_logger", lambda *a: None)
    monkeypatch.setattr(tw, "OUT", str(tmp_path))
    monkeypatch.setattr("builtins.input", lambda *a: pytest.fail("T6 asked a question"))
    tw.test_range()
    out = capsys.readouterr().out
    assert "Live camera: http://10.0.11.20:8080" in out
    assert "pill bottle 1/1 detected at x=1.20 y=-0.30 z=0.02 m" in out
    assert "Spot 1: 1.24 m, height floor" in out
    assert "Spot 2: 2.63 m, height table" in out and "misses: {'low conf': 10}" in out
    assert "Max reliable distance [floor]: 640 -> 1.24 m   1280 -> 1.24 m" in out
    assert "Max reliable distance [table]: 640 -> none m   1280 -> none m" in out
    assert len(list(tmp_path.glob("T6_spot*.jpg"))) == 2 and len(list(tmp_path.glob("T6_range_*.json"))) == 1


def test_t6_q_typed_while_recording_still_stops(monkeypatch, tmp_path):
    import queue
    import time as _t
    import pill_bottle_nodes.testers_world as tw
    from pill_bottle_nodes.range_test import RANGE_SETTINGS
    keys = queue.Queue()
    keys.put((math.inf, ""))
    keys.put((_t.time(), ""))        # extra Enter during recording: ignored
    keys.put((_t.time(), "q"))       # q during recording: still stops

    class FakeView:
        last_view = None

        def __init__(self, tm):
            pass

        def link(self):
            return ""

        def analyze(self, size):
            return {"size": size, "boxes": [], "ms": 50.0}

    p = dict(RANGE_SETTINGS, live_size=1280, print_s=0.0, stream_port=0, repo_testing_dir="", frames=2)
    tm = SimpleNamespace(p=p, c=SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(tw, "make_tm", lambda name, extra: tm)
    monkeypatch.setattr(tw, "RangeView", FakeView)
    monkeypatch.setattr(tw, "_keyboard", lambda: keys)
    monkeypatch.setattr(tw, "_results_logger", lambda *a: None)
    monkeypatch.setattr(tw, "OUT", str(tmp_path))
    tw.test_range()                  # returns (would hang forever if q were lost)
    assert len(list(tmp_path.glob("T6_range_*.json"))) == 1
