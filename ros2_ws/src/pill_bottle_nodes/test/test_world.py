# T18 / T13 / T12 / T15 tests (no robot, no ROS): python3 -m pytest test/test_world.py -v
#
# Unit tests for each piece, then a SIMULATED ROOM: bottles, a bin and obstacles; a robot whose odometry drifts;
# a head camera that only sees what's in front of it (with noise, hidden behind obstacles); a lidar that ray-casts
# the obstacles and the bin. The REAL mission code (mission.Mission, the same code the robot runs through
# world_robot.RosIO) does the whole run: look around, find the bin, fetch every bottle around the obstacles, put
# it in the bin, look for missing ones, report. The tests check it finds everything and never hits anything.
import math

import numpy as np
import pytest

from pill_bottle_nodes.geometry import PLANNER_SETTINGS as P
from pill_bottle_nodes.mission import Mission
from pill_bottle_nodes.world import (WORLD_SETTINGS as S, WorldMemory, corridor_blocked, path_to_moves, plan_drop,
                                     plan_path, report_lines, scan_clusters, scan_points, seg_point_dist, stand_spot,
                                     to_robot, to_world, bin_from_points)

CLEAR = S["robot_radius_m"] + S["clearance_m"]


# ---------------- memory ----------------

def test_same_bottle_seen_twice_is_one_entry():
    m = WorldMemory(expected=2)
    a = m.observe("bottle", (1.0, 0.0, 0.0), orientation="upright")
    b = m.observe("bottle", (1.08, 0.05, 0.0))
    c = m.observe("bottle", (2.0, 1.0, 0.0))
    assert a is b and a["n_seen"] == 2 and c["id"] == "bottle_2"
    assert a["xyz"][0] == pytest.approx(1.04) and a["orientation"] == "upright"
    assert len(m.of_type("bottle")) == 2


def test_bin_merges_from_farther_away_and_has_its_own_id():
    m = WorldMemory()
    m.observe("bin", (2.0, 0.0, 0.0))
    m.observe("bin", (2.3, 0.0, 0.0))
    assert len(m.of_type("bin")) == 1 and m.bin()["id"] == "bin"


def test_lidar_blob_on_a_known_bottle_or_bin_is_not_an_obstacle():
    m = WorldMemory()
    m.observe("bin", (2.0, 0.0, 0.0))
    m.add_obstacles([(2.05, 0.02, 0.15), (1.0, 1.0, 0.2)])
    assert [o[3] for o in m.obstacles(avoid_objects=False)] == ["obstacle_1"]
    assert [o[3] for o in m.obstacles()] == ["obstacle_1", "bin"]          # the bin is still driven around


def test_statuses_summary_and_report():
    m = WorldMemory(expected=3)
    for x in (1, 2, 3):
        m.observe("bottle", (x, 0, 0), orientation="upright")
    m.set_status("bottle_1", "in_bin", label={"status": "read", "fields": {"drug_name": "FUNKITOL", "strength": "200MG"}})
    m.set_status("bottle_2", "blocked", why="obstacle_1 in the way")
    s = m.summary()
    assert s["in_bin"] == 1 and s["blocked"] == 1 and s["not_reached"] == 1 and s["missed"] == 2
    text = "\n".join(report_lines(m))
    assert "FUNKITOL 200MG" in text and "MISSED 2" in text and "obstacle_1 in the way" in text


def test_save_and_load(tmp_path):
    m = WorldMemory(expected=1)
    m.observe("bottle", (1, 2, 0))
    m.observe("obstacle", (0, 1, 0))
    m.bin_camera_count = 1
    m2 = WorldMemory.load(m.save(str(tmp_path / "w.json")))
    assert m2.get("bottle_1")["xyz"] == (1, 2, 0) and m2.bin_camera_count == 1
    assert m2.observe("bottle", (5, 5, 0))["id"] == "bottle_2"          # ids keep counting


def test_frames_round_trip():
    pose = (1.0, -2.0, 0.7)
    assert to_robot(pose, to_world(pose, (0.3, -0.4))) == pytest.approx((0.3, -0.4))


# ---------------- lidar ----------------

def raycast(pose, circles, n=720, max_range=6.0):
    """Fake LaserScan from a robot at pose: ranges to the nearest circle (x, y, r) in odom."""
    ranges = []
    for i in range(n):
        a = pose[2] - math.pi + i * 2 * math.pi / n
        dx, dy = math.cos(a), math.sin(a)
        best = math.inf
        for cx, cy, r in circles:
            fx, fy = pose[0] - cx, pose[1] - cy
            b = fx * dx + fy * dy
            c = fx * fx + fy * fy - r * r
            disc = b * b - c
            if disc >= 0:
                t = -b - math.sqrt(disc)
                if 0 < t < best:
                    best = t
        ranges.append(best if best < max_range else math.inf)
    return ranges, -math.pi, 2 * math.pi / n


def test_scan_clusters_find_each_obstacle():
    circles = [(1.5, 0.0, 0.20), (0.0, 2.0, 0.10), (-1.2, -1.2, 0.15)]
    ranges, a0, inc = raycast((0, 0, 0), circles)
    found = scan_clusters(scan_points(ranges, a0, inc))
    assert len(found) == 3
    for cx, cy, r in circles:
        f = min(found, key=lambda c: math.hypot(c[0] - cx, c[1] - cy))
        assert math.hypot(f[0] - cx, f[1] - cy) < 0.08 and f[2] <= r + 0.05


def test_scan_ignores_too_near_and_too_far():
    pts = scan_points([0.1, 5.0, float("nan"), 1.0], 0.0, 0.01)
    assert pts[:3] == [None, None, None] and pts[3] is not None


# ---------------- paths ----------------

def test_straight_line_when_clear():
    r = plan_path((0, 0), (2, 0), [(1, 1.5, 0.2, "o")])
    assert r["ok"] and not r["detour"] and r["waypoints"] == [(2, 0)]


def _path_clear(start, wps, obstacles, clear=CLEAR):
    pts = [start] + list(wps)
    return all(seg_point_dist(a, b, o[:2]) >= o[2] + clear - 1e-9 for a, b in zip(pts, pts[1:]) for o in obstacles)


def test_detour_around_one_obstacle():
    obs = [(1.0, 0.0, 0.2, "o1")]
    r = plan_path((0, 0), (2, 0), obs)
    assert r["ok"] and r["detour"] and r["blocked_by"] == "o1" and len(r["waypoints"]) == 2
    assert _path_clear((0, 0), r["waypoints"], obs)


def test_detour_around_two_obstacles():
    obs = [(1.0, 0.0, 0.2, "o1"), (2.2, 0.6, 0.2, "o2")]
    r = plan_path((0, 0), (3.2, 0.3), obs)
    assert r["ok"] and _path_clear((0, 0), r["waypoints"], obs)


def test_no_way_around_a_wall():
    wall = [(1.0, y, 0.15, f"w{i}") for i, y in enumerate(np.arange(-3, 3.01, 0.25))]
    r = plan_path((0, 0), (2, 0), wall)
    assert not r["ok"] and r["blocked_by"].startswith("w")


def test_goal_inside_an_obstacle_is_refused():
    assert not plan_path((0, 0), (1, 0), [(1.05, 0, 0.2, "o")])["ok"]


def test_stand_spot_avoids_obstacles_and_none_when_boxed_in():
    target = (2.0, 0.0)
    spot = stand_spot(target, (0, 0), [(1.25, 0.0, 0.2, "o")])
    assert spot is not None and math.hypot(spot[0] - 1.25, spot[1]) >= 0.2 + CLEAR
    assert math.hypot(spot[0] - 2.0, spot[1]) == pytest.approx(S["stand_off_m"])
    ring = [(2 + 0.75 * math.cos(a), 0.75 * math.sin(a), 0.25, f"r{i}")
            for i, a in enumerate(np.linspace(0, 2 * math.pi, 12, endpoint=False))]
    assert stand_spot(target, (0, 0), ring) is None


def test_moves_split_and_reach_the_goal():
    moves = path_to_moves((0, 0, 0), [(0, 1.2)])
    assert moves[0][0] == "rotate" and moves[0][1] == pytest.approx(math.pi / 2)
    drives = [m[1] for m in moves if m[0] == "translate"]
    assert len(drives) == 3 and sum(drives) == pytest.approx(1.2) and max(drives) <= 0.5


def test_corridor_check():
    assert corridor_blocked([(0.6, 0.05), (0.3, 0.6)], 0.4) == pytest.approx(0.6)
    assert corridor_blocked([(1.5, 0.0), (0.4, 0.5)], 0.4) is None


# ---------------- T12: drop ----------------

def test_drop_plan_puts_the_bottle_over_the_rim():
    g0, lift = (0.0, -0.3, 0.55), 0.6
    rim = 0.30
    plan = plan_drop((0.02, -0.6, rim), rim, 0.0, lift, g0, P)
    assert plan["next_step"] == "reach"
    bottom = g0[2] + (plan["lift"] - lift) - P["grasp_frac_upright"] * P["bottle_length"]
    assert bottom == pytest.approx(rim + 0.08, abs=1e-6)
    assert plan["arm_grasp"] == pytest.approx(0.3)                    # center of the bin, not past it


def test_drop_plan_moves_the_base_first_when_the_bin_is_ahead():
    plan = plan_drop((1.5, 0.0, 0.3), 0.3, 0.0, 0.6, (0.0, -0.3, 0.55), P)
    assert plan["next_step"] in ("drive_base", "rotate_base")


def test_bin_center_from_its_near_wall():
    rng = np.random.default_rng(0)
    wall = np.column_stack([np.full(300, 1.0) + rng.normal(0, 0.005, 300), rng.uniform(-0.1, 0.1, 300),
                            rng.uniform(0.0, 0.3, 300)])
    (cx, cy, rim), rim2 = bin_from_points(wall, depth_offset=0.1)
    assert cx == pytest.approx(1.1, abs=0.02) and abs(cy) < 0.02 and rim == pytest.approx(0.29, abs=0.02)


# ---------------- simulated room (T18 + T13 integration) ----------------

class Room:
    """True positions (odom = world at the start). The robot's odometry drifts a little on every move."""

    def __init__(self, bottles, bin_xy, obstacles, drift=0.01, seed=0):
        self.bottles = {f"b{i}": np.array(b, float) for i, b in enumerate(bottles)}
        self.bin = np.array(bin_xy, float)
        self.obstacles = obstacles                    # (x, y, r)
        self.true = np.array([0.0, 0.0, 0.0])
        self.odom = np.array([0.0, 0.0, 0.0])
        self.rng = np.random.default_rng(seed)
        self.drift = drift
        self.collisions = 0
        self.path = [tuple(self.true[:2])]

    def move(self, kind, amount):
        if kind == "rotate":
            self.true[2] += amount * (1 + self.rng.normal(0, self.drift))
            self.odom[2] += amount
            return
        steps = max(1, int(abs(amount) / 0.02))
        for _ in range(steps):                         # check the footprint along the way
            d = amount / steps
            self.true[:2] += d * (1 + self.rng.normal(0, self.drift)) * np.array([math.cos(self.true[2]),
                                                                                  math.sin(self.true[2])])
            self.odom[:2] += d * np.array([math.cos(self.odom[2]), math.sin(self.odom[2])])
            for ox, oy, r in self.obstacles + [(self.bin[0], self.bin[1], 0.15)] + \
                    [(b[0], b[1], 0.02) for b in self.bottles.values()]:          # knocking a bottle counts too
                if math.hypot(self.true[0] - ox, self.true[1] - oy) < r + S["robot_radius_m"] - 0.02:
                    self.collisions += 1
            self.path.append(tuple(self.true[:2]))

    def believed(self, xy_true):
        """True world point -> where the robot's (drifted) odom thinks it is."""
        rel = to_robot(tuple(self.true), xy_true)
        return to_world(tuple(self.odom), rel)

    def camera(self, pan, fov=math.radians(42), max_range=3.0, noise=0.02):
        """What the head camera sees at this pan: (kind, xyz in odom as the robot believes it)."""
        out = []
        items = [("bottle", k, v) for k, v in self.bottles.items()] + [("bin", "bin", self.bin)]
        for kind, key, xy in items:
            rel = to_robot(tuple(self.true), xy)
            ang = math.atan2(rel[1], rel[0])
            if math.hypot(*rel) > max_range or abs((ang - pan + math.pi) % (2 * math.pi) - math.pi) > fov / 2:
                continue
            if any(seg_point_dist((0, 0), rel, to_robot(tuple(self.true), o[:2])) < o[2] for o in self.obstacles):
                continue                                # hidden behind an obstacle
            seen = self.believed(xy + self.rng.normal(0, noise, 2))
            out.append((kind, key, (seen[0], seen[1], 0.0)))
        return out

    def lidar(self):
        ranges, a0, inc = raycast(tuple(self.true), self.obstacles + [(self.bin[0], self.bin[1], 0.15)])
        # raycast's angles start at heading - pi, so with angle_min = -pi the points come out in the robot frame
        circles = scan_clusters(scan_points(ranges, a0, inc))
        out = []
        for cx, cy, r in circles:                        # robot frame -> odom, with the (drifted) odom pose,
            w = to_world(tuple(self.odom), (cx, cy))      # as the real code does with tf
            out.append((w[0], w[1], r))
        return out


class SimIO:
    """The `io` mission.Mission drives, backed by the simulated Room (same interface as world_robot.RosIO)."""

    def __init__(self, room):
        self.room = room
        self.dropped = 0

    def pose(self):
        return tuple(float(v) for v in self.room.odom)

    def look(self, pan):
        return [(kind, xyz, {}) for kind, _key, xyz in self.room.camera(pan)]

    def scan_circles(self):
        return self.room.lidar()

    def scan_robot(self):
        r = self.room
        ranges, a0, inc = raycast(tuple(r.true), r.obstacles + [(r.bin[0], r.bin[1], 0.15)])
        return scan_points(ranges, a0, inc)

    def rotate(self, rad):
        self.room.move("rotate", rad)

    def translate(self, m):
        self.room.move("translate", m)

    def pick(self, item):
        """Look toward the saved spot; grab the bottle seen closest to it (the grasp planner's job on the robot)."""
        rel = to_robot(self.pose(), item["xyz"][:2])
        seen = [o for o in self.room.camera(math.atan2(rel[1], rel[0])) if o[0] == "bottle"]
        near = [o for o in seen if math.hypot(o[2][0] - item["xyz"][0], o[2][1] - item["xyz"][1]) < 0.3]
        if not near:
            return {"status": "missing", "why": "not at the saved spot"}
        best = min(near, key=lambda o: math.hypot(o[2][0] - item["xyz"][0], o[2][1] - item["xyz"][1]))
        del self.room.bottles[best[1]]
        return {"status": "picked", "xyz": best[2], "orientation": "upright"}

    def drop(self, bin_item):
        self.dropped += 1
        return True

    def count_in_bin(self, bin_item):
        return self.dropped


def look_around(room, mem):
    Mission(SimIO(room), mem, say=lambda t: None).look_around()


def run_mission(room, expected):
    mem = WorldMemory(expected=expected)
    m = Mission(SimIO(room), mem, say=lambda t: None)
    m.run()
    return mem, m


ROOM = dict(bottles=[(2.0, 0.5), (1.0, -1.8), (-1.5, 1.0)], bin_xy=(-1.0, -1.5),
            obstacles=[(1.2, -0.6, 0.2), (0.0, -1.2, 0.15)])
# bottle 1 is hidden behind an obstacle from the start, so it's only found after the robot has moved
HIDDEN = dict(bottles=[(2.0, 0.5), (1.0, -1.8), (-1.5, 1.0)], bin_xy=(-1.0, -1.5),
              obstacles=[(1.2, 0.3, 0.2), (0.0, -1.2, 0.15)])


def test_look_around_finds_three_bottles_and_the_bin_without_duplicates():
    room = Room(**ROOM)
    mem = WorldMemory(expected=3)
    look_around(room, mem)
    look_around(room, mem)                             # looking twice must not add duplicates
    bottles = mem.of_type("bottle")
    assert len(bottles) == 3 and mem.bin() is not None
    for b in room.bottles.values():
        assert min(math.hypot(e["xyz"][0] - b[0], e["xyz"][1] - b[1]) for e in bottles) < 0.10
    assert len(mem.obstacles(avoid_objects=False)) == 2   # lidar blobs on the bin are not extra obstacles


def test_saved_spots_hold_after_driving_and_turning_around():
    room = Room(**ROOM, drift=0.01)
    mem = WorldMemory(expected=3)
    look_around(room, mem)
    before = {b["id"]: b["xyz"] for b in mem.of_type("bottle")}
    for kind, amount in path_to_moves(tuple(room.odom), [(0.0, 2.0)]):
        room.move(kind, amount)
    room.move("rotate", math.pi)
    look_around(room, mem)
    assert len(mem.of_type("bottle")) == 3            # re-seen bottles merged, nothing new added
    for bid, xyz in before.items():
        assert math.hypot(*(np.array(mem.get(bid)["xyz"][:2]) - xyz[:2])) < 0.10


@pytest.mark.parametrize("layout", [ROOM, HIDDEN])
def test_fetch_every_bottle_around_obstacles_without_hitting_anything(layout):
    room = Room(**layout, drift=0.005, seed=3)
    mem, mission = run_mission(room, 3)
    s = mem.summary()
    assert s["in_bin"] == 3 and s["missed"] == 0 and room.collisions == 0, (s, room.collisions)


def test_boxed_in_bottle_is_marked_blocked_and_the_rest_still_get_done():
    ring = [(1.5 + 0.55 * math.cos(a), 1.5 + 0.55 * math.sin(a), 0.22)
            for a in np.linspace(0, 2 * math.pi, 10, endpoint=False)]
    room = Room(bottles=[(1.5, 1.5), (-1.0, 1.2)], bin_xy=(-1.2, -1.2), obstacles=ring, drift=0.0)
    mem = WorldMemory(expected=2)
    mem.observe("bottle", (1.5, 1.5, 0.0))              # the camera can't see over the ring; tell it, like a user would
    Mission(SimIO(room), mem, say=lambda t: None).run()
    s = mem.summary()
    assert s["blocked"] == 1 and s["in_bin"] == 1 and room.collisions == 0


def random_room(seed):
    rng = np.random.default_rng(seed)

    def pt():
        r, a = rng.uniform(0.8, 2.5), rng.uniform(-math.pi, math.pi)
        return r * math.cos(a), r * math.sin(a)
    obs = []
    while len(obs) < 3:
        p = pt()
        obs.append((p[0], p[1], rng.uniform(0.1, 0.25)))
    items = []
    while len(items) < 4:
        p = pt()
        if all(math.hypot(p[0] - o[0], p[1] - o[1]) > o[2] + 0.45 for o in obs) and \
                all(math.hypot(p[0] - q[0], p[1] - q[1]) > 0.5 for q in items):
            items.append(p)
    return Room(bottles=items[:3], bin_xy=items[3], obstacles=obs, drift=0.005, seed=seed)


def test_100_random_rooms_no_collisions():
    """3 bottles, a bin and 3 obstacles placed at random, 100 times. Never hits anything (obstacles, the bin or a
    bottle on the floor), and gets at least 95 % of the bottles into the bin; the rest are reported."""
    binned = collisions = 0
    for seed in range(100):
        room = random_room(seed)
        mem, _ = run_mission(room, 3)
        binned += mem.summary()["in_bin"]
        collisions += room.collisions
        s = mem.summary()
        assert s["in_bin"] + s["blocked"] + s["dropped"] + s["not_reached"] + s["missing"] + \
            (3 - s["bottles_found"]) >= 3                 # every bottle is accounted for in the report
    assert collisions == 0 and binned >= 0.95 * 300, (binned, collisions)


def test_map_picture(tmp_path):
    import cv2
    from pill_bottle_nodes.map_drawing import draw_map
    room = Room(**ROOM, drift=0.0)
    mem, _ = run_mission(room, 3)
    img = draw_map(mem, tuple(room.odom), room.path, title="T18 sim")
    assert img.shape == (600, 600, 3) and (img != 255).any()
    assert cv2.imwrite(str(tmp_path / "map.png"), img)
