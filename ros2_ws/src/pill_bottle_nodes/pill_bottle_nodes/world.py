# world.py: what the robot remembers about the room, and how it gets around (tickets T18, T13, T12).
# No ROS in this file, so all of it is unit tested (test/test_world.py, including a simulated room).
#
#   WorldMemory     list of bottles, the bin and obstacles, in the odom frame (meters from where the robot
#                   started). Seeing the same thing again updates it instead of adding a duplicate.
#   scan_clusters   lidar scan -> obstacles (circles)
#   plan_path       straight line to the goal, or around obstacles with up to 2 detour points; None = no way
#   stand_spot      where to stop near a bottle/bin so the grasp planner can take over
#   path_to_moves   waypoints -> base turns and drives (drives split into <= 0.5 m, motion_node's limit)
#   corridor_blocked  live safety check: anything in the lidar right in front of the next drive?
#   plan_drop       T12: arm over the bin, bottle bottom above the rim
import json
import math
import time

from .geometry import plan_grasp, wrap

WORLD_SETTINGS = {
    # same thing seen again closer than this = update, not a new entry (m)
    "same_bottle_m": 0.15, "same_bin_m": 0.35, "same_obstacle_m": 0.30,
    # lidar clustering
    "scan_gap_m": 0.10,            # neighbouring scan points farther apart than this start a new cluster
    "scan_min_points": 3,
    "scan_min_range_m": 0.20, "scan_max_range_m": 4.0,
    "obstacle_min_radius_m": 0.05,
    "known_object_clearance_m": 0.12,   # a lidar blob this close to a known bottle/bin IS that bottle/bin
    # driving
    "robot_radius_m": 0.25,        # Stretch base is about 34 x 33 cm; with the stowed arm, 25 cm is safe
    "clearance_m": 0.08,           # extra room around obstacles
    "max_waypoints": 3,            # detour points per path (2 obstacles in a row need 3)
    "max_step_m": 0.5,             # motion_node's translate limit
    "stand_off_m": 0.75,           # stop this far from a bottle/bin; the grasp planner does the last bit
    "stop_distance_m": 0.30,       # live check: something closer than this ahead (plus the drive) = stop
    "bin_radius_m": 0.15,          # measure your bin (half its width)
    "bottle_avoid_radius_m": 0.05, # bottles on the floor are obstacles too (except the one it's going to)
}


def _d(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ---------------- frames ----------------

def pose_from_matrix(T):
    """(x, y, yaw) of a frame from its 4x4 transform in odom."""
    return float(T[0][3]), float(T[1][3]), math.atan2(T[1][0], T[0][0])


def to_world(pose, xy):
    """Point in the robot's frame (x forward, y left) -> odom, robot at pose (x, y, yaw)."""
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return x + c * xy[0] - s * xy[1], y + s * xy[0] + c * xy[1]


def to_robot(pose, xy):
    """odom point -> the robot's frame."""
    x, y, yaw = pose
    dx, dy = xy[0] - x, xy[1] - y
    c, s = math.cos(yaw), math.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy


# ---------------- memory ----------------

class WorldMemory:
    """Bottles, the bin and obstacles in odom. Statuses for bottles:
    seen -> picked -> in_bin, or blocked / out_of_range / missing / dropped / unreadable_label."""

    def __init__(self, s=None, expected=None):
        self.s = dict(WORLD_SETTINGS, **(s or {}))
        self.expected = expected
        self.items = []
        self.bin_camera_count = None              # T15 step 5: bottles counted in the bin by the head camera
        self._n = {"bottle": 0, "bin": 0, "obstacle": 0}

    # -- adding --
    def _new_id(self, kind):
        self._n[kind] += 1
        return kind if kind == "bin" and self._n[kind] == 1 else f"{kind}_{self._n[kind]}"

    def observe(self, kind, xyz, t=None, **extra):
        """Something seen at xyz (odom). Merges with the nearest same-kind entry within the radius."""
        radius = {"bottle": self.s["same_bottle_m"], "bin": self.s["same_bin_m"]}.get(kind, self.s["same_obstacle_m"])
        t = time.time() if t is None else t
        near = [it for it in self.items if it["type"] == kind and _d(it["xyz"], xyz) <= radius
                and it.get("status") not in ("in_bin",)]
        if near:
            it = min(near, key=lambda i: _d(i["xyz"], xyz))
            n = it["n_seen"]
            w = 1.0 / min(n + 1, 5)                  # running average, recent views count more after 5
            it["xyz"] = tuple(round((1 - w) * a + w * b, 4) for a, b in zip(it["xyz"], xyz))
            it["n_seen"] = n + 1
            it["last_seen"] = t
            it.update({k: v for k, v in extra.items() if v is not None})
            return it
        it = {"id": self._new_id(kind), "type": kind, "xyz": tuple(round(v, 4) for v in xyz), "n_seen": 1,
              "first_seen": t, "last_seen": t}
        if kind == "bottle":
            it["status"] = "seen"
        it.update({k: v for k, v in extra.items() if v is not None})
        self.items.append(it)
        return it

    def add_obstacles(self, circles, t=None):
        """circles: [(x, y, r)] in odom from scan_clusters. Blobs that are really a known bottle or the bin are
        skipped. Returns the obstacles added or updated."""
        out = []
        for cx, cy, r in circles:
            if any(it["type"] in ("bottle", "bin") and _d(it["xyz"], (cx, cy)) <= r + self.s["known_object_clearance_m"]
                   + (0.15 if it["type"] == "bin" else 0.0) for it in self.items):
                continue
            ob = self.observe("obstacle", (cx, cy, 0.0), t)
            ob["radius"] = round(max(ob.get("radius", 0.0) * 0.5 + r * 0.5 if ob["n_seen"] > 1 else r,
                                     self.s["obstacle_min_radius_m"]), 3)
            out.append(ob)
        return out

    def forget_obstacles_near(self, xy, radius):
        """Drop obstacles that turned out to be a bottle/bin (seen by the camera after the lidar)."""
        self.items = [it for it in self.items if not (it["type"] == "obstacle" and _d(it["xyz"], xy) <= radius)]

    # -- reading --
    def get(self, item_id):
        return next((it for it in self.items if it["id"] == item_id), None)

    def of_type(self, kind, statuses=None):
        return [it for it in self.items if it["type"] == kind and (statuses is None or it.get("status") in statuses)]

    def bin(self):
        bins = self.of_type("bin")
        return max(bins, key=lambda b: b["n_seen"]) if bins else None

    def obstacles(self, avoid_objects=True, exclude=()):
        """Circles (x, y, r, id) to drive around. With avoid_objects, the bin and bottles still on the floor count
        too (the lidar is above a bottle, and the robot mustn't knock one over). Keep the target in: the stand
        spot is outside its margin, and leaving it out lets a detour clip the very bottle it's going to."""
        out = [(it["xyz"][0], it["xyz"][1], it.get("radius", self.s["obstacle_min_radius_m"]), it["id"])
               for it in self.of_type("obstacle")]
        if avoid_objects:
            for it in self.items:
                if it["id"] in exclude:
                    continue
                if it["type"] == "bin":
                    out.append((it["xyz"][0], it["xyz"][1], self.s["bin_radius_m"], it["id"]))
                elif it["type"] == "bottle" and it.get("status") in ("seen", "blocked", "out_of_range", "missing"):
                    out.append((it["xyz"][0], it["xyz"][1], self.s["bottle_avoid_radius_m"], it["id"]))
        return out

    def nearest(self, kind, from_xy, statuses=("seen",)):
        cands = self.of_type(kind, statuses if kind == "bottle" else None)
        return min(cands, key=lambda it: _d(it["xyz"], from_xy)) if cands else None

    def set_status(self, item_id, status, **extra):
        it = self.get(item_id)
        if it is not None:
            it["status"] = status
            it.update(extra)
        return it

    def correct(self, item_id, xyz):
        """Re-seen up close (more accurate than from far away): replace the saved spot. Returns the shift (m)."""
        it = self.get(item_id)
        shift = _d(it["xyz"], xyz)
        it["xyz"] = tuple(round(v, 4) for v in xyz)
        it["corrections"] = it.get("corrections", 0) + 1
        return shift

    # -- report --
    def summary(self):
        bottles = self.of_type("bottle")
        count = lambda st: sum(1 for b in bottles if b.get("status") == st)
        out = {"bottles_expected": self.expected, "bottles_found": len(bottles), "in_bin": count("in_bin"),
               "blocked": count("blocked"), "out_of_range": count("out_of_range"), "missing": count("missing"),
               "dropped": count("dropped"), "not_reached": count("seen"), "bin_found": self.bin() is not None,
               "obstacles": len(self.of_type("obstacle"))}
        if self.expected is not None:
            out["missed"] = max(0, self.expected - out["in_bin"])
        return out

    def to_dict(self):
        return {"expected": self.expected, "summary": self.summary(), "bin_camera_count": self.bin_camera_count,
                "items": self.items}

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1, default=str)
        return path

    @classmethod
    def load(cls, path, s=None):
        with open(path) as f:
            d = json.load(f)
        m = cls(s, d.get("expected"))
        m.bin_camera_count = d.get("bin_camera_count")
        for it in d["items"]:
            it["xyz"] = tuple(it["xyz"])
            m.items.append(it)
            kind = it["type"]
            m._n[kind] = max(m._n.get(kind, 0), int(it["id"].split("_")[-1]) if "_" in it["id"] else 1)
        return m


def report_lines(memory):
    """The end-of-run report (T15) as text lines."""
    s = memory.summary()
    lines = ["===== RUN REPORT =====",
             f"Bottles: expected {s['bottles_expected']}, found {s['bottles_found']}, in the bin {s['in_bin']}"
             + (f", MISSED {s['missed']}" if s.get("missed") else "")]
    for b in memory.of_type("bottle"):
        label = b.get("label") or {}
        if label.get("status") == "read":
            f = label.get("fields", {})
            text = f"{f.get('drug_name')} {f.get('strength')}, patient {f.get('patient_name')}, Rx {f.get('rx_number')}"
        elif label:
            text = f"label {label.get('status', 'not read').upper()}"
        else:
            text = "label not read"
        why = f" ({b['why']})" if b.get("why") else ""
        lines.append(f"  {b['id']}: {b.get('status')}{why}, found {b.get('orientation', '?')} at "
                     f"({b['xyz'][0]:+.2f}, {b['xyz'][1]:+.2f}) m, {text}")
    cam = memory.bin_camera_count
    check = "not checked" if cam is None else (f"{cam} (matches)" if cam == s["in_bin"] else
                                               f"{cam} (robot counted {s['in_bin']} drops)")
    lines.append(f"Bin: {'found' if s['bin_found'] else 'NOT found'}; bottles the camera sees in the bin: {check}")
    obs = memory.of_type("obstacle")
    lines.append(f"Obstacles: {len(obs)}" + "".join(f"\n  {o['id']} at ({o['xyz'][0]:+.2f}, {o['xyz'][1]:+.2f}) m, "
                                                    f"{o.get('radius', 0) * 100:.0f} cm"
                                                    + (" - blocked a bottle" if o.get("blocked") else "") for o in obs))
    return lines


# ---------------- lidar ----------------

def scan_points(ranges, angle_min, angle_increment, s=None):
    """LaserScan ranges -> [(x, y)] in the laser frame (in scan order), dropping invalid/too near/too far."""
    s = dict(WORLD_SETTINGS, **(s or {}))
    out = []
    for i, r in enumerate(ranges):
        if r is None or not math.isfinite(r) or r < s["scan_min_range_m"] or r > s["scan_max_range_m"]:
            out.append(None)                   # keeps gaps so clusters split at missing returns
            continue
        a = angle_min + i * angle_increment
        out.append((r * math.cos(a), r * math.sin(a)))
    return out


def scan_clusters(points, s=None):
    """Points in scan order (None = no return) -> circles [(cx, cy, r)]. Neighbours closer than scan_gap_m join."""
    s = dict(WORLD_SETTINGS, **(s or {}))
    clusters, cur = [], []
    for p in points + [None]:
        if p is not None and (not cur or _d(cur[-1], p) <= s["scan_gap_m"]):
            cur.append(p)
            continue
        if len(cur) >= s["scan_min_points"]:
            clusters.append(cur)
        cur = [p] if p is not None else []
    # the scan wraps around: join the last and first clusters if they touch
    if len(clusters) > 1 and _d(clusters[-1][-1], clusters[0][0]) <= s["scan_gap_m"] and points and \
            points[0] is not None and points[-1] is not None:
        clusters[0] = clusters.pop() + clusters[0]
    out = []
    for c in clusters:
        cx = sum(p[0] for p in c) / len(c)
        cy = sum(p[1] for p in c) / len(c)
        r = max(_d(p, (cx, cy)) for p in c)
        # the lidar only sees the near side: push the center back by the radius, away from the sensor
        rng = math.hypot(cx, cy)
        if rng > 1e-6:
            cx, cy = cx + cx / rng * r * 0.5, cy + cy / rng * r * 0.5
        out.append((cx, cy, max(r, s["obstacle_min_radius_m"])))
    return out


# ---------------- paths ----------------

def seg_point_dist(a, b, p):
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def first_blocker(a, b, obstacles, clear):
    """First obstacle (x, y, r, id) whose inflated circle the segment a->b passes through, or None.
    If the robot already starts inside an obstacle's safety margin (it stopped close to it), that obstacle only
    blocks a path that gets any closer to it; moving away is allowed."""
    hits = []
    for o in obstacles:
        d = seg_point_dist(a, b, o[:2])
        start_d = _d(a, o[:2])
        if start_d < o[2] + clear:
            if d < start_d - 0.01:
                hits.append(o)
        elif d < o[2] + clear:
            hits.append(o)
    return min(hits, key=lambda o: _d(a, o[:2])) if hits else None


def inside_any(p, obstacles, clear):
    return any(_d(p, o[:2]) < o[2] + clear for o in obstacles)


def plan_path(start, goal, obstacles, s=None, depth=None):
    """Waypoints from start to goal (both xy in the same frame) around obstacles [(x, y, r, id)].

    Returns dict(ok, waypoints=[... goal], detour=bool, blocked_by=id or None, length_m)."""
    s = dict(WORLD_SETTINGS, **(s or {}))
    clear = s["robot_radius_m"] + s["clearance_m"]
    depth = s["max_waypoints"] if depth is None else depth
    if inside_any(goal, obstacles, clear):
        o = min(obstacles, key=lambda o: _d(goal, o[:2]))
        return dict(ok=False, waypoints=[], detour=False, blocked_by=o[3], length_m=0.0)
    blk = first_blocker(start, goal, obstacles, clear)
    if blk is None:
        return dict(ok=True, waypoints=[tuple(goal)], detour=False, blocked_by=None, length_m=_d(start, goal))
    if depth <= 0:
        return dict(ok=False, waypoints=[], detour=False, blocked_by=blk[3], length_m=0.0)
    dx, dy = goal[0] - start[0], goal[1] - start[1]
    L = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / L, dx / L                        # left of the straight line
    best = None
    for side, scale in ((1, 1.0), (-1, 1.0), (1, 1.7), (-1, 1.7)):   # beside the obstacle, then wider
        if best is not None and scale > 1.0:
            break
        off = (blk[2] + clear + 0.10) * scale
        wp = (blk[0] + side * nx * off, blk[1] + side * ny * off)
        if inside_any(wp, obstacles, clear):
            continue
        a = plan_path(start, wp, obstacles, s, depth - 1)
        if not a["ok"]:
            continue
        b = plan_path(wp, goal, obstacles, s, depth - 1 - (len(a["waypoints"]) - 1))
        if not b["ok"]:
            continue
        cand = dict(ok=True, waypoints=a["waypoints"] + b["waypoints"], detour=True, blocked_by=blk[3],
                    length_m=a["length_m"] + b["length_m"])
        if len(cand["waypoints"]) - 1 <= s["max_waypoints"] and (best is None or cand["length_m"] < best["length_m"]):
            best = cand
    return best or dict(ok=False, waypoints=[], detour=False, blocked_by=blk[3], length_m=0.0)


def stand_spot(target, robot_xy, obstacles, s=None):
    """Where to stop near a target: stand_off_m from it, on the robot's side, or the nearest clear spot around it.
    Returns (x, y) or None if every spot around it is blocked."""
    s = dict(WORLD_SETTINGS, **(s or {}))
    clear = s["robot_radius_m"] + s["clearance_m"]
    base = math.atan2(robot_xy[1] - target[1], robot_xy[0] - target[0])
    for k in range(13):                               # 0, +30, -30, +60, ... 180 deg around the target
        a = base + math.radians(30 * ((k + 1) // 2) * (1 if k % 2 else -1))
        spot = (target[0] + s["stand_off_m"] * math.cos(a), target[1] + s["stand_off_m"] * math.sin(a))
        if not inside_any(spot, obstacles, clear):
            return spot
    return None


def lookout_spots(start_xy, obstacles, s=None, radius=1.2, n=6):
    """Places to look from when fewer bottles were found than you said are out: the start, then n spots on a
    circle around it, skipping any inside an obstacle's margin."""
    s = dict(WORLD_SETTINGS, **(s or {}))
    clear = s["robot_radius_m"] + s["clearance_m"]
    spots = [tuple(start_xy)] + [(start_xy[0] + radius * math.cos(a), start_xy[1] + radius * math.sin(a))
                                 for a in (2 * math.pi * k / n for k in range(n))]
    return [p for p in spots if not inside_any(p, obstacles, clear)]


def path_to_moves(pose, waypoints, s=None):
    """Robot at pose (x, y, yaw) -> [("rotate", rad), ("translate", m), ...]; drives split to <= max_step_m."""
    s = dict(WORLD_SETTINGS, **(s or {}))
    x, y, yaw = pose
    moves = []
    for wx, wy in waypoints:
        heading = math.atan2(wy - y, wx - x)
        turn = wrap(heading - yaw)
        dist = math.hypot(wx - x, wy - y)
        if dist < 0.02:
            continue
        if abs(turn) > math.radians(2):
            moves.append(("rotate", turn))
        n = max(1, math.ceil(dist / s["max_step_m"] - 1e-9))
        moves += [("translate", dist / n)] * n
        x, y, yaw = wx, wy, heading
    return moves


def face_moves(pose, target):
    """Turn in place to face a target (xy)."""
    turn = wrap(math.atan2(target[1] - pose[1], target[0] - pose[0]) - pose[2])
    return [("rotate", turn)] if abs(turn) > math.radians(2) else []


def corridor_blocked(points_robot, distance, s=None):
    """Live check before a drive: any lidar point (robot frame, x forward) in the strip the base will sweep?
    Returns the nearest such point's distance ahead, or None."""
    s = dict(WORLD_SETTINGS, **(s or {}))
    half = s["robot_radius_m"]
    reach = distance + s["stop_distance_m"]
    ahead = [p[0] for p in points_robot if p is not None and 0.0 < p[0] < reach and abs(p[1]) < half]
    return min(ahead) if ahead else None


# ---------------- T12: dropping into the bin ----------------

def plan_drop(bin_xyz, rim_z, ext, lift, g0, p, drop_clearance=0.08):
    """Arm over the bin with the bottle's bottom drop_clearance above the rim.

    Uses the grasp planner's base moves (turn/drive until the bin is on the arm's side within reach), then:
    lift so the bottle (held grasp_frac_upright up from its bottom) clears the rim, arm to the bin's center.
    Returns a PlanGrasp-style dict; lift_clear = travel height over the rim."""
    surface = rim_z + drop_clearance                  # the bottle's bottom ends up here
    plan = plan_grasp(bin_xyz, "upright", surface, 0.0, ext, lift, g0, p)
    if plan["success"] and plan["next_step"] == "reach":
        plan["arm_grasp"] = plan["arm_grasp"] - plan["grasp_depth"]      # bottle over the center, not past it
        plan["arm_pregrasp"] = 0.0
        plan["lift_clear"] = min(max(plan["lift_clear"], plan["lift"] + 0.05), p["lift_max"])
        plan["message"] = (f"Drop: lift {plan['lift']:.3f} m (bottle bottom {drop_clearance * 100:.0f} cm over the "
                           f"rim), arm {plan['arm_grasp']:.3f} m, base {plan['base_translate']:+.3f} m.")
    return plan


def bin_from_points(pts_base, view_from=(0.0, 0.0), depth_offset=0.08):
    """Bin center and rim height from its depth points (base_link, numpy Nx3). The camera mostly sees the near
    wall, so the center is pushed depth_offset farther along the view direction."""
    import numpy as np
    rim = float(np.percentile(pts_base[:, 2], 95))
    cx, cy = float(np.median(pts_base[:, 0])), float(np.median(pts_base[:, 1]))
    dx, dy = cx - view_from[0], cy - view_from[1]
    L = math.hypot(dx, dy) or 1.0
    return (cx + dx / L * depth_offset, cy + dy / L * depth_offset, rim), rim
