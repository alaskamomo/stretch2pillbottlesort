# mission.py: the full-run logic (T15), and the look-around / drive-around-obstacles pieces T18, T13 and T12 use.
# No ROS here: it talks to the robot through an `io` object, so the SAME code runs on the robot (world_robot.RosIO)
# and in the simulated room in test/test_world.py.
#
# io must provide:
#   pose()                 -> (x, y, yaw) in odom
#   look(pan)              -> [(kind, (x, y, z) odom, extra dict)] seen with the head at that pan (kind: bottle | bin)
#   scan_circles()         -> [(x, y, r)] lidar obstacles in odom
#   scan_robot()           -> [(x, y)] lidar points in the robot frame (x forward), for the live check before a drive
#   rotate(rad), translate(m)
#   pick(item)             -> dict(status = picked | missing | out_of_range | missed, plus orientation, label, xyz)
#   drop(bin_item)         -> True if the bottle was let go over the bin
#   count_in_bin(bin_item) -> bottles the head camera sees in the bin, or None
import math
import time

from .world import (WORLD_SETTINGS, corridor_blocked, lookout_spots, path_to_moves, plan_path, report_lines,
                    stand_spot)

MISSION_SETTINGS = {
    # head pans for a look-around: Stretch's head turns from -231 to +99 deg, so -225..95 in 30 deg steps covers
    # almost the whole circle (the D435 sees about 42 deg across in the upright image)
    "look_pans_deg": [-225.0, -195.0, -165.0, -135.0, -105.0, -75.0, -45.0, -15.0, 15.0, 45.0, 75.0, 95.0],
    "max_replans": 6,
    "retry_blocked": 1,            # try a blocked bottle again this many times (from wherever the robot is then)
    "arrive_tol_m": 0.06,
}


class Mission:
    def __init__(self, io, memory, s=None, say=print):
        self.io, self.mem, self.say = io, memory, say
        self.s = dict(WORLD_SETTINGS, **MISSION_SETTINGS, **(s or {}))
        self.visited = set()
        self.start = tuple(io.pose()[:2])
        self.log = []                          # what happened, for the report / test logs
        self.stats = {"replans": 0, "detours": 0, "stopped_by_lidar": 0, "drives": 0}

    def note(self, text):
        self.log.append((round(time.time(), 1), text))
        self.say(text)

    # ---------- seeing ----------
    def look_around(self, pans=None):
        """Head sweep + lidar. Returns the number of new bottles."""
        before = len(self.mem.of_type("bottle"))
        for pan in (pans if pans is not None else [math.radians(p) for p in self.s["look_pans_deg"]]):
            for kind, xyz, extra in self.io.look(pan):
                self.mem.observe(kind, xyz, **(extra or {}))
        self.mem.add_obstacles(self.io.scan_circles())
        new = len(self.mem.of_type("bottle")) - before
        b = self.mem.bin()
        self.note(f"Looked around: {len(self.mem.of_type('bottle'))} bottle(s) ({new} new), "
                  f"bin {'at ({:+.2f}, {:+.2f})'.format(*b['xyz'][:2]) if b else 'not seen'}, "
                  f"{len(self.mem.of_type('obstacle'))} obstacle(s)")
        return new

    # ---------- driving ----------
    def drive_to(self, xy, stand=True, what="target"):
        """Drive near xy (stand_off_m away when stand) around obstacles. Returns (ok, why)."""
        for _ in range(self.s["max_replans"]):
            pose = self.io.pose()
            obs = self.mem.obstacles()
            spot = stand_spot(xy, pose[:2], obs, self.s) if stand else tuple(xy)
            if spot is None:
                return False, f"every spot around the {what} is blocked"
            if math.hypot(pose[0] - spot[0], pose[1] - spot[1]) < self.s["arrive_tol_m"]:
                return True, ""
            plan = plan_path(pose[:2], spot, obs, self.s)
            if not plan["ok"]:
                return False, f"no way around {plan['blocked_by']}"
            if plan["detour"]:
                self.stats["detours"] += 1
                self.note(f"  detour around {plan['blocked_by']} ({len(plan['waypoints']) - 1} point(s))")
            if self._execute(path_to_moves(pose, plan["waypoints"], self.s)):
                self.mem.add_obstacles(self.io.scan_circles())
                return True, ""
            self.stats["replans"] += 1
        return False, "too many replans"

    def _execute(self, moves):
        for kind, amount in moves:
            if kind == "translate":
                hit = corridor_blocked(self.io.scan_robot(), amount, self.s)
                if hit is not None:
                    self.stats["stopped_by_lidar"] += 1
                    self.note(f"  something {hit:.2f} m ahead; stopping to replan")
                    self.mem.add_obstacles(self.io.scan_circles())
                    return False
                self.stats["drives"] += 1
                self.io.translate(amount)
            else:
                self.io.rotate(amount)
        return True

    def go_look(self):
        """Drive to the next lookout spot and look around. False = no lookout spots left."""
        for spot in lookout_spots(self.start, self.mem.obstacles(), self.s):
            key = (round(spot[0], 2), round(spot[1], 2))
            if key in self.visited:
                continue
            self.visited.add(key)
            ok, _ = self.drive_to(spot, stand=False, what="lookout spot")
            if not ok:
                continue
            self.look_around()
            return True
        return False

    def find_bin(self):
        while self.mem.bin() is None:
            if not self.go_look():
                return False
        return True

    # ---------- the run ----------
    def next_bottle(self):
        b = self.mem.nearest("bottle", tuple(self.io.pose()[:2]))
        if b is None:
            b = next((x for x in self.mem.of_type("bottle", ("blocked",))
                      if x.get("tries", 0) <= self.s["retry_blocked"]), None)
        return b

    def run(self):
        """Look around, find the bin, then fetch bottles until all expected are in (or none left to try)."""
        self.look_around()
        if not self.find_bin():
            self.note("Couldn't find the bin anywhere; stopping before picking anything up.")
            return self.mem
        while True:
            b = self.next_bottle()
            if b is None:
                if self.mem.expected and self.mem.summary()["in_bin"] < self.mem.expected and self.go_look():
                    continue
                break
            self.fetch(b)
        binn = self.mem.bin()
        if binn is not None:
            ok, _ = self.drive_to(binn["xyz"][:2], what="bin")
            self.mem.bin_camera_count = self.io.count_in_bin(binn) if ok else None
        for line in report_lines(self.mem):
            self.say(line)
        return self.mem

    def fetch(self, b):
        self.note(f"Going for {b['id']} at ({b['xyz'][0]:+.2f}, {b['xyz'][1]:+.2f})")
        ok, why = self.drive_to(b["xyz"][:2], what=b["id"])
        if not ok:
            self.mem.set_status(b["id"], "blocked", why=why, tries=b.get("tries", 0) + 1)
            self.note(f"  {b['id']} blocked: {why}")
            return
        got = self.io.pick(b) or {}
        status = got.get("status", "missed")
        extra = {k: v for k, v in got.items() if k in ("orientation", "label", "why")}
        if got.get("xyz") is not None:
            self.mem.correct(b["id"], got["xyz"])
        if status != "picked":
            self.mem.set_status(b["id"], status, tries=b.get("tries", 0) + 1, **extra)
            self.note(f"  {b['id']}: {status} {got.get('why', '')}")
            return
        self.mem.set_status(b["id"], "picked", **extra)
        binn = self.mem.bin()
        ok, why = self.drive_to(binn["xyz"][:2], what="bin")
        if not ok or not self.io.drop(binn):
            self.mem.set_status(b["id"], "dropped", why=why or "drop failed")
            self.note(f"  {b['id']}: couldn't get it to the bin ({why or 'drop failed'})")
            return
        self.mem.set_status(b["id"], "in_bin")
        self.note(f"  {b['id']} is in the bin ({self.mem.summary()['in_bin']} so far)")
        self.look_around()                    # new spot: may see bottles that were hidden before
