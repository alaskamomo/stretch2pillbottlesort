# world_robot.py: the robot side of mission.Mission (T18, T13, T12, T15). RosIO is the `io` the mission drives,
# built on task_manager (moves, grasp, verify, label reading), the head camera + depth, the lidar and tf.
#   look(pan)        head to pan -> YOLO boxes -> depth points -> bottles (with orientation) and the bin, in odom
#   scan_circles()   lidar -> obstacles in odom;  scan_robot(): lidar points in base_link for the live check
#   rotate / translate   base moves through motion_node (arm in and lifted clear of the lidar first)
#   pick(item)       look at the saved spot, plan (turn/drive the base), grasp (retries), verify, read the label
#   drop(bin)        T12: turn/drive until the bin is beside the arm, lift over the rim, reach, open, retract
#   count_in_bin     bottles the head camera sees inside the bin's box
import math
import time

import cv2
import numpy as np

from . import grasp
from .common import depth_to_meters
from .geometry import (PLANNER_SETTINGS, box_points, classify_bottle, grasp_center_at_yaw0, look_at,
                       transform_points, wrap)
from .world import bin_from_points, plan_drop, pose_from_matrix, scan_clusters, scan_points, to_robot

ROBOT_WORLD_SETTINGS = {
    "look_tilt": -0.55,            # head tilt for a look-around (rad); about the floor 1.5 m out
    "look_settle_s": 1.2,          # wait after a head move (YOLO runs at about 2 Hz)
    "det_conf": 0.40,              # YOLO confidence to accept a bottle/bin while looking around
    "bottle_class": "pill_bottle",
    "bin_class": "bin",            # the detector needs this class (T11)
    "min_points": 30,              # depth points needed on a box
    "min_range_m": 0.22,           # closer than this to base_link = part of the robot
    "max_see_m": 3.5,              # ignore anything farther (depth is too noisy)
    "require_lidar": True,         # no /scan -> refuse to drive (set false only to test without the lidar)
    "travel_lift": 0.35,           # lift at least this high before driving, so the arm is above the lidar
    "reacquire_m": 0.35,           # a bottle within this of where it should be is the same bottle
    "pick_attempts": 3,
    "verify_pick": True,
    "drop_clearance_m": 0.08,      # bottle bottom this far above the bin rim when it lets go
    "bin_depth_offset_m": 0.08,    # the camera sees the bin's near wall; its center is about this much farther
    "drop_wait_s": 1.0,
}


def _tp(T, pts):
    return transform_points(np.asarray(T, float), np.asarray(pts, float).reshape(-1, 3))


class RosIO:
    def __init__(self, tm, s=None):
        self.tm, self.c = tm, tm.c
        self.p = tm.p
        self.s = dict(ROBOT_WORLD_SETTINGS, **(s or {}))
        self.path = []                 # odom (x, y) after every base move, for the map picture

    def say(self, text):
        self.tm.say(text)

    # ---------- where things are ----------
    def pose(self):
        return pose_from_matrix(self.c.matrix("odom", "base_link"))

    def head(self, pan, tilt):
        pan = max(-4.0, min(1.7, pan))
        self.tm.move("look", joint_head_pan=pan, joint_head_tilt=max(-1.5, min(0.5, tilt)))

    def look_at_base(self, xyz_base):
        pan, tilt = look_at(self.c.position("link_head_tilt"), xyz_base)
        self.head(pan, tilt)

    def frame(self, settle=True):
        """Fresh detections + the latest depth and camera info."""
        if settle:
            time.sleep(self.s["look_settle_s"])
        t = time.time()
        det = self.c.wait_for("detections", 3.0, newer_than=t)
        depth = self.c.latest("depth")
        info = self.c.latest("camera_info")
        return det, depth, info

    def locate(self, det, depth_msg, info, kinds=("bottle", "bin")):
        """[(kind, xyz_base, xyz_odom, extra)] for every bottle/bin box with enough depth points."""
        if det is None or depth_msg is None or info is None:
            return []
        depth = depth_to_meters(depth_msg)
        if depth.shape != (det.image_height, det.image_width):
            self.say(f"   depth {depth.shape} doesn't match the color image; is align_depth on?")
            return []
        K = list(info.k)
        T_base_cam = self.c.matrix("base_link", det.header.frame_id, det.header.stamp)
        T_odom_base = self.c.matrix("odom", "base_link", det.header.stamp)
        cam_xy = (T_base_cam[0][3], T_base_cam[1][3])
        names = {self.s["bottle_class"]: "bottle", self.s["bin_class"]: "bin"}
        out = []
        for d in det.detections:
            kind = names.get(d.name)
            if kind not in kinds or d.confidence < self.s["det_conf"]:
                continue
            band = 0.06 if kind == "bottle" else 0.30
            pts = box_points(depth, (d.x_min, d.y_min, d.x_max, d.y_max), K, band=band)
            if len(pts) < self.s["min_points"]:
                continue
            pts = _tp(T_base_cam, pts)
            extra = {"confidence": round(float(d.confidence), 2)}
            if kind == "bottle":
                info_b = classify_bottle(pts, radius=PLANNER_SETTINGS["bottle_radius"])
                xyz = tuple(float(v) for v in info_b["center"])
                extra.update(orientation=info_b["orientation"], tilt_deg=round(info_b["tilt_deg"]),
                             surface_z=round(info_b["surface_z"], 3), axis_yaw=round(info_b["axis_yaw"], 3),
                             num_points=int(len(pts)))
            else:
                xyz, rim = bin_from_points(pts, cam_xy, self.s["bin_depth_offset_m"])
                extra.update(rim_z=round(rim, 3))
            dist = math.hypot(xyz[0], xyz[1])
            if dist < self.s["min_range_m"] or dist > self.s["max_see_m"]:
                continue
            odom = _tp(T_odom_base, [xyz])[0]
            out.append((kind, xyz, tuple(float(v) for v in odom), extra))
        return out

    def look(self, pan):
        self.head(pan, self.s["look_tilt"])
        det, depth, info = self.frame()
        return [(kind, odom, extra) for kind, _b, odom, extra in self.locate(det, depth, info)]

    # ---------- lidar ----------
    def _scan(self):
        msg = self.c.wait_for("scan", 2.0, newer_than=time.time() - 0.5)
        if msg is None:
            if self.s["require_lidar"]:
                raise RuntimeError("no /scan from the lidar. Start it: ros2 launch pill_bottle_nodes robot.launch.py "
                                   "lidar:=true (or -p require_lidar:=false to test without it, at your own risk)")
            return None, []
        return msg, scan_points(list(msg.ranges), msg.angle_min, msg.angle_increment)

    def scan_circles(self):
        msg, pts = self._scan()
        if msg is None:
            return []
        T = self.c.matrix("odom", msg.header.frame_id, msg.header.stamp)
        out = []
        for cx, cy, r in scan_clusters(pts):
            o = _tp(T, [(cx, cy, 0.0)])[0]
            out.append((float(o[0]), float(o[1]), r))
        return out

    def scan_robot(self):
        msg, pts = self._scan()
        if msg is None:
            return []
        T = self.c.matrix("base_link", msg.header.frame_id, msg.header.stamp)
        good = [(p[0], p[1], 0.0) for p in pts if p is not None]
        return [(float(q[0]), float(q[1])) for q in _tp(T, good)] if good else []

    # ---------- base ----------
    def before_drive(self):
        self.tm.stow_for_base()
        lift = self.c.joints.get("joint_lift", 1.0)
        if lift < self.s["travel_lift"]:
            self.tm.move("raise the arm above the lidar", joint_lift=self.s["travel_lift"])

    def rotate(self, rad):
        self.before_drive()
        self.tm.move(f"turn the base {math.degrees(wrap(rad)):+.0f} deg", rotate_mobile_base=wrap(rad))
        self._mark()

    def translate(self, m):
        self.before_drive()
        self.tm.move(f"drive {m:+.2f} m", translate_mobile_base=max(-0.5, min(0.5, m)))
        self._mark()

    def _mark(self):
        try:
            self.path.append(self.pose()[:2])
        except Exception:
            pass

    # ---------- picking ----------
    def reacquire(self, target_odom, pans=(0.0, 0.3, -0.3)):
        """Look at where the bottle should be; return a BottleState (base_link) for the bottle nearest that spot,
        or None. Uses every bottle box in view (the localizer only publishes the most confident one, which may
        be a different bottle when several are out)."""
        pose = self.pose()
        rel = to_robot(pose, target_odom[:2])
        for dp in pans:
            ang = math.atan2(rel[1], rel[0]) + dp
            r = math.hypot(*rel)
            self.look_at_base((r * math.cos(ang), r * math.sin(ang), target_odom[2]))
            for _ in range(3):
                det, depth, info = self.frame(settle=_ == 0)
                seen = [x for x in self.locate(det, depth, info, kinds=("bottle",))
                        if math.hypot(x[1][0] - rel[0], x[1][1] - rel[1]) <= self.s["reacquire_m"]]
                if seen:
                    kind, xyz, odom, extra = min(seen, key=lambda x: math.hypot(x[1][0] - rel[0], x[1][1] - rel[1]))
                    return make_state(xyz, extra)
        return None

    def to_odom(self, st):
        T = self.c.matrix("odom", "base_link")
        o = _tp(T, [(st.position.x, st.position.y, st.position.z)])[0]
        return tuple(float(v) for v in o)

    def reach_plan(self, st):
        """Plan, turn/drive the base, look again at the same bottle, plan again... until 'reach'."""
        target = self.to_odom(st)
        for _ in range(self.p["max_base_moves"] + 1):
            plan = self.c.plan(st)
            if plan is None:
                raise RuntimeError("grasp_planner isn't running")
            self.say(f"   plan [{plan.next_step}]: {plan.message}")
            if not plan.success:
                return None, st, plan.message
            if plan.next_step == "reach":
                return plan, st, ""
            if plan.next_step == "rotate_base":
                self.rotate(plan.base_rotate)
            else:
                self.translate(plan.base_translate)
            st = self.reacquire(target)
            if st is None:
                return None, None, "lost the bottle after moving the base"
            target = self.to_odom(st)
        return None, st, "too many base moves"

    def pick(self, item):
        st = self.reacquire(item["xyz"])
        if st is None:
            return {"status": "missing", "why": "not at the saved spot"}
        orientation = st.orientation
        p = self.p
        for attempt in range(self.s["pick_attempts"]):
            plan, st2, why = self.reach_plan(st)
            if plan is None:
                status = "missing" if st2 is None else "out_of_range"
                return {"status": status, "why": why, "orientation": orientation}
            st = st2
            log = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "source": "demo_run", "bottle": item["id"],
                   "attempt": attempt + 1, "orientation": st.orientation}
            grasp.approach(self.c, plan, p, self.tm.move, self.say, log)
            grasp.refine(self.c, p, self.tm.move, self.say, log)
            outcome = grasp.close_and_check(self.c, plan, p, self.tm.move, self.say, log,
                                            tag=f"_{item['id']}_try{attempt + 1}")
            log["outcome"] = outcome
            self.tm.save_log(log)
            if outcome in ("held", "slipping"):
                break
            grasp.release(self.c, plan, p, self.tm.move)
            st = self.reacquire(self.to_odom(st))
            if st is None:
                return {"status": "missing", "why": "lost it after a missed grab", "orientation": orientation}
        else:
            return {"status": "missed", "why": f"{self.s['pick_attempts']} grabs missed", "orientation": orientation}
        xyz = self.to_odom(st)
        out = {"status": "picked", "xyz": xyz, "orientation": orientation}
        if self.s["verify_pick"] and not self.tm.verify():
            self.say("   (couldn't confirm it's a pill bottle at the head camera; carrying on)")
            out["verified"] = False
        if p["read_label"]:
            current = {"plan": plan}
            label = self.tm.read_label(current, orientation, log)
            out["label"] = {k: label.get(k) for k in ("status", "fields", "views", "sources")}
            if label.get("status") == "regrasp_failed":
                return {"status": "dropped", "why": "lost it while re-gripping for the label", "orientation": orientation,
                        "label": out["label"]}
        return out

    # ---------- the bin ----------
    def find_bin_here(self, bin_odom):
        """Look at the remembered bin; return (xyz_base, rim_z) from a fresh view, or from memory if not seen."""
        pose = self.pose()
        rel = to_robot(pose, bin_odom[:2])
        self.look_at_base((rel[0], rel[1], 0.2))
        det, depth, info = self.frame()
        seen = [x for x in self.locate(det, depth, info, kinds=("bin",))
                if math.hypot(x[1][0] - rel[0], x[1][1] - rel[1]) < 0.5]
        if seen:
            kind, xyz, odom, extra = max(seen, key=lambda x: x[3]["confidence"])
            return xyz, extra["rim_z"], odom
        return None, None, None

    def drop(self, bin_item):
        rim = bin_item.get("rim_z")
        bin_odom = bin_item["xyz"]
        for _ in range(self.p["max_base_moves"] + 1):
            xyz, rim_now, odom = self.find_bin_here(bin_odom)
            if xyz is None:
                rel = to_robot(self.pose(), bin_odom[:2])
                if rim is None:
                    self.say("   can't see the bin and don't know its height; not dropping")
                    return False
                xyz = (rel[0], rel[1], rim)
            else:
                rim, bin_odom = rim_now, odom
            wrist = self.c.position("link_wrist_yaw")
            grasp_c = self.c.position("link_grasp_center")
            g0 = grasp_center_at_yaw0(wrist, grasp_c)
            plan = plan_drop(xyz, rim, self.c.joints["wrist_extension"], self.c.joints["joint_lift"], g0,
                             PLANNER_SETTINGS, self.s["drop_clearance_m"])
            self.say(f"   drop plan [{plan['next_step']}]: {plan['message']}")
            if not plan["success"]:
                return False
            if plan["next_step"] == "rotate_base":
                self.rotate(plan["base_rotate"])
                continue
            if plan["next_step"] == "drive_base":
                self.translate(plan["base_translate"])
                continue
            m = self.tm.move
            m("retract the arm", wrist_extension=0.0)
            m("lift over the bin rim", joint_lift=plan["lift_clear"])
            m("point the wrist out", joint_wrist_yaw=0.0)
            if plan["base_translate"]:
                m(f"line up with the bin ({plan['base_translate']:+.3f} m)", translate_mobile_base=plan["base_translate"])
            m("reach over the bin", wrist_extension=plan["arm_grasp"])
            m("lower toward the rim", joint_lift=plan["lift"])
            m("let go", gripper_aperture=self.p["gripper_open"])
            time.sleep(self.s["drop_wait_s"])
            m("lift clear", joint_lift=plan["lift_clear"])
            m("retract", wrist_extension=0.0)
            self.tm.stow_for_base()
            return True
        return False

    def count_in_bin(self, bin_item):
        """Bottles whose box center is inside the bin's box, best of 5 frames (bottles hide each other, so the
        robot's own drop count is the main number and this is the double-check)."""
        rel = to_robot(self.pose(), bin_item["xyz"][:2])
        self.look_at_base((rel[0], rel[1], bin_item.get("rim_z", 0.2)))
        best = None
        for _ in range(5):
            det, _, _ = self.frame(settle=False)
            if det is None:
                continue
            bins = [d for d in det.detections if d.name == self.s["bin_class"]]
            if not bins:
                continue
            b = max(bins, key=lambda d: d.confidence)
            n = sum(1 for d in det.detections if d.name == self.s["bottle_class"] and d.confidence >= 0.25
                    and b.x_min <= (d.x_min + d.x_max) / 2 <= b.x_max and b.y_min <= (d.y_min + d.y_max) / 2 <= b.y_max)
            best = n if best is None else max(best, n)
        return best

    def save_map(self, memory, path, title=""):
        from .map_drawing import draw_map
        try:
            pose = self.pose()
        except Exception:
            pose = None
        return cv2.imwrite(path, draw_map(memory, pose, self.path, title=title))


def make_state(xyz_base, extra):
    """BottleState (base_link) for the grasp planner, from a located bottle."""
    from pill_bottle_interfaces.msg import BottleState
    st = BottleState()
    st.header.frame_id = "base_link"
    st.position.x, st.position.y, st.position.z = (float(v) for v in xyz_base)
    st.orientation = extra.get("orientation", "upright")
    st.axis_tilt_deg = float(extra.get("tilt_deg", 0.0))
    st.axis_yaw = float(extra.get("axis_yaw", 0.0))
    st.surface_z = float(extra.get("surface_z", 0.0))
    st.confidence = float(extra.get("confidence", 0.0))
    st.num_points = int(extra.get("num_points", 0))
    return st

