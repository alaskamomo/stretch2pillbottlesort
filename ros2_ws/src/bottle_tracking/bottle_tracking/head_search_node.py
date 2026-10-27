#!/usr/bin/env python3
"""
head_search_node - finds the bottle by moving the HEAD first, the base only if needed.

Service: /head_search/search (std_srvs/Trigger)
  1. If memory already has the bottle (KNOWN), point the head at the saved spot and
     wait look_saved_timeout_s for a detection. Done if seen. (No sweep, no spin.)
  2. Otherwise sweep the head through head_pan_positions x head_tilt_positions while
     the base stays still. Done as soon as the detector sees the bottle.
  3. Only if the whole sweep finds nothing: rotate the base base_step_deg and sweep
     again, up to max_base_rotations times.
  When found, head tracking is switched back on so the camera stays on the bottle.

Response message is JSON:
  {"found": true, "found_with": "saved" | "head_sweep" | "base_rotation",
   "search_time_s": 4.2, "base_rotation_deg": 0.0, "num_base_rotations": 0,
   "x": .., "y": .., "z": ..}

Your state machine (or a tester) calls this instead of spinning the robot.
"""

import json
import math
import time

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from std_srvs.srv import Trigger

from bottle_tracking.common import MemoryWatcher
from bottle_tracking.stretch_interface import StretchInterface, declare


class HeadSearchNode(Node):
    def __init__(self):
        super().__init__("head_search")
        cb = ReentrantCallbackGroup()
        self.robot = StretchInterface(self, cb)
        self.mem = MemoryWatcher(self, cb)

        self.pans = list(declare(self, "head_pan_positions",
                                 [1.4, 0.7, 0.0, -0.7, -1.4, -2.1, -2.8, -3.5]))
        self.tilts = list(declare(self, "head_tilt_positions", [-0.45, -0.85]))
        self.dwell = declare(self, "dwell_s", 0.8)                 # time to let the detector run
        self.look_saved_timeout = declare(self, "look_saved_timeout_s", 3.0)
        self.base_step = math.radians(declare(self, "base_step_deg", 90.0))
        self.max_rot = int(declare(self, "max_base_rotations", 3))
        self.timeout = declare(self, "search_timeout_s", 120.0)

        self.busy = False
        self.create_service(Trigger, "/head_search/search", self.on_search, callback_group=cb)
        self.get_logger().info("head search ready: call /head_search/search")

    def on_search(self, _req, res):
        if self.busy:
            res.success, res.message = False, json.dumps({"error": "search already running"})
            return res
        self.busy = True
        try:
            result = self.search()
        except Exception as e:  # never leave the base moving or the service hanging
            self.robot.stop()
            result = {"found": False, "error": f"{type(e).__name__}: {e}"}
        finally:
            self.busy = False
        res.success = bool(result.get("found"))
        res.message = json.dumps(result)
        return res

    def search(self):
        t0_wall = time.time()
        t0 = self.mem.now()
        total_rot, n_rot = 0.0, 0

        def found():
            return self.mem.seen_since(t0)

        def result(ok, how):
            r = {"found": ok, "found_with": how, "search_time_s": round(time.time() - t0_wall, 2),
                 "base_rotation_deg": round(math.degrees(total_rot), 1), "num_base_rotations": n_rot}
            pos = self.mem.position()
            if ok and pos:
                r.update(x=pos[0], y=pos[1], z=pos[2])
            if ok:
                self.mem.set_tracking(True)
            self.get_logger().info(f"search result: {r}")
            return r

        # 1) look where we saved it
        if self.mem.get().get("status") == "KNOWN":
            self.get_logger().info("checking saved position first")
            self.mem.set_tracking(True)
            if self.mem.wait_for(found, self.look_saved_timeout):
                return result(True, "saved")

        # 2) head sweep, then 3) base rotation + sweep
        self.mem.set_tracking(False)
        for k in range(self.max_rot + 1):
            for row, tilt in enumerate(self.tilts):
                pans = self.pans if row % 2 == 0 else list(reversed(self.pans))  # snake pattern
                for pan in pans:
                    if time.time() - t0_wall > self.timeout:
                        return result(False, "timeout")
                    self.robot.move_head_and_wait(pan, tilt)
                    if self.mem.wait_for(found, self.dwell):
                        return result(True, "head_sweep" if k == 0 else "base_rotation")
            if k < self.max_rot:
                self.get_logger().info(f"head sweep found nothing, rotating base "
                                       f"{math.degrees(self.base_step):.0f} deg")
                if not self.robot.switch_to_navigation_mode():
                    return result(False, "could not switch to navigation mode")
                ok, turned = self.robot.rotate_by(self.base_step)
                total_rot += abs(turned)
                n_rot += 1
                if not ok:
                    return result(False, "base rotation failed")
        return result(False, "not_found")


def main():
    rclpy.init()
    node = HeadSearchNode()
    ex = MultiThreadedExecutor(num_threads=4)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.robot.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
