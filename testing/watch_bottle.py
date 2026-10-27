#!/usr/bin/env python3
"""
watch_bottle.py - prints whether the robot currently SEES the pill bottle.

Run in its own terminal while anything else is running:
    python3 ~/stretch2pillbottlesort/testing/watch_bottle.py
Prints a line the moment the bottle is seen or lost, plus a status line every second.
"""
import json
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PointStamped
from std_msgs.msg import String

SEEN_WINDOW_S = 0.5   # no detection for this long = "does not see it"


class BottleWatcher(Node):
    def __init__(self):
        super().__init__("bottle_watcher")
        self.last_det_time = None    # wall time of the last raw detection
        self.last_det = None         # (x, y, z, frame) of the last raw detection
        self.det_count = 0
        self.state = {}
        self.tracker = {}
        self.seeing = None           # None until the first check
        self.create_subscription(PointStamped, "/pill_bottle/detection", self.on_det, 10)
        self.create_subscription(String, "/bottle_memory/state", self.on_state, 10)
        self.create_subscription(String, "/head_tracker/status", self.on_tracker, 10)
        self.create_timer(0.1, self.check_change)   # catch SEES/LOST quickly
        self.create_timer(1.0, self.status_line)    # steady status once a second

    def on_det(self, msg):
        self.last_det_time = time.time()
        self.last_det = (msg.point.x, msg.point.y, msg.point.z, msg.header.frame_id)
        self.det_count += 1

    def on_state(self, msg):
        self.state = json.loads(msg.data)

    def on_tracker(self, msg):
        self.tracker = json.loads(msg.data)

    def sees(self):
        return self.last_det_time is not None and time.time() - self.last_det_time < SEEN_WINDOW_S

    def saved(self):
        s = self.state
        if "x" not in s:
            return "no saved position"
        return f"saved at x={s['x']:.2f} y={s['y']:.2f} z={s['z']:.2f} ({s.get('frame', 'odom')})"

    def check_change(self):
        now_sees = self.sees()
        if now_sees != self.seeing:
            stamp = time.strftime("%H:%M:%S")
            if now_sees:
                x, y, z, frame = self.last_det
                print(f"\n[{stamp}] >>> SEES the pill bottle at x={x:.2f} y={y:.2f} z={z:.2f} ({frame})", flush=True)
            elif self.seeing is not None:
                print(f"\n[{stamp}] <<< LOST sight of the pill bottle | {self.saved()}", flush=True)
            else:
                print(f"\n[{stamp}] does not see the pill bottle yet", flush=True)
            self.seeing = now_sees

    def status_line(self):
        status = self.state.get("status", "no memory node?")
        err = self.tracker.get("angle_err_deg")
        err_txt = f"{err:.1f} deg off center" if err is not None else "head error n/a"
        seen = "SEES bottle" if self.sees() else "does NOT see bottle"
        print(f"  {seen:<20} | memory: {status:<8} | {self.saved()} | {err_txt} | detections: {self.det_count}",
              flush=True)


def main():
    rclpy.init()
    node = BottleWatcher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
