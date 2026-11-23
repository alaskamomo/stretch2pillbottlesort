#!/usr/bin/env python3
"""
capture_training_images.py - save head camera pictures for YOLO training, with a free camera.

Needs the camera (d435i launch) and the driver (stretch_driver.launch.py) running.
Saves full-size color images, turned upright the same way the detector sees them, with no
boxes or text drawn on them. Prints a browser link to watch the camera while you aim.

  python3 testing/capture_training_images.py                 start file names with "head"
  python3 testing/capture_training_images.py --name box      start file names with "box"
  python3 testing/capture_training_images.py --name empty    pictures with NO bottle in them

Keys (no Enter needed, hold a key to keep moving):
  space / Enter   take a picture
  f               FREE camera on/off: head motors go limp so you can move the camera by hand,
                  press f again to lock it where you left it
  w / s           tilt up / down        W / S / A / D   bigger steps
  a / d           pan left / right      c               center (level, straight ahead)
  q               quit (turns free mode off first)

Free mode uses the driver's runstop, so the wrist and gripper go limp too while it is on and
the arm/base will not take commands. Keep the gripper empty and the arm low before pressing f.
Pictures go to ~/stretch_training_images/.
"""

import argparse
import os
import select
import sys
import termios
import threading
import time
import tty

import cv2
import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_srvs.srv import SetBool, Trigger
from trajectory_msgs.msg import JointTrajectoryPoint

try:
    from pill_bottle_nodes.web_stream import WebStream
except ImportError:  # workspace not sourced: use the copy in this repo
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "ros2_ws", "src", "pill_bottle_nodes"))
    from pill_bottle_nodes.web_stream import WebStream

PAN, TILT = "joint_head_pan", "joint_head_tilt"
LIMITS = {PAN: (-4.0, 1.7), TILT: (-1.5, 0.79)}
STEP = 0.05          # rad per key press (held keys repeat)
BIG = 4              # Shift = bigger steps


def to_bgr(msg):
    """sensor_msgs/Image (rgb8 or bgr8) -> BGR numpy array."""
    if msg.encoding not in ("rgb8", "bgr8"):
        raise ValueError(f"unsupported image encoding {msg.encoding}")
    img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)[:, :msg.width * 3]
    img = img.reshape(msg.height, msg.width, 3)
    return img[:, :, ::-1] if msg.encoding == "rgb8" else img


def upright(img, mode):
    if mode == "cw":
        return np.ascontiguousarray(np.rot90(img, k=-1))
    if mode == "ccw":
        return np.ascontiguousarray(np.rot90(img, k=1))
    return img


def clamp(joint, value):
    lo, hi = LIMITS[joint]
    return min(hi, max(lo, value))


class Capture(Node):
    def __init__(self, topic):
        super().__init__("capture_training_images")
        self.latest, self.stamp = None, 0.0
        self.joints = {}
        self.create_subscription(Image, topic, self.on_image, qos_profile_sensor_data)
        for t in ("/stretch/joint_states", "/joint_states"):
            self.create_subscription(JointState, t, self.on_joints, 10)
        self.traj = ActionClient(self, FollowJointTrajectory, "/stretch_controller/follow_joint_trajectory")
        self.runstop = self.create_client(SetBool, "/runstop")
        self.pos_mode = self.create_client(Trigger, "/switch_to_position_mode")
        self.rejected = False

    def on_image(self, msg):
        self.latest, self.stamp = msg, time.time()

    def on_joints(self, msg):
        for n, p in zip(msg.name, msg.position):
            self.joints[n] = p

    def head(self):
        if PAN in self.joints and TILT in self.joints:
            return self.joints[PAN], self.joints[TILT]
        return None

    # -------------------------------------------------------------- head motion
    def send_head(self, pan, tilt):
        if not self.traj.server_is_ready():
            return False
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = [PAN, TILT]
        goal.trajectory.points = [JointTrajectoryPoint(positions=[float(pan), float(tilt)],
                                                       time_from_start=Duration(sec=0, nanosec=0))]
        self.traj.send_goal_async(goal).add_done_callback(self._on_goal)
        return True

    def _on_goal(self, fut):
        h = fut.result()
        if h is not None and not h.accepted:
            self.rejected = True

    def call(self, client, req, timeout=5.0):
        if not client.wait_for_service(timeout_sec=2.0):
            return None
        fut = client.call_async(req)
        end = time.time() + timeout
        while not fut.done() and time.time() < end:
            time.sleep(0.02)
        return fut.result() if fut.done() else None

    def set_free(self, on):
        """Free = runstop on (Dynamixel head torque off, move it by hand)."""
        r = self.call(self.runstop, SetBool.Request(data=bool(on)))
        return r is not None and r.success, (r.message if r is not None else "no /runstop service")

    def position_mode(self):
        r = self.call(self.pos_mode, Trigger.Request())
        return r is not None and r.success


class Keys:
    """Single key presses without Enter; Ctrl-C still works."""

    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def get(self, timeout=0.1):
        r, _, _ = select.select([sys.stdin], [], [], timeout)
        return sys.stdin.read(1) if r else None

    def __exit__(self, *a):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="head", help="start of each file name")
    p.add_argument("--topic", default="/camera/color/image_raw")
    p.add_argument("--rotate", default="cw", choices=["cw", "ccw", "none"], help="same as rotate_image")
    p.add_argument("--dir", default=os.path.expanduser("~/stretch_training_images"))
    p.add_argument("--port", type=int, default=8080, help="browser link port")
    args = p.parse_args()

    os.makedirs(args.dir, exist_ok=True)
    rclpy.init()
    node = Capture(args.topic)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    stream = WebStream(args.port, f"Training pictures: {args.name}")

    state = {"count": 0, "free": False, "flash": 0.0, "run": True}

    def feed():
        """Send camera pictures to the browser link with a small status line."""
        last = 0.0
        while state["run"]:
            if node.latest is not None and node.stamp != last:
                last = node.stamp
                try:
                    img = upright(to_bgr(node.latest), args.rotate).copy()
                except ValueError:
                    time.sleep(0.1)
                    continue
                h = node.head()
                txt = f"{state['count']} saved"
                if h:
                    txt += f"  pan {h[0]:+.2f} tilt {h[1]:+.2f}"
                if state["free"]:
                    txt += "  FREE (move by hand)"
                cv2.putText(img, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                cv2.putText(img, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 200, 255) if state["free"] else (255, 255, 255), 2)
                if time.time() - state["flash"] < 0.3:
                    cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (0, 255, 0), 12)
                stream.update(img)
            time.sleep(0.03)

    threading.Thread(target=feed, daemon=True).start()

    print("Waiting for the camera and driver ...", flush=True)
    end = time.time() + 10.0
    while time.time() < end and (node.latest is None or node.head() is None or not node.traj.server_is_ready()):
        time.sleep(0.1)
    if node.latest is None:
        print("  No camera image yet. Is the d435i launch running?")
    driver_ok = node.traj.server_is_ready() and node.head() is not None
    if not driver_ok:
        print("  No driver (follow_joint_trajectory / joint_states). Aiming keys and free mode are off;")
        print("  you can still take pictures. Start stretch_driver.launch.py to aim the head.")

    target = list(node.head() or (0.0, 0.0))
    print(f"\nLive camera: {stream.url}   (open on your Mac or phone)")
    print(f"Saving to {args.dir}")
    print("space/Enter picture | f free camera | w/s tilt | a/d pan | c center | q quit\n")

    def status(msg=""):
        mode = "FREE " if state["free"] else ""
        sys.stdout.write(f"\r\033[K[{state['count']} saved] {mode}{msg}")
        sys.stdout.flush()

    def snap():
        if node.latest is None or time.time() - node.stamp > 2.0:
            status("No fresh camera image. Is the camera launch running?")
            return
        img = upright(to_bgr(node.latest), args.rotate)
        path = os.path.join(args.dir, f"{args.name}_{time.strftime('%Y%m%d_%H%M%S')}_{state['count']:03d}.jpg")
        if cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            state["count"] += 1
            state["flash"] = time.time()
            status(os.path.basename(path))
        else:
            status("Could not write the picture.")

    def lock_here():
        """Leave free mode and hold the head where it was moved by hand."""
        h = node.head()               # where it was moved by hand (it stays put while limp)
        ok, msg = node.set_free(False)
        if not ok:
            status(f"Could not turn free mode off: {msg}. Press the runstop button to clear it.")
            return False
        state["free"] = False
        if h:
            target[:] = [clamp(PAN, h[0]), clamp(TILT, h[1])]
            node.send_head(*target)   # hold here instead of springing back to the old goal
        status("Locked.")
        return True

    status()
    try:
        with Keys() as keys:
            while True:
                k = keys.get()
                if k is None:
                    if node.rejected:
                        node.rejected = False
                        status("Head goal rejected; switching driver to position mode ...")
                        status("Position mode on. Try again." if node.position_mode()
                               else "Could not switch to position mode.")
                    continue
                if k in ("q", "Q"):
                    break
                if k in (" ", "\n", "\r"):
                    snap()
                    continue
                if not driver_ok:
                    status("Driver not running: only pictures work.")
                    continue
                if k in ("f", "F"):
                    if state["free"]:
                        lock_here()
                    else:
                        ok, msg = node.set_free(True)
                        if ok:
                            state["free"] = True
                            status("Head is limp: move the camera by hand, f to lock it.")
                        else:
                            status(f"Could not turn free mode on: {msg}")
                    continue
                if state["free"]:
                    status("Free mode: move it by hand (f to lock, then keys work).")
                    continue
                big = BIG if k.isupper() else 1
                k = k.lower()
                if k == "w":
                    target[1] += STEP * big
                elif k == "s":
                    target[1] -= STEP * big
                elif k == "a":
                    target[0] += STEP * big
                elif k == "d":
                    target[0] -= STEP * big
                elif k == "c":
                    target[:] = [0.0, 0.0]
                else:
                    continue
                target[:] = [clamp(PAN, target[0]), clamp(TILT, target[1])]
                node.send_head(*target)
                status(f"pan {target[0]:+.2f}  tilt {target[1]:+.2f}")
    except KeyboardInterrupt:
        pass
    finally:
        if state["free"]:
            print()
            lock_here()
        state["run"] = False
        print(f"\nSaved {state['count']} pictures in {args.dir}")
        stream.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
