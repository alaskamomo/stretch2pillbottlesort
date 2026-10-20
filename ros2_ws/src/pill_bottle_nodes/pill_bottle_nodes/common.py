# common.py: small helpers the nodes share (loading the existing scripts, images, tf, waiting on futures).
import importlib
import os
import sys
import threading

import numpy as np

from .geometry import quat_to_matrix

# The existing scripts (detect_bottles.py, gripper_cam.py, stretch_fk.py + the model) stay where they are
# on the robot. Every node takes a `legacy_dir` parameter pointing at that folder.
DEFAULT_LEGACY_DIR = os.path.expanduser("~/pill_bottle_robot")


def load_legacy(module_name, legacy_dir):
    """Import one of the existing scripts (e.g. 'detect_bottles') from legacy_dir."""
    legacy_dir = os.path.expanduser(legacy_dir)
    if not os.path.isdir(legacy_dir):
        raise RuntimeError(f"legacy_dir {legacy_dir} doesn't exist. Point it at the folder with "
                           "detect_bottles.py, gripper_cam.py and the model, e.g. -p legacy_dir:=/path")
    if legacy_dir not in sys.path:
        sys.path.insert(0, legacy_dir)
    cwd = os.getcwd()
    os.chdir(legacy_dir)               # load_model() looks for the model next to the script
    try:
        return importlib.import_module(module_name)
    finally:
        os.chdir(cwd)


def load_model(detect_bottles, legacy_dir):
    cwd = os.getcwd()
    os.chdir(os.path.expanduser(legacy_dir))
    try:
        return detect_bottles.load_model()
    finally:
        os.chdir(cwd)


# ---------- sensor_msgs/Image <-> numpy (no cv_bridge, so numpy 2 on the robot doesn't matter) ----------

_ENC = {"rgb8": (np.uint8, 3), "bgr8": (np.uint8, 3), "mono8": (np.uint8, 1),
        "16UC1": (np.uint16, 1), "mono16": (np.uint16, 1), "32FC1": (np.float32, 1)}


def image_to_numpy(msg):
    if msg.encoding not in _ENC:
        raise ValueError(f"Unsupported image encoding {msg.encoding}")
    dtype, ch = _ENC[msg.encoding]
    item = np.dtype(dtype).itemsize
    row = np.frombuffer(bytes(msg.data), dtype=dtype).reshape(msg.height, msg.step // item)
    img = row[:, :msg.width * ch]
    img = img.reshape(msg.height, msg.width, ch) if ch > 1 else img.reshape(msg.height, msg.width)
    if msg.encoding == "rgb8":
        img = img[:, :, ::-1]           # to BGR like OpenCV
    return np.ascontiguousarray(img)


def numpy_to_image(img, stamp, frame_id):
    from sensor_msgs.msg import Image
    img = np.ascontiguousarray(img)
    msg = Image()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height, msg.width = img.shape[:2]
    if img.ndim == 2:
        msg.encoding = "mono8" if img.dtype == np.uint8 else "16UC1"
        msg.step = msg.width * img.dtype.itemsize
    else:
        msg.encoding = "bgr8"
        msg.step = msg.width * 3
    msg.is_bigendian = 0
    msg.data = img.tobytes()
    return msg


def depth_to_meters(msg):
    d = image_to_numpy(msg)
    if d.dtype == np.uint16:
        return d.astype(np.float32) / 1000.0
    return d.astype(np.float32)


# ---------- tf ----------

def lookup_matrix(tf_buffer, target, source, stamp=None, timeout_s=0.5):
    """4x4 transform that maps points in `source` frame into `target` frame."""
    from rclpy.duration import Duration
    from rclpy.time import Time
    t = tf_buffer.lookup_transform(target, source, stamp if stamp is not None else Time(),
                                   timeout=Duration(seconds=timeout_s))
    tr, q = t.transform.translation, t.transform.rotation
    return quat_to_matrix(tr.x, tr.y, tr.z, q.x, q.y, q.z, q.w)


def frame_position(tf_buffer, target, source):
    return tuple(lookup_matrix(tf_buffer, target, source)[:3, 3])


# ---------- waiting on futures from a worker thread (node spun by a MultiThreadedExecutor) ----------

def wait_future(future, timeout_s):
    if future.done():
        return future.result()
    done = threading.Event()
    future.add_done_callback(lambda _: done.set())
    if not done.wait(timeout_s):
        return None
    return future.result()
