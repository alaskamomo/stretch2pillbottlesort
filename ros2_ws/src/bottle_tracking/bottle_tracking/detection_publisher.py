"""
detection_publisher.py - the ONE thing your existing perception code needs to add.

bottle_memory listens on /pill_bottle/detection (geometry_msgs/PointStamped).
Every time your YOLO/SAM3 detector finds the pill bottle in a head camera frame,
publish the bottle center in the camera frame with the IMAGE's timestamp.
Run the detector continuously (every frame or a few Hz) so the memory can tell
when the bottle has gone missing.

Example, inside your perception node:

    from bottle_tracking.detection_publisher import BottleDetectionPublisher, deproject

    self.bottle_pub = BottleDetectionPublisher(self)          # self = your rclpy Node
    ...
    def on_image(self, color_msg, depth_msg):
        box = self.detector.detect(color_image)                # your PillBottleDetector / YOLO
        if box is not None:
            u, v = box.center                                   # pixel of the bottle center
            depth_m = median_depth_in_box(depth_image, box)     # median handles depth holes
            x, y, z = deproject(u, v, depth_m, self.camera_info.k)
            self.bottle_pub.publish(x, y, z, color_msg.header)  # same header = right frame + time

Note the Stretch head camera is mounted sideways. That doesn't matter here: just use
the camera_info from the same image and the header's frame_id, and TF handles the rest.
"""

from geometry_msgs.msg import PointStamped


def deproject(u, v, depth_m, k):
    """Pixel (u, v) + depth (meters) -> 3D point in the camera optical frame.
    k is CameraInfo.k (row-major 3x3: fx 0 cx / 0 fy cy / 0 0 1)."""
    fx, cx, fy, cy = k[0], k[2], k[4], k[5]
    return ((u - cx) * depth_m / fx, (v - cy) * depth_m / fy, depth_m)


class BottleDetectionPublisher:
    def __init__(self, node, topic="/pill_bottle/detection"):
        self.pub = node.create_publisher(PointStamped, topic, 10)

    def publish(self, x, y, z, header):
        """header: the std_msgs/Header of the image the detection came from."""
        msg = PointStamped()
        msg.header = header
        msg.point.x, msg.point.y, msg.point.z = float(x), float(y), float(z)
        self.pub.publish(msg)
