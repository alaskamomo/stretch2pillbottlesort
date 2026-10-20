# All YOLO detections for one head-camera frame (header = that frame's stamp and frame_id).
std_msgs/Header header
uint32 image_width
uint32 image_height
Detection[] detections
