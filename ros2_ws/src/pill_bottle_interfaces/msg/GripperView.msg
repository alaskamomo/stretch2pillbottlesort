# What the black-and-white claw camera sees (from gripper_cam.analyze).
std_msgs/Header header
bool seen
string orientation   # upright | across | end_on | tilted (empty if not seen)
float32 confidence
float32 center_u     # bottle center pixel
float32 center_v
float32 offset_x_m   # how far the bottle is right of the claw's grasp spot in the image (m)
float32 offset_y_m   # how far it is below it (m)
float32 distance_m   # camera to bottle surface (m); -1 if unknown
