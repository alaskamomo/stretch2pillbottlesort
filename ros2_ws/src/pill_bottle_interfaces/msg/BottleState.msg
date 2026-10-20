# Where the pill bottle is, in header.frame_id (base_link).
std_msgs/Header header
geometry_msgs/Point position   # center of the visible bottle surface (m)
string orientation             # upright | tilted | lying
float32 axis_tilt_deg          # bottle's long axis from vertical: 0 = upright, 90 = lying
float32 axis_yaw               # long axis direction in the floor plane (rad, from +x); used for lying bottles
float32 surface_z              # height of what it's sitting on (m); about 0 = floor
float32 confidence             # YOLO confidence of the box used
uint32 num_points              # depth points used (low = unreliable)
