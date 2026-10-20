# One YOLO box. Pixel coordinates are in the RAW (unrotated) camera image,
# so they line up with the aligned depth image and camera_info.
string name          # pill_bottle, cap or label
float32 confidence
float32 x_min
float32 y_min
float32 x_max
float32 y_max
