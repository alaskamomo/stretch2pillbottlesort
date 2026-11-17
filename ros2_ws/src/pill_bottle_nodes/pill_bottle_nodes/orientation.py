# orientation.py: combine the two ways of telling how the bottle is sitting (ticket T17). No ROS, unit tested.
#
#   depth   bottle_localizer fits the bottle's depth points (geometry.classify_bottle): exact tilt angle, but it
#           needs enough points. Few points (far away, shiny, small in the image, on a high shelf) -> noisy.
#   vision  orientation_classifier runs a YOLOv8 classification model on the bottle crop: works with any number
#           of depth points, but only gives a class and a confidence.
#
# Rule: depth wins when it has enough points AND a clear angle; otherwise a confident vision answer wins;
# otherwise depth (it's what the grasp planner always used).
import math

ORIENTATION_SETTINGS = {
    "use_vision_orientation": True,
    "vision_max_age_s": 1.5,          # ignore a vision answer older than this
    "vision_min_conf": 0.6,           # vision answers below this are ignored
    "depth_trust_points": 150,        # depth needs at least this many points to overrule vision...
    "depth_clear_margin_deg": 10.0,   # ...and a tilt this far from the upright/lying thresholds
}


def label_from_class(name):
    """Map a model's class name to upright | lying | tilted (names like 'upright', 'on_side', 'knocked_over')."""
    n = str(name).lower()
    if "tilt" in n or "lean" in n:
        return "tilted"
    if "up" in n or "stand" in n or "straight" in n:
        return "upright"
    if any(k in n for k in ("ly", "side", "knock", "down", "flat", "over")):
        return "lying"
    return None


def depth_label(tilt_deg, upright_max_deg=30.0, lying_min_deg=60.0):
    if tilt_deg < upright_max_deg:
        return "upright"
    if tilt_deg > lying_min_deg:
        return "lying"
    return "tilted"


def depth_is_clear(tilt_deg, num_points, s, upright_max_deg=30.0, lying_min_deg=60.0):
    m = s["depth_clear_margin_deg"]
    return num_points >= s["depth_trust_points"] and (tilt_deg <= upright_max_deg - m or tilt_deg >= lying_min_deg + m)


def fuse_orientation(tilt_deg, num_points, vision, s, upright_max_deg=30.0, lying_min_deg=60.0):
    """-> (orientation, source, agree).

    tilt_deg, num_points: from the depth fit. vision: None or dict(label, conf, age_s).
    source: depth | vision | both.  agree: None if there was no usable vision answer.
    """
    d = depth_label(tilt_deg, upright_max_deg, lying_min_deg)
    usable = (s["use_vision_orientation"] and vision is not None and vision.get("label") in ("upright", "lying", "tilted")
              and vision.get("age_s", math.inf) <= s["vision_max_age_s"] and vision.get("conf", 0.0) >= s["vision_min_conf"])
    if not usable:
        return d, "depth", None
    v = vision["label"]
    if v == d:
        return d, "both", True
    if depth_is_clear(tilt_deg, num_points, s, upright_max_deg, lying_min_deg):
        return d, "depth", False
    return v, "vision", False
