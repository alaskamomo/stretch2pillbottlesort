# T17 orientation tests (no robot or ROS needed): python3 -m pytest test/
import math

import numpy as np
import pytest

from pill_bottle_nodes.geometry import classify_bottle
from pill_bottle_nodes.orientation import (ORIENTATION_SETTINGS as S, depth_is_clear, depth_label, fuse_orientation,
                                           label_from_class)


@pytest.mark.parametrize("name,label", [
    ("upright", "upright"), ("standing", "upright"), ("Upright_bottle", "upright"),
    ("lying", "lying"), ("on_side", "lying"), ("knocked_over", "lying"), ("lying_down", "lying"),
    ("tilted", "tilted"), ("leaning", "tilted"), ("cap", None),
])
def test_class_names_map(name, label):
    assert label_from_class(name) == label


def test_depth_label_thresholds():
    assert depth_label(10) == "upright" and depth_label(45) == "tilted" and depth_label(80) == "lying"


def test_no_vision_means_depth():
    assert fuse_orientation(80, 500, None, S) == ("lying", "depth", None)


def test_old_or_unsure_vision_is_ignored():
    assert fuse_orientation(10, 50, {"label": "lying", "conf": 0.95, "age_s": 5.0}, S)[1] == "depth"
    assert fuse_orientation(10, 50, {"label": "lying", "conf": 0.40, "age_s": 0.1}, S)[1] == "depth"
    assert fuse_orientation(10, 50, {"label": None, "conf": 0.99, "age_s": 0.1}, S)[1] == "depth"
    off = dict(S, use_vision_orientation=False)
    assert fuse_orientation(10, 50, {"label": "lying", "conf": 0.99, "age_s": 0.1}, off)[1] == "depth"


def test_agreement():
    assert fuse_orientation(5, 400, {"label": "upright", "conf": 0.9, "age_s": 0.2}, S) == ("upright", "both", True)


def test_clear_depth_beats_vision():
    # plenty of points and 5 deg from vertical: the depth fit is trustworthy
    assert fuse_orientation(5, 400, {"label": "lying", "conf": 0.9, "age_s": 0.2}, S) == ("upright", "depth", False)


def test_vision_wins_when_depth_is_thin_or_borderline():
    v = {"label": "lying", "conf": 0.9, "age_s": 0.2}
    assert fuse_orientation(5, 60, v, S) == ("lying", "vision", False)           # too few points
    assert fuse_orientation(25, 400, v, S) == ("lying", "vision", False)         # 25 deg: too close to 30
    assert fuse_orientation(45, 400, v, S) == ("lying", "vision", False)         # depth said tilted


def test_depth_is_clear():
    assert depth_is_clear(5, 400, S) and depth_is_clear(85, 400, S)
    assert not depth_is_clear(25, 400, S) and not depth_is_clear(5, 10, S)


def _cylinder(center, axis, length=0.08, radius=0.02, n=600, seed=0, noise=0.0):
    rng = np.random.default_rng(seed)
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    a = np.cross(axis, [0, 0, 1]) if abs(axis[2]) < 0.9 else np.cross(axis, [1, 0, 0])
    a /= np.linalg.norm(a)
    b = np.cross(axis, a)
    t = rng.uniform(-length / 2, length / 2, n)
    th = rng.uniform(0, math.pi, n)
    pts = center + np.outer(t, axis) + radius * (np.outer(np.cos(th), a) + np.outer(np.sin(th), b))
    return pts + rng.normal(0, noise, pts.shape)


def test_noisy_few_points_is_where_vision_should_help():
    """An upright bottle seen with only a few noisy depth points: the depth fit can come out at any angle, which
    is exactly the case the vision model covers (few points -> depth isn't 'clear' -> a confident vision wins)."""
    pts = _cylinder(np.array([0.6, -0.4, 0.8]), (0, 0, 1), n=40, noise=0.015, seed=3)
    info = classify_bottle(pts)
    v = {"label": "upright", "conf": 0.92, "age_s": 0.3}
    got, source, _ = fuse_orientation(info["tilt_deg"], len(pts), v, S)
    assert got == "upright" and source in ("both", "vision")
