# T6 range test math (no robot or ROS): python3 -m pytest test/test_range.py
import pytest

from pill_bottle_nodes.range_test import (RANGE_SETTINGS as S, box_reasons, expected_box_px, max_reliable,
                                          orientation_guess, suggestions, summarize_spot)


def frames(size, n, seen, conf=0.7, depth_px=50, dist=2.0, w=15, h=30, ms=100):
    out = []
    for i in range(n):
        boxes = [dict(conf=conf, w=w, h=h, depth_px=depth_px, dist=dist)] if i < seen else []
        out.append(dict(size=size, boxes=boxes, ms=ms))
    return out


def test_box_reasons():
    assert box_reasons(0.7, 50, 2.0, S) == []
    assert box_reasons(0.3, 5, 3.5, S) == ["low conf", "no depth", "too far"]


def test_orientation_guess():
    assert orientation_guess(15, 30) == "upright" and orientation_guess(30, 15) == "lying"
    assert orientation_guess(20, 20) == "unsure"


def test_spot_summary_and_pass():
    f = frames(640, 20, 10, conf=0.4) + frames(1280, 20, 19, dist=2.02, ms=400)
    s = summarize_spot(f, 2.0, S)
    assert s[640]["seen"] == 0 and s[640]["reasons"]["low conf"] == 10 and not s[640]["passed"]
    assert s[1280]["seen"] == 19 and s[1280]["err_cm"] == pytest.approx(2.0) and s[1280]["passed"]
    assert s[1280]["fps"] == pytest.approx(2.5)


def test_far_spots_get_a_looser_error_limit():
    assert summarize_spot(frames(640, 10, 10, dist=2.84), 2.8, S)[640]["passed"]       # 4 cm at 2.8 m: OK
    assert not summarize_spot(frames(640, 10, 10, dist=2.04), 2.0, S)[640]["passed"]   # 4 cm at 2 m: too big


def test_max_reliable_stops_at_the_first_miss():
    res = [(1.0, summarize_spot(frames(640, 10, 10, dist=1.0), 1.0, S)),
           (2.0, summarize_spot(frames(640, 10, 10, dist=2.0), 2.0, S)),
           (2.5, summarize_spot(frames(640, 10, 2, dist=2.5), 2.5, S)),
           (3.0, summarize_spot(frames(640, 10, 10, dist=3.0), 3.0, S))]   # a lucky far mark doesn't count
    assert max_reliable(res, 640) == 2.0


def test_suggestions_name_the_fix():
    res = [(1.0, summarize_spot(frames(640, 10, 10, dist=1.0) + frames(1280, 10, 10, dist=1.0), 1.0, S)),
           (2.5, summarize_spot(frames(640, 10, 10, conf=0.35, dist=2.5) + frames(1280, 10, 10, dist=2.5), 2.5, S)),
           (3.5, summarize_spot(frames(640, 10, 10, conf=0.3, dist=3.5) + frames(1280, 10, 10, dist=3.5), 3.5, S))]
    text = "\n".join(suggestions(res, S))
    assert "imgsz: 1280" in text and "conf_threshold" in text and "max_bottle_distance_m" in text


def test_expected_box_size_matches_the_table_in_the_ticket():
    assert round(expected_box_px(2.0)) == 37 and round(expected_box_px(4.0, size_m=0.04)) == 9


# ---------------------------------------------------------------- live printout and measured spots (no prompts)
from pill_bottle_nodes.range_test import detection_lines, height_bucket, spot_distance, spot_height  # noqa: E402


def box(conf=0.8, xyz=(1.2, -0.3, 0.02), depth_px=50, **k):
    d = None if xyz is None else (xyz[0] ** 2 + xyz[1] ** 2) ** 0.5
    return dict(conf=conf, xyz=xyz, dist=d, depth_px=depth_px, w=20, h=40, **k)


def test_detection_lines_list_every_bottle_nearest_first():
    lines = detection_lines([box(0.21, (2.6, 0.4, 0.75)), box(0.83), box(0.5, None, depth_px=0)], S)
    assert lines[0].startswith("pill bottle 1/3 detected at x=1.20 y=-0.30 z=0.02 m | 1.24 m away | conf 0.83")
    assert lines[1].startswith("pill bottle 2/3 detected at x=2.60 y=0.40 z=0.75 m") and "(low conf)" in lines[1]
    assert lines[2].startswith("pill bottle 3/3 detected, no depth") and "no depth" in lines[2]
    assert detection_lines([], S) == ["no pill bottle"]
    assert "from the bin" in detection_lines([box(bin_m=0.5, orientation="upright?")], S)[0]


def test_spot_is_measured_not_asked():
    fr = [dict(size=640, boxes=[], ms=100), dict(size=1280, boxes=[box(0.3, (2.0, 0.0, 0.76))], ms=200),
          dict(size=1280, boxes=[box(0.7, (2.1, 0.0, 0.74))], ms=200)]
    assert spot_distance(fr) == 2.05 and spot_height(fr) == 0.75
    assert height_bucket(0.02) == "floor" and height_bucket(0.75) == "table" and height_bucket(1.0) == "raised"
    assert spot_distance([dict(size=640, boxes=[], ms=1)]) is None and height_bucket(None) == "unknown"
    s = summarize_spot(fr, None, S)
    assert s[640]["seen"] == 0 and s[1280]["err_cm"] is None and s[1280]["dist"] == 2.1
