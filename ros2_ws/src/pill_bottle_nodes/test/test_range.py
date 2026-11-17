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
