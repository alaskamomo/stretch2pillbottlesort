"""
multi_find.py - several pill bottles at once (T2). No ROS, unit tested (test/test_multi_find.py).

  box_status       what the detector does with one YOLO box: tracked / ok / low_conf / no_depth / too_far
  box_label        text + color for that box on the debug picture
  cluster          merge sightings from every head position into one entry per bottle (odom frame)
  bottle_lines     the report: "Pill bottle 1/2 at x, y, z ..." and "Possible pill bottle ..." lines
  score            SUCCESS / MISS / FALSE_POSITIVE from how many bottles you put out
"""

import math

GREEN, YELLOW, RED = (0, 200, 0), (0, 200, 255), (0, 0, 255)


def box_status(conf, depth_m, conf_threshold, max_depth_m):
    if depth_m is None:
        return "no_depth"
    if depth_m > max_depth_m:
        return "too_far"
    if conf < conf_threshold:
        return "low_conf"
    return "ok"


def box_label(e):
    """(text, BGR color) for one detections_all entry."""
    d = f"{e['depth_m']:.2f} m" if e.get("depth_m") is not None else "no depth"
    s = e["status"]
    if s in ("ok", "tracked"):
        return f"pill_bottle {e['conf']:.2f}  {d}" + ("  (tracked)" if s == "tracked" else ""), GREEN
    if s == "low_conf":
        return f"{e['conf']:.2f}  {d}  low confidence", YELLOW
    return f"{e['conf']:.2f}  {d}  " + ("too far" if s == "too_far" else "no depth"), RED


def cluster(sightings, radius_m=0.10):
    """sightings: dicts with "fixed" (x, y, z) odom, "base" (x, y, z), "conf", "view", "status".
    Most confident first, each joins the nearest bottle within radius_m (in x, y; height is noisier),
    else starts a new one. Returns bottles sorted by distance from the base:
    {"fixed", "base", "distance_m", "bearing_deg", "conf" (best), "mean_conf", "n", "views", "confident"}
    "confident" means at least one sighting reached the threshold (status ok/tracked)."""
    groups = []
    for s in sorted(sightings, key=lambda s: -s["conf"]):
        best, best_d = None, radius_m
        for g in groups:
            c = _mean([q["fixed"] for q in g])
            d = math.hypot(s["fixed"][0] - c[0], s["fixed"][1] - c[1])
            if d <= best_d:
                best, best_d = g, d
        if best is None:
            groups.append([s])
        else:
            best.append(s)
    out = []
    for g in groups:
        base = _mean([q["base"] for q in g])
        confs = [q["conf"] for q in g]
        out.append({"fixed": _mean([q["fixed"] for q in g]), "base": base,
                    "distance_m": math.hypot(base[0], base[1]),
                    "bearing_deg": math.degrees(math.atan2(base[1], base[0])),
                    "conf": max(confs), "mean_conf": sum(confs) / len(confs), "n": len(g),
                    "views": sorted({q["view"] for q in g}),
                    "confident": any(q["status"] in ("ok", "tracked") for q in g)})
    return sorted(out, key=lambda b: b["distance_m"])


def _mean(pts):
    return tuple(sum(p[i] for p in pts) / len(pts) for i in range(3))


def _where(b, frame):
    x, y, z = b["fixed"]
    side = "left" if b["bearing_deg"] > 0 else "right"
    return (f"x={x:.2f} y={y:.2f} z={z:.2f} m ({frame}) | {b['distance_m']:.2f} m away, "
            f"{abs(b['bearing_deg']):.0f} deg {side} | conf {b['conf']:.2f} | seen {b['n']}x in "
            f"{len(b['views'])} view{'s' if len(b['views']) != 1 else ''}")


def bottle_lines(bottles, conf_threshold, no_depth=(), frame="odom"):
    """Report lines. no_depth: [(view, conf)] boxes that had no depth (can't be placed)."""
    sure = [b for b in bottles if b["confident"]]
    maybe = [b for b in bottles if not b["confident"]]
    lines = [f"Found {len(sure)} pill bottle{'s' if len(sure) != 1 else ''}"
             + (f" (+{len(maybe)} possible, low confidence)" if maybe else "")]
    for i, b in enumerate(sure, 1):
        lines.append(f"  Pill bottle {i}/{len(sure)} at " + _where(b, frame))
    for i, b in enumerate(maybe, 1):
        lines.append(f"  Possible pill bottle {i}/{len(maybe)} at " + _where(b, frame)
                     + f"  (below {conf_threshold:.2f})")
    if no_depth:
        views = sorted({v for v, _ in no_depth})
        lines.append(f"  Also saw {len(no_depth)} box(es) with no depth, so no position "
                     f"(best conf {max(c for _, c in no_depth):.2f}, view{'s' if len(views) > 1 else ''} "
                     + ", ".join(str(v) for v in views) + ")")
    if not bottles and not no_depth:
        lines[0] = "Does NOT see any pill bottle (not even a low-confidence one)"
    return lines


def score(n_expected, bottles):
    """(outcome, note). n_expected None/0 = unknown: SUCCESS if anything confident was found."""
    sure = sum(1 for b in bottles if b["confident"])
    maybe = sum(1 for b in bottles if not b["confident"])
    if not n_expected:
        return ("SUCCESS" if sure else "NOT_FOUND"), f"found {sure} (+{maybe} possible)"
    if sure == n_expected:
        return "SUCCESS", f"found all {n_expected}"
    if sure > n_expected:
        return "FALSE_POSITIVE", f"found {sure} but only {n_expected} out"
    if sure == 0 and maybe == 0:
        return "NOT_FOUND", f"found none of {n_expected}"
    note = f"found {sure} of {n_expected}"
    if sure + maybe >= n_expected:
        note += f"; {maybe} more seen with low confidence (retrain / lower conf_threshold)"
    return "MISS", note
