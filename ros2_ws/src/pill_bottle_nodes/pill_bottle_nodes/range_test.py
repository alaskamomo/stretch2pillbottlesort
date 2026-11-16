# range_test.py: how far away can the robot see the pill bottle? (ticket T6). No ROS, unit tested.
#
#   box_reasons        why a YOLO box was or wasn't used (low confidence / no depth / too far)
#   orientation_guess  upright / lying from the box shape, for the live view when no orientation model is loaded
#   summarize_spot     one tape mark: how often it was seen at each YOLO size, confidence, box size, distance error
#   max_reliable       the farthest mark where every mark up to it passed, per YOLO size
#   suggestions        settings to change, from the results
import math
import statistics
from collections import Counter

RANGE_SETTINGS = {
    "distances": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],   # tape marks (m from the base center)
    "sizes": [640, 1280],                              # YOLO input sizes to compare
    "frames": 20,                                      # frames per size per mark
    "show_conf": 0.10,                                 # draw boxes down to this confidence (yellow when below conf)
    "conf_threshold": 0.50,                            # what the detector uses now (params.yaml)
    "min_depth_px": 10,                                # detector needs this many good depth pixels
    "max_distance_m": 3.0,                             # bottle_memory's max_bottle_distance_m
    "pass_rate": 0.8,                                  # seen in 80 % of frames
    "max_error_cm": 3.0, "max_error_far_cm": 5.0, "far_m": 2.5,
}


def box_reasons(conf, depth_px, distance_m, s):
    """[] if the detector would use the box, else the reasons it wouldn't."""
    why = []
    if conf < s["conf_threshold"]:
        why.append("low conf")
    if depth_px < s["min_depth_px"]:
        why.append("no depth")
    if distance_m is not None and distance_m > s["max_distance_m"]:
        why.append("too far")
    return why


def orientation_guess(w, h):
    """Box in the UPRIGHT image: taller than wide = upright, wider than tall = lying. End-on lying bottles look
    round, so in-between shapes say 'unsure'. A guess for the live view only; T17 has the real answer."""
    if w <= 0 or h <= 0:
        return "unsure"
    r = h / w
    return "upright" if r >= 1.3 else ("lying" if r <= 0.8 else "unsure")


def spot_pass(rate, err_cm, true_m, s):
    if rate < s["pass_rate"]:
        return False
    if err_cm is None or true_m is None:
        return True
    return err_cm <= (s["max_error_far_cm"] if true_m > s["far_m"] else s["max_error_cm"])


def summarize_spot(frames, true_m, s):
    """frames: [{"size", "boxes": [{"conf", "w", "h", "depth_px", "dist"}], "ms"}] (boxes = pill_bottle boxes
    at show_conf or above). Returns {size: dict(seen, n, rate, conf, box, dist, err_cm, reasons, fps, pass)}."""
    out = {}
    for size in sorted({f["size"] for f in frames}):
        fs = [f for f in frames if f["size"] == size]
        used, near, reasons = [], [], Counter()
        for f in fs:
            best = max(f["boxes"], key=lambda b: b["conf"]) if f["boxes"] else None
            if best is None:
                reasons["not detected"] += 1
                continue
            why = box_reasons(best["conf"], best["depth_px"], best["dist"], s)
            if why:
                reasons.update(why)
                near.append(best)
            else:
                used.append(best)
        n = len(fs)
        dists = [b["dist"] for b in used if b["dist"] is not None]
        dist = statistics.median(dists) if dists else None
        err = abs(dist - true_m) * 100 if (dist is not None and true_m is not None) else None
        allb = used + near
        ms = [f["ms"] for f in fs if f.get("ms")]
        out[size] = dict(
            seen=len(used), n=n, rate=len(used) / n if n else 0.0,
            conf=round(statistics.mean(b["conf"] for b in allb), 2) if allb else None,
            box=(round(statistics.median(b["w"] for b in allb)), round(statistics.median(b["h"] for b in allb)))
            if allb else None,
            depth_px=round(statistics.median(b["depth_px"] for b in allb)) if allb else None,
            dist=round(dist, 3) if dist is not None else None, err_cm=round(err, 1) if err is not None else None,
            reasons=dict(reasons), fps=round(1000.0 / statistics.mean(ms), 1) if ms else None,
            passed=spot_pass(len(used) / n if n else 0.0, err, true_m, s))
    return out


def max_reliable(results, size):
    """results: [(true_m, {size: summary})] sorted or not. Farthest mark with every nearer mark passing."""
    best = None
    for true_m, summ in sorted(results, key=lambda r: r[0]):
        if size in summ and summ[size]["passed"]:
            best = true_m
        else:
            break
    return best


def main_miss_reason(results, size):
    c = Counter()
    for true_m, summ in results:
        if size in summ and not summ[size]["passed"]:
            c.update(summ[size]["reasons"])
    return c.most_common(1)[0][0] if c else None


def expected_box_px(distance_m, hfov_deg=69.0, width_px=1280, size_m=0.08):
    """How many pixels an object of size_m spans at distance_m (D435 color: about 69 deg across 1280 px)."""
    f = width_px / 2 / math.tan(math.radians(hfov_deg / 2))
    return f * size_m / distance_m


def suggestions(results, s):
    """Settings to change, as text lines."""
    lines = []
    sizes = sorted({k for _, summ in results for k in summ})
    reach = {z: max_reliable(results, z) for z in sizes}
    if len(sizes) > 1:
        lo, hi = sizes[0], sizes[-1]
        if (reach[hi] or 0) > (reach[lo] or 0):
            lines.append(f"imgsz: {hi}   (sees reliably to {reach[hi]} m vs {reach[lo]} m at {lo}; check the fps is OK)")
    reasons = Counter()
    for _, summ in results:
        for z in summ:
            if not summ[z]["passed"]:
                reasons.update(summ[z]["reasons"])
    if reasons.get("low conf"):
        confs = [summ[z]["conf"] for _, summ in results for z in summ if summ[z]["conf"] is not None
                 and not summ[z]["passed"]]
        if confs:
            lines.append(f"conf_threshold: {max(0.25, round(min(confs) - 0.05, 2))}   (far bottles score "
                         f"{min(confs):.2f}-{max(confs):.2f})")
    if reasons.get("too far"):
        far = max(r[0] for r in results)
        lines.append(f"max_bottle_distance_m: {far + 0.5}   (it's filtering out real bottles past {s['max_distance_m']} m)")
    if reasons.get("no depth"):
        lines.append("no depth on far bottles: more light, or a matte label; the depth camera can't measure shiny "
                     "orange plastic far away")
    if reasons.get("not detected"):
        lines.append("not detected at all past the limit: add training pictures of far bottles from the head camera")
    return lines
