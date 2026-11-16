# T16 label reading tests (no robot, no ROS, no internet):  python3 -m pytest test/ -v
#
# Unit tests for each piece, then INTEGRATION tests that run the real robot-side code (LabelSteps: grab image ->
# find box -> crop -> turn upright -> flatten -> read, plus the wrist/lift moves) together with the real state
# machine (run_label_sequence) against a simulated robot. The simulated head camera draws a round bottle whose
# label is five colored fields; the wrist turns it, the fingers hide part of it, and glare can wash out the middle.
# A fake reader "reads" a field when enough of its color is visible, so the tests check the real geometry: what the
# camera sees at each wrist angle, how the curve squeezes the edges, and whether flattening undoes it.
import math
from types import SimpleNamespace

import numpy as np
import pytest

from pill_bottle_nodes import label_reading as lr
from pill_bottle_nodes.label_steps import LabelSteps, copy_plan, regrasp_plan

S = dict(lr.LABEL_SETTINGS, label_settle_s=0.0)


# ---------------- the simulated robot ----------------

FIELD_COLORS = {                       # BGR, one per label field
    "drug_name": (0, 0, 255), "strength": (0, 255, 0), "patient_name": (255, 0, 0),
    "rx_number": (0, 255, 255), "directions": (255, 0, 255),
}
UP_W, UP_H = 480, 640                  # upright head image (the raw camera image is 640 x 480, turned cw)


class World:
    """A bottle in the claw in front of the head camera.

    label: {field: (center_deg, width_deg)} around the bottle (0 deg faces the camera at wrist yaw 0).
    hidden: fields under the fingers (they turn with the bottle, so turning the wrist never shows them;
            a regrasp at a different height does).
    glare: the middle of the bottle is washed out until the lift moves.
    """

    def __init__(self, label, hidden=(), glare=False, lying=False, radius=40, length=160, regrasp_ok=True):
        self.label, self.hidden, self.glare, self.lying = dict(label), set(hidden), glare, lying
        self.R, self.L = radius, length
        self.joints = {"joint_wrist_yaw": 0.3, "joint_lift": 0.9}    # verify pose (yaw 0.3 rad on purpose)
        self.yaw0, self.lift0 = 0.3, 0.9
        self.regrasp_ok = regrasp_ok
        self.moves = []
        self.views = 0

    # -- what the camera sees --
    def column_color(self, theta_deg):
        """Color of the bottle surface at angle theta from the camera's line of sight."""
        yaw = math.degrees(self.joints["joint_wrist_yaw"] - self.yaw0)
        on_bottle = (theta_deg + yaw) % 360
        if self.glare and abs(self.joints["joint_lift"] - self.lift0) < 1e-6 and abs(theta_deg) < 50:
            return (255, 255, 255)
        for f, (c, w) in self.label.items():
            if f in self.hidden:
                continue
            if abs((on_bottle - c + 180) % 360 - 180) <= w / 2:
                return FIELD_COLORS[f]
        return (230, 230, 230)

    def upright_image(self):
        img = np.full((UP_H, UP_W, 3), 90, np.uint8)               # gray background
        patch = np.zeros((self.L, 2 * self.R, 3), np.uint8)       # bottle, axis vertical
        for x in range(2 * self.R):
            s = (x + 0.5 - self.R) / self.R
            patch[:, x] = self.column_color(math.degrees(math.asin(max(-1.0, min(1.0, s)))))
        if self.lying:
            patch = np.ascontiguousarray(np.rot90(patch, k=-1))
        h, w = patch.shape[:2]
        y0, x0 = 260, 200
        img[y0:y0 + h, x0:x0 + w] = patch
        return img, (x0, y0, x0 + w, y0 + h)

    def raw_image_and_box(self):
        up, box_up = self.upright_image()
        raw = np.ascontiguousarray(np.rot90(up, k=1))             # undo the cw turn
        return raw, lr._box_in_rotated_crop(box_up, UP_W, UP_H, "ccw")


class FakeClient:
    def __init__(self, world, detections=True):
        self.w, self.detections = world, detections
        self.joints = world.joints

    def position(self, frame):
        return (0.0, 0.0, 1.2) if frame == "link_head_tilt" else (0.1, -0.3, 0.9)

    def wait_for(self, key, timeout_s=5.0, newer_than=None):
        raw, box = self.w.raw_image_and_box()
        if key == "head_color":
            h, w = raw.shape[:2]
            return SimpleNamespace(encoding="bgr8", height=h, width=w, step=w * 3, data=raw.tobytes())
        if key == "detections":
            dets = [SimpleNamespace(name="pill_bottle", confidence=0.9, x_min=box[0], y_min=box[1], x_max=box[2],
                                    y_max=box[3])] if self.detections else []
            return SimpleNamespace(detections=dets)
        return None


class ColorReader:
    """Reads a field when at least min_cols columns of the image show its color (squeezed text is unreadable)."""
    name = "fake"

    def __init__(self, min_cols=20):
        self.min_cols = min_cols
        self.images = []

    def read(self, img):
        self.images.append(img)
        out = lr.empty_read()
        for f, col in FIELD_COLORS.items():
            match = np.abs(img.astype(int) - np.array(col)).sum(axis=2) < 60
            n = int((match.mean(axis=0) >= 0.2).sum())            # columns where the field shows
            if n >= self.min_cols:
                out[f] = f.upper()
        out["readable"] = any(out[f] for f in lr.FIELDS)
        out["confidence"] = "high" if out["readable"] else "low"
        return out


def run_world(world, settings=S, reader=None, detections=True):
    c = FakeClient(world, detections)
    reader = reader or ColorReader()

    def move(label, **joints):
        world.moves.append(dict(joints))
        world.joints.update({k: v for k, v in joints.items() if k in world.joints})

    steps = LabelSteps(c, settings, move, lambda t: None, reader, rotate="cw")

    def regrasp():
        world.hidden.clear()
        world.joints.update(joint_wrist_yaw=world.yaw0, joint_lift=world.lift0)
        steps.reset_reference()
        return world.regrasp_ok

    res = lr.run_label_sequence(steps.read_view, steps.set_yaw, steps.set_lift, regrasp, settings, lambda t: None)
    return res, reader


# ---------------- integration: simulated robot + real LabelSteps + real state machine ----------------

def test_reads_first_view_when_label_faces_camera():
    w = World({"drug_name": (-20, 40), "strength": (20, 40)})
    res, _ = run_world(w)
    assert res["status"] == "read" and res["views"] == 1
    assert res["fields"]["drug_name"] == "DRUG_NAME" and res["fields"]["strength"] == "STRENGTH"
    assert res["trace"] == ["READ_VIEW", "DONE"]


def test_turns_wrist_and_merges_views():
    # drug and strength on opposite sides, both too far round to read from the front: +45 deg shows the strength,
    # -45 deg shows the drug name, and the two views together are a complete read
    w = World({"drug_name": (-80, 30), "strength": (80, 30)})
    res, _ = run_world(w)
    assert res["status"] == "read" and res["views"] == 3
    assert res["trace"] == ["READ_VIEW", "ROTATE_WRIST +45", "READ_VIEW", "ROTATE_WRIST -45", "READ_VIEW",
                            "ROTATE_WRIST +0 (restore)", "DONE"]
    assert w.joints["joint_wrist_yaw"] == pytest.approx(w.yaw0)      # wrist put back for place()


def test_narrow_label_on_the_back_is_out_of_reach_of_90_deg_turns():
    # +-90 deg turns plus the +-65 deg flattening window see the bottle from -155 to +155 deg; a narrow label
    # right at the back (150-210 deg) is never seen, and a regrasp doesn't spin the bottle, so it ends unreadable.
    # Real pharmacy labels wrap most of the way round, so this is rare; label_yaw_offsets_deg can add 135 if the
    # bottle clears the mast at that angle (check by hand first).
    w = World({"drug_name": (165, 30), "strength": (195, 30)})
    res, _ = run_world(w)
    assert res["status"] == "unreadable" and "REGRASP" in res["trace"]
    assert w.joints["joint_wrist_yaw"] == pytest.approx(w.yaw0) and w.joints["joint_lift"] == pytest.approx(w.lift0)
    wider, _ = run_world(World({"drug_name": (165, 30), "strength": (195, 30)}),
                         dict(S, label_yaw_offsets_deg=[0.0, 45.0, -45.0, 90.0, -90.0, 135.0, -135.0]))
    assert wider["status"] == "read"


def test_glare_is_fixed_by_moving_the_lift():
    w = World({"drug_name": (-15, 30), "strength": (15, 30)}, glare=True)
    res, _ = run_world(w)
    assert res["status"] == "read" and res["lift_adjusted"] and res["regrasps"] == 0
    assert any(t.startswith("ADJUST_LIFT +6") for t in res["trace"])
    assert w.joints["joint_lift"] == pytest.approx(w.lift0)          # lift put back


def test_fingers_over_the_label_need_a_regrasp():
    w = World({"drug_name": (-20, 40), "strength": (20, 40)}, hidden={"strength"})
    res, _ = run_world(w)
    assert res["status"] == "read" and res["regrasps"] == 1 and res["lift_adjusted"]
    # yaw sweep (5) + lift views (2) + first view after the regrasp
    assert res["views"] == 8


def test_gives_up_and_says_so():
    w = World({"drug_name": (0, 40)}, hidden={"drug_name"})          # never visible, strength never printed
    res, _ = run_world(w)
    assert res["status"] == "unreadable"
    assert res["views"] == 5 + 2 + 5
    assert res["trace"][-1] == "MARK_UNREADABLE"
    assert w.joints["joint_wrist_yaw"] == pytest.approx(w.yaw0) and w.joints["joint_lift"] == pytest.approx(w.lift0)


def test_failed_regrasp_stops_the_ladder():
    w = World({"drug_name": (0, 40)}, hidden={"drug_name"}, regrasp_ok=False)
    res, _ = run_world(w)
    assert res["status"] == "regrasp_failed" and res["views"] == 7


def test_lying_bottle_is_turned_upright_before_reading():
    w = World({"drug_name": (-20, 40), "strength": (20, 40)}, lying=True)
    res, reader = run_world(w)
    assert res["status"] == "read" and res["views"] == 1
    h, wd = reader.images[0].shape[:2]
    assert h > wd                                                     # axis runs up and down after prepare_view


def test_no_detection_still_reads_the_whole_image():
    w = World({"drug_name": (-20, 40), "strength": (20, 40)})
    res, reader = run_world(w, detections=False)
    assert res["status"] == "read"
    assert reader.images[0].shape[:2] == (UP_H, UP_W)               # whole upright frame, not flattened


def test_flattening_rescues_text_near_the_edge():
    # strength sits 30-80 deg round the bottle. Unflattened, the curve squeezes it into ~19 columns (unreadable);
    # flattened, 30-65 deg becomes ~24 columns (readable). Wrist turns are off so only flattening can help.
    label = {"drug_name": (-15, 30), "strength": (55, 50)}
    no_turns = dict(S, label_yaw_offsets_deg=[0.0], label_lift_offsets_m=[], label_max_regrasps=0)
    flat, _ = run_world(World(label), no_turns)
    plain, _ = run_world(World(label), dict(no_turns, label_unwarp=False))
    assert flat["status"] == "read"
    assert plain["status"] == "unreadable" and plain["fields"]["strength"] is None


def test_wrist_limit_skips_that_view():
    w = World({"drug_name": (-80, 30), "strength": (80, 30)})
    c = FakeClient(w)
    steps = LabelSteps(c, S, lambda label, **j: (_ for _ in ()).throw(RuntimeError("outside limits"))
                       if "joint_wrist_yaw" in j else w.joints.update(j), lambda t: None, ColorReader(), rotate="cw")
    res = lr.run_label_sequence(steps.read_view, steps.set_yaw, steps.set_lift, lambda: False, S, lambda t: None)
    assert any("wrist couldn't reach" in e for e in res["errors"])
    assert res["views"] >= 1


def test_stop_request_is_not_swallowed():
    w = World({"drug_name": (-80, 30), "strength": (80, 30)})
    c = FakeClient(w)

    def move(label, **j):
        raise RuntimeError("stopped at your request")
    steps = LabelSteps(c, S, move, lambda t: None, ColorReader(), rotate="cw")
    with pytest.raises(RuntimeError, match="stopped at your request"):
        lr.run_label_sequence(steps.read_view, steps.set_yaw, steps.set_lift, lambda: True, S, lambda t: None)


def test_camera_error_counts_as_an_empty_view():
    calls = {"n": 0}

    def read_view():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("no head image")
        return {"drug_name": "X", "strength": "1MG", "confidence": "high"}
    res = lr.run_label_sequence(read_view, lambda y: True, lambda z: True, lambda: True, S, lambda t: None)
    assert res["status"] == "read" and res["views"] == 2 and "no head image" in res["errors"][0]


def test_low_confidence_is_not_enough():
    reads = iter([{"drug_name": "X", "strength": "1MG", "confidence": "low"}] * 3 +
                 [{"drug_name": "X", "strength": "1MG", "confidence": "medium"}] * 20)
    res = lr.run_label_sequence(lambda: next(reads), lambda y: True, lambda z: True, lambda: True, S, lambda t: None)
    assert res["status"] == "read" and res["views"] == 4


# ---------------- the curve ----------------

def _wrap_label_on_cylinder(flat, radius):
    """Render a flat label wrapped round a cylinder, as an orthographic camera sees it (centered on the label)."""
    h, wf = flat.shape[:2]
    out = np.zeros((h, 2 * radius), flat.dtype)
    for x in range(2 * radius):
        theta = math.asin(max(-1.0, min(1.0, (x + 0.5 - radius) / radius)))
        u = int(round(wf / 2 + radius * theta))
        if 0 <= u < wf:
            out[:, x] = flat[:, u]
    return out


def _stripe_centers(img):
    col = img.mean(axis=0) > 127
    centers, start = [], None
    for i, on in enumerate(list(col) + [False]):
        if on and start is None:
            start = i
        elif not on and start is not None:
            centers.append((start + i - 1) / 2)
            start = None
    return np.array(centers)


def test_unwarp_restores_even_spacing():
    radius = 120
    flat = np.zeros((40, int(math.pi * radius)), np.uint8)
    flat[:, ::12] = 255                                             # evenly spaced stripes
    flat[:, 1::12] = 255
    seen = _wrap_label_on_cylinder(flat, radius)
    fixed = lr.unwarp_cylinder(seen, max_angle_deg=65)
    gaps_seen, gaps_fixed = np.diff(_stripe_centers(seen)), np.diff(_stripe_centers(fixed))
    assert gaps_seen.min() < 0.6 * gaps_seen.max()                  # squeezed at the edges
    assert gaps_fixed.std() < 1.0 and gaps_fixed.mean() == pytest.approx(12, abs=0.6)
    assert fixed.shape[1] == pytest.approx(2 * math.radians(65) * (2 * radius - 1) / 2, abs=1)


def test_unwarp_keeps_the_middle_column():
    img = np.zeros((10, 81, 3), np.uint8)
    img[:, 40] = (255, 255, 255)
    out = lr.unwarp_cylinder(img, 65)
    assert out[:, out.shape[1] // 2].mean() > 100


# ---------------- pieces ----------------

def test_box_in_rotated_crop_matches_numpy():
    img = np.zeros((30, 50), np.uint8)
    img[5:12, 20:28] = 1                                            # box x 20..28, y 5..12 (end-exclusive)
    for rot, k in (("cw", -1), ("ccw", 1)):
        r = np.rot90(img, k=k)
        ys, xs = np.nonzero(r)
        x1, y1, x2, y2 = lr._box_in_rotated_crop((20, 5, 28, 12), 50, 30, rot)
        assert (x1, y1, x2, y2) == (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)


def test_pick_box_prefers_the_bottle_in_the_claw():
    dets = [("pill_bottle", 0.95, (0, 0, 20, 40)),        # small: on the table behind
            ("pill_bottle", 0.60, (100, 100, 200, 300)),  # big: in the claw
            ("label", 0.99, (110, 150, 190, 250))]
    assert lr.pick_box(dets)[2] == (100, 100, 200, 300)
    assert lr.pick_box([("label", 0.5, (1, 2, 3, 4))])[0] == "label"
    assert lr.pick_box([("pill_bottle", 0.1, (1, 2, 3, 4))]) is None


def test_parse_json_reply_handles_fences():
    assert lr.parse_json_reply('```json\n{"drug_name": "A"}\n```') == {"drug_name": "A"}
    assert lr.parse_json_reply('Here you go: {"strength": "5MG"} done') == {"strength": "5MG"}


def test_normalize_read():
    r = lr.normalize_read({"drug_name": " null ", "strength": "200MG", "confidence": "HIGH", "_source": "x"})
    assert r["drug_name"] is None and r["strength"] == "200MG" and r["confidence"] == "high" and r["_source"] == "x"
    assert set(lr.FIELDS) <= set(r) and r["readable"]
    assert lr.normalize_read("garbage")["confidence"] == "low"


def test_merge_keeps_the_most_confident_value():
    m = lr.merge_reads([{"drug_name": "FUNKIT0L", "confidence": "low"},
                        {"drug_name": "FUNKITOL", "strength": "200MG", "confidence": "high"},
                        {"strength": "2OOMG", "confidence": "medium"}])
    assert m["drug_name"] == "FUNKITOL" and m["strength"] == "200MG"
    assert lr.is_complete(m)
    assert not lr.is_complete(lr.merge_reads([{"drug_name": "A", "strength": "1MG", "confidence": "low"}]))


def test_ocr_fields_from_the_test_bottle():
    lines = [("PHONE: (555) 555-5626", 0.8), ("YONKERS PHARMACY", 0.9), ("Rx: 05242024", 0.85),
             ("ASHLEY J. NICHOLS", 0.9), ("TAKE ONE TABLET BY MOUTH TWICE DAILY AS NEEDED", 0.8),
             ("FUNKITOL 200MG TABLETS", 0.9)]
    r = lr.ocr_lines_to_fields(lines)
    assert r["drug_name"] == "FUNKITOL" and r["strength"] == "200MG" and r["rx_number"] == "05242024"
    assert r["directions"].upper().startswith("TAKE ONE TABLET") and r["confidence"] == "medium"
    assert r["patient_name"] is None


class _Boom:
    def __init__(self, name):
        self.name, self.calls = name, 0

    def read(self, img):
        self.calls += 1
        raise RuntimeError("503 UNAVAILABLE")


class _Ok:
    name = "easyocr"

    def read(self, img):
        return {"drug_name": "A", "strength": "1MG", "confidence": "medium"}


def test_fallback_uses_easyocr_and_skips_gemini_for_a_while():
    t = {"now": 0.0}
    g = _Boom("gemini")
    fr = lr.FallbackReader([g, _Ok()], cooldown_s=60, clock=lambda: t["now"])
    r = fr.read(None)
    assert r["_source"] == "easyocr" and "503" in r["_errors"][0]
    fr.read(None)
    assert g.calls == 1                                             # skipped while cooling down
    t["now"] = 61
    fr.read(None)
    assert g.calls == 2                                             # tried again later


def test_fallback_never_skips_the_last_reader_and_raises_when_all_fail():
    only = _Boom("easyocr")
    fr = lr.FallbackReader([only], cooldown_s=60)
    for _ in range(2):
        with pytest.raises(RuntimeError, match="every label reader failed"):
            fr.read(None)
    assert only.calls == 2
    with pytest.raises(RuntimeError):
        lr.FallbackReader([])


def test_gemini_backend_moves_on_from_a_busy_model(monkeypatch):
    pytest.importorskip("google.genai")
    monkeypatch.setattr(lr.time, "sleep", lambda s: None)
    calls = []

    class Models:
        def list(self):
            return [SimpleNamespace(name=f"models/{n}", supported_actions=["generateContent"])
                    for n in ("gemini-flash-lite-latest", "gemini-9-flash", "gemini-9-flash-image", "gemini-9-pro")]

        def generate_content(self, model, contents, config):
            calls.append(model)
            if model == "gemini-flash-lite-latest":
                raise RuntimeError("503 UNAVAILABLE")
            return SimpleNamespace(text='{"drug_name": "FUNKITOL", "strength": "200MG", "confidence": "high"}')

    g = lr.GeminiBackend(models=["gemini-flash-lite-latest"], client=SimpleNamespace(models=Models()))
    assert g.models() == ["gemini-flash-lite-latest", "gemini-9-flash"]   # no image/pro models
    r = g.read(np.zeros((20, 20, 3), np.uint8))
    assert r["drug_name"] == "FUNKITOL" and r["_source"] == "gemini:gemini-9-flash"
    assert calls == ["gemini-flash-lite-latest", "gemini-9-flash"]


def test_gemini_needs_a_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        lr.GeminiBackend()


def test_regrasp_plan_moves_the_grip():
    plan = SimpleNamespace(lift=0.75, lift_clear=0.87, arm_grasp=0.35, arm_pregrasp=0.2, base_translate=0.0,
                           wrist_yaw=0.0, grasp_depth=0.05)
    s = {"regrasp_dz_m": 0.025, "regrasp_shift_m": 0.02}
    up = regrasp_plan(plan, "upright", s)
    assert up.lift == pytest.approx(0.775) and up.lift_clear == pytest.approx(0.895) and up.arm_grasp == 0.35
    lying = regrasp_plan(plan, "lying", s)
    assert lying.arm_grasp == pytest.approx(0.33) and lying.lift == 0.75
    assert plan.lift == 0.75                                        # original untouched
    assert copy_plan(plan, lift=1.0).lift == 1.0


# ---------------- optional: the real Gemini on a real picture ----------------

@pytest.mark.skipif(not (__import__("os").environ.get("GEMINI_API_KEY") and __import__("os").environ.get("LABEL_TEST_IMAGE")),
                    reason="set GEMINI_API_KEY and LABEL_TEST_IMAGE=/path/to/bottle.jpg (+ LABEL_TEST_DRUG) to run")
def test_real_gemini_reads_the_test_bottle():
    import os

    import cv2
    img = cv2.imread(os.environ["LABEL_TEST_IMAGE"])
    r = lr.FallbackReader([lr.GeminiBackend(models=S["label_gemini_models"])]).read(img)
    assert r["readable"]
    if os.environ.get("LABEL_TEST_DRUG"):
        assert os.environ["LABEL_TEST_DRUG"].upper() in str(r["drug_name"]).upper()
