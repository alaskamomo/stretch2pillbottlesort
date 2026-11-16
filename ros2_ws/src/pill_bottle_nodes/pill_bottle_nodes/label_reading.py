# label_reading.py: read the pill bottle label at the head camera (ticket T16).
# No ROS in this file, so all of it is unit tested (test/test_label_reading.py).
#
#   prepare_view        crop the bottle from the RAW head image, turn it upright, flatten the curved label
#   unwarp_cylinder     the flattening: undoes the squeeze at the left and right edges of a round bottle
#   GeminiBackend       Gemini reads the label and returns the fields as JSON (needs GEMINI_API_KEY)
#   EasyOcrBackend      offline OCR, used when Gemini is down, busy or has no key
#   FallbackReader      tries the backends in order
#   merge_reads         combine what each view read (the fingers hide a different strip in each view)
#   run_label_sequence  rotate the wrist -> adjust the lift -> regrasp -> give up (the T16 state machine)
import json
import math
import os
import re
import time

import numpy as np

FIELDS = ("drug_name", "strength", "patient_name", "rx_number", "directions")
CONF_RANK = {"low": 0, "medium": 1, "high": 2}

# Defaults; task_manager and test_label expose each as a ROS parameter.
LABEL_SETTINGS = {
    "read_label": True,
    "label_required": ["drug_name", "strength"],    # the read counts only when these are found...
    "label_min_confidence": "medium",               # ...at this confidence or better
    "label_yaw_offsets_deg": [0.0, 45.0, -45.0, 90.0, -90.0],   # wrist yaw views, relative to the verify pose
    "label_lift_offsets_m": [0.06],                 # after the yaw views: move the lift (changes glare)...
    "label_lift_yaws_deg": [0.0, 90.0],             # ...and read again at these yaws
    "label_max_regrasps": 1,                        # then put it down and grab it at a different height
    "label_unwarp": True,                           # flatten the curved label before reading
    "label_unwarp_max_deg": 65.0,                   # use the bottle surface up to this angle from the middle
    "label_crop_pad": 0.15,                         # pad the bottle box by this fraction above and below
    "label_settle_s": 0.6,                          # wait after a move so the image isn't blurred
    "label_gemini_models": ["gemini-flash-lite-latest", "gemini-3.8-flash"],   # tried first, then any other Flash
    "label_gemini_timeout_s": 20.0,
    "label_max_models": 3,                          # Gemini models to try per view before EasyOCR
    "label_use_easyocr": True,
    "label_ocr_gpu": False,
}

PROMPT = """Read this pill bottle label. Return JSON only:
{"drug_name": str|null, "strength": str|null, "patient_name": str|null,
 "rx_number": str|null, "directions": str|null, "readable": bool, "confidence": "high"|"medium"|"low"}
Use null for anything you can't read clearly. Do not guess. The label may be partly hidden by the gripper."""


# ---------------- image preparation ----------------

def rotate_upright(img, rotate):
    """Same rotation as geometry.rotate_image (kept here so this file has no package imports)."""
    if rotate == "cw":
        return np.ascontiguousarray(np.rot90(img, k=-1))
    if rotate == "ccw":
        return np.ascontiguousarray(np.rot90(img, k=1))
    if rotate == "180":
        return np.ascontiguousarray(img[::-1, ::-1])
    return img


def unwarp_cylinder(img, max_angle_deg=65.0):
    """Flatten a cylinder whose axis runs up and down the image and whose sides touch the image's left and right
    edges (crop tightly to the bottle's width first).

    A point at angle theta around the bottle shows up at x = cx + r*sin(theta), so text near the edges is squeezed.
    This samples the image at equal steps of theta instead, which gives equal spacing along the label. Edges past
    max_angle_deg are dropped: they're too squeezed to read and the next wrist view sees them face-on.
    """
    import cv2
    h, w = img.shape[:2]
    if w < 8:
        return img
    a = math.radians(max(5.0, min(89.0, max_angle_deg)))
    r = (w - 1) / 2.0
    out_w = max(8, int(round(2 * a * r)))           # arc length of the kept part, in pixels at radius r
    theta = np.linspace(-a, a, out_w, dtype=np.float64)
    map_x = np.tile((r + r * np.sin(theta)).astype(np.float32), (h, 1))
    map_y = np.tile(np.arange(h, dtype=np.float32)[:, None], (1, out_w))
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def pick_box(detections, names=("pill_bottle", "label"), min_conf=0.25):
    """Box to read from. detections: [(name, conf, (x1, y1, x2, y2)), ...] in RAW image pixels.
    Prefers the whole bottle (its width gives the radius for flattening), then the label box. Among those, the
    BIGGEST box wins: the bottle in the claw is 30-50 cm from the camera, so it's much bigger than any bottle
    still on the table behind it."""
    def area(d):
        x1, y1, x2, y2 = d[2]
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)
    for name in names:
        cands = [d for d in detections if d[0] == name and d[1] >= min_conf]
        if cands:
            return max(cands, key=area)
    return None


def prepare_view(raw_bgr, box_raw=None, rotate="cw", unwarp=True, max_angle_deg=65.0, pad=0.15):
    """RAW head image (+ bottle box in raw pixels) -> dict(image, crop, axis, unwarped).

    The crop is padded above and below the bottle but kept tight at the sides, so its width is the bottle's
    diameter (what unwarp_cylinder needs). A bottle held lying down shows up wider than tall; it's turned 90 deg
    so its axis runs up and down before flattening.
    """
    H, W = raw_bgr.shape[:2]
    if box_raw is None:
        return {"image": rotate_upright(raw_bgr, rotate), "crop": None, "axis": "unknown", "unwarped": False}
    x1, y1, x2, y2 = [float(v) for v in box_raw]
    # Pad along the bottle's axis only. In the RAW image the axis direction depends on the camera rotation,
    # so pad after turning upright: crop generously in raw, rotate, then trim the sides back to the box.
    bw, bh = x2 - x1, y2 - y1
    px, py = pad * bw, pad * bh
    cx1, cy1 = int(max(0, math.floor(x1 - px))), int(max(0, math.floor(y1 - py)))
    cx2, cy2 = int(min(W, math.ceil(x2 + px))), int(min(H, math.ceil(y2 + py)))
    crop = rotate_upright(raw_bgr[cy1:cy2, cx1:cx2], rotate)
    # where the box sits inside the rotated crop
    bx1, by1, bx2, by2 = _box_in_rotated_crop((x1 - cx1, y1 - cy1, x2 - cx1, y2 - cy1),
                                              cx2 - cx1, cy2 - cy1, rotate)
    ch, cw = crop.shape[:2]
    if (bx2 - bx1) > 1.1 * (by2 - by1):              # wider than tall: bottle is lying -> make its axis vertical
        crop = np.ascontiguousarray(np.rot90(crop, k=1))
        bx1, by1, bx2, by2 = by1, cw - bx2, by2, cw - bx1
        axis = "horizontal"
    else:
        axis = "vertical"
    ch, cw = crop.shape[:2]
    tight = crop[:, int(max(0, round(bx1))):int(min(cw, round(bx2)))]
    if tight.shape[1] < 8 or tight.shape[0] < 8:
        return {"image": crop, "crop": crop, "axis": axis, "unwarped": False}
    flat = unwarp_cylinder(tight, max_angle_deg) if unwarp else tight
    return {"image": flat, "crop": tight, "axis": axis, "unwarped": bool(unwarp)}


def _box_in_rotated_crop(box, w, h, rotate):
    """Box (x1, y1, x2, y2) in a w x h image -> the same box after rotate_upright."""
    x1, y1, x2, y2 = box
    if rotate == "cw":          # raw (u, v) -> rotated (h-1-v, u)
        return h - y2, x1, h - y1, x2
    if rotate == "ccw":         # raw (u, v) -> rotated (v, w-1-u)
        return y1, w - x2, y2, w - x1
    if rotate == "180":
        return w - x2, h - y2, w - x1, h - y1
    return x1, y1, x2, y2


# ---------------- readers ----------------

def empty_read():
    out = {f: None for f in FIELDS}
    out.update(readable=False, confidence="low")
    return out


def normalize_read(d):
    """Make any backend's answer look the same: every field present, blanks -> None, valid confidence."""
    out = empty_read()
    if isinstance(d, dict):
        for f in FIELDS:
            v = d.get(f)
            if isinstance(v, str):
                v = v.strip()
                if v.lower() in ("", "null", "none", "n/a", "unknown"):
                    v = None
            out[f] = v if v not in ("", []) else None
        c = str(d.get("confidence", "low")).lower()
        out["confidence"] = c if c in CONF_RANK else "low"
        out["readable"] = bool(d.get("readable", any(out[f] for f in FIELDS)))
        for k, v in d.items():
            if k.startswith("_") or k == "raw_text":
                out[k] = v
    return out


def parse_json_reply(text):
    """Gemini's JSON, even if it came wrapped in ``` fences or with extra words around it."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", t, re.S)
        if m:
            return json.loads(m.group(0))
        raise


def _is_busy(err):
    s = str(err)
    return any(k in s for k in ("503", "429", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "DEADLINE", "imeout"))


class GeminiBackend:
    """Gemini reads the label. The key comes from GEMINI_API_KEY (or GOOGLE_API_KEY); never put it in the repo."""
    name = "gemini"

    def __init__(self, api_key=None, models=(), timeout_s=20.0, max_models=3, client=None):
        if client is None:
            key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise RuntimeError("no Gemini key: run  export GEMINI_API_KEY=...  (see the T16 ticket)")
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))
        self.client = client
        self.preferred = list(models)
        self.max_models = max_models
        self._models = None

    def models(self):
        """Preferred models first, then any other Flash model this key can use (the list changes over time)."""
        if self._models is None:
            names = []
            try:
                for m in self.client.models.list():
                    if "generateContent" in (getattr(m, "supported_actions", None) or []):
                        names.append(m.name.replace("models/", ""))
            except Exception:
                names = []
            skip = ("image", "tts", "live", "audio", "embedding")
            flash = sorted((n for n in names if "flash" in n and not any(s in n for s in skip)), reverse=True)
            ordered = []
            for n in self.preferred + flash:
                if n not in ordered and (not names or n in names or n in self.preferred):
                    ordered.append(n)
            self._models = ordered
        return self._models

    def read(self, img_bgr):
        import cv2
        from google.genai import types
        ok, jpg = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if not ok:
            raise RuntimeError("couldn't encode the image")
        part = types.Part.from_bytes(data=jpg.tobytes(), mime_type="image/jpeg")
        cfg = types.GenerateContentConfig(response_mime_type="application/json")
        last = None
        for model in self.models()[:self.max_models]:
            try:
                resp = self.client.models.generate_content(model=model, contents=[part, PROMPT], config=cfg)
                out = normalize_read(parse_json_reply(resp.text))
                out["_source"] = f"gemini:{model}"
                return out
            except Exception as e:      # busy -> next model; anything else -> also next model, keep the error
                last = e
                if _is_busy(e):
                    time.sleep(1.0)
        raise RuntimeError(f"Gemini failed on {self.models()[:self.max_models]}: {last}")


_STRENGTH = re.compile(r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|g|ml|iu|%)\b", re.I)
_DRUG_BEFORE_STRENGTH = re.compile(r"\b([A-Za-z][A-Za-z\-]{3,})\s+\d+(?:\.\d+)?\s?(?:mg|mcg|g|ml|iu)\b", re.I)
_RX = re.compile(r"\bRx\s*(?:#|no\.?|num(?:ber)?)?\s*:?\s*([A-Z0-9][A-Z0-9\-]{3,})", re.I)
_DIRECTIONS = re.compile(r"\b(take|apply|use|inhale|place|instill|chew|dissolve)\b.{3,120}?"
                         r"\b(daily|a day|hours|needed|bedtime|morning|evening|meals?)\b", re.I)
_NOT_DRUG = {"take", "tablet", "tablets", "capsule", "capsules", "each", "every", "daily", "twice", "once"}


def ocr_lines_to_fields(lines):
    """[(text, conf), ...] from OCR -> label fields with simple patterns. Confidence is medium only when it found
    both a drug name and a strength with good OCR confidence; OCR can't tell who the patient is, so that's None."""
    texts = [t for t, _ in lines]
    full = " ".join(texts)
    out = empty_read()
    s = _STRENGTH.search(full)
    out["strength"] = s.group(0).replace(" ", "").upper() if s else None
    d = next((m for m in _DRUG_BEFORE_STRENGTH.finditer(full) if m.group(1).lower() not in _NOT_DRUG), None)
    out["drug_name"] = d.group(1).upper() if d else None
    r = _RX.search(full)
    out["rx_number"] = r.group(1) if r else None
    dr = _DIRECTIONS.search(full)
    out["directions"] = dr.group(0) if dr else None
    mean_conf = float(np.mean([c for _, c in lines])) if lines else 0.0
    out["readable"] = bool(texts)
    out["confidence"] = "medium" if (out["drug_name"] and out["strength"] and mean_conf >= 0.6) else "low"
    out["raw_text"] = texts
    return out


class EasyOcrBackend:
    """Offline OCR (no key, no internet). Loads its model once (the first time is slow: it downloads it)."""
    name = "easyocr"

    def __init__(self, gpu=False, reader=None, min_conf=0.3):
        if reader is None:
            import easyocr
            reader = easyocr.Reader(["en"], gpu=gpu)
        self.reader = reader
        self.min_conf = min_conf

    def read(self, img_bgr):
        res = self.reader.readtext(img_bgr[:, :, ::-1].copy(), rotation_info=[180])   # label may be upside down
        lines = [(t, float(c)) for _, t, c in res if c >= self.min_conf]
        out = ocr_lines_to_fields(lines)
        out["_source"] = "easyocr"
        return out


class FallbackReader:
    """Tries each backend in order; the first that answers wins. Records what failed in out['_errors'].

    A backend that fails is skipped for cooldown_s (unless it's the last one left), so a Gemini outage costs one
    slow view, not one per view, and the robot keeps going on EasyOCR."""

    def __init__(self, backends, cooldown_s=60.0, clock=time.monotonic):
        if not backends:
            raise RuntimeError("no label reader available (no Gemini key and EasyOCR isn't installed)")
        self.backends = list(backends)
        self.cooldown_s = cooldown_s
        self.clock = clock
        self.down_until = {}

    def read(self, img_bgr):
        errors = []
        now = self.clock()
        for i, b in enumerate(self.backends):
            last = i == len(self.backends) - 1
            if not last and self.down_until.get(b.name, 0.0) > now:
                errors.append(f"{b.name}: skipped (failed recently)")
                continue
            try:
                out = normalize_read(b.read(img_bgr))
                out.setdefault("_source", b.name)
                self.down_until.pop(b.name, None)
                if errors:
                    out["_errors"] = errors
                return out
            except Exception as e:
                errors.append(f"{b.name}: {e}")
                self.down_until[b.name] = self.clock() + self.cooldown_s
        raise RuntimeError("every label reader failed: " + " | ".join(errors))


def build_reader(s, say=print):
    """Gemini first (if there's a key), EasyOCR second (if installed), from LABEL_SETTINGS-style settings."""
    backends = []
    try:
        backends.append(GeminiBackend(models=s["label_gemini_models"], timeout_s=s["label_gemini_timeout_s"],
                                      max_models=s["label_max_models"]))
    except Exception as e:
        say(f"   Gemini off: {e}")
    if s["label_use_easyocr"]:
        try:
            backends.append(EasyOcrBackend(gpu=s["label_ocr_gpu"]))
        except Exception as e:
            say(f"   EasyOCR off: {e}  (pip3 install easyocr)")
    say("   Label readers: " + (", ".join(b.name for b in backends) or "NONE"))
    return FallbackReader(backends)


# ---------------- combining views ----------------

def merge_reads(reads):
    """Combine the reads from several views. For each field keep the value from the most confident read
    (earliest on a tie). Returns the fields plus '_field_conf' (confidence each field came with)."""
    merged = empty_read()
    field_conf = {}
    for f in FIELDS:
        best = None
        for r in reads:
            v = r.get(f)
            if v in (None, ""):
                continue
            rank = CONF_RANK.get(r.get("confidence", "low"), 0)
            if best is None or rank > best[0]:
                best = (rank, v, r.get("confidence", "low"))
        if best:
            merged[f] = best[1]
            field_conf[f] = best[2]
    merged["readable"] = any(merged[f] for f in FIELDS)
    merged["confidence"] = min(field_conf.values(), key=lambda c: CONF_RANK[c]) if field_conf else "low"
    merged["_field_conf"] = field_conf
    return merged


def is_complete(merged, required=("drug_name", "strength"), min_conf="medium"):
    fc = merged.get("_field_conf", {})
    need = CONF_RANK.get(min_conf, 1)
    return all(merged.get(f) and CONF_RANK.get(fc.get(f, "low"), 0) >= need for f in required)


# ---------------- the T16 state machine ----------------

def run_label_sequence(read_view, set_yaw, set_lift, regrasp, s, say=print):
    """Read the label with the bottle held at the head camera (the verify pose).

    read_view()        -> one read dict (may raise: no image, every reader failed)
    set_yaw(offset)    -> True if the wrist moved to verify_yaw + offset (rad)
    set_lift(offset)   -> True if the lift moved to verify_lift + offset (m)
    regrasp()          -> True if the bottle was put down, grabbed at a different height and is back at the
                          verify pose; False if that failed (the bottle may not be in the claw any more)

    States: READ_VIEW -> ROTATE_WRIST (more yaws) -> ADJUST_LIFT -> REGRASP -> MARK_UNREADABLE.
    Stops as soon as the merged read has the required fields. Always puts the wrist and lift back to the verify
    pose before returning (place() doesn't touch the wrist yaw).

    Returns dict(status = read | unreadable | regrasp_failed, fields, views, lift_adjusted, regrasps, sources,
                 trace, errors)
    """
    required = tuple(s["label_required"])
    reads, trace, errors = [], [], []
    state = {"yaw": 0.0, "lift": 0.0, "lift_adjusted": False, "regrasps": 0}

    def done():
        return is_complete(merge_reads(reads), required, s["label_min_confidence"])

    def view(yaw_deg):
        yaw = math.radians(yaw_deg)
        if abs(yaw - state["yaw"]) > 1e-6:
            trace.append(f"ROTATE_WRIST {yaw_deg:+.0f}")
            if not set_yaw(yaw):
                errors.append(f"wrist couldn't reach {yaw_deg:+.0f} deg")
                return False
            state["yaw"] = yaw
        trace.append("READ_VIEW")
        try:
            r = normalize_read(read_view())
        except Exception as e:
            errors.append(f"view at {yaw_deg:+.0f} deg: {e}")
            r = empty_read()
        r.update(_yaw_deg=yaw_deg, _lift_m=state["lift"], _grasp=state["regrasps"])
        reads.append(r)
        got = [f for f in FIELDS if r.get(f)]
        say(f"   view {len(reads)} (yaw {yaw_deg:+.0f}, lift {state['lift'] * 100:+.0f} cm): "
            f"{', '.join(got) or 'nothing'} [{r.get('_source', '?')}, {r.get('confidence')}]")
        return done()

    def sweep(yaws):
        return any(view(y) for y in yaws)

    def finish(status):
        if abs(state["yaw"]) > 1e-6:
            trace.append("ROTATE_WRIST +0 (restore)")
            set_yaw(0.0)
        if abs(state["lift"]) > 1e-6:
            trace.append("ADJUST_LIFT +0 (restore)")
            set_lift(0.0)
        trace.append({"read": "DONE", "unreadable": "MARK_UNREADABLE", "regrasp_failed": "REGRASP_FAILED"}[status])
        merged = merge_reads(reads)
        return {"status": status, "fields": {f: merged[f] for f in FIELDS}, "field_conf": merged["_field_conf"],
                "views": len(reads), "lift_adjusted": state["lift_adjusted"], "regrasps": state["regrasps"],
                "sources": sorted({r.get("_source", "?") for r in reads}), "trace": trace, "errors": errors}

    # 1) yaw sweep at the verify pose
    if sweep(s["label_yaw_offsets_deg"]):
        return finish("read")
    # 2) lift change (glare / reflections), a couple of yaws each
    for dz in s["label_lift_offsets_m"]:
        trace.append(f"ADJUST_LIFT {dz * 100:+.0f} cm")
        if not set_lift(dz):
            errors.append(f"lift couldn't move {dz * 100:+.0f} cm")
            continue
        state["lift"], state["lift_adjusted"] = dz, True
        if sweep(s["label_lift_yaws_deg"]):
            return finish("read")
    if abs(state["lift"]) > 1e-6:
        set_lift(0.0)
        state["lift"] = 0.0
    # 3) regrasp at a different height, then the yaw sweep again
    for _ in range(int(s["label_max_regrasps"])):
        if abs(state["yaw"]) > 1e-6:
            set_yaw(0.0)
            state["yaw"] = 0.0
        trace.append("REGRASP")
        state["regrasps"] += 1
        if not regrasp():
            return finish("regrasp_failed")
        if sweep(s["label_yaw_offsets_deg"]):
            return finish("read")
    return finish("unreadable")


# ---------------- offline check on saved pictures (no robot, no ROS) ----------------

def main(argv=None):
    """Run the image half of T16 on saved pictures: crop -> flatten -> read. Works on a laptop or in Colab.

      python3 -m pill_bottle_nodes.label_reading pics/*.jpg --detector best.pt --compare \\
          --expected-drug FUNKITOL --expected-strength 200MG

    --compare reads every picture with and without flattening, to show whether flattening helps.
    Pictures are used as they are (--rotate none); for raw robot frames use --rotate cw.
    """
    import argparse
    import glob

    import cv2
    ap = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--detector", default="", help="YOLO detection model with a pill_bottle (and label) class")
    ap.add_argument("--rotate", default="none", choices=["none", "cw", "ccw", "180"])
    ap.add_argument("--compare", action="store_true", help="read each picture flattened AND not flattened")
    ap.add_argument("--no-unwarp", action="store_true")
    ap.add_argument("--max-deg", type=float, default=LABEL_SETTINGS["label_unwarp_max_deg"])
    ap.add_argument("--expected-drug", default="")
    ap.add_argument("--expected-strength", default="")
    ap.add_argument("--no-easyocr", action="store_true")
    ap.add_argument("--out", default=os.path.expanduser("~/pill_tests/offline"))
    a = ap.parse_args(argv)

    s = dict(LABEL_SETTINGS, label_use_easyocr=not a.no_easyocr)
    reader = build_reader(s)
    model = None
    if a.detector:
        from ultralytics import YOLO
        model = YOLO(a.detector)
    paths = sorted({p for pat in a.images for p in glob.glob(os.path.expanduser(pat))})
    os.makedirs(a.out, exist_ok=True)
    modes = [True, False] if a.compare else [not a.no_unwarp]
    score = {m: [0, 0] for m in modes}
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            print(f"{path}: can't open")
            continue
        box = None
        if model is not None:
            up = rotate_upright(img, a.rotate)
            res = model(up, verbose=False)[0]
            dets = []
            for b in res.boxes:
                x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
                dets.append((model.names[int(b.cls)], float(b.conf), (x1, y1, x2, y2)))
            box = pick_box(dets)
            if box is not None and a.rotate != "none":
                # boxes were found on the upright picture; prepare_view wants raw pixels
                inv = {"cw": "ccw", "ccw": "cw", "180": "180"}[a.rotate]
                h, w = up.shape[:2]
                box = (box[0], box[1], _box_in_rotated_crop(box[2], w, h, inv))
        print(f"\n{os.path.basename(path)}  box: {box[0] + ' ' + format(box[1], '.2f') if box else 'none (whole image)'}")
        for unwarp in modes:
            v = prepare_view(img, box[2] if box else None, a.rotate, unwarp, a.max_deg, s["label_crop_pad"])
            tag = "flat" if unwarp else "plain"
            cv2.imwrite(os.path.join(a.out, f"{os.path.splitext(os.path.basename(path))[0]}_{tag}.jpg"), v["image"])
            try:
                r = reader.read(v["image"])
            except Exception as e:
                print(f"  {tag:5s}: ERROR {e}")
                continue
            ok = None
            if a.expected_drug or a.expected_strength:
                drug = re.sub(r"[^A-Z0-9.]", "", str(r.get("drug_name") or "").upper())
                strength = re.sub(r"[^A-Z0-9.]", "", str(r.get("strength") or "").upper())
                ed = re.sub(r"[^A-Z0-9.]", "", a.expected_drug.upper())
                es = re.sub(r"[^A-Z0-9.]", "", a.expected_strength.upper())
                ok = (not ed or (bool(drug) and (ed in drug or drug in ed))) and (not es or strength == es)
                score[unwarp][0] += ok
                score[unwarp][1] += 1
            print(f"  {tag:5s}: {r.get('drug_name')} | {r.get('strength')} | {r.get('patient_name')} | "
                  f"Rx {r.get('rx_number')} | {r.get('confidence')} [{r.get('_source')}]"
                  + ("" if ok is None else ("  CORRECT" if ok else "  WRONG")))
    if any(n for _, n in score.values()):
        print("\nCorrect drug + strength:")
        for unwarp, (right, n) in score.items():
            print(f"  {'flattened' if unwarp else 'not flattened':14s} {right}/{n}")
    print(f"\nImages that were read: {a.out}")


if __name__ == "__main__":
    main()
