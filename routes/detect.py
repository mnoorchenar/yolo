import os, base64, io, traceback
import numpy as np
import torch
from flask import Blueprint, request, jsonify, render_template
from PIL import Image

detect_bp = Blueprint("detect", __name__)

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE = 0 if torch.cuda.is_available() else "cpu"
DEVICE_LABEL = f"GPU (cuda:{DEVICE})" if DEVICE != "cpu" else "CPU"

# ── Model cache ───────────────────────────────────────────────────────────────
_cache: dict = {"path": None, "model": None}


def _load_model(model_path: str | None = None):
    from ultralytics import YOLO

    # Only use a custom model when the caller explicitly supplies a valid path.
    # An empty / missing path ALWAYS falls back to the pretrained COCO weights so
    # that selecting "Pretrained YOLO11n" in the UI never silently loads a custom
    # model that was trained on a different, narrow dataset.
    if model_path and os.path.exists(str(model_path)):
        target    = model_path
        is_custom = True
    else:
        target    = "yolo11n.pt"   # pretrained COCO 80-class fallback
        is_custom = False

    # Reload if path changed or cache is empty
    if _cache["path"] != target or _cache["model"] is None:
        _cache["model"] = YOLO(str(target))
        _cache["path"]  = target

    label = f"custom  [{DEVICE_LABEL}]" if is_custom else f"pretrained YOLO11n COCO  [{DEVICE_LABEL}]"
    return _cache["model"], label


def _pil_to_np(img: Image.Image) -> np.ndarray:
    """PIL RGB → numpy RGB uint8 — the only format YOLO predict reliably accepts."""
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


# ── Routes ────────────────────────────────────────────────────────────────────

@detect_bp.route("/")
def index():
    from utils.model_manager import list_trained_models, get_best_model_path
    return render_template(
        "detect.html",
        trained_models=list_trained_models(),
        active_model=get_best_model_path(),
        device_label=DEVICE_LABEL,
    )


@detect_bp.route("/image", methods=["POST"])
def detect_image():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    conf       = float(request.form.get("conf", 0.20))
    model_path = request.form.get("model_path") or None

    try:
        img    = Image.open(request.files["image"].stream).convert("RGB")
        model, src = _load_model(model_path)
        results    = model.predict(source=_pil_to_np(img), conf=conf,
                                   device=DEVICE, verbose=False)
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500

    # Annotated output
    annotated = results[0].plot()                       # numpy BGR
    pil_out   = Image.fromarray(annotated[..., ::-1])   # → RGB

    buf = io.BytesIO()
    pil_out.save(buf, format="JPEG", quality=88)
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    dets = [
        {
            "class": results[0].names[int(b.cls)],
            "conf":  round(float(b.conf), 3),
            "bbox":  [round(v, 1) for v in b.xyxy[0].tolist()],
        }
        for b in results[0].boxes
    ]

    return jsonify({
        "image":        img_b64,
        "detections":   dets,
        "count":        len(dets),
        "model_source": src,
    })


@detect_bp.route("/frame", methods=["POST"])
def detect_frame():
    """WebRTC webcam frame: base64 JPEG in → annotated base64 JPEG out."""
    data = request.json
    if not data or "frame" not in data:
        return jsonify({"error": "No frame data"}), 400

    conf       = float(data.get("conf", 0.20))
    model_path = data.get("model_path") or None

    try:
        raw = base64.b64decode(data["frame"].split(",")[-1])
        img = Image.open(io.BytesIO(raw)).convert("RGB")

        # Downscale for speed
        w, h = img.size
        if w > 640:
            img = img.resize((640, int(h * 640 / w)), Image.BILINEAR)

        model, src = _load_model(model_path)
        results    = model.predict(source=_pil_to_np(img), conf=conf,
                                   device=DEVICE, verbose=False)
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500

    annotated = results[0].plot()
    pil_out   = Image.fromarray(annotated[..., ::-1])

    buf = io.BytesIO()
    pil_out.save(buf, format="JPEG", quality=75)
    out_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    dets = [
        {"class": results[0].names[int(b.cls)], "conf": round(float(b.conf), 3)}
        for b in results[0].boxes
    ]

    return jsonify({
        "frame":      out_b64,
        "detections": dets,
        "count":      len(dets),
        "source":     src,
    })