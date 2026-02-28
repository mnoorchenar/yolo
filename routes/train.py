import os, json, time, threading, shutil, csv
import torch
from flask import (Blueprint, request, jsonify, render_template,
                   Response, stream_with_context)

train_bp = Blueprint("train", __name__)

# ── Device ────────────────────────────────────────────────────────────────────
DEVICE       = 0 if torch.cuda.is_available() else "cpu"
DEVICE_LABEL = f"GPU (cuda:{DEVICE})" if DEVICE != "cpu" else "CPU"

# ── Global training state ─────────────────────────────────────────────────────
_lock       = threading.Lock()
_stop_event = threading.Event()
_state = {
    "status":       "idle",
    "epoch":        0,
    "total_epochs": 0,
    # live callback metrics
    "box_loss":     "—",
    "cls_loss":     "—",
    "dfl_loss":     "—",
    "mAP50":        "—",
    "mAP50_95":     "—",
    "precision":    "—",
    "recall":       "—",
    # history for sparklines (list of dicts)
    "history":      [],
    "log":          [],
    "model_path":   None,
    "plots":        [],      # list of relative URLs for saved PNG plots
    "error":        None,
    "model_size":   "n",
    "device_label": DEVICE_LABEL,
}

RUNS_DIR    = "runs/detect/custom"
PLOTS_DIR   = "static/results/plots"
RESULTS_CSV = os.path.join(RUNS_DIR, "results.csv")


def _set(**kw):
    with _lock:
        _state.update(kw)


def _log(msg: str):
    with _lock:
        _state["log"].append(msg)
        if len(_state["log"]) > 200:
            _state["log"] = _state["log"][-200:]


def _cleanup_dataset():
    dataset_dir = "data/datasets/current"
    if os.path.exists(dataset_dir):
        try:
            shutil.rmtree(dataset_dir)
            _log("🗑  Dataset removed from disk (weights kept in runs/).")
        except Exception as exc:
            _log(f"⚠️  Cleanup warning: {exc}")


def _copy_plots():
    """Copy Ultralytics PNG plots to static/ so they can be served."""
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_names = [
        "results.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
        "PR_curve.png",
        "P_curve.png",
        "R_curve.png",
        "F1_curve.png",
    ]
    saved = []
    for name in plot_names:
        src = os.path.join(RUNS_DIR, name)
        if os.path.exists(src):
            dst = os.path.join(PLOTS_DIR, name)
            shutil.copy2(src, dst)
            saved.append(f"/{dst}")
    return saved


def _parse_csv_history() -> list:
    """Read results.csv and return a list of per-epoch metric dicts."""
    if not os.path.exists(RESULTS_CSV):
        return []
    rows = []
    try:
        with open(RESULTS_CSV, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                clean = {k.strip(): v.strip() for k, v in row.items()}
                rows.append({
                    "epoch":     int(float(clean.get("epoch", 0))),
                    "box_loss":  _safe_float(clean.get("train/box_loss")),
                    "cls_loss":  _safe_float(clean.get("train/cls_loss")),
                    "dfl_loss":  _safe_float(clean.get("train/dfl_loss")),
                    "precision": _safe_float(clean.get("metrics/precision(B)")),
                    "recall":    _safe_float(clean.get("metrics/recall(B)")),
                    "mAP50":     _safe_float(clean.get("metrics/mAP50(B)")),
                    "mAP50_95":  _safe_float(clean.get("metrics/mAP50-95(B)")),
                })
    except Exception:
        pass
    return rows


def _safe_float(val) -> str:
    try:
        return str(round(float(val), 4))
    except Exception:
        return "—"


# ── Routes ────────────────────────────────────────────────────────────────────

@train_bp.route("/")
def index():
    yaml_exists = os.path.exists("data/datasets/current/data.yaml")
    classes = []
    if yaml_exists:
        import yaml
        with open("data/datasets/current/data.yaml") as f:
            classes = yaml.safe_load(f).get("names", [])
    return render_template("train.html", yaml_exists=yaml_exists,
                           classes=classes, device_label=DEVICE_LABEL)


@train_bp.route("/status")
def status():
    with _lock:
        snap = dict(_state)
    snap["log"] = snap["log"][-30:]
    return jsonify(snap)


@train_bp.route("/progress")
def progress():
    def _gen():
        while True:
            with _lock:
                snap = dict(_state)
            snap["log_tail"] = snap["log"][-30:]
            snap.pop("log", None)
            yield f"data: {json.dumps(snap)}\n\n"
            if snap["status"] in ("done", "error", "idle"):
                break
            time.sleep(1.5)
    return Response(
        stream_with_context(_gen()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@train_bp.route("/start", methods=["POST"])
def start():
    with _lock:
        if _state["status"] == "running":
            return jsonify({"error": "Training is already running"}), 400

    cfg        = request.json or {}
    epochs     = max(1, int(cfg.get("epochs", 30)))
    imgsz      = int(cfg.get("imgsz", 640))
    batch      = int(cfg.get("batch", 8))
    model_size = cfg.get("model_size", "n")

    yaml_path = "data/datasets/current/data.yaml"
    if not os.path.exists(yaml_path):
        return jsonify({"error": "No dataset found — please upload first."}), 400

    _set(
        status="running", epoch=0, total_epochs=epochs,
        box_loss="—", cls_loss="—", dfl_loss="—",
        mAP50="—", mAP50_95="—", precision="—", recall="—",
        history=[], log=[], model_path=None, plots=[], error=None,
        model_size=model_size, device_label=DEVICE_LABEL,
    )
    _stop_event.clear()
    threading.Thread(
        target=_run_training,
        args=(yaml_path, epochs, imgsz, batch, model_size),
        daemon=True,
    ).start()
    return jsonify({"success": True, "device": DEVICE_LABEL})


@train_bp.route("/stop", methods=["POST"])
def stop():
    _stop_event.set()
    _log("⛔ Stop requested — finishing current epoch…")
    return jsonify({"success": True})


@train_bp.route("/plots")
def plots():
    """Return list of available training plot URLs."""
    with _lock:
        return jsonify({"plots": _state.get("plots", [])})


# ── Background training ───────────────────────────────────────────────────────

def _run_training(yaml_path, epochs, imgsz, batch, size):
    try:
        from ultralytics import YOLO

        model_name = f"yolo11{size}.pt"
        _log(f"↓ Loading base model: {model_name}")
        model = YOLO(model_name)
        _log(f"✓ Model loaded  |  Device: {DEVICE_LABEL}")

        # ── Callbacks ──────────────────────────────────────────────────────────
        def on_train_start(trainer):
            dev = str(trainer.device) if hasattr(trainer, "device") else DEVICE_LABEL
            _log(f"🚀 Training started — {epochs} ep | imgsz={imgsz} | batch={batch} | {dev}")

        def on_fit_epoch_end(trainer):
            if _stop_event.is_set():
                trainer.stop = True
            ep = trainer.epoch + 1
            try:
                li  = trainer.loss_items
                box = round(float(li[0]), 4) if li is not None else "—"
                cls = round(float(li[1]), 4) if li is not None and len(li) > 1 else "—"
                dfl = round(float(li[2]), 4) if li is not None and len(li) > 2 else "—"
            except Exception:
                box = cls = dfl = "—"

            mAP50 = mAP50_95 = prec = rec = "—"
            try:
                m = trainer.metrics
                if m:
                    mAP50    = round(float(m.get("metrics/mAP50(B)",    0)), 4)
                    mAP50_95 = round(float(m.get("metrics/mAP50-95(B)", 0)), 4)
                    prec     = round(float(m.get("metrics/precision(B)", 0)), 4)
                    rec      = round(float(m.get("metrics/recall(B)",    0)), 4)
            except Exception:
                pass

            entry = dict(epoch=ep, box_loss=box, cls_loss=cls, dfl_loss=dfl,
                         mAP50=mAP50, mAP50_95=mAP50_95,
                         precision=prec, recall=rec)
            with _lock:
                _state.update(entry)
                _state["history"].append(entry)

            _log(
                f"  Ep {ep:>3}/{epochs} | "
                f"box={box} cls={cls} dfl={dfl} | "
                f"P={prec} R={rec} mAP50={mAP50} mAP50-95={mAP50_95}"
            )

        def on_train_end(trainer):
            _log("✅ Training complete — copying result plots…")

        model.add_callback("on_train_start",   on_train_start)
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        model.add_callback("on_train_end",     on_train_end)

        model.train(
            data=yaml_path,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            project="runs/detect",
            name="custom",
            exist_ok=True,
            device=DEVICE,
            workers=0,
            verbose=False,
        )

        # ── Post-training: copy plots + parse CSV ──────────────────────────────
        plots  = _copy_plots()
        hist   = _parse_csv_history()
        best   = os.path.join(RUNS_DIR, "weights", "best.pt")

        _set(
            status="done",
            epoch=epochs,
            model_path=best if os.path.exists(best) else None,
            plots=plots,
            history=hist or _state.get("history", []),
        )
        _log(f"💾 Best weights → {best}")
        _log(f"📊 {len(plots)} plot(s) saved to static/results/plots/")

    except Exception as exc:
        import traceback
        if _stop_event.is_set():
            _set(status="idle")
            _log("⛔ Training stopped by user.")
        else:
            _set(status="error", error=str(exc))
            _log(f"❌ Error: {exc}")
            _log(traceback.format_exc())

    finally:
        _stop_event.clear()
        _cleanup_dataset()