import os
from flask import Blueprint, request, jsonify, render_template, current_app
from werkzeug.utils import secure_filename
from utils.dataset import extract_labelstudio_zip, build_data_yaml

upload_bp = Blueprint("upload", __name__)

MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MB


@upload_bp.route("/")
def index():
    yaml_exists = os.path.exists("data/datasets/current/data.yaml")
    classes = []
    if yaml_exists:
        import yaml
        with open("data/datasets/current/data.yaml") as f:
            d = yaml.safe_load(f)
            classes = d.get("names", [])
    return render_template("upload.html", yaml_exists=yaml_exists, classes=classes)


@upload_bp.route("/submit", methods=["POST"])
def submit():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    f = request.files["file"]
    if not f.filename.lower().endswith(".zip"):
        return jsonify({"error": "Please upload a .zip file exported from Label Studio"}), 400

    # ── Size check before touching disk ──────────────────────────────────────
    f.seek(0, 2)          # seek to end
    size = f.tell()
    f.seek(0)             # rewind
    if size > MAX_UPLOAD_BYTES:
        mb = round(size / 1_048_576, 2)
        return jsonify({
            "error": f"File is {mb} MB — maximum allowed is 10 MB. "
                     "Try reducing image resolution or splitting the dataset."
        }), 413

    # ── Replace: wipe any previous uploads & dataset ─────────────────────────
    upload_dir  = current_app.config["UPLOAD_FOLDER"]
    dataset_dir = os.path.join(current_app.config["DATASET_FOLDER"], "current")

    was_replaced = os.path.exists(dataset_dir)

    for old in os.listdir(upload_dir):          # remove old zips
        try:
            os.remove(os.path.join(upload_dir, old))
        except OSError:
            pass

    # ── Save & extract ────────────────────────────────────────────────────────
    fname    = secure_filename(f.filename)
    zip_path = os.path.join(upload_dir, fname)
    f.save(zip_path)

    try:
        info = extract_labelstudio_zip(zip_path, dataset_dir)
    except Exception as e:
        return jsonify({"error": f"Extraction failed: {str(e)}"}), 500
    finally:
        # Delete the zip immediately — we only need the extracted dataset
        try:
            os.remove(zip_path)
        except OSError:
            pass

    if not info["classes"]:
        return jsonify({
            "error": "Could not find class names. Make sure classes.txt is in the zip."
        }), 400

    yaml_path = os.path.join(dataset_dir, "data.yaml")
    build_data_yaml(dataset_dir, info["classes"], yaml_path)

    return jsonify({
        "success":     True,
        "replaced":    was_replaced,
        "classes":     info["classes"],
        "num_classes": info["num_classes"],
        "image_count": info["image_count"],
        "train_count": info["train_count"],
        "val_count":   info["val_count"],
    })