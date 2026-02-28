import os
from flask import Flask


def create_app():
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.environ.get("SECRET_KEY", "yolo-playground-secret-key"),
        UPLOAD_FOLDER="data/uploads",
        DATASET_FOLDER="data/datasets",
        RESULTS_FOLDER="static/results",
        MAX_CONTENT_LENGTH=10 * 1024 * 1024,  # ← 10 MB hard limit
    )
    for d in [
        app.config["UPLOAD_FOLDER"],
        app.config["DATASET_FOLDER"],
        app.config["RESULTS_FOLDER"],
        "runs",
    ]:
        os.makedirs(d, exist_ok=True)

    from routes.upload import upload_bp
    from routes.train import train_bp
    from routes.detect import detect_bp

    app.register_blueprint(upload_bp, url_prefix="/upload")
    app.register_blueprint(train_bp, url_prefix="/train")
    app.register_blueprint(detect_bp, url_prefix="/detect")

    @app.errorhandler(413)
    def too_large(e):
        from flask import jsonify
        return jsonify({"error": "File exceeds the 10 MB upload limit."}), 413

    @app.route("/")
    def index():
        from flask import render_template
        return render_template("index.html")

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=7860, debug=False, threaded=True)