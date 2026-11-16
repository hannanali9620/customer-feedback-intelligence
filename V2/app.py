import os
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from analyzer import SchemaValidationError, analyze_feedback

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
load_dotenv(ROOT_DIR / ".env")

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "dev-local-v2-key")
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

SAMPLE_CSV_PATH = BASE_DIR / "sample_data" / "sample_feedback.csv"


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/api/sample", methods=["GET"])
def analyze_sample():
    try:
        result = analyze_feedback(SAMPLE_CSV_PATH)
        result["dataset_name"] = "sample_feedback.csv (Built-in Sample)"
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": f"Sample analysis failed: {exc}"}), 500


@app.route("/api/analyze", methods=["POST"])
def analyze_upload():
    if "file" not in request.files:
        return jsonify({"error": "No CSV file uploaded under form field 'file'."}), 400

    uploaded = request.files["file"]
    if not uploaded or uploaded.filename == "":
        return jsonify({"error": "Please select a non-empty .csv file."}), 400

    if not uploaded.filename.lower().endswith(".csv"):
        return jsonify({"error": "Invalid file type. Only .csv files are supported."}), 400

    try:
        result = analyze_feedback(uploaded)
        result["dataset_name"] = uploaded.filename
        return jsonify(result)
    except SchemaValidationError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": f"Failed to process CSV: {exc}"}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5002, debug=True)