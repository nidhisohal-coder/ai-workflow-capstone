from flask import Flask, request, jsonify
from pathlib import Path
import json
from datetime import datetime

from app.model_service import train_model, predict_next

app = Flask(__name__)

REPO = Path(__file__).resolve().parents[1]
LOG_PATH = REPO / "logs" / "app.log"

def log_event(event: dict):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    event["ts"] = datetime.utcnow().isoformat()
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(event) + "\n")

@app.route("/train", methods=["POST"])
def train():
    try:
        payload = request.get_json(silent=True) or {}
        train_dir = payload.get("train_dir", str(REPO / "cs-train"))

        result = train_model(Path(train_dir))
        log_event({"endpoint": "train", "payload": payload, "result": result})
        return jsonify(result), 200
    except Exception as e:
        log_event({"endpoint": "train", "error": str(e)})
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        data_dir = payload.get("data_dir", str(REPO / "cs-production"))

        result = predict_next(payload, Path(data_dir))
        log_event({"endpoint": "predict", "payload": payload, "result": result})
        return jsonify(result), 200
    except Exception as e:
        log_event({"endpoint": "predict", "error": str(e)})
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/logs", methods=["GET"])
def logs():
    if not LOG_PATH.exists():
        return jsonify({"status": "ok", "logs": ""}), 200
    return LOG_PATH.read_text(), 200, {"Content-Type": "text/plain; charset=utf-8"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)