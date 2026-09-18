#!/usr/bin/env python3
"""
DeepFER web application (the project brief's "Application Development"
requirement: "integrate the emotion recognition system into a user-friendly
application or interface").

Two ways to use it once running:
1. Upload a photo -> POST /predict (multipart file)
2. Live browser webcam -> client JS grabs frames from getUserMedia and
    POSTs them as base64 JPEGs to /predict_frame, throttled client-side.

Both routes share the exact same detection + preprocessing + classification
code as realtime_webcam.py (imported, not duplicated), so results are
identical whether you run the desktop OpenCV script or this web app.

Run:
    python webapp/app.py --model transfer --checkpoint saved_models/transfer_finetune_best.keras
Then open http://127.0.0.1:5000
"""
import argparse
import base64
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from deepfer import config  # noqa: E402
from realtime_webcam import FACE_CASCADE_PATH, EmotionClassifier  # noqa: E402

app = Flask(__name__)

CLASSIFIER: EmotionClassifier = None

FACE_CASCADE = cv2.CascadeClassifier(FACE_CASCADE_PATH)


def decode_image_from_bytes(raw_bytes: bytes):

    arr = np.frombuffer(raw_bytes, dtype = np.uint8)

    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def run_detection(frame_bgr):

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    faces = FACE_CASCADE.detectMultiScale(gray, scaleFactor = 1.1, minNeighbors = 5, minSize = (48, 48))

    results = []

    for (x, y, w, h) in faces:

        face = frame_bgr[y:y + h, x:x + w]

        label, conf, probs = CLASSIFIER.predict(face)

        results.append({

            "box": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},

            "label": label,

            "confidence": round(conf, 4),

            "probabilities": {cls: round(float(p), 4) for cls, p in zip(config.CLASS_NAMES, probs)}

        })

    return results


@app.route("/")

def index():

    return render_template("index.html", class_names = config.CLASS_NAMES, model_kind = CLASSIFIER.kind if CLASSIFIER else "unknown")


@app.route("/predict", methods = ["POST"])

def predict():

    if "image" not in request.files:

        return jsonify({"error": "no image field in form-data"}), 400

    file = request.files["image"]

    frame = decode_image_from_bytes(file.read())

    if frame is None:

        return jsonify({"error": "could not decode image"}), 400

    t0 = time.time()

    results = run_detection(frame)

    return jsonify({"faces": results, "inference_ms": round((time.time() - t0) * 1000, 1)})


@app.route("/predict_frame", methods = ["POST"])

def predict_frame():

    data = request.get_json(silent = True) or {}

    data_url = data.get("image", "")

    if "," in data_url:

        data_url = data_url.split(",", 1)[1]

    try:

        raw = base64.b64decode(data_url)

    except Exception:

        return jsonify({"error": "invalid base64 image"}), 400

    frame = decode_image_from_bytes(raw)

    if frame is None:

        return jsonify({"error": "could not decode frame"}), 400

    t0 = time.time()

    results = run_detection(frame)

    return jsonify({"faces": results, "inference_ms": round((time.time() - t0) * 1000, 1)})


@app.route("/healthz")

def healthz():

    return jsonify({"status": "ok", "model": CLASSIFIER.kind if CLASSIFIER else None})


def main():

    global CLASSIFIER

    ap = argparse.ArgumentParser()

    ap.add_argument("--model", choices = ["scratch", "transfer"], default = "transfer")

    ap.add_argument("--checkpoint", default = None)

    ap.add_argument("--host", default = "127.0.0.1")

    ap.add_argument("--port", type=int, default = 5000)

    ap.add_argument("--debug", action = "store_true")

    args = ap.parse_args()

    checkpoint = args.checkpoint

    if checkpoint is None:

        candidates = [

            config.SAVED_MODELS_DIR / "transfer_finetune_best.keras",

            config.SAVED_MODELS_DIR / "transfer_head_best.keras",

            config.SAVED_MODELS_DIR / "scratch_best.keras"

        ]

        checkpoint = next((str(c) for c in candidates if c.exists()), None)

        if checkpoint is None:

            raise SystemExit("No trained checkpoint found in saved_models/. Train a model first (see README).")

        kind = "scratch" if "scratch" in checkpoint else "transfer"

    else:

        kind = args.model

    print(f"Loading {kind} model from {checkpoint}")

    CLASSIFIER = EmotionClassifier(checkpoint, kind)

    print("Model loaded. Starting Flask server ...")

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    
    main()
