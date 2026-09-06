import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, request, jsonify, Response
from lib.pipeline import warmup_status, swap_and_enhance

app = Flask(__name__)


@app.route("/api/warmup", methods=["GET"])
def warmup():
    try:
        status = warmup_status()
        ok = all(v == 200 for v in status.values())
        return jsonify({"status": "listo" if ok else "error", "services": status}), (200 if ok else 500)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"status": "error", "detail": str(exc)}), 500


@app.route("/api/swap-upload", methods=["POST"])
def swap_upload():
    try:
        source = request.files.get("source")
        target = request.files.get("target")
        if not source or not target:
            return jsonify({"detail": "Faltan los campos 'source' y/o 'target'"}), 400

        result_jpeg = swap_and_enhance(source.read(), target.read())
        return Response(result_jpeg, mimetype="image/jpeg")
    except Exception as exc:  # noqa: BLE001
        return jsonify({"detail": str(exc)}), 400
