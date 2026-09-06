import json
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from http.server import BaseHTTPRequestHandler
from lib.pipeline import warmup_status


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        try:
            status = warmup_status()
            ok = all(v == 200 for v in status.values())
            self.send_response(200 if ok else 500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({
                "status": "listo" if ok else "error",
                "services": status,
            }).encode())
        except Exception as exc:  # noqa: BLE001
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"status": "error", "detail": str(exc)}).encode())
