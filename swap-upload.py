import json
import sys
import os
import cgi
from io import BytesIO

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from http.server import BaseHTTPRequestHandler
from lib.pipeline import swap_and_enhance


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        try:
            ctype, pdict = cgi.parse_header(self.headers.get("Content-Type", ""))
            if ctype != "multipart/form-data":
                raise ValueError("Se esperaba multipart/form-data con 'source' y 'target'")

            pdict["boundary"] = pdict["boundary"].encode()
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)

            fields = cgi.parse_multipart(BytesIO(body), pdict)
            source_bytes = fields.get("source", [None])[0]
            target_bytes = fields.get("target", [None])[0]
            if not source_bytes or not target_bytes:
                raise ValueError("Faltan los campos 'source' y/o 'target'")

            result_jpeg = swap_and_enhance(source_bytes, target_bytes)

            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.end_headers()
            self.wfile.write(result_jpeg)

        except Exception as exc:  # noqa: BLE001
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"detail": str(exc)}).encode())
