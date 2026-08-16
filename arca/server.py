from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from arca.agent_api import ARCAAgentBackend


class ARCAHTTPHandler(BaseHTTPRequestHandler):
    server_version = "ARCA/0.4"

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/v1/models":
            self._json(200, {"object": "list", "data": [{"id": "arca-native", "object": "model", "owned_by": "enrrutador", "capabilities": ["chat"]}]})
            return
        self._json(404, {"error": {"message": "not found", "type": "invalid_request_error"}})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/v1/chat/completions":
            self._json(404, {"error": {"message": "not found", "type": "invalid_request_error"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length))
            messages = body.get("messages", [])
            user = next((message.get("content", "") for message in reversed(messages) if message.get("role") == "user"), "")
            session_id = self.headers.get("X-ARCA-Session", body.get("user", "default"))
            result = self.server.backend.respond(str(user), str(session_id))
            now = int(time.time())
            self._json(200, {"id": f"arca-{now}", "object": "chat.completion", "created": now, "model": "arca-native", "choices": [{"index": 0, "message": {"role": "assistant", "content": result.text}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 0, "completion_tokens": len(result.text.encode("utf-8")), "total_tokens": len(result.text.encode("utf-8"))}, "arca": {"success": result.success, "telemetry": result.telemetry, "trace": list(result.trace)}})
        except Exception as exc:
            self._json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})

    def log_message(self, format: str, *args: Any) -> None:
        return


def serve(model: str | Path, db: str | Path = "arca.db", host: str = "127.0.0.1", port: int = 8787) -> None:
    backend = ARCAAgentBackend(model, db)
    server = ThreadingHTTPServer((host, port), ARCAHTTPHandler)
    server.backend = backend
    print(f"ARCA OpenAI-compatible API listening on http://{host}:{port}/v1")
    server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve ARCA for OpenCode and OpenAI-compatible clients")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--db", default="arca.db")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    args = parser.parse_args()
    serve(args.model, args.db, args.host, args.port)


if __name__ == "__main__":
    main()
