from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
DATA = ROOT / "data"
ALLOWED_UPLOADS = {
    "edges.parquet",
    "nodes.parquet",
    "transactions.parquet",
}


def _utf8_env():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env


def run_capture(args):
    started = time.perf_counter()
    proc = subprocess.run(
        args,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env=_utf8_env(),
    )
    elapsed = time.perf_counter() - started
    return proc, elapsed


class AppHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        # Static files are always served from web/
        parsed = urllib.parse.urlparse(path)
        rel = parsed.path.lstrip("/") or "index.html"
        return str((WEB / rel).resolve())

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()

    def json_response(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/api/status":
            files = {name: (DATA / name).exists() for name in sorted(ALLOWED_UPLOADS)}
            self.json_response(
                200,
                {
                    "files": files,
                    "ready": all(files.values()),
                    "has_results": (WEB / "graph_data.json").exists(),
                },
            )
            return

        super().do_GET()

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/api/upload":
            query = urllib.parse.parse_qs(parsed.query)
            name = query.get("name", [""])[0]

            if name not in ALLOWED_UPLOADS:
                self.json_response(400, {"ok": False, "error": "Недопустимое имя файла"})
                return

            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0:
                self.json_response(400, {"ok": False, "error": "Пустой файл"})
                return

            DATA.mkdir(parents=True, exist_ok=True)
            target = DATA / name

            with target.open("wb") as f:
                remaining = length
                while remaining:
                    chunk = self.rfile.read(min(1024 * 1024, remaining))
                    if not chunk:
                        break
                    f.write(chunk)
                    remaining -= len(chunk)

            self.json_response(
                200,
                {"ok": True, "name": name, "bytes": target.stat().st_size},
            )
            return

        if parsed.path == "/api/analyze":
            missing = [name for name in ALLOWED_UPLOADS if not (DATA / name).exists()]
            if missing:
                self.json_response(
                    400,
                    {"ok": False, "error": "Не хватает файлов: " + ", ".join(sorted(missing))},
                )
                return

            logs = []
            total_started = time.perf_counter()

            for cmd in [
                [sys.executable, "starter.py"],
                [sys.executable, "validate_outputs.py"],
                [sys.executable, "build_web_data.py"],
            ]:
                proc, elapsed = run_capture(cmd)
                logs.append(
                    {
                        "command": " ".join(cmd),
                        "seconds": round(elapsed, 3),
                        "stdout": proc.stdout,
                        "stderr": proc.stderr,
                    }
                )
                if proc.returncode != 0:
                    self.json_response(
                        500,
                        {
                            "ok": False,
                            "error": f"Ошибка команды: {' '.join(cmd)}",
                            "logs": logs,
                        },
                    )
                    return

            total = time.perf_counter() - total_started
            self.json_response(
                200,
                {
                    "ok": True,
                    "seconds": round(total, 3),
                    "under_5_minutes": total <= 300,
                    "logs": logs,
                },
            )
            return

        self.json_response(404, {"ok": False, "error": "Unknown API endpoint"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--auto-analyze",
        action="store_true",
        help="сразу пересчитать данные при запуске, если три parquet уже лежат в data/",
    )
    args = parser.parse_args()

    if args.auto_analyze and all((DATA / n).exists() for n in ALLOWED_UPLOADS):
        for cmd in [
            [sys.executable, "starter.py"],
            [sys.executable, "validate_outputs.py"],
            [sys.executable, "build_web_data.py"],
        ]:
            proc = subprocess.run(cmd, cwd=ROOT, env=_utf8_env())
            if proc.returncode != 0:
                raise SystemExit(proc.returncode)

    server = ThreadingHTTPServer(("127.0.0.1", args.port), AppHandler)
    url = f"http://127.0.0.1:{args.port}/"

    print(f"Сайт запущен: {url}")
    print("Можно загрузить 3 parquet прямо через сайт.")
    print("Ctrl+C — остановить сервер.")

    threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nСервер остановлен.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
