from __future__ import annotations

import argparse
import os
import subprocess
import sys
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"


class NoCacheHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


def run_cmd(args):
    print(">", " ".join(map(str, args)))
    result = subprocess.run(args, cwd=ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-analysis", action="store_true",
                        help="не пересчитывать starter.py, использовать существующий out/")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if not args.skip_analysis:
        run_cmd([sys.executable, "starter.py"])

    run_cmd([sys.executable, "build_web_data.py"])

    os.chdir(WEB)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), NoCacheHandler)
    url = f"http://127.0.0.1:{args.port}/"

    print(f"\nСайт запущен: {url}")
    print("Для остановки нажми Ctrl+C.\n")

    threading.Timer(0.7, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nСервер остановлен.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
