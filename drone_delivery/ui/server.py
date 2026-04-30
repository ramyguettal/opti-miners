"""
server.py
=========
Python HTTP server bridging the UI with the backend optimisation models.

Usage (from ANY directory):
    python drone_delivery/ui/server.py
    
Then open http://localhost:5050
"""
import http.server
import json
import os
import socketserver
import subprocess
import sys
import urllib.parse
from pathlib import Path

PORT = 5050
# Resolve paths regardless of where the script is invoked from
SCRIPT_DIR = Path(__file__).parent.resolve()        # drone_delivery/ui/
PROJECT_ROOT = SCRIPT_DIR.parent.parent.resolve()   # NMO/


class OptimizationHandler(http.server.SimpleHTTPRequestHandler):
    """Serves static files from the ui/ folder and handles API calls."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SCRIPT_DIR), **kwargs)

    def do_POST(self):
        """Handle POST /api/run_optimization from the UI."""
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/api/run_optimization":
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)

            try:
                params = json.loads(body.decode("utf-8"))
            except json.JSONDecodeError:
                params = {}


            mode        = params.get("mode", "real")          # 'real' or 'random'
            customers   = params.get("customers", 20)
            drones      = params.get("drones", 4)
            battery     = params.get("battery", 150)
            payload     = params.get("payload", 5)
            seed        = params.get("seed", 42)
            generations = params.get("generations", 80)

            output_file = SCRIPT_DIR / "solution.json"

            if mode == "random":
                # Random synthetic instance — all slider values used directly
                cmd = [
                    sys.executable, "-m", "drone_delivery.main",
                    "--random",
                    "--customers",   str(customers),
                    "--drones",      str(drones),
                    "--battery",     str(battery),
                    "--payload",     str(payload),
                    "--seed",        str(seed),
                    "--generations", str(generations),
                    "--pop-size",    "60",
                    "--output",      str(output_file),
                ]
                print(f"[API] RANDOM mode | customers={customers} drones={drones} seed={seed}")
            else:
                # Real dataset — sliders override fleet config
                data_dir = PROJECT_ROOT / "data pre processing"
                cmd = [
                    sys.executable, "-m", "drone_delivery.main",
                    "--data-dir",       str(data_dir),
                    "--max-customers",  str(customers),
                    "--drones",         str(drones),
                    "--battery",        str(battery),
                    "--payload",        str(payload),
                    "--generations",    str(generations),
                    "--pop-size",       "60",
                    "--output",         str(output_file),
                ]
                print(f"[API] REAL mode | customers={customers} drones={drones} battery={battery} payload={payload}")


            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, check=True,
                    cwd=str(PROJECT_ROOT),
                )
                print(f"[API] Finished OK ({output_file.stat().st_size} bytes)")

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(output_file.read_bytes())

            except subprocess.CalledProcessError as e:
                print(f"[API] ERROR:\n{e.stderr}")
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "error": "Optimization failed",
                    "details": e.stderr[-2000:],
                }).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, fmt, *args):
        """Quieter logging — skip 304 and favicon noise."""
        msg = fmt % args
        if "304" not in msg and "favicon" not in msg:
            print(f"[HTTP] {msg}")


def main():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), OptimizationHandler) as httpd:
        print(f"{'='*50}")
        print(f"  DroneOpt server running on http://localhost:{PORT}")
        print(f"  Serving UI from: {SCRIPT_DIR}")
        print(f"  Project root   : {PROJECT_ROOT}")
        print(f"{'='*50}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    main()
