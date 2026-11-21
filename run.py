#!/usr/bin/env python3
"""Start the whole local app from the repository root with one command.

The launcher owns three localhost ports and wires the services together:

    browser -> frontend/ (Vite) -> api/ (FastAPI) -> modelService/ (classifier)

Press Ctrl+C to stop every child process. Ports and commands can be overridden
with the environment variables documented in the root README.
"""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PORT = 8000
DEFAULT_API_PORT = 8001
DEFAULT_WEB_PORT = 5173


def _port(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        port = int(raw)
    except ValueError:
        raise SystemExit(f"{name} must be an integer, got {raw!r}") from None
    if not 1 <= port <= 65535:
        raise SystemExit(f"{name} must be between 1 and 65535, got {port}")
    return port


def _port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _pump(pipe, label: str) -> None:
    """Prefix each child output line so interleaved logs stay readable."""
    try:
        for line in iter(pipe.readline, ""):
            sys.stdout.write(f"[{label}] {line}")
            sys.stdout.flush()
    except (OSError, ValueError):
        pass


def _spawn(label: str, cwd: Path, command: list[str], env: dict[str, str]) -> subprocess.Popen[str]:
    kwargs: dict[str, object] = {
        "cwd": str(cwd),
        "env": env,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "text": True,
        "bufsize": 1,
    }
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True

    process = subprocess.Popen(command, **kwargs)  # type: ignore[arg-type]
    if process.stdout is not None:
        threading.Thread(target=_pump, args=(process.stdout, label), daemon=True).start()
    return process


def _kill(process: subprocess.Popen[str], force: bool = False) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        flags = ["/T", "/F"] if force else ["/T"]
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), *flags],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return
    try:
        os.killpg(os.getpgid(process.pid), signal.SIGKILL if force else signal.SIGTERM)
    except ProcessLookupError:
        pass


def _terminate(processes: list[subprocess.Popen[str]]) -> None:
    for process in processes:
        _kill(process)
    deadline = time.monotonic() + 5
    for process in processes:
        remaining = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            _kill(process, force=True)
    for process in processes:
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass


def _http_reachable(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=1):
            return True
    except urllib.error.HTTPError:
        return True
    except (OSError, ValueError):
        return False


def _wait_until_listening(processes: list[tuple[str, subprocess.Popen[str]]], urls: list[str]) -> None:
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        for label, process in processes:
            code = process.poll()
            if code is not None:
                print(f"[run] {label} exited with code {code} during startup.", file=sys.stderr, flush=True)
                return
        if all(_http_reachable(url) for url in urls):
            print("[run] API and frontend are listening.", flush=True)
            return
        time.sleep(0.25)
    print("[run] Startup check timed out; see the service logs above.", file=sys.stderr, flush=True)


def _request_stop(_signum, _frame) -> None:
    raise KeyboardInterrupt


def main() -> int:
    model_port = _port("MODEL_SERVICE_PORT", DEFAULT_MODEL_PORT)
    api_port = _port("API_PORT", DEFAULT_API_PORT)
    web_port = _port("FRONTEND_PORT", DEFAULT_WEB_PORT)

    for port, name, variable in (
        (model_port, "modelService", "MODEL_SERVICE_PORT"),
        (api_port, "api", "API_PORT"),
        (web_port, "frontend", "FRONTEND_PORT"),
    ):
        if _port_in_use(port):
            raise SystemExit(
                f"Port {port} for {name} is already in use. Stop that process or set "
                f"{variable}=<free-port> and rerun."
            )

    uv = shutil.which("uv")
    npm = shutil.which("npm")
    if uv is None or npm is None:
        missing = ", ".join(name for name, path in (("uv", uv), ("npm", npm)) if path is None)
        raise SystemExit(f"Missing required command(s): {missing}. Install them and rerun.")

    if not (ROOT / "frontend" / "node_modules").is_dir():
        raise SystemExit("Frontend dependencies are missing. Run: cd frontend && npm install")

    env = os.environ.copy()
    env.update(
        {
            "MODEL_SERVICE_PORT": str(model_port),
            "API_PORT": str(api_port),
            "API_MODEL_SERVICE_URL": f"http://127.0.0.1:{model_port}",
            "VITE_API_BASE_URL": f"http://127.0.0.1:{api_port}",
            "API_CORS_ORIGINS": (
                f"http://localhost:{web_port},http://127.0.0.1:{web_port},"
                "http://localhost:4173,http://127.0.0.1:4173"
            ),
        }
    )

    processes: list[tuple[str, subprocess.Popen[str]]] = []
    try:
        signal.signal(signal.SIGTERM, _request_stop)
        print(f"[run] modelService  http://127.0.0.1:{model_port}", flush=True)
        print(f"[run] api          http://127.0.0.1:{api_port}", flush=True)
        print(f"[run] frontend     http://127.0.0.1:{web_port}", flush=True)
        print("[run] Ctrl+C stops all three.\n", flush=True)

        processes.append(
            ("model", _spawn("model", ROOT / "modelService", [uv, "run", "model-service"], env))
        )
        processes.append(
            ("api", _spawn("api", ROOT / "api", [uv, "run", "python", "-m", "api_service"], env))
        )
        processes.append(
            (
                "web",
                _spawn(
                    "web",
                    ROOT / "frontend",
                    [npm, "run", "dev", "--", "--host", "127.0.0.1", "--port", str(web_port)],
                    env,
                ),
            )
        )

        _wait_until_listening(
            processes,
            [
                f"http://127.0.0.1:{api_port}/api/v1/health",
                f"http://127.0.0.1:{web_port}/",
            ],
        )

        while True:
            for label, process in processes:
                code = process.poll()
                if code is not None:
                    print(f"[run] {label} exited with code {code}.", file=sys.stderr, flush=True)
                    return code or 1
            time.sleep(0.25)
    except KeyboardInterrupt:
        print("\n[run] Stopping services...", flush=True)
        return 130
    finally:
        _terminate([process for _, process in processes])
        for _, process in processes:
            if process.stdout is not None:
                process.stdout.close()


if __name__ == "__main__":
    raise SystemExit(main())
