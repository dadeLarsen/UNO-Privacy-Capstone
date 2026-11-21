#!/usr/bin/env python3
"""Set up and start the whole local app from the repository root.

`python run.py setup` installs every part's dependencies and downloads the pinned
model files, once, while online. `python run.py` then starts the app. The launcher
owns three localhost ports and wires the services together:

    browser -> frontend/ (Vite) -> api/ (FastAPI) -> modelService/ (classifier)

Press Ctrl+C to stop every child process. Ports and commands can be overridden
with the environment variables documented in the root README.
"""

from __future__ import annotations

import argparse
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
# Every default condition in modelService/src/model_service/settings.py, relative to modelService/.
# A model's GPU and CPU conditions pin the same file, so the second download finds it cached.
MODEL_CONDITIONS = [
    "../privacy-research/conditions/classification/appsi139/embeddinggemma2-gguf-q8-gpu-head-linear-balanced-tuned-limit2048-appsi139-test-half.toml",
    "../privacy-research/conditions/classification/appsi139/embeddinggemma2-gguf-q8-cpu-head-linear-balanced-tuned-limit2048-appsi139-test-half.toml",
    "../privacy-research/conditions/rewrite/qwen3.5-4b-q4km-gpu-summary-v3.toml",
    "../privacy-research/conditions/rewrite/qwen3.5-4b-q4km-cpu-summary-v3.toml",
]


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
        # Vite and llama.cpp write UTF-8; a byte that is not UTF-8 must not stop the log reader.
        "encoding": "utf-8",
        "errors": "replace",
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
        # Windows refuses taskkill without /F for these console processes, so trying it first only delays the stop.
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
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


def _require_commands() -> tuple[str, str]:
    uv = shutil.which("uv")
    npm = shutil.which("npm")
    if uv is None or npm is None:
        missing = ", ".join(name for name, path in (("uv", uv), ("npm", npm)) if path is None)
        raise SystemExit(f"Missing required command(s): {missing}. Install them and rerun.")
    return uv, npm


def setup() -> int:
    """Install the dependencies and download the models in order; stop at the first failure."""
    uv, npm = _require_commands()
    # nvidia-smi comes with the NVIDIA driver; without one, the CUDA build has no GPU to use.
    cuda = shutil.which("nvidia-smi") is not None
    print(
        "[setup] nvidia-smi found: installing the CUDA 13.2 build of llama.cpp."
        if cuda
        else "[setup] nvidia-smi not found: compiling llama.cpp for the CPU, which needs CMake and a C++ compiler.",
        flush=True,
    )
    steps = [
        ("modelService", [uv, "sync"] if cuda else [uv, "sync", "--no-group", "cu132", "--group", "cpu"]),
        ("api", [uv, "sync"]),
        ("frontend", [npm, "ci"]),
        *(("modelService", [uv, "run", "--no-sync", "pbench", "download", path]) for path in MODEL_CONDITIONS),
    ]
    for folder, command in steps:
        shown = " ".join([Path(command[0]).stem, *command[1:]])
        print(f"\n[setup] {folder}> {shown}", flush=True)
        code = subprocess.run(command, cwd=ROOT / folder, check=False).returncode
        if code != 0:
            print(f"\n[setup] Failed in {folder} with exit code {code}: {shown}", file=sys.stderr, flush=True)
            return code
    print("\n[setup] Done. Start the app with: python run.py", flush=True)
    return 0


def main() -> int:
    # Piped or redirected on Windows, stdout is cp1252, which cannot encode every character a service logs.
    sys.stdout.reconfigure(errors="replace")
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

    uv, npm = _require_commands()
    if not (ROOT / "modelService" / ".venv").is_dir() or not (ROOT / "frontend" / "node_modules").is_dir():
        raise SystemExit("Dependencies are missing. Run: python run.py setup")

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

        # A sync here would swap the CPU build of llama.cpp that setup chose for the default CUDA one.
        processes.append(
            ("model", _spawn("model", ROOT / "modelService", [uv, "run", "--no-sync", "model-service"], env))
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
    parser = argparse.ArgumentParser(description="Start the local app, or set it up first.")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["setup"],
        help="install the dependencies and download the pinned models, then exit",
    )
    raise SystemExit(setup() if parser.parse_args().command == "setup" else main())
