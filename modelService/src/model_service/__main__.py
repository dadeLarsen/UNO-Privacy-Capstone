"""Start the localhost model service."""

import argparse
from dataclasses import replace
from pathlib import Path

from .app import create_app
from .settings import Settings


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="model-service",
        description="Classify uploaded privacy policies with the pinned local classifier.",
    )
    parser.add_argument("--condition", type=Path, help="Classification condition TOML to run.")
    parser.add_argument("--device", choices=("cuda", "cpu"), help="Override the condition's device.")
    parser.add_argument("--host", help="Loopback address to bind (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, help="Port to bind (default: 8000).")
    parser.add_argument("--no-preload", action="store_true", help="Load the model on the first request instead.")
    args = parser.parse_args(argv)

    settings = Settings.from_env()
    changes = {}
    if args.condition is not None:
        changes["condition"] = args.condition
    if args.device is not None:
        changes["device"] = args.device
    if args.host is not None:
        changes["host"] = args.host
    if args.port is not None:
        changes["port"] = args.port
    if args.no_preload:
        changes["preload"] = False
    if changes:
        settings = replace(settings, **changes)

    import uvicorn

    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="info")


if __name__ == "__main__":
    main()
