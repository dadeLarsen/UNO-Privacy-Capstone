"""Run the API with ``python -m api_service``."""

import argparse

import uvicorn

from .app import create_app
from .settings import LOOPBACK_HOSTS, Settings


def main() -> None:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description="Privacy Policy Reader API")
    parser.add_argument("--host", default=settings.host, help="Loopback host to bind")
    parser.add_argument("--port", type=int, default=settings.port, help="Port to bind")
    args = parser.parse_args()
    if args.host not in LOOPBACK_HOSTS:
        parser.error("--host must be a loopback address")
    uvicorn.run(create_app(settings), host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
