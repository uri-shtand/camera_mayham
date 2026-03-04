"""
Camera Mayham — entry point.

Run with::

    python main.py

Optional CLI arguments::

    python main.py --camera 0 --width 1280 --height 720 --log-level INFO
"""

from __future__ import annotations

import argparse
import logging
import sys


def _parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed argument values.
    """
    parser = argparse.ArgumentParser(
        prog="camera_mayham",
        description=(
            "Camera Mayham — GPU-accelerated live camera playground "
            "with face tracking, filters, 3D overlays and mini-games."
        ),
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=0,
        metavar="DEVICE",
        help="Camera device index (default: 0)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Render resolution width in pixels (default: 1280)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Render resolution height in pixels (default: 720)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args()


def _configure_logging(level: str) -> None:
    """
    Configure the root logger with a simple timestamp format.

    Parameters:
        level (str): One of DEBUG, INFO, WARNING, ERROR.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def main() -> None:
    """
    Parse arguments, configure logging, and run the application.

    Entry point registered in setup.cfg / pyproject.toml if used as a
    package.
    """
    args = _parse_args()
    _configure_logging(args.log_level)

    # Deferred import so logging is configured before any module-level
    # code runs inside the app package.
    from app.application import Application  # noqa: PLC0415

    app = Application(
        camera_device_id=args.camera,
        width=args.width,
        height=args.height,
    )
    app.run()


if __name__ == "__main__":
    main()
