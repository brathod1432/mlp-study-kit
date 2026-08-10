"""
nn_core.logger -- single, self-contained ObjLogger for mlp-study-kit.

Replaces the copy-pasted inline ObjLogger that appeared in every
homework file (hw_01, hw_02, hw_03, testing_nn/*.py).

Usage:
    from nn_core.logger import ObjLogger, title_message

    logger = ObjLogger("MyScript")
    logger("Script started", color="green")
    title_message("Section Title", color="blue")
"""

from __future__ import annotations

import datetime


class ObjLogger:
    """
    Lightweight ANSI-colored console logger.

    Instantiate once per script/module:
        logger = ObjLogger("MyModule")
        logger("message", color="cyan")

    Available colors: blue, cyan, yellow, red, green, magenta, white
    """

    ANSI_COLORS: dict[str, str] = {
        "blue":    "\033[34m",
        "cyan":    "\033[36m",
        "yellow":  "\033[33m",
        "red":     "\033[31m",
        "green":   "\033[32m",
        "magenta": "\033[35m",
        "white":   "\033[37m",
        "reset":   "\033[0m",
    }

    def __init__(self, name: str = "Logger") -> None:
        self.name = name

    def __call__(self, message: object, color: str = "white") -> None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prefix = f"{timestamp}  [{self.name}]  "
        color_code = self.ANSI_COLORS.get(color.lower(), self.ANSI_COLORS["white"])
        reset_code = self.ANSI_COLORS["reset"]
        print(f"{prefix}{color_code}{message}{reset_code}")

    def info(self, message: object) -> None:
        self(message, color="white")

    def debug(self, message: object) -> None:
        self(message, color="cyan")

    def warning(self, message: object) -> None:
        self(message, color="yellow")

    def error(self, message: object) -> None:
        self(message, color="red")

    def success(self, message: object) -> None:
        self(message, color="green")


def title_message(msg: str, color: str = "blue", logger: ObjLogger | None = None) -> None:
    """
    Print a bordered title box.

        title_message("Training Started", color="magenta")

    If no logger is supplied a temporary one is created.
    """
    _log = logger or ObjLogger("Title")
    border = "#" * (len(str(msg)) + 10)
    _log(border, color=color)
    _log(f"#    {msg}    #", color=color)
    _log(border, color=color)
