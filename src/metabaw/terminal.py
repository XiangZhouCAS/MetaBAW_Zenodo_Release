"""Terminal status colors and columns measured without escape sequences."""
from __future__ import annotations

import os
import re
import sys


_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def plain_text(text: str) -> str:
    return _ANSI.sub("", text)


def status_tag(state: str, width: int = 0, stream=None) -> str:
    label = f"[{state}]"
    stream = sys.stdout if stream is None else stream
    color = ("31" if state.endswith("MISSING") else
             "32" if state.endswith("OK") else None)
    if (color and getattr(stream, "isatty", lambda: False)()
            and "NO_COLOR" not in os.environ and os.environ.get("TERM") != "dumb"):
        rendered = f"\x1b[{color}m{label}\x1b[0m"
    else:
        rendered = label
    return rendered + " " * max(0, width - len(label))


def status_row(state: str, name: str, description: str, name_width: int = 34) -> str:
    # Keep one space after the tag; compensate for short tags after the name.
    name_width += max(0, 9 - len(f"[{state}]"))
    return f"{status_tag(state)} {name:<{name_width}} {description}"
