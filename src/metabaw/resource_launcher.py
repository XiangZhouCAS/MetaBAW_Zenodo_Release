"""Set inherited CPU affinity before exec; no preexec_fn in worker threads."""
from __future__ import annotations

import os
import sys


def main():
    if len(sys.argv) < 4 or sys.argv[2] != "--":
        raise SystemExit("Expected CPU_IDS -- COMMAND...")
    identifiers = {int(value) for value in sys.argv[1].split(",") if value}
    if identifiers and hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, identifiers)
    os.execvpe(sys.argv[3], sys.argv[3:], os.environ)


if __name__ == "__main__":
    main()
