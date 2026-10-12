"""Visible terminal columns, color boundaries and readable startup logs."""
from argparse import Namespace
from contextlib import redirect_stdout
import io
import os
from unittest.mock import patch

import pytest

from metabaw.dependencies import SoftwareRequirement, print_software_status
from metabaw.startup_log import capture_startup
from metabaw.terminal import plain_text, status_tag


class Terminal(io.StringIO):
    def isatty(self):
        return True


@pytest.mark.parametrize("terminal", [False, True])
@pytest.mark.parametrize("long_name", [False, True])
def test_tags_have_one_space_before_names_and_descriptions_align(terminal, long_name):
    output = Terminal() if terminal else io.StringIO()
    present = "a_very_long_executable_name_that_exceeds_the_old_column" if long_name else "samtools"
    requirements = [SoftwareRequirement(present, present, "present description"),
                    SoftwareRequirement("SemiBin2", "semibin", "missing description")]
    with patch.dict(os.environ, {"TERM": "xterm"}, clear=True), redirect_stdout(output), \
            patch("metabaw.dependencies.shutil.which", side_effect=lambda name: "/bin/tool" if name == present else None):
        print_software_status(requirements)
    colored = output.getvalue()
    rows = plain_text(colored).splitlines()
    assert rows[0].index("present description") == rows[1].index("missing description")
    assert rows[0].startswith(f"[OK] {present}")
    assert rows[1].startswith("[MISSING] SemiBin2")
    assert "Conda package: semibin" in rows[1]
    if terminal:
        assert "\x1b[31m[MISSING]\x1b[0m" in colored
        assert "\x1b[32m[OK]\x1b[0m" in colored
    else:
        assert "\x1b" not in colored


@pytest.mark.parametrize("environment", [{"NO_COLOR": "1"}, {"TERM": "dumb"}])
def test_terminal_color_can_be_disabled(environment):
    with patch.dict(os.environ, environment, clear=True), redirect_stdout(Terminal()):
        assert status_tag("MISSING") == "[MISSING]"


def test_colored_startup_statuses_keep_existing_routing_and_plain_logs(tmp_path):
    args = Namespace(output=tmp_path, _invocation=["metabaw", "bin"])
    output = Terminal()

    @capture_startup()
    def command(args):
        print(status_tag("OK") + " healthy software", flush=True)
        print(status_tag("MISSING") + " missing software", flush=True)

    with patch.dict(os.environ, {"TERM": "xterm"}, clear=True), redirect_stdout(output):
        command(args)
    assert "healthy software" not in output.getvalue()
    assert "\x1b[31m[MISSING]\x1b[0m missing software" in output.getvalue()
    recorded = (tmp_path / "start_info.txt").read_text(encoding="utf-8")
    assert "[OK] healthy software" in recorded
    assert "[MISSING] missing software" in recorded
    assert "\x1b" not in recorded
