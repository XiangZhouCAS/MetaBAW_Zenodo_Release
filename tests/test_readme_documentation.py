from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
from pathlib import Path
import re
import shlex
import unittest
from unittest.mock import patch

from metabaw.cli import build_parser
from metabaw.dependencies import ISOLATED_TOOLS


class ReadmeDocumentationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path(__file__).resolve().parents[1]
        self.readme = (self.root / "README.md").read_text(encoding="utf-8")

    def test_all_bash_metabaw_examples_parse_without_running_workflows(self):
        commands: list[str] = []
        for block in re.findall(r"^```bash\n(.*?)^```", self.readme, re.M | re.S):
            lines = iter(block.splitlines())
            for raw in lines:
                command = raw.strip()
                if not command.startswith("metabaw "):
                    continue
                while command.endswith("\\"):
                    continuation = next(lines, None)
                    self.assertIsNotNone(continuation, command)
                    command = command[:-1] + " " + continuation.strip()
                commands.append(command)
        self.assertGreaterEqual(len(commands), 16)
        for command in commands:
            with self.subTest(command=command):
                captured = io.StringIO()
                with (
                    redirect_stdout(captured), redirect_stderr(captured),
                    patch("metabaw.cli._page_text"),
                ):
                    try:
                        build_parser().parse_args(shlex.split(command, comments=True)[1:])
                    except SystemExit as exc:
                        self.assertEqual(exc.code, 0, captured.getvalue())

    def test_every_visible_public_option_is_documented(self):
        parser = build_parser()
        subparsers = next(
            action for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        for name, command in {"metabaw": parser, **subparsers.choices}.items():
            for action in command._actions:
                if not action.option_strings or action.help == argparse.SUPPRESS:
                    continue
                with self.subTest(module=name, options=action.option_strings):
                    self.assertTrue(any(
                        re.search(re.escape(option) + r"(?![A-Za-z0-9_-])", self.readme)
                        for option in action.option_strings
                    ))

    def test_markdown_code_fences_are_complete(self):
        inside = False
        for line in self.readme.splitlines():
            if line.startswith("```"):
                if inside:
                    self.assertEqual(line.strip(), "```")
                inside = not inside
        self.assertFalse(inside)

    def test_local_links_and_contents_targets_exist(self):
        prose = re.sub(r"^```[^\n]*\n.*?^```[ \t]*$", "", self.readme, flags=re.M | re.S)
        anchors = {
            re.sub(r"[^\w -]", "", heading.lower()).replace(" ", "-")
            for heading in re.findall(r"^#{1,6} (.+)$", prose, re.M)
        }
        links = re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", prose)
        self.assertGreaterEqual(len(links), 15)
        for target in links:
            with self.subTest(target=target):
                if target.startswith("#"):
                    self.assertIn(target[1:], anchors)
                elif "://" not in target:
                    local = (self.root / target.split("#", 1)[0]).resolve()
                    self.assertTrue(local.is_relative_to(self.root))
                    self.assertTrue(local.is_file())

    def test_manifest_examples_use_real_tabs_and_two_fields(self):
        checked = 0
        for block in re.findall(r"^```text\n(.*?)^```", self.readme, re.M | re.S):
            if not block.startswith("#sample"):
                continue
            checked += 1
            for row in block.splitlines():
                if not row:
                    continue
                self.assertNotIn(r"\t", row)
                self.assertEqual(len(row.split("\t")), 2, row)
        self.assertEqual(checked, 3)

    def test_all_managed_environment_names_and_options_are_documented(self):
        for spec in ISOLATED_TOOLS.values():
            with self.subTest(tool=spec.key):
                self.assertIn(spec.default_environment, self.readme)
                self.assertIn(spec.option, self.readme)
                self.assertIn(f"Python {spec.python_version}", self.readme)


if __name__ == "__main__":
    unittest.main()
