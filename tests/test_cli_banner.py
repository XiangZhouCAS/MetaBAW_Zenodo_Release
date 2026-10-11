from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
import re
import unittest
from unittest.mock import Mock, patch

from metabaw.cli import build_parser, command_bin, command_annotation, command_check, main


class CliBannerTests(unittest.TestCase):
    def test_root_help_uses_capitalized_workflow_name(self) -> None:
        help_text = build_parser().format_help()
        self.assertIn("MetaBAW: Metagenome Binning Automated Workflow", help_text)
        self.assertNotIn("metagenome Binning Automated Workflow", help_text)

    def test_only_canonical_subcommands_are_registered_and_documented(self) -> None:
        parser = build_parser()
        action = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
        self.assertEqual(tuple(action.choices), ("bin", "annotation", "check"))
        help_text = parser.format_help()
        self.assertNotIn("{bin,annotation,check}", help_text)
        self.assertIn("Module options:", help_text)
        self.assertNotIn("positional arguments:", help_text)
        self.assertNotIn("binning", action.choices)
        self.assertNotIn("checking", action.choices)
        self.assertEqual(parser.parse_args(["bin", "--input_reads_files", "reads.tsv"]).func, command_bin)
        self.assertEqual(parser.parse_args([
            "annotation", "--input_reads_files", "reads.tsv", "--input_genome_files", "genomes.txt",
        ]).func, command_annotation)
        args = parser.parse_args(["check", "--scope", "bin"])
        self.assertEqual(args.command, "check")
        self.assertEqual(args.func, command_check)

    def test_root_help_lists_modules_directly_without_a_choice_header(self) -> None:
        text = build_parser().format_help()
        self.assertEqual(text.count("Module options:"), 1)
        self.assertLess(text.index("Module options:"), text.index("\noptions:"))
        self.assertIn("usage: metabaw [-h] [-v] MODULE ...", text)
        self.assertNotRegex(text, r"(?m)^\s*MODULE\s*$")
        self.assertNotIn("{bin,annotation,check}", text)
        for name, description in (
            ("bin", "run sequence mapping"), ("annotation", "run taxonomic assignment"),
            ("check", "check software"),
        ):
            pattern = rf"(?m)^  {name}\s+{re.escape(description)}"
            self.assertEqual(len(re.findall(pattern, text)), 1)
        self.assertIn("options:", text)
        self.assertIn("-h, --help", text)
        self.assertIn("-v, --version", text)

    def test_root_help_uses_requested_module_descriptions(self) -> None:
        parser = build_parser()
        action = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
        expected = {
            "bin": "run sequence mapping, binning, refinement, quality control and dereplication",
            "annotation": (
                "run taxonomic assignment, abundance estimation, niche classification, "
                "KEGG Orthologues (KOs), Carbohydrate-Active enZYmes (CAZymes) and "
                "hydrogen metabolism annotations"
            ),
            "check": "check software, runtime environments and databases",
        }
        self.assertEqual({child.dest: child.help for child in action._get_subactions()}, expected)
        help_text = " ".join(parser.format_help().split())
        for name, description in expected.items():
            with self.subTest(module=name):
                self.assertIn(f"{name} {description}", help_text)

    def test_root_short_and_long_help_use_the_same_layout(self) -> None:
        expected = build_parser().format_help()
        for flag in ("-h", "--help"):
            with self.subTest(flag=flag):
                output = io.StringIO()
                with redirect_stdout(output), self.assertRaises(SystemExit) as stopped:
                    main([flag])
                self.assertEqual(stopped.exception.code, 0)
                self.assertEqual(output.getvalue(), expected)

    def test_help_layout_does_not_make_module_optional_or_change_subcommand_usage(self) -> None:
        error = io.StringIO()
        with redirect_stderr(error), self.assertRaises(SystemExit) as stopped:
            build_parser().parse_args([])
        self.assertEqual(stopped.exception.code, 2)
        self.assertIn("the following arguments are required: MODULE", error.getvalue())
        for name in ("bin", "annotation", "check"):
            with self.subTest(module=name):
                parser = build_parser()
                action = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
                self.assertEqual(action.choices[name].prog, "metabaw " + name)
                self.assertNotIn("Module options:", action.choices[name].format_help())

    def test_removed_aliases_are_rejected_before_execution(self) -> None:
        for arguments in (["binning"], ["binning", "-h"], ["binning", "--input_reads_files", "reads.tsv"],
                          ["checking"], ["checking", "-h"], ["checking", "--all"],
                          ["check", "--scope", "binning"]):
            with self.subTest(arguments=arguments):
                stdout, stderr = io.StringIO(), io.StringIO()
                with (patch("metabaw.cli.command_bin") as binning,
                      patch("metabaw.cli.command_check") as checking,
                      redirect_stdout(stdout), redirect_stderr(stderr),
                      self.assertRaises(SystemExit) as stopped):
                    main(arguments)
                self.assertEqual(stopped.exception.code, 2)
                self.assertIn("invalid choice", stderr.getvalue())
                self.assertEqual(stdout.getvalue(), "")
                binning.assert_not_called()
                checking.assert_not_called()

    def test_check_dispatch_and_help_preserve_dependency_options(self) -> None:
        stdout, stderr = io.StringIO(), io.StringIO()
        with (patch("metabaw.cli.command_check", return_value=0) as checking,
              redirect_stdout(stdout), redirect_stderr(stderr), self.assertRaises(SystemExit) as stopped):
            main(["check", "--all"])
        self.assertEqual(stopped.exception.code, 0)
        checking.assert_called_once()
        self.assertTrue(checking.call_args.args[0].all)
        self.assertIn("Command: metabaw check --all", stdout.getvalue())
        self.assertEqual(stderr.getvalue(), "")
        for flag in ("-h", "--help"):
            with self.subTest(flag=flag):
                stdout = io.StringIO()
                with redirect_stdout(stdout), self.assertRaises(SystemExit) as stopped:
                    build_parser().parse_args(["check", flag])
                self.assertEqual(stopped.exception.code, 0)
                text = stdout.getvalue()
                self.assertIn("usage: metabaw check", text)
                for option in ("--all", "--essential", "--scope", "--checkm2-db", "--gtdbtk-db"):
                    self.assertIn(option, text)

    def test_check_scope_only_accepts_workflow_modules(self) -> None:
        parser = build_parser()
        subparsers = next(action for action in parser._actions if isinstance(action, argparse._SubParsersAction))
        checking = subparsers.choices["check"]
        scope = next(action for action in checking._actions if action.dest == "scope")
        self.assertEqual(scope.choices, ("bin", "annotation"))
        self.assertIn("--scope {bin,annotation}", checking.format_help())
        for module in scope.choices:
            with self.subTest(module=module):
                self.assertEqual(parser.parse_args(["check", "--scope", module]).scope, module)
        for invalid in (
            "kegg", "cazy", "hydrogenase", "checkm", "minibwa", "comebin",
            "checkm2", "metawrap", "lorbin", "all", "essential",
        ):
            with self.subTest(invalid=invalid):
                stdout, stderr = io.StringIO(), io.StringIO()
                with (patch("metabaw.cli.command_check") as check,
                      redirect_stdout(stdout), redirect_stderr(stderr),
                      self.assertRaises(SystemExit) as stopped):
                    main(["check", "--scope", invalid])
                self.assertEqual(stopped.exception.code, 2)
                self.assertIn("invalid choice", stderr.getvalue())
                self.assertEqual(stdout.getvalue(), "")
                check.assert_not_called()
        self.assertTrue(parser.parse_args(["check", "--all"]).all)
        self.assertTrue(parser.parse_args(["check", "--essential"]).essential)

    def _run_main(self, return_code: int) -> tuple[int, list[str], str]:
        parsed = argparse.Namespace(
            command="bin",
            func=lambda _args: return_code,
        )
        parser = Mock()
        parser.parse_args.return_value = parsed
        stdout = io.StringIO()
        stderr = io.StringIO()
        with patch("metabaw.cli.build_parser", return_value=parser):
            with redirect_stdout(stdout), redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as stopped:
                    main(["bin", "-p", "/data/reads with spaces"])
        return int(stopped.exception.code), stdout.getvalue().splitlines(), stderr.getvalue()

    def test_success_prints_banner_command_and_final_marker(self) -> None:
        code, lines, stderr = self._run_main(0)

        self.assertEqual(code, 0)
        self.assertEqual(lines[0], "MetaBAW is Running.")
        self.assertEqual(
            lines[1],
            "Command: metabaw bin -p '/data/reads with spaces'",
        )
        self.assertEqual(lines[-1], "ALL DONE.")
        self.assertEqual(stderr, "")

    def test_failed_workflow_does_not_print_final_marker(self) -> None:
        code, lines, _stderr = self._run_main(1)

        self.assertEqual(code, 1)
        self.assertEqual(lines[:2], [
            "MetaBAW is Running.",
            "Command: metabaw bin -p '/data/reads with spaces'",
        ])
        self.assertNotIn("ALL DONE.", lines)


if __name__ == "__main__":
    unittest.main()
