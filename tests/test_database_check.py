from __future__ import annotations

import argparse
import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from metabaw.cli import (
    _preflight_bin,
    _prompt_database_path,
    build_parser,
    command_check,
)
from metabaw.dependencies import (
    CudaRuntimeStatus,
    HostCudaStatus,
    checkm_database_missing,
    checkm_database_valid,
    diamond_database_missing,
    diamond_database_valid,
    gtdbtk_database_missing,
    gtdbtk_database_valid,
    gtdbtk_classification_database_missing,
    gtdbtk_classification_database_valid,
)


class DatabaseCheckTests(unittest.TestCase):
    def test_diamond_database_must_be_a_nonempty_dmnd_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            database = root / "database.dmnd"
            database.write_bytes(b"")
            self.assertFalse(diamond_database_valid(database))
            self.assertIn("empty", " ".join(diamond_database_missing(database)))
            database.write_bytes(b"DIAMOND")
            self.assertTrue(diamond_database_valid(database))
            with patch.object(Path, "open", side_effect=PermissionError("not readable")):
                self.assertFalse(diamond_database_valid(database))
                self.assertIn("unreadable", " ".join(diamond_database_missing(database)))
            wrong_suffix = root / "database.bin"
            wrong_suffix.write_bytes(b"DIAMOND")
            self.assertFalse(diamond_database_valid(wrong_suffix))

    def test_checkm_database_requires_its_data_root_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertFalse(checkm_database_valid(root))
            for path in (
                root / ".dmanifest",
                root / "hmms" / "phylo.hmm",
                root / "hmms" / "checkm.hmm",
                root / "pfam" / "Pfam-A.hmm.dat",
                root / "genome_tree" / "reference.tre",
                root / "distributions" / "marker.tsv",
                root / "selected_marker_sets.tsv",
                root / "taxon_marker_sets.tsv",
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("data", encoding="utf-8")
            self.assertTrue(checkm_database_valid(root))
            self.assertEqual(checkm_database_missing(root), ())

    def test_gtdbtk_database_rejects_a_parent_or_incomplete_root(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "release232"
            root.mkdir()
            self.assertFalse(gtdbtk_database_valid(parent))
            required_files = (
                root / "metadata" / "metadata.txt",
                root / "taxonomy" / "gtdb_taxonomy.tsv",
                root / "markers" / "marker.hmm",
                root / "masks" / "marker.mask",
                root / "msa" / "marker.faa",
                root / "pplacer" / "reference.refpkg" / "tree",
                root / "radii" / "radii.tsv",
                root / "skani" / "database" / "genome.msh",
            )
            for path in required_files:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("data", encoding="utf-8")
            self.assertTrue(gtdbtk_database_valid(root))
            self.assertEqual(gtdbtk_database_missing(root), ())
            self.assertFalse(gtdbtk_classification_database_valid(root))
            self.assertEqual(gtdbtk_classification_database_missing(root), ("mrca_red/",))
            red = root / "mrca_red" / "bac120.tsv"
            red.parent.mkdir()
            red.write_text("data", encoding="utf-8")
            self.assertTrue(gtdbtk_classification_database_valid(root))

    def test_interactive_path_is_validated_and_saved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary) / "checkm2.dmnd"
            database.write_bytes(b"DIAMOND")
            with (
                patch("metabaw.cli.sys.stdin.isatty", return_value=True),
                patch("builtins.input", return_value=str(database)),
                patch("metabaw.cli.save_database_path") as save,
            ):
                selected = _prompt_database_path(
                    "CheckM2",
                    None,
                    diamond_database_valid,
                    "checkm2",
                    artifact_suffix=".dmnd",
                )
            self.assertEqual(selected, database.resolve())
            save.assert_called_once_with("checkm2", database.resolve())

    def test_check_parser_accepts_legacy_checkm_database_path(self) -> None:
        args = build_parser().parse_args(
            ["check", "--all", "--checkm-db", "/db/checkm"]
        )
        self.assertEqual(args.checkm_db, "/db/checkm")

    def test_checkm_database_is_exported_for_downstream_tools(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            database = Path(temporary)
            for path in (
                database / ".dmanifest",
                database / "hmms" / "phylo.hmm",
                database / "hmms" / "checkm.hmm",
                database / "pfam" / "Pfam-A.hmm.dat",
                database / "genome_tree" / "reference.tre",
                database / "distributions" / "marker.tsv",
                database / "selected_marker_sets.tsv",
                database / "taxon_marker_sets.tsv",
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("data", encoding="utf-8")
            args = argparse.Namespace(
                align_tool="bowtie2",
                tools=["metabat2"],
                type="short",
                refinement="das_tool",
                quality_control="checkm",
                gunc=False,
                trna=False,
                rrna=False,
                dereplication_tool="galah",
                gpu=False,
            )
            with (
                patch("metabaw.cli._ensure_software"),
                patch("metabaw.cli.configured_database_path", return_value=database),
                patch.dict(os.environ, {"CHECKM_DATA_PATH": "old"}),
            ):
                _preflight_bin(args)
                self.assertEqual(os.environ["CHECKM_DATA_PATH"], str(database))

    def test_all_mode_reports_every_supported_database(self) -> None:
        args = build_parser().parse_args(["check", "--all"])
        host = HostCudaStatus(None, (), None, "not available")
        runtime = CudaRuntimeStatus(
            "MetaBAW environment",
            None,
            None,
            None,
            None,
            0,
            (),
            False,
            "not available",
        )
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch("metabaw.cli._check_software_interactively", return_value=True),
            patch("metabaw.cli.ISOLATED_TOOLS", {}),
            patch("metabaw.cli.host_cuda_status", return_value=host),
            patch("metabaw.cli._main_cuda_status", return_value=runtime),
            patch("metabaw.cli.magscot_files", return_value=()),
            patch("metabaw.cli.save_magscot_directory"),
            patch("metabaw.cli.configured_database_path", return_value=None),
            patch("metabaw.cli.sys.stdin.isatty", return_value=False),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            code = command_check(args)
        report = stdout.getvalue() + stderr.getvalue()
        self.assertEqual(code, 1)
        for name in (
            "CheckM database",
            "CheckM2 database",
            "GUNC database",
            "GTDB-Tk database",
            "KOfam/KEGG database",
            "dbCAN/CAZy database",
            "Hydrogenase database",
        ):
            self.assertIn(name, report)

    def test_annotation_check_routes_gtdbtk_to_database_path_prompt(self) -> None:
        args = build_parser().parse_args(["check", "--scope", "annotation"])
        database = Path("/db/release232")
        host = HostCudaStatus(None, (), None, "not available")
        with (
            patch("metabaw.cli._check_software_interactively", return_value=True),
            patch("metabaw.cli.host_cuda_status", return_value=host),
            patch("metabaw.cli.configured_database_path", return_value=database),
            patch("metabaw.cli.gtdbtk_database_valid", return_value=True),
            patch("metabaw.cli.kofam_database_valid", return_value=True),
            patch("metabaw.cli.dbcan_database_valid", return_value=True),
            patch("metabaw.cli.hydrogenase_database_valid", return_value=True),
            patch("metabaw.cli.save_database_path"),
            patch(
                "metabaw.cli._prompt_database_path",
                return_value=database,
            ) as prompt,
        ):
            code = command_check(args)

        self.assertEqual(code, 0)
        prompt.assert_called_once()
        self.assertEqual(prompt.call_args.args[0], "GTDB-Tk")
        self.assertIs(prompt.call_args.kwargs["missing_reasons"], gtdbtk_database_missing)

    def test_check_bin_scope_also_validates_checkm2_database(self) -> None:
        args = build_parser().parse_args(["check", "--scope", "bin"])
        host = HostCudaStatus(None, (), None, "not available")
        runtime = CudaRuntimeStatus("MetaBAW environment", None, None, None, None, 0, (), False, "not available")
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            patch("metabaw.cli._check_software_interactively", return_value=True),
            patch("metabaw.cli._check_isolated_tools_interactively", return_value=({}, True)),
            patch("metabaw.cli.host_cuda_status", return_value=host),
            patch("metabaw.cli._main_cuda_status", return_value=runtime),
            patch("metabaw.cli.magscot_files", return_value=()),
            patch("metabaw.cli.save_magscot_directory"),
            patch("metabaw.cli.configured_database_path", return_value=None) as configured,
            patch("metabaw.cli.confirm_install", return_value=False),
            contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr),
        ):
            code = command_check(args)
        self.assertEqual(code, 1)
        self.assertIn("CheckM2 database", stdout.getvalue() + stderr.getvalue())
        self.assertTrue(any(call.args[2] == "checkm2" for call in configured.call_args_list))


if __name__ == "__main__":
    unittest.main()
