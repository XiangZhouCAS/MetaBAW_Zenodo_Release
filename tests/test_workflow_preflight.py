from __future__ import annotations

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from metabaw.cli import (
    _ensure_isolated_tool, _ensure_software, _missing_r_dependency_groups, _preflight_annotation,
    _preflight_bin, build_parser, main,
)
from metabaw.dependencies import (
    ISOLATED_TOOLS, SOFTWARE, IsolatedEnvironmentStatus,
    SoftwareRuntimeIssue, annotation_requirements, bin_requirements,
)


class WorkflowPreflightTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        # Starting binning/annotation must never reach a prompt or an installer.
        stack = ExitStack()
        self.addCleanup(stack.close)
        self.installers = {}
        for name in (
            "confirm_install", "install_software", "install_isolated_environment",
            "install_magscot", "install_checkm2_database", "install_gunc_database",
            "install_gtdbtk_database", "install_kofam_database", "install_dbcan_database",
            "install_main_cuda_runtime", "install_isolated_cuda_runtime",
            "install_main_cupy_runtime",
            "save_database_path", "save_magscot_directory",
        ):
            self.installers[name] = stack.enter_context(
                patch(f"metabaw.cli.{name}", side_effect=AssertionError(f"unexpected {name}"))
            )

    def bin_args(self, *flags):
        args = build_parser().parse_args([
            "bin", "--input_reads_files", "reads.tsv", "--tools", "metabat2",
            "--refinement", "das_tool", "--quality-control", "checkm",
            "--no-gpu", "-o", str(self.root / "output"), *flags,
        ])
        args.align_tool = "bowtie2"
        args.type = "short"
        args.required_assemblers = ()
        args.magscot_dir = str(self.root / "magscot")
        for key, spec in ISOLATED_TOOLS.items():
            setattr(args, f"{key}_env", spec.default_environment)
        return args

    def annotation_args(self, *flags):
        return build_parser().parse_args([
            "annotation", "--input_reads_files", "reads.tsv",
            "--input_genome_files", "genomes.txt", "-o", str(self.root / "output"),
            *flags,
        ])

    def unavailable(self, key, environment):
        return IsolatedEnvironmentStatus(
            ISOLATED_TOOLS[key], environment, None, None, None, "environment missing",
        )

    def test_software_check_lists_programs_r_packages_and_broken_runtime(self):
        with (
            patch("metabaw.cli.missing_software", return_value=[SOFTWARE["megahit"], SOFTWARE["samtools"]]),
            patch("metabaw.cli._missing_r_dependency_groups", return_value={"MAGScoT": ["dplyr"]}),
            patch("metabaw.cli.software_runtime_issues", return_value=[
                SoftwareRuntimeIssue("barrnap", "cannot import dependency", ("barrnap",)),
            ]),
        ):
            with self.assertRaises(RuntimeError) as failed:
                _ensure_software([SOFTWARE["megahit"]], "bin")
        for name in ("megahit", "samtools", "dplyr", "barrnap", "cannot import dependency"):
            self.assertIn(name, str(failed.exception))

    def test_missing_isolated_environment_does_not_prompt_or_install(self):
        with patch("metabaw.cli.isolated_environment_status", return_value=self.unavailable("comebin", "missing")):
            with self.assertRaisesRegex(RuntimeError, "COMEBin.*missing"):
                _ensure_isolated_tool(ISOLATED_TOOLS["comebin"], "missing")

    def test_isolated_repair_hints_use_supported_environment_options(self):
        for key, spec in ISOLATED_TOOLS.items():
            environment = f"/envs/missing-{key}"
            with (
                self.subTest(tool=key),
                patch("metabaw.cli.isolated_environment_status", return_value=self.unavailable(key, environment)),
                self.assertRaises(RuntimeError) as failed,
            ):
                _ensure_isolated_tool(spec, environment)
            self.assertIn(f"metabaw check {spec.option} {environment}", str(failed.exception))
            self.assertNotIn("--scope", str(failed.exception))
            args = build_parser().parse_args(["check", spec.option, environment])
            self.assertIsNone(args.scope)
            self.assertEqual(getattr(args, f"{key}_env"), environment)

    def test_wrong_python_lorbin_preflight_stays_read_only_and_points_to_check(self):
        spec = ISOLATED_TOOLS["lorbin"]
        wrong = IsolatedEnvironmentStatus(
            spec, spec.default_environment, "conda", "3.11", None,
            "Python 3.11 is incompatible; LorBin requires Python 3.10",
        )
        with patch("metabaw.cli.isolated_environment_status", return_value=wrong):
            with self.assertRaisesRegex(RuntimeError, "metabaw check --lorbin-env"):
                _ensure_isolated_tool(spec, spec.default_environment)
        for installer in self.installers.values():
            installer.assert_not_called()

    def test_bin_collects_all_missing_components_and_skips_cuda(self):
        args = self.bin_args(
            "--tools", "comebin", "--refinement", "magscot",
            "--quality-control", "checkm2", "--gunc", "--trna", "--rrna",
        )
        args.gpu = None
        errors = io.StringIO()
        with (
            patch("metabaw.cli._ensure_software", side_effect=RuntimeError("bowtie2 is missing")),
            patch("metabaw.cli.isolated_environment_status", side_effect=lambda spec, env: self.unavailable(spec.key, env)) as isolated,
            patch("metabaw.cli.configured_database_path", return_value=None) as databases,
            patch("metabaw.cli._enable_bin_cuda") as cuda,
            redirect_stderr(errors),
        ):
            with self.assertRaisesRegex(RuntimeError, "no workflow tasks were started"):
                _preflight_bin(args)
        for name in ("bowtie2", "COMEBin", "CheckM2", "MAGScoT.R", "gtdbtk_rel207_Pfam-A.hmm", "GUNC", "GTDB-Tk"):
            self.assertIn(name, errors.getvalue())
        self.assertEqual({call.args[0].key for call in isolated.call_args_list}, {"comebin", "checkm2"})
        self.assertEqual({call.args[2] for call in databases.call_args_list}, {"checkm2", "gunc", "gtdbtk"})
        cuda.assert_not_called()

    def test_annotation_reports_all_missing_database_assets(self):
        args = self.annotation_args()
        empty = self.root / "databases"
        empty.mkdir()
        errors = io.StringIO()
        with (
            patch("metabaw.cli._ensure_software", side_effect=RuntimeError("coverm is missing")),
            patch("metabaw.cli.configured_database_path", return_value=empty) as databases,
            redirect_stderr(errors),
        ):
            with self.assertRaises(RuntimeError):
                _preflight_annotation(args)
        for name in (
            "coverm", "GTDB-Tk", "metadata.txt", "KOfam/KEGG", "ko_list",
            "dbCAN/CAZy", "CAZy.dmnd", "Hydrogenase", "hyddb.all.fa", "Terminal.dmnd",
        ):
            self.assertIn(name, errors.getvalue())
        self.assertEqual({call.args[2] for call in databases.call_args_list}, {"gtdbtk", "kofam", "dbcan", "hydrogenase"})

    def test_disabled_modules_and_reused_taxonomy_need_no_databases(self):
        args = self.annotation_args("--gtdbtk_res", "existing", "--no-kegg", "--no-cazy", "--no-hyd")
        with (
            patch("metabaw.cli._ensure_software") as software,
            patch("metabaw.cli.configured_database_path") as databases,
            redirect_stdout(io.StringIO()),
        ):
            _preflight_annotation(args)
        executables = {item.executable for item in software.call_args.args[0]}
        self.assertEqual(executables, {"bash", "coverm", "minimap2", "samtools"})
        databases.assert_not_called()

    def test_empty_magscot_marker_file_is_rejected_before_tasks(self):
        args = self.bin_args("--refinement", "magscot")
        directory = Path(args.magscot_dir)
        (directory / "hmm").mkdir(parents=True)
        (directory / "MAGScoT.R").write_text("script")
        (directory / "hmm" / "gtdbtk_rel207_Pfam-A.hmm").touch()
        (directory / "hmm" / "gtdbtk_rel207_tigrfam.hmm").write_text("HMM")
        errors = io.StringIO()
        with (
            patch("metabaw.cli._ensure_software"),
            patch("metabaw.cli.configured_database_path", return_value=None),
            redirect_stdout(io.StringIO()), redirect_stderr(errors),
        ):
            with self.assertRaises(RuntimeError):
                _preflight_bin(args)
        self.assertIn("gtdbtk_rel207_Pfam-A.hmm (empty)", errors.getvalue())

    def test_description_table_format_failure_is_caught_at_startup(self):
        args = self.annotation_args("--gtdbtk_res", "existing", "--no-cazy", "--no-hyd")
        database = self.root / "kofam"
        (database / "profiles").mkdir(parents=True)
        (database / "profiles" / "K00001.hmm").write_text("HMM")
        (database / "ko_list").write_text("knum\tdefinition\nK00001\tenzyme\n")
        (database / "K.descript.txt").write_text("not_a_KO\tenzyme\n")
        errors = io.StringIO()
        with (
            patch("metabaw.cli._ensure_software"),
            patch("metabaw.cli.configured_database_path", return_value=database),
            redirect_stdout(io.StringIO()), redirect_stderr(errors),
        ):
            with self.assertRaises(RuntimeError):
                _preflight_annotation(args)
        self.assertIn("KEGG description table", errors.getvalue())
        self.assertIn("Invalid KO identifier", errors.getvalue())

    def test_old_kofam_database_can_derive_descriptions_locally(self):
        args = self.annotation_args("--gtdbtk_res", "existing", "--no-cazy", "--no-hyd")
        database = self.root / "kofam"
        (database / "profiles").mkdir(parents=True)
        (database / "profiles" / "K00001.hmm").write_text("HMM")
        (database / "ko_list").write_text("knum\tdefinition\nK00001\tenzyme\n")
        with (
            patch("metabaw.cli._ensure_software"),
            patch("metabaw.cli.configured_database_path", return_value=database),
            redirect_stdout(io.StringIO()),
        ):
            _preflight_annotation(args)
        self.assertEqual(args.kegg_db, str(database))
        self.assertIn("K00001\tenzyme", (database / "K.descript.txt").read_text())

    def test_drep_with_existing_quality_report_needs_no_checkm_database(self):
        args = self.bin_args("--quality-control", "checkm2", "--dereplication-tool", "drep")
        database = self.root / "checkm2.dmnd"
        database.write_bytes(b"DIAMOND")
        status = IsolatedEnvironmentStatus(
            ISOLATED_TOOLS["checkm2"], args.checkm2_env, "conda", "3.12", "checkm2",
        )
        with (
            patch("metabaw.cli._ensure_software"),
            patch("metabaw.cli._ensure_isolated_tool", return_value=status),
            patch("metabaw.cli.configured_database_path", return_value=database) as databases,
            redirect_stdout(io.StringIO()),
        ):
            _preflight_bin(args)
        self.assertEqual([call.args[2] for call in databases.call_args_list], ["checkm2"])
        self.assertEqual(args.checkm2_db, str(database))

    def test_requirements_include_selected_assemblers_and_search_helpers(self):
        args = self.bin_args("--refinement", "magscot", "--gunc")
        args.required_assemblers = ("megahit", "flye")
        binaries = {item.executable for item in bin_requirements(args)}
        self.assertTrue({"megahit", "flye", "gunc", "diamond", "prodigal"} <= binaries)
        self.assertNotIn("git", binaries)
        annotation = self.annotation_args("--gtdbtk_res", "existing", "--no-hyd")
        helpers = {item.executable for item in annotation_requirements(annotation)}
        self.assertTrue({"exec_annotation", "ruby", "parallel", "hmmsearch", "run_dbcan", "diamond"} <= helpers)
        self.assertNotIn("blastp", helpers)

    def test_classification_helpers_are_checked_only_for_new_taxonomy(self):
        args = self.annotation_args("--no-kegg", "--no-cazy", "--no-hyd")
        executables = {item.executable for item in annotation_requirements(args)}
        self.assertTrue({"prodigal", "hmmsearch", "hmmalign", "pplacer", "guppy", "skani"} <= executables)
        self.assertNotIn("FastTree", executables)
        args.gtdbtk_res = "existing"
        reused = {item.executable for item in annotation_requirements(args)}
        self.assertEqual(reused, {"bash", "coverm", "minimap2", "samtools"})

    def test_shared_helpers_do_not_require_unselected_refiner_r_packages(self):
        with (
            patch("metabaw.cli.missing_magscot_r_packages", return_value=["dplyr"]) as magscot,
            patch("metabaw.cli.missing_dastool_r_packages", return_value=["docopt"]) as dastool,
        ):
            self.assertEqual(_missing_r_dependency_groups(annotation_requirements(self.annotation_args())), {})
            magscot.assert_not_called()
            dastool.assert_not_called()
            self.assertEqual(
                _missing_r_dependency_groups(bin_requirements(self.bin_args("--refinement", "magscot"))),
                {"MAGScoT": ["dplyr"]},
            )
            dastool.assert_not_called()
            self.assertEqual(
                _missing_r_dependency_groups(bin_requirements(self.bin_args())),
                {"DAS Tool": ["docopt"]},
            )

    def test_read_error_does_not_hide_other_missing_dependencies(self):
        args = self.annotation_args()
        errors = io.StringIO()
        def database_path(_explicit, _environment, config, **_kwargs):
            if config == "gtdbtk":
                raise PermissionError("cannot read GTDB-Tk reference assets")
            return None
        with (
            patch("metabaw.cli._ensure_software"),
            patch("metabaw.cli.configured_database_path", side_effect=database_path),
            redirect_stderr(errors),
        ):
            with self.assertRaises(RuntimeError):
                _preflight_annotation(args)
        for problem in ("cannot read GTDB-Tk", "KOfam/KEGG", "dbCAN/CAZy", "Hydrogenase"):
            self.assertIn(problem, errors.getvalue())

    def test_cli_failure_is_logged_and_executor_is_never_started(self):
        (self.root / "r1.fastq").touch()
        (self.root / "r2.fastq").touch()
        (self.root / "contigs.fa").write_text(">contig_1\n" + "ACGT" * 500 + "\n")
        (self.root / "reads.tsv").write_text("S1\tr1.fastq,r2.fastq\n")
        (self.root / "contigs.tsv").write_text("S1\tcontigs.fa\n")
        (self.root / "genomes.txt").write_text("contigs.fa\n")
        for module in ("bin", "annotation"):
            output = self.root / module
            flags = (
                ["--input_contig_files", str(self.root / "contigs.tsv"),
                 "--tools", "metabat2", "--refinement", "das_tool", "--quality-control", "checkm", "--no-gpu"]
                if module == "bin" else ["--input_genome_files", str(self.root / "genomes.txt")]
            )
            errors = io.StringIO()
            with (
                self.subTest(module=module),
                patch("metabaw.cli._ensure_software", side_effect=RuntimeError("missing test_program")),
                patch("metabaw.cli.configured_database_path", return_value=None),
                patch("metabaw.cli.Executor") as executor,
                redirect_stdout(io.StringIO()), redirect_stderr(errors),
            ):
                for attempt in range(2):
                    with self.assertRaises(SystemExit) as failed:
                        main([module, "--input_reads_files", str(self.root / "reads.tsv"), *flags, "-o", str(output)])
                    self.assertNotEqual(failed.exception.code, 0)
            executor.assert_not_called()
            record = (output / "start_info.txt").read_text()
            self.assertEqual(record.count("=== MetaBAW startup:"), 2)
            self.assertIn("[MISSING]", record)
            self.assertIn("missing test_program", record)
            self.assertIn("[STARTUP ERROR]", record)
            self.assertIn(str(output.resolve() / "start_info.txt"), errors.getvalue())
            self.assertFalse((output / "tmp").exists())
            self.assertFalse((output / "run_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
