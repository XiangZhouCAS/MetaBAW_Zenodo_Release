from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import csv
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from metabaw.dependencies import (
    annotation_requirements,
    dbcan_database_missing,
    dbcan_database_valid,
    hydrogenase_database_missing,
    hydrogenase_database_valid,
    kofam_database_missing,
    kofam_database_valid,
)
from metabaw.cli import _preflight_annotation, build_parser, command_annotation
from metabaw.direct import AnnotationBuilder, AnnotationOptions
from metabaw.discovery import ReadSample
from metabaw.internal import (
    build_functional_gene_map,
    combine_functional_proteins,
    format_functional_annotations,
    gtdbtk_result_missing,
    gtdbtk_result_valid,
    merge_dbcan_annotations,
    filter_hydrogenase_hits,
    finalize_hydrogenase_hits,
    merge_hydrogenase_annotations,
    merge_kofam_annotations,
    filter_terminal_enzyme_hits,
    merge_terminal_enzyme_annotations,
    prepare_dbcan_database,
    prepare_samtools_compat,
    run_gtdbtk_classify,
    _gtdbtk_temp_environment,
    _multiprocessing_socket_path_too_long,
    _samtools_version,
)


class FunctionalAnnotationTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output = Path(temporary.name) / "annotation"

    @staticmethod
    def _write_gtdbtk_summary(directory: Path, genomes: tuple[str, ...]) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        summary = directory / "gtdbtk.bac120.summary.tsv"
        classification = (
            "d__Bacteria;p__Pseudomonadota;c__Gammaproteobacteria;"
            "o__Enterobacterales;f__Enterobacteriaceae;g__Escherichia;s__"
        )
        summary.write_text(
            "user_genome\tclassification\n"
            + "".join(f"{genome}\t{classification}\n" for genome in genomes),
            encoding="utf-8",
        )
        return summary

    def test_cli_enables_functional_annotation_and_accepts_check_paths(self) -> None:
        parser = build_parser()
        annotation = parser.parse_args(
            ["annotation", "--input_genome_files", "mags", "--input_reads_files", "reads"]
        )
        self.assertGreater(annotation.max_memory, 0)
        self.assertTrue(annotation.kegg)
        self.assertTrue(annotation.cazy)
        self.assertTrue(annotation.hydrogenase)
        self.assertTrue(annotation.place_species)
        self.assertNotIn("--no-place-species", annotation._provided_options)
        disabled = parser.parse_args(
            [
                "annotation",
                "--input_genome_files",
                "mags",
                "--input_reads_files",
                "reads",
                "--no-kegg",
                "--no-cazy",
                "--no-hyd",
            ]
        )
        self.assertFalse(disabled.kegg)
        self.assertFalse(disabled.cazy)
        self.assertFalse(disabled.hydrogenase)
        no_species = parser.parse_args(
            [
                "annotation",
                "--input_genome_files",
                "mags",
                "--input_reads_files",
                "reads",
                "--no-place-species",
            ]
        )
        self.assertFalse(no_species.place_species)
        self.assertIn("--no-place-species", no_species._provided_options)
        reused = parser.parse_args(
            [
                "annotation",
                "--input_genome_files",
                "mags",
                "--input_reads_files",
                "reads",
                "--gtdbtk_res",
                "/results/gtdbtk",
            ]
        )
        self.assertEqual(reused.gtdbtk_res, "/results/gtdbtk")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "annotation",
                    "--input_genome_files",
                    "mags",
                    "--input_reads_files",
                    "reads",
                    "--gtdbtk-db",
                    "/db/gtdbtk",
                    "--gtdbtk_res",
                    "/results/gtdbtk",
                ]
            )
        for removed in ("--kegg", "--cazy", "--hydrogenase", "--no-hydrogenase"):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser.parse_args(["annotation", "--input_genome_files", "/mags", "--input_reads_files", "/reads", removed])
        for removed in ("--dry-run", "--force"):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
                parser.parse_args(["annotation", "--input_genome_files", "/mags", "--input_reads_files", "/reads", removed])
            self.assertEqual(raised.exception.code, 2)
        check = parser.parse_args(
            [
                "check",
                "--scope",
                "annotation",
                "--kegg-db",
                "/db/kofam",
                "--dbcan-db",
                "/db/dbcan",
                "--hydrogenase-db",
                "/db/hydrogenase",
            ]
        )
        self.assertEqual(check.kegg_db, "/db/kofam")
        self.assertEqual(check.dbcan_db, "/db/dbcan")
        self.assertEqual(check.hydrogenase_db, "/db/hydrogenase")

    def test_annotation_command_builds_default_functional_tasks(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "annotation",
                "--input_genome_files",
                "mags",
                "--input_reads_files",
                "reads",
                "--no-niche",
            ]
        )
        mags = [Path("mags/A.fa").resolve(), Path("mags/B.fa").resolve()]
        reads = [ReadSample("S1", Path("reads/S1.fastq.gz").resolve())]
        with patch("metabaw.cli.read_genome_files", return_value=mags):
            with patch("metabaw.cli.read_named_reads", return_value=reads):
                with patch("metabaw.cli._preflight_annotation"):
                    with patch("metabaw.cli._run_direct", return_value=0) as run:
                        args.output = str(self.output)
                        self.assertEqual(command_annotation(args), 0)
        task_ids = {task.id for task in run.call_args.args[1]}
        self.assertIn("05.annotation.proteins.merge", task_ids)
        self.assertIn("06.annotation.kegg", task_ids)
        self.assertIn("07.annotation.cazy", task_ids)
        self.assertIn("08.annotation.hydrogenase.finalize", task_ids)
        self.assertIn("09.annotation.terminal.filter", task_ids)
        self.assertFalse(any(task_id.startswith("06.annotation.kegg.A") for task_id in task_ids))

    def test_annotation_task_divides_total_threads_across_sample_slots(self) -> None:
        parser = build_parser()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            args = parser.parse_args(
                [
                    "annotation",
                    "--input_genome_files",
                    str(root / "mags"),
                    "--input_reads_files",
                    str(root / "reads"),
                    "--no-niche",
                    "--no-kegg",
                    "--no-cazy",
                    "--no-hyd",
                    "-t",
                    "32",
                    "--task",
                    "2",
                    "-o",
                    str(root / "output"),
                ]
            )
            mags = [
                (root / "mags" / f"M{number}.fa").resolve()
                for number in range(1, 9)
            ]
            reads = [
                ReadSample(
                    f"S{number}",
                    (root / "reads" / f"S{number}.fastq.gz").resolve(),
                )
                for number in range(1, 9)
            ]
            output = io.StringIO()
            with (
                patch("metabaw.cli.read_genome_files", return_value=mags),
                patch("metabaw.cli.read_named_reads", return_value=reads),
                patch("metabaw.cli._preflight_annotation"),
                patch("metabaw.cli._run_direct", return_value=0) as run,
                redirect_stdout(output),
            ):
                self.assertEqual(command_annotation(args), 0)

            startup = (root / "output" / "start_info.txt").read_text(encoding="utf-8")

        self.assertEqual(args.threads_per_task, 16)
        self.assertEqual(args.concurrent_sample_limit, 2)
        coverm_tasks = [
            task
            for task in run.call_args.args[1]
            if task.id.startswith("02.annotation.coverm.")
        ]
        self.assertEqual(
            {task.sample for task in coverm_tasks},
            {f"S{i}" for i in range(1, 9)},
        )
        self.assertTrue(coverm_tasks)
        self.assertEqual({task.cpus for task in coverm_tasks}, {16})
        self.assertNotIn("up to 2 concurrent sample task(s)", output.getvalue())
        self.assertIn("up to 2 concurrent sample task(s)", startup)
        self.assertIn("16 thread(s) each", startup)

    def test_annotation_help_exposes_every_option(self) -> None:
        parser = build_parser()
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as stopped:
            parser.parse_args(["annotation", "-h"])

        self.assertEqual(stopped.exception.code, 0)
        help_text = " ".join(output.getvalue().split())
        self.assertIn("--task", help_text)
        self.assertIn("divided evenly", help_text)
        self.assertIn("--max-memory", help_text)
        self.assertIn("maximum combined resident memory", help_text)
        self.assertNotIn("--place-species", help_text)
        self.assertIn("--no-place-species", help_text)
        self.assertIn("placement is enabled when this option is omitted", help_text)
        self.assertIn("--threads", help_text)
        self.assertIn("also passed directly to GTDB-Tk --cpus", help_text)
        self.assertNotIn("--gtdbtk-threads", help_text)
        self.assertIn("--gtdbtk-tmpdir", help_text)
        self.assertNotIn("--full-help", help_text)
        self.assertNotIn("--dry-run", help_text)
        self.assertNotIn("--force", help_text)

        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stopped:
            parser.parse_args(["annotation", "--full-help"])
        self.assertEqual(stopped.exception.code, 2)

    def _options(
        self,
        root: Path,
        *,
        kegg: bool = True,
        cazy: bool = True,
        hydrogenase: bool = True,
    ) -> AnnotationOptions:
        mags = root / "mags"
        mags.mkdir()
        mag_paths = (mags / "A.fa", mags / "B.fa")
        for path in mag_paths:
            path.write_text(">contig\nATGAAATAG\n", encoding="utf-8")
        reads = root / "reads"
        reads.mkdir()
        read_paths = (reads / "S1.fastq.gz", reads / "S2.fastq.gz")
        for path in read_paths:
            path.write_bytes(b"test")
        return AnnotationOptions(
            mag_dir=mags,
            mags=mag_paths,
            mag_suffix="fa",
            reads=tuple(
                ReadSample(name, path)
                for name, path in zip(("S1", "S2"), read_paths)
            ),
            output=root / "output",
            output_suffix=".tsv",
            threads=8,
            read_type="short",
            methods=("relative_abundance", "count"),
            place_species=True,
            niche_rank="family",
            niche_method="cv",
            no_niche=False,
            gtdbtk_data=root / "gtdbtk",
            run_kegg=kegg,
            run_cazy=cazy,
            run_hydrogenase=hydrogenase,
            kegg_db=root / "kofam",
            dbcan_db=root / "dbcan",
            hydrogenase_db=root / "hydrogenase_db",
        )

    def test_builder_creates_one_database_search_per_function(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks = AnnotationBuilder(self._options(root), root / "tmp" / "work").build()
        by_id = {task.id: task for task in tasks}
        self.assertIn("run-gtdbtk-classify", by_id["01.annotation.gtdbtk"].command)
        self.assertIn("--place-species", by_id["01.annotation.gtdbtk"].command)
        self.assertEqual(by_id["01.annotation.gtdbtk"].minimum_memory_gb, 160.0)
        self.assertIn("--pplacer-threads", by_id["01.annotation.gtdbtk"].command)
        self.assertEqual(
            by_id["01.annotation.gtdbtk"].command[
                by_id["01.annotation.gtdbtk"].command.index("--pplacer-threads") + 1
            ],
            "1",
        )
        self.assertIn("--scratch-dir", by_id["01.annotation.gtdbtk"].command)
        self.assertEqual(
            by_id["01.annotation.gtdbtk"].outputs[-1].name,
            ".metabaw.complete",
        )
        self.assertIn("02.annotation.samtools.compat", by_id)
        coverm_task = by_id["02.annotation.coverm.S1"]
        self.assertIn("02.annotation.samtools.compat", coverm_task.deps)
        self.assertIn("PATH=", coverm_task.command)
        for genome in ("A", "B"):
            self.assertIn(f"05.annotation.prodigal.{genome}", by_id)
            self.assertIn(f"05.annotation.gene_map.{genome}", by_id)
        self.assertIn("05.annotation.proteins.merge", by_id)
        self.assertEqual(
            set(by_id["05.annotation.proteins.merge"].deps),
            {"05.annotation.gene_map.A", "05.annotation.gene_map.B"},
        )
        self.assertIn("exec_annotation --profile", by_id["06.annotation.kegg"].command)
        self.assertIn("all_mags.faa", by_id["06.annotation.kegg"].command)
        self.assertIn("run_dbcan CAZyme_annotation", by_id["07.annotation.cazy"].command)
        self.assertIn("dbcan_database", by_id["07.annotation.cazy"].command)
        self.assertIn("07.annotation.cazy.database", by_id["07.annotation.cazy"].deps)
        self.assertIn("blastp -query", by_id["08.annotation.hydrogenase.search"].command)
        self.assertIn("diamond blastp", by_id["08.annotation.hydrogenase.fefe"].command)
        self.assertIn("Terminal.dmnd", by_id["09.annotation.terminal.search"].command)
        self.assertFalse(any(task_id.startswith("06.annotation.kegg.") and task_id.rsplit(".", 1)[-1] in {"A", "B"} for task_id in by_id))
        self.assertFalse(any(task_id.startswith("07.annotation.cazy.") and task_id.rsplit(".", 1)[-1] in {"A", "B"} for task_id in by_id))
        self.assertEqual(
            by_id["06.annotation.kegg.format"].outputs[0].name,
            "kegg_annotations.tsv",
        )
        self.assertEqual(
            by_id["07.annotation.cazy.format"].outputs[0].name,
            "cazy_annotations.tsv",
        )
        self.assertEqual(
            by_id["08.annotation.hydrogenase.format"].outputs[0].name,
            "hydrogenase_annotations.tsv",
        )
        self.assertEqual(
            by_id["09.annotation.terminal.format"].outputs[0].name,
            "terminal_enzyme_annotations.tsv",
        )

    def test_combined_functional_searches_use_total_thread_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            options = replace(self._options(root), threads=8, total_threads=32)
            tasks = AnnotationBuilder(options, root / "tmp" / "work").build()
        by_id = {task.id: task for task in tasks}
        for task_id in (
            "01.annotation.gtdbtk",
            "06.annotation.kegg",
            "07.annotation.cazy",
            "08.annotation.hydrogenase.search",
            "08.annotation.hydrogenase.fefe",
            "09.annotation.terminal.search",
        ):
            self.assertEqual(by_id[task_id].cpus, 32, task_id)
        self.assertIn("--cpu 32", by_id["06.annotation.kegg"].command)
        taxonomy_command = by_id["01.annotation.gtdbtk"].command
        self.assertEqual(taxonomy_command[taxonomy_command.index("--threads") + 1], "32")
        self.assertIn("--threads 32", by_id["07.annotation.cazy"].command)
        self.assertIn("-num_threads 32", by_id["08.annotation.hydrogenase.search"].command)
        self.assertIn("-p 32", by_id["09.annotation.terminal.search"].command)

    def test_complete_external_gtdbtk_results_skip_classification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            options = self._options(root)
            result = root / "existing_gtdbtk"
            self._write_gtdbtk_summary(result, ("A", "B"))
            self.assertTrue(gtdbtk_result_valid(result, options.mags))
            tasks = AnnotationBuilder(
                replace(options, gtdbtk_result=result),
                root / "tmp" / "work",
            ).build()

        by_id = {task.id: task for task in tasks}
        self.assertNotIn("01.annotation.gtdbtk", by_id)
        merge = by_id["03.annotation.merge"]
        self.assertEqual(merge.inputs[0], result)
        self.assertIn(str(result), merge.command)
        self.assertNotIn("01.annotation.gtdbtk", merge.deps)

    def test_external_gtdbtk_results_require_every_mag_and_nonempty_classification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mags = (root / "A.fa", root / "B.fa")
            result = root / "existing_gtdbtk"
            summary = self._write_gtdbtk_summary(result, ("A",))

            missing = gtdbtk_result_missing(result, mags)
            self.assertTrue(any("missing MAG classifications: B" in item for item in missing))

            summary.write_text(
                "user_genome\tclassification\n"
                "A\tUnclassified Bacteria\n"
                "B\tUnclassified Bacteria\n",
                encoding="utf-8",
            )
            self.assertEqual(gtdbtk_result_missing(result, mags), ())

            summary.write_text(
                "user_genome\tclassification\n"
                "A\tUnclassified Bacteria\n"
                "B\t\n",
                encoding="utf-8",
            )
            missing = gtdbtk_result_missing(result, mags)
            self.assertTrue(
                any("rows missing user_genome or classification" in item for item in missing)
            )

    def test_annotation_command_reuses_valid_gtdbtk_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = root / "existing_gtdbtk"
            self._write_gtdbtk_summary(result, ("A", "B"))
            parser = build_parser()
            args = parser.parse_args(
                [
                    "annotation",
                    "--input_genome_files",
                    str(root / "mags"),
                    "--input_reads_files",
                    str(root / "reads"),
                    "--gtdbtk_res",
                    str(result),
                    "--no-niche",
                    "--no-kegg",
                    "--no-cazy",
                    "--no-hyd",
                ]
            )
            mags = [root / "mags" / "A.fa", root / "mags" / "B.fa"]
            reads = [ReadSample("S1", root / "reads" / "S1.fastq.gz")]
            with (
                patch("metabaw.cli.read_genome_files", return_value=mags),
                patch("metabaw.cli.read_named_reads", return_value=reads),
                patch("metabaw.cli._preflight_annotation"),
                patch("metabaw.cli._run_direct", return_value=0) as run,
                redirect_stdout(io.StringIO()),
            ):
                args.output = str(self.output)
                self.assertEqual(command_annotation(args), 0)

        tasks = run.call_args.args[1]
        self.assertFalse(any(task.id == "01.annotation.gtdbtk" for task in tasks))
        payload = run.call_args.args[4]
        self.assertTrue(payload["gtdbtk_classification_skipped"])
        self.assertEqual(payload["gtdbtk_result_source"], str(result.resolve()))

    def test_annotation_command_rejects_incomplete_gtdbtk_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = root / "existing_gtdbtk"
            self._write_gtdbtk_summary(result, ("A",))
            args = build_parser().parse_args(
                [
                    "annotation",
                    "--input_genome_files",
                    str(root / "mags"),
                    "--input_reads_files",
                    str(root / "reads"),
                    "--gtdbtk_res",
                    str(result),
                    "--no-niche",
                ]
            )
            mags = [root / "mags" / "A.fa", root / "mags" / "B.fa"]
            with patch("metabaw.cli.read_genome_files", return_value=mags):
                with self.assertRaisesRegex(
                    ValueError,
                    "incomplete species annotation information.*GTDB-Tk must be rerun",
                ):
                    command_annotation(args)

    def test_reused_gtdbtk_results_skip_software_and_database_preflight(self) -> None:
        args = build_parser().parse_args(
            [
                "annotation",
                "--input_genome_files",
                "mags",
                "--input_reads_files",
                "reads",
                "--gtdbtk_res",
                "/results/gtdbtk",
                "--no-kegg",
                "--no-cazy",
                "--no-hyd",
            ]
        )
        with (
            patch("metabaw.cli._ensure_software") as ensure,
            patch("metabaw.cli.configured_database_path") as configured,
        ):
            _preflight_annotation(args)

        requirements = ensure.call_args.args[0]
        self.assertNotIn("gtdbtk", {item.executable for item in requirements})
        configured.assert_not_called()
        self.assertIsNone(args.gtdbtk_data)

    def test_coverm_splits_threshold_incompatible_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            options = replace(
                self._options(root),
                methods=(
                    "relative_abundance",
                    "rpkm",
                    "tpm",
                    "mean",
                    "count",
                    "length",
                ),
            )
            tasks = AnnotationBuilder(options, root / "tmp" / "work").build()
        by_id = {task.id: task for task in tasks}
        filtered = by_id["02.annotation.coverm.S1"]
        unfiltered = by_id["02.annotation.coverm.S1.unfiltered"]
        self.assertIn("--methods relative_abundance mean", filtered.command)
        self.assertIn("--min-covered-fraction 10", filtered.command)
        self.assertNotIn(" rpkm", filtered.command)
        self.assertIn("--methods rpkm tpm count length", unfiltered.command)
        self.assertIn("--min-covered-fraction 0", unfiltered.command)
        merge = by_id["03.annotation.merge"]
        self.assertIn(f"S1={unfiltered.outputs[0]}", merge.command)

    def test_gtdbtk_temp_path_length_detection(self) -> None:
        with patch.dict("metabaw.internal.os.environ", {"TMPDIR": "/tmp"}, clear=False):
            environment, alias = _gtdbtk_temp_environment()
        self.assertEqual(environment["TMPDIR"], "/tmp")
        self.assertIsNone(alias)

    def test_gtdbtk_temp_path_length_detection_is_conservative(self) -> None:
        self.assertFalse(_multiprocessing_socket_path_too_long(Path("/tmp")))
        self.assertTrue(
            _multiprocessing_socket_path_too_long(Path("/") / ("long-segment-" * 8))
        )

    def test_samtools_version_shim_ignores_loader_warning(self) -> None:
        diagnostic = (
            "samtools: libncurses.so: no version information available (required by samtools)\n"
            "samtools 1.19.2\nUsing htslib 1.19.1\n"
        )
        self.assertEqual(_samtools_version(diagnostic), "1.19.2")
        with tempfile.TemporaryDirectory() as temporary:
            shim = Path(temporary) / "bin" / "samtools"
            probe = subprocess.CompletedProcess(
                ["/usr/bin/samtools", "--version"],
                0,
                "samtools 1.19.2\nUsing htslib 1.19.1\n",
                "loader warning (required by samtools)\n",
            )
            with (
                patch("metabaw.internal.shutil.which", return_value="/usr/bin/samtools"),
                patch("metabaw.internal.subprocess.run", return_value=probe),
            ):
                prepare_samtools_compat(shim)
            contents = shim.read_text(encoding="utf-8")
        self.assertIn("samtools 1.19.2", contents)
        self.assertIn("exec /usr/bin/samtools", contents)

    def test_old_gtdbtk_omits_unsupported_species_option(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "taxonomy"
            (root / "A.fa").write_text(">A\nACGT\n", encoding="utf-8")
            output.mkdir()
            (output / "stale.txt").write_text("stale", encoding="utf-8")
            commands: list[list[str]] = []
            environments: list[dict[str, str] | None] = []

            def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
                commands.append(command)
                environments.append(kwargs.get("env"))  # type: ignore[arg-type]
                if "--help" in command:
                    return subprocess.CompletedProcess(
                        command,
                        0,
                        "usage: classify_wf --pplacer_cpus N --scratch_dir DIR",
                        "",
                    )
                output.mkdir(parents=True, exist_ok=True)
                self._write_gtdbtk_summary(output, ("A",))
                return subprocess.CompletedProcess(command, 0, "", "")

            warning = io.StringIO()
            temporary_environment = {
                "TMPDIR": "/tmp/mbw-test",
                "TMP": "/tmp/mbw-test",
                "TEMP": "/tmp/mbw-test",
            }
            with (
                patch("metabaw.internal.shutil.which", return_value="/usr/bin/gtdbtk"),
                patch(
                    "metabaw.internal._gtdbtk_temp_environment",
                    return_value=(temporary_environment, None),
                ),
                patch("metabaw.internal.subprocess.run", side_effect=fake_run),
                redirect_stdout(warning),
            ):
                run_gtdbtk_classify(
                    root,
                    output,
                    "fa",
                    64,
                    True,
                    1,
                    root / "scratch",
                )
            marker_created = (output / ".metabaw.complete").is_file()
        self.assertNotIn("--place_species", commands[-1])
        self.assertEqual(commands[-1][commands[-1].index("--cpus") + 1], "64")
        self.assertIn("--pplacer_cpus", commands[-1])
        self.assertIn("--scratch_dir", commands[-1])
        self.assertEqual(environments, [temporary_environment, temporary_environment])
        self.assertIn("does not support --place_species", warning.getvalue())
        self.assertTrue(marker_created)

    def test_functional_annotation_can_be_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks = AnnotationBuilder(
                self._options(root, kegg=False, cazy=False, hydrogenase=False),
                root / "tmp" / "work",
            ).build()
        self.assertFalse(any("prodigal" in task.id for task in tasks))
        self.assertFalse(any("kegg" in task.id for task in tasks))
        self.assertFalse(any("cazy" in task.id for task in tasks))
        self.assertFalse(any("hydrogenase" in task.id for task in tasks))
        self.assertFalse(any("terminal" in task.id for task in tasks))

    def test_annotation_dependencies_follow_enabled_functions(self) -> None:
        class Options:
            kegg = True
            cazy = False
            hydrogenase = False

        executables = {item.executable for item in annotation_requirements(Options())}
        self.assertIn("exec_annotation", executables)
        self.assertIn("prodigal", executables)
        self.assertIn("samtools", executables)
        self.assertNotIn("run_dbcan", executables)

        class ReuseOptions(Options):
            gtdbtk_res = "/results/gtdbtk"

        reused_executables = {
            item.executable for item in annotation_requirements(ReuseOptions())
        }
        self.assertNotIn("gtdbtk", reused_executables)

    def test_database_validation_requires_upstream_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            kofam = root / "kofam"
            (kofam / "profiles").mkdir(parents=True)
            (kofam / "profiles" / "K00001.hmm").write_text("HMM", encoding="utf-8")
            (kofam / "ko_list").write_text("knum\n", encoding="utf-8")
            self.assertTrue(kofam_database_valid(kofam))
            self.assertEqual(kofam_database_missing(kofam), ())

            dbcan = root / "dbcan"
            dbcan.mkdir()
            for name in (
                "CAZy.dmnd",
                "dbCAN.hmm",
                "dbCAN-sub.hmm",
                "fam-substrate-mapping.tsv",
            ):
                (dbcan / name).write_text("data", encoding="utf-8")
            self.assertTrue(dbcan_database_valid(dbcan))
            self.assertEqual(dbcan_database_missing(dbcan), ())

            legacy = root / "dbcan_legacy"
            legacy.mkdir()
            for name in (
                "CAZy.dmnd",
                "dbCAN.txt",
                "dbCAN_sub.hmm",
                "fam-substrate-mapping.tsv",
            ):
                (legacy / name).write_text("legacy", encoding="utf-8")
            for suffix in (".h3f", ".h3i", ".h3m", ".h3p"):
                (legacy / f"dbCAN.txt{suffix}").write_text("index", encoding="utf-8")
            self.assertTrue(dbcan_database_valid(legacy))
            self.assertEqual(dbcan_database_missing(legacy), ())

            view = root / "dbcan_view"
            marker = view / ".metabaw_dbcan_view.json"
            prepare_dbcan_database(legacy, view, marker)
            for name in (
                "CAZy.dmnd",
                "dbCAN.hmm",
                "dbCAN.txt",
                "dbCAN-sub.hmm",
                "dbCAN_sub.hmm",
                "fam-substrate-mapping.tsv",
                "dbCAN.hmm.h3f",
                "dbCAN.txt.h3f",
            ):
                self.assertTrue((view / name).is_file(), name)
            self.assertTrue(marker.is_file())

            hydrogenase = root / "hydrogenase"
            hydrogenase.mkdir()
            (hydrogenase / "hyddb.all.fa").write_text(
                ">WP_000000001.1\nMPEPTIDE\n", encoding="utf-8"
            )
            (hydrogenase / "FeFe.dmnd").write_bytes(b"diamond")
            (hydrogenase / "hyd_id-name.script.txt").write_text(
                "id\tgene\nWP_000000001.1\tFe\n", encoding="utf-8"
            )
            self.assertFalse(hydrogenase_database_valid(hydrogenase))
            self.assertIn("Terminal.dmnd", hydrogenase_database_missing(hydrogenase))
            (hydrogenase / "Terminal.dmnd").write_bytes(b"diamond")
            self.assertTrue(hydrogenase_database_valid(hydrogenase))
            self.assertEqual(hydrogenase_database_missing(hydrogenase), ())

    def test_hydrogenase_filter_confirmation_and_merge(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mapping = root / "hyd_id-name.script.txt"
            mapping.write_text(
                "id\tgene\nWP_FE\tFe\nWP_NI\tNiFe\nWP_FF\tFeFe\n",
                encoding="utf-8",
            )
            proteins = root / "MAG.faa"
            proteins.write_text(
                ">gene_fe\n" + "A" * 100 + "\n"
                ">gene_nife\n" + "C" * 100 + "\n"
                ">gene_fefe\n" + "G" * 100 + "\n"
                ">gene_low\n" + "T" * 100 + "\n",
                encoding="utf-8",
            )
            hits = root / "MAG.m8"
            hits.write_text(
                "gene_fe\tWP_FE\t60\t90\t0\t0\t1\t90\t100\t1\t90\t1e-70\t250\n"
                "gene_nife\tWP_NI\t60\t95\t0\t0\t1\t95\t100\t1\t95\t1e-60\t220\n"
                "gene_nife\tWP_FE\t80\t95\t0\t0\t1\t95\t100\t1\t95\t1e-50\t180\n"
                "gene_fefe\tWP_FF\t55\t90\t0\t0\t1\t90\t100\t1\t90\t1e-80\t300\n"
                "gene_low\tWP_FE\t49\t100\t0\t0\t1\t100\t100\t1\t100\t1e-90\t400\n",
                encoding="utf-8",
            )
            filtered = root / "HydDB.filtered.tsv"
            fefe_fasta = root / "Filtered.MAG.FeFe.faa"
            filter_hydrogenase_hits(
                hits,
                proteins,
                mapping,
                "MAG",
                filtered,
                fefe_fasta,
                0.9,
                0.5,
            )
            filtered_text = filtered.read_text(encoding="utf-8")
            self.assertIn("gene_fe\tFe", filtered_text)
            self.assertIn("gene_nife\tNiFe", filtered_text)
            self.assertIn("gene_fefe\tFeFe", filtered_text)
            self.assertNotIn("gene_low", filtered_text)
            self.assertIn(">gene_fefe", fefe_fasta.read_text(encoding="utf-8"))

            fefe_hits = root / "FeFe_filtered_results.m6"
            fefe_hits.write_text(
                "gene_fefe\tFeFe_reference_1\t70\t90\t0\t0\t1\t90\t1\t90\t1e-100\t500\n",
                encoding="utf-8",
            )
            nife = root / "MAG.NiFe.tsv"
            fe = root / "MAG.Fe.tsv"
            fefe = root / "MAG.FeFe.tsv"
            combined = root / "MAG.hydrogenases.tsv"
            finalize_hydrogenase_hits(filtered, fefe_hits, nife, fe, fefe, combined)
            self.assertIn("gene_nife", nife.read_text(encoding="utf-8"))
            self.assertIn("gene_fe", fe.read_text(encoding="utf-8"))
            self.assertIn("FeFe_reference_1", fefe.read_text(encoding="utf-8"))

            merged = root / "hydrogenase_annotations.tsv"
            merge_hydrogenase_annotations([f"MAG={combined}"], merged)
            merged_text = merged.read_text(encoding="utf-8")
            self.assertIn("gene_fe", merged_text)
            self.assertIn("gene_nife", merged_text)
            self.assertIn("gene_fefe", merged_text)

    def test_terminal_enzyme_marker_specific_filters_and_merge(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            hits = root / "Terminal.m8"
            hits.write_text(
                "gene_psaa\tref_psaa\tphotosystem I PsaA\t81\t90\t1\t90\t100\t1e-90\t300\n"
                "gene_hbst\tref_hbst\t4-hydroxybutyryl-CoA synthase HbsT\t75\t80\t1\t80\t100\t1e-80\t280\n"
                "gene_atpa\tref_atpa\tF-type ATP synthase AtpA\t69\t90\t1\t90\t100\t1e-100\t400\n"
                "gene_atpa\tref_other\tcurated terminal enzyme\t55\t90\t1\t90\t100\t1e-60\t250\n"
                "gene_nife\tref_nife\tgroup 4 NiFe-hydrogenase\t60\t85\t1\t85\t100\t1e-70\t260\n"
                "gene_rho\tref_rho\trhodopsin RHO\t40\t80\t1\t80\t100\t1e-50\t200\n"
                "gene_low_cov\tref_psaa\tPsaA\t99\t79\t1\t79\t100\t1e-100\t500\n",
                encoding="utf-8",
            )
            output = root / "MAG.terminal_enzymes.tsv"
            filter_terminal_enzyme_hits(hits, "MAG", output)
            text = output.read_text(encoding="utf-8")
            self.assertIn("gene_psaa\tPsaA", text)
            self.assertIn("gene_hbst\tHbsT", text)
            self.assertIn("gene_atpa\tOther", text)
            self.assertIn("gene_nife\tNiFe", text)
            self.assertIn("gene_rho\tRHO", text)
            self.assertNotIn("gene_low_cov", text)
            merged = root / "terminal_enzyme_annotations.tsv"
            merge_terminal_enzyme_annotations([f"MAG={output}"], merged)
            self.assertIn("gene_psaa", merged.read_text(encoding="utf-8"))

    def test_combined_tables_retain_genome_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first_kegg = root / "A.tsv"
            second_kegg = root / "B.tsv"
            first_kegg.write_text("gene1\tK00001\n", encoding="utf-8")
            second_kegg.write_text("gene1\tK00002\n", encoding="utf-8")
            merged_kegg = root / "kegg.tsv"
            merge_kofam_annotations(
                [f"A={first_kegg}", f"B={second_kegg}"], merged_kegg
            )
            kegg_text = merged_kegg.read_text(encoding="utf-8")
            self.assertIn("A\tgene1\tK00001", kegg_text)
            self.assertIn("B\tgene1\tK00002", kegg_text)

            dbcan_a = root / "dbcan_a"
            dbcan_b = root / "dbcan_b"
            dbcan_a.mkdir()
            dbcan_b.mkdir()
            header = "Gene ID\tdbCAN_hmm\t#ofTools\n"
            (dbcan_a / "overview.tsv").write_text(
                header + "gene1\tGH1\t2\n", encoding="utf-8"
            )
            (dbcan_b / "overview.txt").write_text(
                header + "gene1\tGH2\t3\n", encoding="utf-8"
            )
            merged_cazy = root / "cazy.tsv"
            merge_dbcan_annotations(
                [f"A={dbcan_a}", f"B={dbcan_b}"], merged_cazy
            )
            cazy_text = merged_cazy.read_text(encoding="utf-8")
            self.assertIn("A\tgene1\tGH1\t2", cazy_text)
            self.assertIn("B\tgene1\tGH2\t3", cazy_text)

    def test_all_final_functional_tables_use_common_mag_gene_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            genome = "SampleA_MetaBAT2_4"
            mag = root / f"{genome}.fa"
            mag.write_text(
                ">SampleA_256\nATGAAATAG\n>SampleA_257\nATGCCCTAG\n",
                encoding="utf-8",
            )
            proteins = root / f"{genome}.faa"
            proteins.write_text(
                ">SampleA_256_1\nMK\n>SampleA_257_2\nMP\n",
                encoding="utf-8",
            )
            gene_map = root / "gene_map.tsv"
            build_functional_gene_map(mag, proteins, genome, gene_map)

            cases = {
                "kegg": (
                    "Genome\tGene\tKO\n"
                    f"{genome}\tSampleA_257_2\tK00001\n",
                    "K00001",
                ),
                "cazy": (
                    "Genome\tGene ID\tRecommend Results\t#ofTools\n"
                    f"{genome}\tSampleA_257_2\tGH1\t3\n",
                    "GH1",
                ),
                "hydrogenase": (
                    "Genome\tGene\tHydrogenase_type\tIdentity\n"
                    f"{genome}\tSampleA_257_2\tNiFe\t65\n",
                    "NiFe",
                ),
                "terminal": (
                    "Genome\tGene\tTerminal_enzyme\tIdentity\n"
                    f"{genome}\tSampleA_257_2\tMcrA\t75\n",
                    "McrA",
                ),
            }
            expected_prefix = [
                "MAG",
                "MAG_contig_raw_id",
                "MAG_contig_id",
                "MAG_contig_gene_id",
                "Gene",
            ]
            for kind, (content, expected_gene) in cases.items():
                raw = root / f"{kind}.raw.tsv"
                raw.write_text(content, encoding="utf-8")
                output = root / f"{kind}.tsv"
                format_functional_annotations(
                    raw,
                    [f"{genome}={gene_map}"],
                    output,
                    kind,
                )
                lines = output.read_text(encoding="utf-8").splitlines()
                header = lines[0].split("\t")
                row = lines[1].split("\t")
                self.assertEqual(header[:5], expected_prefix, kind)
                self.assertEqual(
                    row[:5],
                    [
                        genome,
                        "SampleA_257",
                        f"{genome}_2",
                        f"{genome}_2_2",
                        expected_gene,
                    ],
                    kind,
                )

    def test_combined_proteins_use_unique_ids_and_restore_mag_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            protein_items: list[str] = []
            map_items: list[str] = []
            for genome in ("A", "B"):
                mag = root / f"{genome}.fa"
                proteins = root / f"{genome}.faa"
                gene_map = root / f"{genome}.map.tsv"
                mag.write_text(">contig\nATGAAATAG\n", encoding="utf-8")
                proteins.write_text(">contig_1\nMK\n", encoding="utf-8")
                build_functional_gene_map(mag, proteins, genome, gene_map)
                protein_items.append(f"{genome}={proteins}")
                map_items.append(f"{genome}={gene_map}")

            combined = root / "all_mags.faa"
            combined_map = root / "all_mags.tsv"
            combine_functional_proteins(
                protein_items,
                map_items,
                combined,
                combined_map,
            )
            identifiers = [
                line[1:]
                for line in combined.read_text(encoding="utf-8").splitlines()
                if line.startswith(">")
            ]
            self.assertEqual(identifiers, ["MBWPROT000000000001", "MBWPROT000000000002"])
            mapping_text = combined_map.read_text(encoding="utf-8")
            self.assertIn("A\tcontig\tA_1\tA_1_1\tMBWPROT000000000001\tcontig_1", mapping_text)
            self.assertIn("B\tcontig\tB_1\tB_1_1\tMBWPROT000000000002\tcontig_1", mapping_text)

            raw = root / "combined.kegg.tsv"
            raw.write_text(
                "Genome\tGene\tKO\n"
                "combined\tMBWPROT000000000001\tK00001\n"
                "combined\tMBWPROT000000000002\tK00002\n",
                encoding="utf-8",
            )
            formatted = root / "kegg.tsv"
            format_functional_annotations(
                raw,
                [f"combined={combined_map}"],
                formatted,
                "kegg",
            )
            text = formatted.read_text(encoding="utf-8")
            self.assertIn("A\tcontig\tA_1\tA_1_1\tK00001", text)
            self.assertIn("B\tcontig\tB_1\tB_1_1\tK00002", text)

    def test_missing_kegg_description_is_na_without_losing_ko_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mag = root / "SampleA.fa"
            proteins = root / "SampleA.faa"
            gene_map = root / "gene_map.tsv"
            mag.write_text(">contig\nATGAAATAG\n", encoding="utf-8")
            proteins.write_text(">contig_1\nMK\n>contig_2\nMP\n", encoding="utf-8")
            build_functional_gene_map(mag, proteins, "SampleA", gene_map)
            raw = root / "kofam.tsv"
            raw.write_text(
                "Genome\tGene\tKO\n"
                "SampleA\tcontig_1\tK00001\n"
                "SampleA\tcontig_2\tK03886\n",
                encoding="utf-8",
            )
            descriptions = root / "K.descript.txt"
            descriptions.write_text(
                "Kid\tdescript\nK00001\tKnown enzyme; detailed function\n",
                encoding="utf-8",
            )
            output = root / "kegg_annotations.tsv"
            warning = io.StringIO()
            with redirect_stderr(warning):
                format_functional_annotations(
                    raw, [f"SampleA={gene_map}"], output, "kegg", descriptions
                )
            with output.open(encoding="utf-8", newline="") as handle:
                rows = {row["Gene"]: row for row in csv.DictReader(handle, delimiter="\t")}
            self.assertEqual(rows["K00001"]["KO_gene_name"], "Known enzyme")
            self.assertEqual(rows["K00001"]["KO_description"],
                             "Known enzyme; detailed function")
            self.assertEqual(rows["K03886"]["KO_gene_name"], "NA")
            self.assertEqual(rows["K03886"]["KO_description"], "NA")
            self.assertEqual(rows["K03886"]["MAG"], "SampleA")
            self.assertIn("1 annotated KO(s)", warning.getvalue())
            self.assertIn("K03886", warning.getvalue())

    def test_previously_renamed_contigs_keep_matching_raw_and_final_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            genome = "SampleA_MetaBAT2_4"
            mag = root / f"{genome}.fa"
            mag.write_text(f">{genome}_3\nATGAAATAG\n", encoding="utf-8")
            proteins = root / f"{genome}.faa"
            proteins.write_text(f">{genome}_3_2\nMK\n", encoding="utf-8")
            gene_map = root / "gene_map.tsv"
            build_functional_gene_map(mag, proteins, genome, gene_map)
            rows = gene_map.read_text(encoding="utf-8").splitlines()
            values = rows[1].split("\t")
            self.assertEqual(values[1], f"{genome}_3")
            self.assertEqual(values[2], f"{genome}_3")
            self.assertEqual(values[3], f"{genome}_3_2")


if __name__ == "__main__":
    unittest.main()
