from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from metabaw.cli import build_parser, command_bin, main
from metabaw.direct import BINNER_ORDER, BinOptions, DirectBinBuilder
from metabaw.discovery import Analysis, ReadSample, build_analyses
from metabaw.model import Task


class BinningHelpTests(unittest.TestCase):
    def help_text(self, flag):
        stream = io.StringIO()
        with redirect_stdout(stream), self.assertRaises(SystemExit) as raised:
            build_parser().parse_args(("bin", flag))
        self.assertEqual(raised.exception.code, 0)
        return stream.getvalue()

    def test_short_and_full_help(self):
        short = self.help_text("-h")
        full = self.help_text("--full-help")
        self.assertNotIn("usage: metabaw bin", full)
        for flag in ("--input_reads_files", "--input_contig_files", "--align-tool",
                     "--tmp-files", "--full-help",
                     "--assembly-strategy", "--coassembly-file", "--tools"):
            self.assertIn(flag, short)
        for flag in ("--nano-raw", "--pacbio-raw", "--pacbio-corr",
                     "--pacbio-hifi", "--nano-corr", "--nano-hq"):
            self.assertNotIn(flag, short)
            self.assertIn(flag, full)
        self.assertNotIn("--long-read-preset", full)
        for flag in ("--multi-files", "--failure-policy"):
            self.assertNotIn(flag, short)
            self.assertIn(flag, full)
        for flag in ("--path", "--contig-suffix", "--suffix", "--type",
                     "--separate-sample-name", "--cohort-size", "--single",
                     "--dry-run", "--force"):
            self.assertNotIn(flag, full)

    def test_removed_options_and_abbreviations_are_rejected(self):
        for option in ("-p", "-c", "-s", "-f", "--separate-sample-name", "--type",
                       "--multi", "--single", "--cohort-size",
                       "--dry-run", "--force", "--input_reads",
                       "--long-read-preset"):
            with self.subTest(option=option), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    build_parser().parse_args((
                        "bin", "--input_reads_files", "reads.tsv",
                        "--input_contig_files", "contigs.tsv", option, "value"))
                self.assertEqual(raised.exception.code, 2)

    def test_tools_are_shown_in_both_short_help_flags_with_choices_and_defaults(self):
        for flag in ("-h", "--help"):
            with self.subTest(flag=flag):
                text = " ".join(self.help_text(flag).split())
                self.assertIn("--tools BINNER [BINNER ...]", text)
                for tool in BINNER_ORDER:
                    self.assertIn(tool, text)
                self.assertIn("default: short=metabat2,metadecoder,vamb; long=metadecoder,vamb,lorbin", text)
        args = build_parser().parse_args([
            "bin", "--input_reads_files", "reads.tsv", "--tools", *BINNER_ORDER,
        ])
        self.assertEqual(args.tools, list(BINNER_ORDER))

    def test_reads_are_required_and_contigs_are_conditional(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            build_parser().parse_args(("bin", "--input_contig_files", "contigs.tsv"))
        args = build_parser().parse_args(("bin", "--input_reads_files", "reads.tsv"))
        self.assertIsNone(args.input_contig_files)
        args = build_parser().parse_args(("bin", "--input_reads_files", "reads.tsv",
                                          "--input_contig_files", "contigs.tsv"))
        self.assertEqual(args.func, command_bin)

    def test_flye_read_type_defaults_and_modes_are_mutually_exclusive(self):
        args = build_parser().parse_args(("bin", "--input_reads_files", "reads.tsv"))
        self.assertEqual(args.flye_read_type, "--nano-raw")
        args = build_parser().parse_args((
            "bin", "--input_reads_files", "reads.tsv", "--pacbio-hifi"
        ))
        self.assertEqual(args.flye_read_type, "--pacbio-hifi")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            build_parser().parse_args((
                "bin", "--input_reads_files", "reads.tsv",
                "--pacbio-hifi", "--nano-hq",
            ))
        self.assertEqual(raised.exception.code, 2)

    def test_both_modules_default_to_detected_machine_memory(self):
        with patch("metabaw.cli.detected_total_memory_gib", return_value=384.5):
            parser = build_parser()
        binning = parser.parse_args(("bin", "--input_reads_files", "reads.tsv"))
        annotation = parser.parse_args((
            "annotation", "--input_reads_files", "reads.tsv",
            "--input_genome_files", "genomes.txt",
        ))
        self.assertEqual(binning.max_memory, 384.5)
        self.assertEqual(binning.max_gpu_memory, "100%")
        self.assertEqual(annotation.max_memory, 384.5)
        explicit = parser.parse_args((
            "bin", "--input_reads_files", "reads.tsv", "--max-memory", "200",
        ))
        self.assertEqual(explicit.max_memory, 200.0)

    def test_bare_commands_show_usage_error_and_full_help_hint_without_running(self):
        for command in ("bin", "annotation"):
            with self.subTest(command=command):
                stdout, stderr = io.StringIO(), io.StringIO()
                with (patch("metabaw.cli.command_bin") as run,
                      redirect_stdout(stdout), redirect_stderr(stderr),
                      self.assertRaises(SystemExit) as raised):
                    main([command])
                self.assertEqual(raised.exception.code, 2)
                self.assertEqual(stdout.getvalue(), "")
                error = stderr.getvalue()
                self.assertIn("usage: metabaw", error)
                self.assertIn("the following arguments are required", error)
                self.assertNotIn("SAMPLE<TAB>", error)
                help_flag = "-h" if command == "annotation" else "--full-help"
                self.assertEqual(
                    error.rstrip().splitlines()[-1],
                    f"Hint: run 'metabaw {command} {help_flag}' to show detailed "
                    "help for all parameters.",
                )
                run.assert_not_called()


class PublicBinNamingTests(unittest.TestCase):
    def _options(self, root: Path) -> BinOptions:
        return BinOptions(
            outdir=root / "result",
            workdir=root / "work",
            threads=4,
            read_type="short",
            align_tool="bowtie2",
            flye_read_type="--nano-raw",
            binners=("metabat2",),
            min_contig_length=1500,
            min_fasta_kbs=200,
            batch_size=1024,
            refiner="magscot",
            quality_control="checkm2",
            min_completeness=50.0,
            max_contamination=10.0,
            min_quality_score=None,
            run_gunc=False,
            run_trna=False,
            trna_pass=None,
            run_rrna=False,
            rrna_pass=False,
            dereplicator="galah",
            environment=None,
            tag_contigs=False,
            gpu=False,
            max_gpu_memory="4G",
            assembly_strategy="default",
            magscot_dir=root / "MAGScoT",
            checkm2_db=None,
            gunc_db=None,
            gtdbtk_data=None,
            extra_args={},
        )

    def test_explicit_mapping_is_used_in_every_published_binner_prefix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contigs = root / "coassembly.fa"
            sample = ReadSample("S1", root / "S1.fq.gz", contigs=contigs)
            analysis = Analysis(
                "coassembly",
                (sample,),
                (contigs,),
                False,
                public_name="coassembly_to_S1",
                explicit_mapping=True,
            )
            builder = DirectBinBuilder([], self._options(root), root)
            prepare = Task("prepare", "01", ("true",), root)
            mapping_task = Task("map", "03", ("true",), root)
            for binner in BINNER_ORDER:
                with self.subTest(binner=binner):
                    _name, task, _mapping, _bins = builder._publish_binner(
                        analysis,
                        binner,
                        prepare,
                        contigs,
                        mapping_task,
                        root / f"{binner}.map.tsv",
                    )
                    self.assertIn(
                        f"--prefix coassembly_to_S1_{binner}",
                        task.display_command(),
                    )
                    self.assertIn("--completion-marker", task.display_command())
                    self.assertTrue(
                        any(path.name == ".metabaw.naming-v2.complete" for path in task.outputs)
                    )

    def test_comebin_uses_total_pss_limit_and_user_controlled_retries(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contigs = root / "prepared.fna"
            sample = ReadSample("S1", root / "S1.fq.gz", contigs=contigs)
            analysis = Analysis("S1", (sample,), (contigs,), False)
            options = replace(self._options(root), binners=("comebin",))
            builder = DirectBinBuilder([], options, root)
            prepare = Task("prepare", "01", ("true",), root)
            bam_task = Task("bamset", "02", ("true",), root)
            builder._binners(
                analysis,
                prepare,
                contigs,
                bam_task,
                [root / "S1.bam"],
                root / "bamset",
            )
            run = next(task for task in builder.tasks if task.id == "03.bin.comebin.S1")
            self.assertEqual(run.automatic_retries, 0)
            self.assertFalse(run.enforce_memory_limit)


if __name__ == "__main__":
    unittest.main()
