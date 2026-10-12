from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from metabaw.cli import main
from metabaw.startup_log import capture_startup, finish_startup


class StartupLogTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name) / "results"
        self.args = Namespace(output=self.root, _invocation=["metabaw", "bin", "-o", str(self.root)])

    def test_details_move_to_file_but_summary_prompt_error_and_progress_stay_visible(self):
        hidden = ["[INPUT] S1: short; aligner=bowtie2", "[RESOURCES] CPU budget",
                  "[OK] COMEBin environment", "[CUDA OK] driver", "[GPU] slots",
                  "[WARNING] CPU fallback", "[COMEBIN CPU] batch size=128",
                  "[GPU PROBE] MetaDecoder executable=/env/bin/metadecoder",
                  "[REFINEMENT THRESHOLDS] magscot: completeness >= 40%; contamination <= 12%",
                  "[00:00:00] [GPU PROBE] Python launcher=/env/bin/python",
                  "[00:00:00] [REFINEMENT THRESHOLDS] metawrap: relative relaxation=20%",
                  "[00:00:00] [MEMORY ESTIMATE] memory", "[00:00:00] [WORKFLOW DETAILS] methods",
                  "[00:00:00] [FAILURE POLICY] strict", "[CONFIG] database", "[KEGG] metadata"]
        visible = ["[INPUT] Named manifests: 2 samples", "[COASSEMBLY] group1: S1,S2",
                   "[INPUT] Annotation: 4 MAGs", "[NICHE] Skipped", "[MISSING] database"]

        @capture_startup()
        def command(args):
            for line in hidden + visible:
                print(line, flush=True)
            for line in ("[WARNING] stderr detail", "[GPU PROBE] stderr detail",
                         "[REFINEMENT THRESHOLDS] stderr detail"):
                print(line, file=sys.stderr, flush=True)
            print("Install missing dependencies? [y/N] ", end="", flush=True)
            print("[ERROR] fatal", file=sys.stderr, flush=True)
            finish_startup(args)
            print("[STEP 1/10] Running alignment", flush=True)
            print("[WARNING] runtime warning", file=sys.stderr, flush=True)
            return 7

        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            self.assertEqual(command(self.args), 7)
        record = (self.root / "start_info.txt").read_text()
        for line in hidden:
            self.assertIn(line, record)
            self.assertNotIn(line, stdout.getvalue())
        for line in visible:
            self.assertIn(line, stdout.getvalue())
        self.assertIn("Command: metabaw bin", record)
        self.assertIn("Install missing dependencies? [y/N]", stdout.getvalue())
        self.assertIn("[ERROR] fatal", stderr.getvalue())
        self.assertIn("[ERROR] fatal", record)
        for line in ("[WARNING] stderr detail", "[GPU PROBE] stderr detail",
                     "[REFINEMENT THRESHOLDS] stderr detail"):
            self.assertIn(line, record)
            self.assertNotIn(line, stderr.getvalue())
        self.assertIn("[STEP 1/10]", stdout.getvalue())
        self.assertIn("[WARNING] runtime warning", stderr.getvalue())
        self.assertNotIn("[STEP 1/10]", record)
        self.assertNotIn("runtime warning", record)
        self.assertFalse(hasattr(self.args, "_startup_log"))

    def test_reruns_append_and_nested_runners_share_the_log(self):
        @capture_startup(output_position=2)
        def runner(args, tasks, output):
            print("[OK] nested", flush=True)
            finish_startup(args)

        @capture_startup()
        def command(args):
            print("[OK] outer", flush=True)
            runner(args, [], self.root)

        with redirect_stdout(io.StringIO()):
            command(self.args)
            command(self.args)
        record = (self.root / "start_info.txt").read_text()
        self.assertEqual(record.count("=== MetaBAW startup:"), 2)
        self.assertEqual(record.count("[OK] nested"), 2)
        self.assertEqual(record.count("[OK] outer"), 2)

    def test_early_invalid_input_does_not_create_output(self):
        @capture_startup()
        def command(args):
            raise ValueError("invalid manifest")

        original = sys.stdout
        with self.assertRaisesRegex(ValueError, "invalid manifest"):
            command(self.args)
        self.assertIs(sys.stdout, original)
        self.assertFalse(self.root.exists())
        self.assertFalse(hasattr(self.args, "_startup_log"))

    def test_preflight_exception_is_recorded_and_streams_are_restored(self):
        @capture_startup()
        def command(args):
            print("[OK] partial preflight", flush=True)
            raise RuntimeError("missing dependency")

        original = sys.stderr
        with self.assertRaisesRegex(RuntimeError, "missing dependency"):
            command(self.args)
        self.assertIs(sys.stderr, original)
        self.assertIn("[STARTUP ERROR] RuntimeError: missing dependency",
                      (self.root / "start_info.txt").read_text())
        self.assertFalse(hasattr(self.args, "_startup_log"))

    def test_standalone_runner_uses_its_output_and_dry_run_does_not_write(self):
        @capture_startup(output_position=2)
        def runner(args, tasks, output):
            print("[RESOURCES] allocated", flush=True)

        with redirect_stdout(io.StringIO()):
            runner(Namespace(dry_run=False), [], self.root)
        self.assertTrue((self.root / "start_info.txt").is_file())
        other = self.root / "dry-run"
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            runner(Namespace(dry_run=True), [], other)
        self.assertIn("[RESOURCES] allocated", stdout.getvalue())
        self.assertFalse(other.exists())

    def test_both_cli_modules_keep_progress_visible_after_preflight(self):
        inputs = self.root.parent
        for filename in ("r1.fastq", "r2.fastq"):
            (inputs / filename).touch()
        (inputs / "assembly.fa").write_text(">contig_1\n" + "ACGT" * 500 + "\n")
        (inputs / "reads.tsv").write_text("S1\tr1.fastq,r2.fastq\n")
        (inputs / "contigs.tsv").write_text("S1\tassembly.fa\n")
        (inputs / "genomes.txt").write_text("assembly.fa\n")

        def preflight(*args):
            print("[OK] test environment", flush=True)
            print("[GPU PROBE] test runtime", flush=True)
            print("[WARNING] test fallback", file=sys.stderr, flush=True)

        def execute(tasks, **kwargs):
            print("[STEP 1/1] test execution", flush=True)
            print("[WARNING] test runtime notice", file=sys.stderr, flush=True)
            return {task.id: "success" for task in tasks}

        for module in ("bin", "annotation"):
            output = self.root / module
            stdout, stderr = io.StringIO(), io.StringIO()
            flags = (["--input_contig_files", str(inputs / "contigs.tsv"), "--no-gpu"] if module == "bin"
                     else ["--input_genome_files", str(inputs / "genomes.txt")])
            with (self.subTest(module=module),
                  patch("metabaw.cli._preflight_bin", side_effect=preflight),
                  patch("metabaw.cli._preflight_annotation", side_effect=preflight),
                  patch("metabaw.cli.Executor") as executor,
                  redirect_stdout(stdout), redirect_stderr(stderr)):
                executor.return_value.run.side_effect = execute
                with self.assertRaises(SystemExit) as stopped:
                    main([module, "--input_reads_files", str(inputs / "reads.tsv"),
                          *flags, "--max-memory", "1024", "-o", str(output)])
                self.assertEqual(stopped.exception.code, 0)
            record = (output / "start_info.txt").read_text()
            manifest = json.loads((output / "run_manifest.json").read_text())
            self.assertEqual(manifest["module"], "binning" if module == "bin" else "annotation")
            self.assertIn("[OK] test environment", record)
            self.assertIn("[WARNING] test fallback", record)
            self.assertIn("[RESOURCES]", record)
            self.assertIn("[GPU PROBE] test runtime", record)
            if module == "bin":
                self.assertIn("[REFINEMENT THRESHOLDS] magscot:", record)
            for stream in (stdout.getvalue(), stderr.getvalue()):
                self.assertNotIn("[GPU PROBE]", stream)
                self.assertNotIn("[REFINEMENT THRESHOLDS]", stream)
            self.assertNotIn("[OK]", stdout.getvalue())
            self.assertNotIn("test fallback", stderr.getvalue())
            self.assertIn("MetaBAW is Running.", stdout.getvalue())
            self.assertIn("[START INFO]", stdout.getvalue())
            self.assertIn("[STEP 1/1] test execution", stdout.getvalue())
            self.assertIn("[WARNING] test runtime notice", stderr.getvalue())
            self.assertIn("ALL DONE.", stdout.getvalue())
            self.assertNotIn("test execution", record)
            self.assertNotIn("test runtime notice", record)


if __name__ == "__main__":
    unittest.main()
