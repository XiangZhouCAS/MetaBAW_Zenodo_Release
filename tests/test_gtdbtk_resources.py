from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from metabaw.cli import build_parser, command_annotation
from metabaw.direct import AnnotationBuilder, AnnotationOptions
from metabaw.internal import _filesystem_type, _gtdbtk_temp_environment, run_gtdbtk_classify
from metabaw.workflow_report import _method_calls


class GtdbtkResourceTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.options = AnnotationOptions(
            mag_dir=self.root / "mags", mags=(self.root / "mags" / "A.fa",), mag_suffix="fa",
            reads=(), output=self.root / "out", output_suffix=".tsv", threads=128,
            total_threads=128, read_type="short", methods=("mean",), place_species=True,
            niche_rank="family", niche_method="cv", no_niche=True, gtdbtk_data=None,
            run_kegg=False, run_cazy=False, run_hydrogenase=False, kegg_db=None,
            dbcan_db=None, hydrogenase_db=None)

    def test_gtdbtk_uses_total_budget_without_a_separate_cap(self):
        for per_sample, total, expected in (
            (1, 1, 1), (8, 8, 8), (16, 64, 64), (32, 128, 128), (64, None, 64)
        ):
            with self.subTest(per_sample=per_sample, total=total):
                options = replace(self.options, threads=per_sample, total_threads=total)
                tasks = AnnotationBuilder(options, self.root / "work").build()
                task = next(task for task in tasks if task.id == "01.annotation.gtdbtk")
                self.assertEqual(task.cpus, expected)
                self.assertEqual(task.command[task.command.index("--threads") + 1], str(expected))
                self.assertEqual(task.command[task.command.index("--pplacer-threads") + 1], "1")
                self.assertIn("--ipc-tmpdir", task.command)
                self.assertFalse(task.enforce_memory_limit)  # PSS guard, not virtual-address-space limit.

    def test_nonpositive_total_budget_is_rejected(self):
        for total in (0, -1):
            with self.subTest(total=total), self.assertRaisesRegex(ValueError, "GTDB-Tk threads must be positive"):
                AnnotationBuilder(replace(self.options, total_threads=total), self.root / "work").build()

    def test_annotation_threads_for_other_steps_are_unchanged(self):
        options = replace(self.options, run_kegg=True, kegg_db=self.root / "kegg")
        tasks = AnnotationBuilder(options, self.root / "work").build()
        by_id = {task.id: task for task in tasks}
        self.assertEqual(by_id["01.annotation.gtdbtk"].cpus, 128)
        self.assertEqual(by_id["06.annotation.kegg"].cpus, 128)

    def test_workflow_details_reports_global_gtdbtk_cpus(self):
        options = replace(self.options, threads=16, total_threads=64)
        task = next(task for task in AnnotationBuilder(options, self.root / "work").build()
                    if task.id == "01.annotation.gtdbtk")
        calls = _method_calls(task)
        self.assertEqual(calls[0][0], "gtdbtk")
        self.assertIn("--cpus 64", calls[0][1])
        self.assertIn("--pplacer_cpus 1", calls[0][1])

    def test_annotation_help_has_one_thread_option(self):
        for flag in ("-h", "--help"):
            output = io.StringIO()
            with redirect_stdout(output), self.assertRaises(SystemExit):
                build_parser().parse_args(["annotation", flag])
            for option in ("--threads", "--gtdbtk-tmpdir"):
                self.assertIn(option, output.getvalue())
            self.assertNotIn("--gtdbtk-threads", output.getvalue())

    def test_removed_gtdbtk_thread_flag_is_rejected(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as stopped:
            build_parser().parse_args([
                "annotation", "--input_genome_files", "genomes.txt",
                "--input_reads_files", "reads.tsv", "--gtdbtk-threads", "8",
            ])
        self.assertEqual(stopped.exception.code, 2)

    def test_removed_positive_species_placement_flag_is_rejected(self):
        error = io.StringIO()
        with redirect_stderr(error), self.assertRaises(SystemExit) as stopped:
            build_parser().parse_args([
                "annotation", "--input_genome_files", "genomes.txt",
                "--input_reads_files", "reads.tsv", "--place-species",
            ])
        self.assertEqual(stopped.exception.code, 2)
        self.assertIn("unrecognized arguments: --place-species", error.getvalue())

    def test_single_negative_switch_preserves_generated_gtdbtk_commands(self):
        (self.root / "A.fa").write_text(">c\nACGT\n")
        (self.root / "genomes.txt").write_text("A.fa\n")
        (self.root / "r1.fastq").touch()
        (self.root / "r2.fastq").touch()
        (self.root / "reads.tsv").write_text("S1\tr1.fastq,r2.fastq\n")
        for flags, placement in (([], True), (["--no-place-species"], False)):
            with self.subTest(flags=flags):
                args = build_parser().parse_args([
                    "annotation", "--input_genome_files", str(self.root / "genomes.txt"),
                    "--input_reads_files", str(self.root / "reads.tsv"),
                    "--no-kegg", "--no-cazy", "--no-hyd",
                    "-o", str(self.root / ("enabled" if placement else "disabled")), *flags,
                ])
                with patch("metabaw.cli._preflight_annotation"), \
                     patch("metabaw.cli._run_direct", return_value=0) as run, redirect_stdout(io.StringIO()):
                    self.assertEqual(command_annotation(args), 0)
                task = next(task for task in run.call_args.args[1] if task.id == "01.annotation.gtdbtk")
                # This private helper flag is not a public annotation option.
                # Keep it unchanged so removing the public switch alone does
                # not invalidate existing GTDB-Tk analysis checkpoints.
                self.assertEqual("--place-species" in task.command, placement)
                previous = next(task for task in AnnotationBuilder(
                    replace(self.options, mag_dir=task.inputs[0], output=Path(args.output).resolve(),
                            threads=task.cpus, total_threads=task.cpus, place_species=placement,
                            gtdbtk_tmpdir=Path(args.gtdbtk_tmpdir).resolve()), task.cwd).build()
                                if task.id == "01.annotation.gtdbtk")
                self.assertEqual(task.command, previous.command)
                self.assertEqual(task.resume_fingerprint([]), previous.resume_fingerprint([]))

    def test_cli_passes_global_threads_and_records_effective_parameters(self):
        (self.root / "A.fa").write_text(">c\nACGT\n")
        (self.root / "genomes.txt").write_text("A.fa\n")
        for number in range(1, 5):
            (self.root / f"r{number}.fastq").touch()
        (self.root / "reads.tsv").write_text("".join(
            f"S{number}\tr{number}.fastq\n" for number in range(1, 5)
        ))
        for flag, total in (("-t", 64), ("--threads", 128)):
            with self.subTest(flag=flag, total=total):
                args = build_parser().parse_args([
                    "annotation", "--input_genome_files", str(self.root / "genomes.txt"),
                    "--input_reads_files", str(self.root / "reads.tsv"), flag, str(total),
                    "--task", "4", "--gtdbtk-tmpdir", str(self.root / "local"),
                    "-o", str(self.root / "out")])
                with patch("metabaw.cli._preflight_annotation"), patch("metabaw.cli._run_direct", return_value=0) as run, redirect_stdout(io.StringIO()):
                    self.assertEqual(command_annotation(args), 0)
                task = next(task for task in run.call_args.args[1] if task.id == "01.annotation.gtdbtk")
                self.assertEqual(task.cpus, total)
                self.assertEqual(task.command[task.command.index("--threads") + 1], str(total))
                params = run.call_args.args[4]["gtdbtk_parameters"]
                self.assertEqual(params["cpus"], total)
                self.assertEqual(params["thread_source"], "--threads")
                self.assertNotIn("requested_thread_cap", params)
                self.assertEqual(Path(params["subprocess_tmp_parent"]), (self.root / "local").resolve())
                self.assertEqual(args.threads_per_task, total // 4)
                profile_tasks = [task for task in run.call_args.args[1]
                                 if task.id.startswith("02.annotation.coverm.")]
                self.assertEqual({task.cpus for task in profile_tasks}, {total // 4})
                startup = (self.root / "out" / "start_info.txt").read_text(encoding="utf-8")
                self.assertIn(f"GTDB-Tk cpus={total} from -t/--threads", startup)
                args.threads = 0
                with self.assertRaisesRegex(ValueError, "--threads and --task must be positive"):
                    command_annotation(args)

    def test_private_tmpdir_is_real_unique_and_ignores_inherited_nfs_tmpdir(self):
        parent = self.root / "local"
        with (patch("metabaw.internal.sys.platform", "linux"),
              patch("metabaw.internal._filesystem_type", return_value="ext4"),
              patch("metabaw.internal._multiprocessing_socket_path_too_long", return_value=False),
              patch.dict("os.environ", {"TMPDIR": "/nfs/workflow/tmp"}), redirect_stdout(io.StringIO())):
            env1, tmp1 = _gtdbtk_temp_environment(parent)
            env2, tmp2 = _gtdbtk_temp_environment(parent)
        self.assertNotEqual(tmp1, tmp2)
        for environment, path in ((env1, tmp1), (env2, tmp2)):
            self.assertTrue(path.is_dir())
            self.assertFalse(path.is_symlink())
            self.assertEqual(path.parent, parent.resolve())
            self.assertEqual(environment["TMPDIR"], str(path))
            self.assertEqual(environment["TMP"], str(path))
            self.assertEqual(environment["TEMP"], str(path))

    def test_nfs_parent_is_rejected_instead_of_aliased(self):
        parent = self.root / "remote"
        with patch("metabaw.internal.sys.platform", "linux"), patch("metabaw.internal._filesystem_type", return_value="nfs4"):
            with self.assertRaisesRegex(ValueError, "is on nfs4"):
                _gtdbtk_temp_environment(parent)
        self.assertEqual(list(parent.iterdir()), [])

    def test_long_socket_parent_fails_without_allocating_a_directory(self):
        parent = self.root / "long"
        with (patch("metabaw.internal.sys.platform", "linux"),
              patch("metabaw.internal._filesystem_type", return_value="ext4"),
              patch("metabaw.internal._multiprocessing_socket_path_too_long", return_value=True)):
            with self.assertRaisesRegex(ValueError, "too long"):
                _gtdbtk_temp_environment(parent)
        self.assertEqual(list(parent.iterdir()), [])

    def test_mount_detection_selects_the_deepest_mount(self):
        root = self.root.as_posix()
        escaped_root = root.replace(" ", r"\040")
        remote = (self.root / "mounted" / "with spaces").as_posix().replace(" ", r"\040")
        mounts = f"1 0 8:1 / {escaped_root} rw - ext4 /dev/root rw\n2 1 0:1 / {remote} rw - nfs4 server:/share rw\n"
        with patch.object(Path, "read_text", return_value=mounts):
            self.assertEqual(_filesystem_type(self.root / "mounted" / "with spaces" / "job"), "nfs4")
            self.assertEqual(_filesystem_type(self.root / "local"), "ext4")

    def test_wrapper_cleans_only_its_private_directory_after_tool_failure(self):
        parent = self.root / "local"
        private = parent / "mbw-gtdb-owned"
        private.mkdir(parents=True)
        (private / "ipc").touch()
        sibling = parent / "keep.txt"
        sibling.write_text("user data")
        environment = {"TMPDIR": str(private), "TMP": str(private), "TEMP": str(private)}
        probe = subprocess.CompletedProcess([], 0, "--pplacer_cpus --scratch_dir --place_species", "")
        failed = subprocess.CompletedProcess([], 1, "", "failed")
        with (patch("metabaw.internal.shutil.which", return_value="gtdbtk"),
              patch("metabaw.internal._gtdbtk_temp_environment", return_value=(environment, private)),
              patch("metabaw.internal.subprocess.run", side_effect=[probe, failed])):
            with self.assertRaisesRegex(RuntimeError, "exited with code 1"):
                run_gtdbtk_classify(self.root, self.root / "taxonomy", "fa", 16, True)
        self.assertFalse(private.exists())
        self.assertEqual(sibling.read_text(), "user data")


if __name__ == "__main__":
    unittest.main()
