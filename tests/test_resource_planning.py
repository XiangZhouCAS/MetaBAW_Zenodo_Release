from __future__ import annotations

from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from metabaw.cli import (
    _comebin_minimum_memory_gib,
    _apply_gpu_memory_minima,
    _gpu_task_slots,
    _run_direct,
    _thread_budget,
)
from metabaw.dependencies import CudaDevice, HostCudaStatus
from metabaw.direct import DirectBinBuilder
from metabaw.model import Task


class ResourcePlanningTests(unittest.TestCase):
    def test_total_threads_are_divided_across_concurrent_samples(self) -> None:
        self.assertEqual(_thread_budget(128, 3, 3), (42, 3))
        self.assertEqual(_thread_budget(128, 8, 3), (42, 3))
        self.assertEqual(_thread_budget(128, 3, 320), (42, 3))
        self.assertEqual(_thread_budget(2, 3, 3), (1, 2))

    def test_gpu_concurrency_uses_shared_aggregate_budget(self) -> None:
        status = HostCudaStatus(
            executable="nvidia-smi",
            devices=(
                CudaDevice("0", "GPU 0", 48 * 1024, "580", 40 * 1024),
                CudaDevice("1", "GPU 1", 48 * 1024, "580", 40 * 1024),
            ),
            advertised_cuda_version="13.0",
        )
        self.assertEqual(_gpu_task_slots(status, 1, "4G", 3), 3)
        self.assertEqual(_gpu_task_slots(status, 2, "4G", 3), 3)
        self.assertEqual(_gpu_task_slots(status, 2, "4G", 1), 1)

    def test_low_gpu_budget_disables_only_binners_below_their_minimum(self) -> None:
        args = Namespace(
            tools=["vamb", "comebin", "semibin2", "lorbin"],
            max_gpu_memory="4G",
            batch_size=1024,
            requested_batch_size=1024,
            parameter_warnings=[],
        )
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            enabled = _apply_gpu_memory_minima(args, 48 * 1024)

        self.assertEqual(enabled, ("vamb", "semibin2", "lorbin"))
        self.assertEqual(args.gpu_binners, enabled)
        self.assertIn("COMEBin will use CPU", stderr.getvalue())
        self.assertNotIn("VAMB will use CPU", stderr.getvalue())

    def test_full_detected_gpu_memory_budget_keeps_default_comebin_on_gpu(self) -> None:
        args = Namespace(
            tools=["comebin"],
            max_gpu_memory="100%",
            batch_size=1024,
            requested_batch_size=1024,
            parameter_warnings=[],
        )

        enabled = _apply_gpu_memory_minima(args, 48 * 1024)

        self.assertEqual(enabled, ("comebin",))
        self.assertEqual(args.gpu_binners, enabled)
        self.assertEqual(args.parameter_warnings, [])

    def test_budget_below_four_gib_disables_all_gpu_binners(self) -> None:
        args = Namespace(
            tools=["vamb", "comebin", "semibin2", "lorbin"],
            max_gpu_memory="3G",
            batch_size=1024,
            requested_batch_size=1024,
            parameter_warnings=[],
        )
        with redirect_stderr(io.StringIO()):
            enabled = _apply_gpu_memory_minima(args, 48 * 1024)
        self.assertEqual(enabled, ())

    def test_comebin_minimum_memory_tracks_requested_batch_size(self) -> None:
        expected = {
            128: 1.0,
            256: 2.0,
            512: 4.0,
            1024: 8.0,
            2048: 16.0,
        }
        for batch_size, minimum_gib in expected.items():
            with self.subTest(batch_size=batch_size):
                self.assertEqual(
                    _comebin_minimum_memory_gib(batch_size),
                    minimum_gib,
                )

    def test_comebin_falls_back_to_cpu_when_dynamic_minimum_is_not_met(self) -> None:
        cases = (
            (512, "4G", True),
            (512, "3G", False),
            (1024, "8G", True),
            (1024, "4G", False),
            (2048, "16G", True),
            (2048, "8G", False),
        )
        for batch_size, budget, should_use_gpu in cases:
            with self.subTest(batch_size=batch_size, budget=budget):
                args = Namespace(
                    tools=["comebin"],
                    max_gpu_memory=budget,
                    batch_size=batch_size,
                    requested_batch_size=batch_size,
                    parameter_warnings=[],
                )
                stderr = io.StringIO()
                with redirect_stderr(stderr):
                    enabled = _apply_gpu_memory_minima(args, 48 * 1024)

                self.assertEqual(bool(enabled), should_use_gpu)
                if not should_use_gpu:
                    self.assertIn(
                        f"at --batch-size {batch_size}",
                        stderr.getvalue(),
                    )

    def test_builder_hides_cuda_only_from_cpu_fallback_binner(self) -> None:
        options = Namespace(
            gpu=True,
            gpu_binners=("vamb", "semibin2"),
            max_gpu_memory="4G",
            threads=42,
            total_threads=128,
        )
        builder = DirectBinBuilder([], options, Path("."))

        self.assertTrue(builder._uses_gpu("vamb"))
        self.assertFalse(builder._uses_gpu("comebin"))
        self.assertNotEqual(
            builder._gpu_env("vamb")["CUDA_VISIBLE_DEVICES"],
            "",
        )
        self.assertEqual(
            builder._gpu_env("comebin")["CUDA_VISIBLE_DEVICES"],
            "",
        )
        self.assertEqual(builder._cohort_threads(), 128)

    def test_executor_receives_total_threads_without_parallel_multiplier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = Namespace(
                dry_run=False,
                threads=128,
                threads_per_task=42,
                task=3,
                concurrent_sample_limit=3,
                max_memory=100.0,
                gpu=False,
                cuda_device_count=0,
                cuda_task_slots=0,
                delete_tmp_files=False,
                retries=0,
                force=False,
            )
            task = Task(
                id="01.sample.A",
                stage="01_sample",
                command=("true",),
                cwd=root,
                sample="A",
                cpus=42,
            )
            state = MagicMock()
            executor = MagicMock()
            executor.run.return_value = {task.id: "success"}
            output = io.StringIO()
            with (
                patch("metabaw.cli.StateStore", return_value=state),
                patch("metabaw.cli.Executor", return_value=executor) as constructor,
                redirect_stdout(output),
            ):
                result = _run_direct(
                    args,
                    [task],
                    root / "output",
                    root / "output" / "tmp",
                    {"module": "binning"},
                )

            self.assertEqual(result, 0)
            self.assertEqual(constructor.call_args.kwargs["max_cpus"], 128)
            self.assertEqual(constructor.call_args.kwargs["max_parallel"], 3)
            manifest = json.loads(
                (root / "output" / "run_manifest.json").read_text(encoding="utf-8")
            )
            workflow_summary = json.loads(
                (root / "output" / "tmp" / "runtime" / "workflow_summary.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertNotIn("resolved_parameters", workflow_summary)
            self.assertNotIn("workflow_summary", manifest["result_files"])
            self.assertNotIn("memory_diagnostics", manifest["result_files"])
            self.assertFalse((root / "output" / "workflow_summary.json").exists())
            self.assertFalse((root / "output" / "memory_diagnostics.jsonl").exists())
            self.assertFalse(
                (root / "output" / "resolved_parameters.tsv").exists()
            )
            self.assertEqual(manifest["resources"]["total_memory_gb"], 100.0)
            self.assertEqual(
                manifest["resources"]["memory_scope"],
                "combined_pss_of_workflow_task_process_trees",
            )
            self.assertEqual(
                manifest["resources"]["estimated_minimum_total_memory_gb"],
                0.0,
            )
            self.assertNotIn("total workflow PSS budget", output.getvalue())
            self.assertIn("[START INFO]", output.getvalue())
            self.assertIn("total workflow PSS budget", (root / "output" / "start_info.txt").read_text())

    def test_known_memory_shortfall_is_reported_before_executor_starts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = Namespace(dry_run=False, max_memory=100.0)
            task = Task(
                id="01.annotation.gtdbtk",
                stage="01_taxonomy",
                command=("gtdbtk",),
                cwd=root,
                description="Classify MAGs with GTDB-Tk",
                minimum_memory_gb=128.0,
                memory_requirement_hint=(
                    "increase --max-memory to at least 128 GiB, or provide "
                    "a complete matching --gtdbtk_res path to skip GTDB-Tk"
                ),
            )
            stdout = io.StringIO()
            stderr = io.StringIO()
            with (
                patch("metabaw.cli.Executor") as constructor,
                redirect_stdout(stdout),
                redirect_stderr(stderr),
            ):
                result = _run_direct(
                    args,
                    [task],
                    root / "output",
                    root / "output" / "tmp",
                    {"module": "annotation"},
                )

            self.assertEqual(result, 1)
            constructor.assert_not_called()
            startup = (root / "output" / "start_info.txt").read_text()
            self.assertNotIn("[MEMORY ESTIMATE]", stdout.getvalue())
            self.assertIn("[MEMORY ESTIMATE]", startup)
            self.assertIn("estimated minimum total workflow memory=128 GiB", startup)
            self.assertIn("shortfall=28 GiB", startup)
            self.assertIn("[MEMORY CONFIG ERROR]", stderr.getvalue())
            self.assertIn("No workflow task process was started", stderr.getvalue())
            log = root / "output" / "tmp" / "runtime" / "logs" / "memory_preflight.log"
            self.assertIn("[MEMORY CONFIG ERROR]", log.read_text(encoding="utf-8"))
            runtime_summary = root / "output" / "tmp" / "runtime" / "workflow_summary.json"
            self.assertEqual(json.loads(runtime_summary.read_text())["final_status"], "failed_preflight")
            self.assertFalse((root / "output" / "workflow_summary.json").exists())
            self.assertFalse((root / "output" / "memory_diagnostics.jsonl").exists())

    def test_final_failure_summary_repeats_runtime_memory_reason(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = Namespace(
                dry_run=False,
                threads=8,
                threads_per_task=4,
                task=2,
                concurrent_sample_limit=2,
                max_memory=100.0,
                gpu=False,
                cuda_device_count=0,
                cuda_task_slots=0,
                delete_tmp_files=False,
                retries=0,
                force=False,
            )
            task = Task(
                id="01.sample.A",
                stage="01_sample",
                command=("true",),
                cwd=root,
            )
            state = MagicMock()
            executor = MagicMock()
            executor.run.return_value = {task.id: "blocked"}
            executor.peak_memory_gb = 103.43
            executor.memory_abort_reason = (
                "total workflow PSS=103.43 GiB exceeded --max-memory=100 GiB"
            )
            stderr = io.StringIO()
            with (
                patch("metabaw.cli.StateStore", return_value=state),
                patch("metabaw.cli.Executor", return_value=executor),
                redirect_stdout(io.StringIO()),
                redirect_stderr(stderr),
            ):
                result = _run_direct(
                    args,
                    [task],
                    root / "output",
                    root / "output" / "tmp",
                    {"module": "annotation"},
                )

            self.assertEqual(result, 1)
            self.assertIn("Workflow stopped by --max-memory", stderr.getvalue())
            self.assertIn("total workflow PSS=103.43 GiB", stderr.getvalue())


class RuntimeReportTests(unittest.TestCase):
    def _run_case(
        self,
        root: Path,
        module: str = "annotation",
        *,
        status: str = "success",
        temp_files: Path | None = None,
        delete_tmp_files: bool = False,
        write_diagnostics: bool = False,
        dry_run: bool = False,
    ) -> tuple[int, str, str, MagicMock]:
        output = root / "output"
        args = Namespace(
            dry_run=dry_run,
            threads=8,
            threads_per_task=4,
            task=2,
            concurrent_sample_limit=2,
            max_memory=100.0,
            gpu=False,
            cuda_device_count=0,
            cuda_task_slots=0,
            delete_tmp_files=delete_tmp_files,
            retries=0,
            force=False,
        )
        task = Task(id="01.sample.A", stage="01_sample", command=("true",), cwd=root)
        state = MagicMock()
        state.records.return_value = []
        executor = MagicMock()
        executor.run.return_value = {task.id: status}
        executor.peak_memory_gb = 0.25
        executor.memory_abort_reason = None

        def create_executor(**kwargs):
            if write_diagnostics:
                with kwargs["memory_diagnostic_path"].open("a", encoding="utf-8") as handle:
                    handle.write('{"event": "test_diagnostic"}\n')
            return executor

        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch("metabaw.cli.StateStore", return_value=state),
            patch("metabaw.cli.Executor", side_effect=create_executor) as constructor,
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            result = _run_direct(
                args,
                [task],
                output,
                temp_files if temp_files is not None else output / "tmp",
                {"module": module, "failure_policy": "strict"},
            )
        return result, stdout.getvalue(), stderr.getvalue(), constructor

    def test_both_modules_keep_reports_in_default_or_custom_runtime_on_success_and_failure(self) -> None:
        for module in ("binning", "annotation"):
            for status in ("success", "failed"):
                for custom_tmp in (False, True):
                    with self.subTest(module=module, status=status, custom_tmp=custom_tmp):
                        with tempfile.TemporaryDirectory() as directory:
                            root = Path(directory)
                            output = root / "output"
                            temporary = root / "external_tmp" if custom_tmp else output / "tmp"
                            result, stdout, _, constructor = self._run_case(
                                root, module, status=status, temp_files=temporary,
                                write_diagnostics=status == "failed",
                            )
                            self.assertEqual(result, 0 if status == "success" else 1)
                            runtime = temporary / "runtime"
                            for name in ("workflow_summary.json", "memory_diagnostics.jsonl"):
                                self.assertFalse((output / name).exists())
                            summary = json.loads((runtime / "workflow_summary.json").read_text())
                            self.assertEqual(summary["final_status"], status)
                            self.assertEqual(summary["task_status_counts"], {status: 1})
                            self.assertEqual(summary["run_manifest"], str(output / "run_manifest.json"))
                            self.assertEqual(
                                summary["binner_summary"],
                                str(output / "binner_summary.tsv") if module == "binning" else None,
                            )
                            self.assertEqual((runtime / "memory_diagnostics.jsonl").exists(), status == "failed")
                            manifest = json.loads((output / "run_manifest.json").read_text())
                            for key, name in (
                                ("workflow_summary", "workflow_summary.json"),
                                ("memory_diagnostics", "memory_diagnostics.jsonl"),
                            ):
                                self.assertNotIn(key, manifest["result_files"])
                                self.assertEqual(manifest["temporary_files"][key], str(runtime / name))
                            self.assertEqual(manifest["execution"]["final_status"], status)
                            kwargs = constructor.call_args.kwargs
                            self.assertEqual(kwargs["memory_diagnostic_path"], runtime / "memory_diagnostics.jsonl")
                            self.assertEqual(kwargs["max_memory_gb"], 100.0)
                            self.assertEqual(kwargs["fail_fast"], module == "binning")
                            self.assertNotIn("workflow_summary.json", stdout)
                            self.assertNotIn("memory_diagnostics.jsonl", stdout)

    def test_resume_moves_legacy_reports_and_preserves_existing_diagnostics(self) -> None:
        for collision in (False, True):
            with self.subTest(collision=collision), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                output = root / "output"
                output.mkdir()
                temporary = root / "external_tmp"
                runtime = temporary / "runtime"
                runtime.mkdir(parents=True)
                (output / "memory_diagnostics.jsonl").write_text("legacy diagnostic\n")
                (output / "workflow_summary.json").write_text('{"final_status": "old"}\n')
                unrelated = output / "user_notes.txt"
                unrelated.write_text("keep user data")
                if collision:
                    (runtime / "memory_diagnostics.jsonl").write_text("existing diagnostic\n")
                    (runtime / "workflow_summary.json").write_text('{"final_status": "older"}\n')
                result, _, _, _ = self._run_case(
                    root, temp_files=temporary, write_diagnostics=True,
                )
                self.assertEqual(result, 0)
                self.assertFalse((output / "workflow_summary.json").exists())
                self.assertFalse((output / "memory_diagnostics.jsonl").exists())
                diagnostics = "".join(path.read_text() for path in runtime.glob("memory_diagnostics*.jsonl"))
                self.assertIn("legacy diagnostic", diagnostics)
                self.assertIn("test_diagnostic", diagnostics)
                if collision:
                    self.assertIn("existing diagnostic", diagnostics)
                    previous_summary = list(runtime.glob("workflow_summary.previous-*.json"))
                    self.assertEqual(len(previous_summary), 1)
                    self.assertEqual(json.loads(previous_summary[0].read_text())["final_status"], "old")
                self.assertEqual(json.loads((runtime / "workflow_summary.json").read_text())["final_status"], "success")
                self.assertEqual(unrelated.read_text(), "keep user data")

    def test_successful_cleanup_removes_runtime_reports_but_failure_retains_them(self) -> None:
        for module in ("binning", "annotation"):
            for status in ("success", "failed"):
                with self.subTest(module=module, status=status), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    output = root / "output"
                    output.mkdir()
                    result_file = output / "example_MAG.fa"
                    result_file.write_text(">example_MAG\nACGT\n")
                    result, _, _, _ = self._run_case(
                        root, module, status=status, delete_tmp_files=True,
                        write_diagnostics=True,
                    )
                    self.assertEqual(result, 0 if status == "success" else 1)
                    self.assertEqual((output / "tmp").exists(), status == "failed")
                    self.assertEqual(result_file.read_text(), ">example_MAG\nACGT\n")
                    manifest = json.loads((output / "run_manifest.json").read_text())
                    self.assertEqual(manifest["execution"]["final_status"], status)
                    for name in ("workflow_summary.json", "memory_diagnostics.jsonl"):
                        self.assertFalse((output / name).exists())
                        self.assertEqual((output / "tmp" / "runtime" / name).exists(), status == "failed")

    def test_dry_run_does_not_move_legacy_reports_or_create_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            output.mkdir()
            for name in ("workflow_summary.json", "memory_diagnostics.jsonl"):
                (output / name).write_text("legacy report")
            result, _, _, constructor = self._run_case(root, dry_run=True)
            self.assertEqual(result, 0)
            constructor.assert_not_called()
            self.assertFalse((output / "tmp").exists())
            for name in ("workflow_summary.json", "memory_diagnostics.jsonl"):
                self.assertEqual((output / name).read_text(), "legacy report")


if __name__ == "__main__":
    unittest.main()
