from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from metabaw.dependencies import isolated_run_prefix
from metabaw.executor import Executor
from metabaw.model import Task
from metabaw.state import StateStore


class IsolatedRunPrefixTests(unittest.TestCase):
    def test_conda_live_output_disables_internal_capture(self) -> None:
        prefix = isolated_run_prefix(
            "comebin-env",
            "/opt/conda/bin/conda",
            live_output=True,
        )

        self.assertEqual(
            prefix,
            (
                "/opt/conda/bin/conda",
                "run",
                "--no-capture-output",
                "--name",
                "comebin-env",
            ),
        )

    def test_non_conda_frontend_does_not_receive_conda_live_output_flag(self) -> None:
        prefix = isolated_run_prefix(
            "comebin-env",
            "/opt/mamba/bin/mamba",
            live_output=True,
        )

        self.assertNotIn("--no-capture-output", prefix)


class ExecutorResilienceTests(unittest.TestCase):
    def test_worker_exception_becomes_a_logged_task_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = Task(
                id="unexpected.worker",
                stage="01_test",
                command=(sys.executable, "-c", "print('not reached')"),
                cwd=root,
            )
            state = StateStore(root / "state.sqlite3")
            output = io.StringIO()
            try:
                executor = Executor(
                    state=state,
                    log_dir=root / "logs",
                    max_cpus=1,
                    max_parallel=1,
                    show_progress=True,
                )
                with patch(
                    "metabaw.executor._execute_task",
                    side_effect=RuntimeError("simulated worker crash"),
                ):
                    with redirect_stdout(output):
                        statuses = executor.run([task])
                record = state.get(task.id)
            finally:
                state.close()

            self.assertEqual(statuses[task.id], "failed")
            self.assertIsNotNone(record)
            assert record is not None
            self.assertEqual(record.exit_code, 70)
            self.assertIn("RuntimeError: simulated worker crash", record.message or "")
            self.assertIn("[ERROR", output.getvalue())
            log = (root / "logs" / "unexpected.worker.log").read_text(
                encoding="utf-8"
            )
            self.assertIn("# executor_exception: RuntimeError", log)
            self.assertIn("# exit_code: 70", log)

    def test_retry_archives_the_previous_attempt_log(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / "attempted"
            script = (
                "from pathlib import Path\n"
                f"marker = Path({str(marker)!r})\n"
                "if marker.exists():\n"
                "    print('second-attempt-output', flush=True)\n"
                "else:\n"
                "    marker.write_text('attempted', encoding='utf-8')\n"
                "    print('first-attempt-output', flush=True)\n"
                "    raise SystemExit(1)\n"
            )
            task = Task(
                id="retry.task",
                stage="01_test",
                command=(sys.executable, "-c", script),
                cwd=root,
                automatic_retries=1,
            )
            state = StateStore(root / "state.sqlite3")
            try:
                statuses = Executor(
                    state=state,
                    log_dir=root / "logs",
                    max_cpus=1,
                    max_parallel=1,
                    retries=0,
                ).run([task])
            finally:
                state.close()

            self.assertEqual(statuses[task.id], "success")
            first_log = root / "logs" / "retry.task.attempt-1.log"
            final_log = root / "logs" / "retry.task.log"
            self.assertIn(
                "first-attempt-output",
                first_log.read_text(encoding="utf-8"),
            )
            self.assertIn(
                "second-attempt-output",
                final_log.read_text(encoding="utf-8"),
            )

    def test_success_record_with_changed_fingerprint_is_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "result.txt"
            first = Task(
                id="fingerprint.task",
                stage="01_test",
                command=(
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(result)!r}).write_text('v1')",
                ),
                cwd=root,
                outputs=(result,),
            )
            second = Task(
                id=first.id,
                stage=first.stage,
                command=(
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(result)!r}).write_text('v2')",
                ),
                cwd=root,
                outputs=(result,),
            )
            state = StateStore(root / "state.sqlite3")
            try:
                first_status = Executor(
                    state=state,
                    log_dir=root / "logs",
                    max_cpus=1,
                    max_parallel=1,
                ).run([first])
                second_status = Executor(
                    state=state,
                    log_dir=root / "logs",
                    max_cpus=1,
                    max_parallel=1,
                ).run([second])
            finally:
                state.close()

            self.assertEqual(first_status[first.id], "success")
            self.assertEqual(second_status[second.id], "success")
            self.assertEqual(result.read_text(encoding="utf-8"), "v2")

    def test_complete_output_without_state_is_still_adopted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "external.txt"
            result.write_text("external", encoding="utf-8")
            task = Task(
                id="external.output",
                stage="01_test",
                command=(
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(result)!r}).write_text('ran')",
                ),
                cwd=root,
                outputs=(result,),
            )
            state = StateStore(root / "state.sqlite3")
            try:
                statuses = Executor(
                    state=state,
                    log_dir=root / "logs",
                    max_cpus=1,
                    max_parallel=1,
                ).run([task])
                record = state.get(task.id)
            finally:
                state.close()

            self.assertEqual(statuses[task.id], "skipped")
            self.assertEqual(result.read_text(encoding="utf-8"), "external")
            self.assertIsNotNone(record)
            assert record is not None
            self.assertEqual(record.status, "success")

    def test_complete_output_with_non_success_state_is_rerun(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = root / "stale.txt"
            result.write_text("stale", encoding="utf-8")
            task = Task(
                id="blocked.output",
                stage="01_test",
                command=(
                    sys.executable,
                    "-c",
                    f"from pathlib import Path; Path({str(result)!r}).write_text('rerun')",
                ),
                cwd=root,
                outputs=(result,),
            )
            state = StateStore(root / "state.sqlite3")
            try:
                state.start(task.id, task.fingerprint([]), root / "old.log")
                state.finish(task.id, "blocked", 130, "interrupted")
                statuses = Executor(
                    state=state,
                    log_dir=root / "logs",
                    max_cpus=1,
                    max_parallel=1,
                ).run([task])
            finally:
                state.close()

            self.assertEqual(statuses[task.id], "success")
            self.assertEqual(result.read_text(encoding="utf-8"), "rerun")


if __name__ == "__main__":
    unittest.main()
