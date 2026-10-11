from __future__ import annotations

from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from metabaw.executor import (
    Executor,
    RunResult,
    _elapsed_clock,
    _execute_task,
    _failure_diagnosis,
    _linux_process_tree_memory_bytes,
    _memory_limit_exceeded,
    _WorkflowMemoryBudget,
)
from metabaw.model import Task
from metabaw.state import StateStore


class ExecutorProgressTests(unittest.TestCase):
    def test_process_tree_pss_does_not_double_count_shared_pages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)

            def write_process(pid: int, parent: int, rss_kb: int, pss_kb: int) -> None:
                process = proc / str(pid)
                process.mkdir()
                (process / "stat").write_text(
                    f"{pid} (worker) S {parent} " + "0 " * 17 + "123\n",
                    encoding="utf-8",
                )
                (process / "status").write_text(
                    f"Name:\tworker\nVmRSS:\t{rss_kb} kB\n",
                    encoding="utf-8",
                )
                (process / "smaps_rollup").write_text(
                    f"00400000-00401000 r--p 00000000 00:00 0\nPss: {pss_kb} kB\n",
                    encoding="utf-8",
                )

            write_process(100, 1, 100_000, 60_000)
            write_process(101, 100, 100_000, 40_000)
            write_process(999, 1, 80_000, 80_000)

            measured = _linux_process_tree_memory_bytes((100,), proc_root=proc)

            self.assertEqual(measured, 100_000 * 1024)

    def test_process_tree_memory_falls_back_to_rss_without_smaps_rollup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)
            process = proc / "100"
            process.mkdir()
            (process / "stat").write_text(
                "100 (worker) S 1 " + "0 " * 17 + "123\n",
                encoding="utf-8",
            )
            (process / "status").write_text(
                "Name:\tworker\nVmRSS:\t1234 kB\n",
                encoding="utf-8",
            )

            measured = _linux_process_tree_memory_bytes((100,), proc_root=proc)

            self.assertEqual(measured, 1234 * 1024)

    def test_elapsed_clock_uses_hours_minutes_and_seconds(self) -> None:
        self.assertEqual(_elapsed_clock(10.0, now=3671.9), "01:01:01")

    def test_long_unix_socket_path_has_a_specific_diagnosis(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "vamb.log"
            log.write_text("OSError: AF_UNIX path too long\n", encoding="utf-8")

            diagnosis = _failure_diagnosis(log)

            self.assertIn("temporary path is too long", diagnosis)

    def test_morecore_error_is_recognized_as_a_cpu_memory_breach(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "coverm.log"
            log.write_text(
                "[morecore] insufficient memory\n"
                "samtools sort: failed to read header from '-'\n",
                encoding="utf-8",
            )

            self.assertTrue(_memory_limit_exceeded(log, 1))
            self.assertIn("--max-memory", _failure_diagnosis(log))

            log.write_text("CUDA out of memory\n", encoding="utf-8")
            self.assertFalse(_memory_limit_exceeded(log, 1))

    def test_task_log_marks_memory_breach_and_sets_global_abort(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "memory.log"
            task = Task(
                id="memory",
                stage="memory",
                command=(
                    sys.executable,
                    "-c",
                    "print('[morecore] insufficient memory'); raise SystemExit(1)",
                ),
                cwd=root,
            )
            memory_budget = _WorkflowMemoryBudget(100, sampler=lambda _roots: 0)

            result = _execute_task(
                task,
                log,
                "/bin/bash",
                100,
                root / "tmp",
                memory_budget,
            )

            self.assertTrue(result.memory_limit_exceeded)
            self.assertTrue(memory_budget.event.is_set())
            self.assertIn("# memory_limit_exceeded:", log.read_text(encoding="utf-8"))

    def test_progress_groups_run_and_print_sequentially(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = [
                Task(
                    id=f"01.first.{sample}",
                    stage="01_first",
                    command=("unused",),
                    cwd=root,
                    sample=sample,
                )
                for sample in ("A", "B")
            ] + [
                Task(
                    id=f"02.second.{sample}",
                    stage="02_second",
                    command=("unused",),
                    cwd=root,
                    sample=sample,
                )
                for sample in ("A", "B")
            ]
            events: list[tuple[str, str]] = []
            lock = threading.Lock()

            def fake_execute(task, *_args, **_kwargs):
                group = "first" if task.id.startswith("01.") else "second"
                with lock:
                    events.append(("start", group))
                time.sleep(0.02)
                with lock:
                    events.append(("end", group))
                return RunResult(task.id, 0, "ok")

            state = StateStore(root / "state.sqlite3")
            output = io.StringIO()
            try:
                executor = Executor(
                    state=state,
                    log_dir=root / "logs",
                    max_cpus=4,
                    max_parallel=4,
                    show_progress=True,
                    started_at=time.monotonic(),
                )
                with patch("metabaw.executor._execute_task", side_effect=fake_execute):
                    with redirect_stdout(output):
                        statuses = executor.run(tasks)
            finally:
                state.close()

            first_second_start = events.index(("start", "second"))
            self.assertEqual(
                sum(
                    event == ("end", "first")
                    for event in events[:first_second_start]
                ),
                2,
            )
            self.assertEqual(set(statuses.values()), {"success"})

            text = output.getvalue()
            markers = [
                "[STEP 1/2]",
                "[STEP 2/2]",
            ]
            positions = [text.index(marker) for marker in markers]
            self.assertEqual(positions, sorted(positions))
            self.assertNotIn("[START", text)
            self.assertNotIn("[DONE", text)
            self.assertNotIn("T", text.splitlines()[0].split("]", 1)[0])
            self.assertNotIn("+08:00", text)

    def test_internal_semibin_input_is_not_reported_as_a_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            samples = ("A", "B", "semibin2_multisample_input")
            tasks = [
                Task(
                    id=f"01.prepare.{sample}",
                    stage="01_prepare",
                    command=("unused",),
                    cwd=root,
                    sample=sample,
                )
                for sample in samples
            ]

            def fake_execute(task, *_args, **_kwargs):
                return RunResult(task.id, 0, "ok")

            state = StateStore(root / "state.sqlite3")
            output = io.StringIO()
            try:
                executor = Executor(
                    state=state,
                    log_dir=root / "logs",
                    max_cpus=3,
                    max_parallel=3,
                    show_progress=True,
                    started_at=time.monotonic(),
                )
                with patch("metabaw.executor._execute_task", side_effect=fake_execute):
                    with redirect_stdout(output):
                        executor.run(tasks)
            finally:
                state.close()

            text = output.getvalue()
            self.assertIn("across 2 sample(s) [A,B]", text)
            self.assertNotIn("[A,B,semibin2_multisample_input]", text)

    def test_two_sample_slots_share_a_32_thread_budget_across_eight_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = [
                Task(
                    id=f"02.annotation.coverm.S{number}",
                    stage="02_abundance",
                    command=("unused",),
                    cwd=root,
                    sample=f"S{number}",
                    cpus=16,
                )
                for number in range(1, 9)
            ]
            active = 0
            maximum_active = 0
            lock = threading.Lock()

            def fake_execute(task, *_args, **_kwargs):
                nonlocal active, maximum_active
                with lock:
                    active += 1
                    maximum_active = max(maximum_active, active)
                time.sleep(0.02)
                with lock:
                    active -= 1
                return RunResult(task.id, 0, "ok")

            state = StateStore(root / "state.sqlite3")
            try:
                executor = Executor(
                    state=state,
                    log_dir=root / "logs",
                    max_cpus=32,
                    max_parallel=2,
                    show_progress=False,
                )
                with patch("metabaw.executor._execute_task", side_effect=fake_execute):
                    statuses = executor.run(tasks)
            finally:
                state.close()

            self.assertEqual(maximum_active, 2)
            self.assertEqual(len(statuses), 8)
            self.assertEqual(set(statuses.values()), {"success"})

    def test_memory_breach_stops_retries_running_tasks_and_pending_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = [
                Task(
                    id=f"01.memory.{sample}",
                    stage="01_memory",
                    command=("unused",),
                    cwd=root,
                    sample=sample,
                )
                for sample in ("A", "B", "C")
            ]
            started: list[str] = []
            attempts = 0
            lock = threading.Lock()

            def fake_execute(task, *_args):
                nonlocal attempts
                memory_budget = _args[-1]
                with lock:
                    started.append(task.sample)
                if task.sample == "A":
                    attempts += 1
                    time.sleep(0.02)
                    return RunResult(
                        task.id,
                        1,
                        "insufficient memory",
                        memory_limit_exceeded=True,
                    )
                memory_budget.event.wait(timeout=1)
                return RunResult(
                    task.id,
                    130,
                    "terminated by memory abort",
                    aborted_by_memory_limit=True,
                )

            state = StateStore(root / "state.sqlite3")
            output = io.StringIO()
            try:
                executor = Executor(
                    state=state,
                    log_dir=root / "logs",
                    max_cpus=2,
                    max_parallel=2,
                    retries=3,
                    show_progress=True,
                )
                with patch("metabaw.executor._execute_task", side_effect=fake_execute):
                    with redirect_stdout(output):
                        statuses = executor.run(tasks)
            finally:
                state.close()

            self.assertEqual(attempts, 1)
            self.assertEqual(set(started), {"A", "B"})
            self.assertEqual(statuses["01.memory.A"], "failed")
            self.assertEqual(statuses["01.memory.B"], "blocked")
            self.assertEqual(statuses["01.memory.C"], "blocked")
            self.assertIn("[MEMORY LIMIT]", output.getvalue())
            self.assertIn("disabling retries", output.getvalue())

    def test_memory_breach_terminates_a_real_concurrent_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = [
                Task(
                    id="01.memory.A",
                    stage="01_memory",
                    command=(
                        sys.executable,
                        "-c",
                        (
                            "import time; time.sleep(0.2); "
                            "print('[morecore] insufficient memory', flush=True); "
                            "raise SystemExit(1)"
                        ),
                    ),
                    cwd=root,
                    sample="A",
                ),
                Task(
                    id="01.memory.B",
                    stage="01_memory",
                    command=(sys.executable, "-c", "import time; time.sleep(30)"),
                    cwd=root,
                    sample="B",
                ),
                Task(
                    id="01.memory.C",
                    stage="01_memory",
                    command=(sys.executable, "-c", "print('must not run')"),
                    cwd=root,
                    sample="C",
                ),
            ]

            state = StateStore(root / "state.sqlite3")
            started = time.monotonic()
            try:
                executor = Executor(
                    state=state,
                    log_dir=root / "logs",
                    max_cpus=2,
                    max_parallel=2,
                    max_memory_gb=100,
                    show_progress=False,
                )
                statuses = executor.run(tasks)
            finally:
                state.close()

            self.assertLess(time.monotonic() - started, 5)
            self.assertEqual(statuses["01.memory.A"], "failed")
            self.assertEqual(statuses["01.memory.B"], "blocked")
            self.assertEqual(statuses["01.memory.C"], "blocked")
            self.assertIn(
                "# workflow_abort:",
                (root / "logs" / "01.memory.B.log").read_text(encoding="utf-8"),
            )
            self.assertFalse((root / "logs" / "01.memory.C.log").exists())

    def test_combined_workflow_rss_limit_terminates_all_task_processes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = [
                Task(
                    id=f"01.total.{sample}",
                    stage="01_total",
                    command=(sys.executable, "-c", "import time; time.sleep(30)"),
                    cwd=root,
                    sample=sample,
                )
                for sample in ("A", "B", "C")
            ]

            def simulated_combined_rss(root_pids):
                return 2 * 1024**3 if root_pids else 0

            state = StateStore(root / "state.sqlite3")
            output = io.StringIO()
            started = time.monotonic()
            try:
                executor = Executor(
                    state=state,
                    log_dir=root / "logs",
                    max_cpus=2,
                    max_parallel=2,
                    max_memory_gb=1,
                    show_progress=True,
                    memory_sampler=simulated_combined_rss,
                )
                with redirect_stdout(output):
                    statuses = executor.run(tasks)
            finally:
                state.close()

            self.assertLess(time.monotonic() - started, 5)
            self.assertEqual(set(statuses.values()), {"blocked"})
            self.assertIn("total workflow PSS=2.00 GiB", output.getvalue())
            self.assertIn("--max-memory=1 GiB", output.getvalue())
            self.assertEqual(
                executor.memory_abort_reason,
                "total workflow PSS=2.00 GiB exceeded --max-memory=1 GiB",
            )
            self.assertFalse((root / "logs" / "01.total.C.log").exists())


if __name__ == "__main__":
    unittest.main()
