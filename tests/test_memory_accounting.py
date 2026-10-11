from __future__ import annotations

import json
from contextlib import redirect_stdout
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from metabaw.executor import Executor, _WorkflowMemoryBudget
from metabaw.memory import (
    MemorySnapshot,
    detected_total_memory_gib,
    linux_memory_snapshot,
)
from metabaw.model import Task
from metabaw.state import StateStore


GIB = 1024**3


class MemoryAccountingTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.proc = self.root / "proc"
        self.proc.mkdir()

    def process(self, pid, parent, rss=600 * GIB, pss=300 * GIB, start=123, state="S", name="python worker"):
        folder = self.proc / str(pid)
        folder.mkdir()
        (folder / "stat").write_text(f"{pid} ({name}) {state} {parent} " + "0 " * 17 + f"{start}\n")
        (folder / "status").write_text(f"VmRSS:\t{rss // 1024} kB\n" if rss is not None else "")
        if pss is not None:
            (folder / "smaps_rollup").write_text(f"Pss:\t{pss // 1024} kB\nPss_Anon: 999 kB\n")
        return folder

    def test_detected_total_memory_honors_a_smaller_cgroup_limit(self):
        meminfo = self.root / "meminfo"
        cgroup = self.root / "memory.max"
        meminfo.write_text("MemTotal:       402653184 kB\n", encoding="utf-8")
        cgroup.write_text(str(256 * GIB), encoding="utf-8")
        self.assertEqual(
            detected_total_memory_gib(meminfo, (cgroup,)),
            256.0,
        )
        cgroup.write_text("max\n", encoding="utf-8")
        self.assertEqual(
            detected_total_memory_gib(meminfo, (cgroup,)),
            384.0,
        )

    def test_shared_pages_and_overlapping_roots_are_counted_once(self):
        self.process(100, 1)
        self.process(101, 100)
        self.process(999, 1, pss=999 * GIB)
        sample = linux_memory_snapshot([100, 101], self.proc)
        self.assertEqual(sample.pss_bytes, 600 * GIB)
        self.assertEqual(sample.upper_bound_bytes, 600 * GIB)
        self.assertEqual(sample.missing_pss, 0)

    def test_rollup_unavailable_uses_summed_smaps_pss_not_rss(self):
        folder = self.process(100, 1, pss=None)
        (folder / "smaps").write_text("Pss: 20 kB\nPss_Anon: 888 kB\nRss: 900 kB\nPss: 30 kB\n")
        sample = linux_memory_snapshot([100], self.proc)
        self.assertEqual(sample.pss_bytes, 50 * 1024)
        self.assertEqual(sample.processes[0]["source"], "smaps")

    def test_rollup_permission_error_still_tries_smaps(self):
        folder = self.process(100, 1)
        (folder / "smaps").write_text("Pss: 50 kB\n")
        original = Path.read_text

        def read(path, *args, **kwargs):
            if path.name == "smaps_rollup":
                raise PermissionError(13, "test denied")
            return original(path, *args, **kwargs)

        with patch.object(Path, "read_text", read):
            sample = linux_memory_snapshot([100], self.proc)
        self.assertEqual(sample.pss_bytes, 50 * 1024)
        self.assertIn("PermissionError", sample.processes[0]["error"])

    def test_process_exiting_during_pss_read_does_not_reuse_stale_rss(self):
        self.process(100, 1)
        self.process(101, 100)
        original = Path.read_text
        exited = False

        def read(path, *args, **kwargs):
            nonlocal exited
            if path.parent.name == "101":
                if path.name == "smaps_rollup":
                    exited = True
                    raise ProcessLookupError(3, "test exited")
                if path.name == "stat" and exited:
                    raise FileNotFoundError(2, "test disappeared")
            return original(path, *args, **kwargs)

        with patch.object(Path, "read_text", read):
            sample = linux_memory_snapshot([100], self.proc)
        self.assertEqual(sample.pss_bytes, 300 * GIB)
        self.assertEqual(sample.upper_bound_bytes, 300 * GIB)
        self.assertEqual(sample.processes[1]["state"], "exited")

    def test_zombies_and_recycled_pids_are_excluded(self):
        self.process(100, 1, state="Z")
        self.assertEqual(linux_memory_snapshot([100], self.proc).upper_bound_bytes, 0)
        folder = self.process(101, 1)
        original = Path.read_text
        calls = 0

        def read(path, *args, **kwargs):
            nonlocal calls
            result = original(path, *args, **kwargs)
            if path == folder / "stat":
                calls += 1
                if calls > 1:
                    result = result.replace("123\n", "456\n")
            return result

        with patch.object(Path, "read_text", read):
            sample = linux_memory_snapshot([101], self.proc)
        self.assertEqual(sample.upper_bound_bytes, 0)
        self.assertEqual(sample.processes[0]["state"], "pid_reused")

    def test_reported_skani_esrch_with_visible_pid_and_empty_smaps_is_not_unknown_memory(self):
        self.process(1348655, 1328898, rss=23040000, pss=14622720,
                     start=338792006, name="python3.11")
        self.process(1348784, 1348655, rss=186650624, pss=177790976,
                     start=338792029, name="gtdbtk")
        skani = self.process(1348939, 1348784, rss=None, pss=None,
                             start=338792092, name="skani")
        (skani / "smaps").write_text("")
        original = Path.read_text

        def read(path, *args, **kwargs):
            if path == skani / "smaps_rollup":
                raise ProcessLookupError(3, "No such process")
            return original(path, *args, **kwargs)

        diagnostics = self.root / "memory_diagnostics.jsonl"
        guard = _WorkflowMemoryBudget(160, sampler=lambda roots: linux_memory_snapshot(roots, self.proc),
                                      diagnostic_path=diagnostics)
        guard.register("01.annotation.gtdbtk", 1348655)
        with patch.object(Path, "read_text", read):
            sample = linux_memory_snapshot([1348655], self.proc)
            for _ in range(4):
                self.assertEqual(guard.sample(), 192413696)
        self.assertFalse(guard.event.is_set())
        self.assertFalse(guard.awaiting_confirmation)
        self.assertFalse(sample.unknown_memory)
        self.assertEqual(sample.missing_pss, 0)
        self.assertEqual(sample.processes[-1]["state"], "no_address_space")
        self.assertEqual(sample.processes[-1]["proc_state"], "S")
        self.assertIn("errno=3", sample.processes[-1]["error"])
        records = [json.loads(line) for line in diagnostics.read_text().splitlines()]
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["event"], "no_address_space_excluded")
        self.assertEqual(records[0]["upper_bound_bytes"], 192413696)

    def test_no_address_space_is_resampled_and_descendants_are_still_counted(self):
        parent = self.process(100, 1, rss=None, pss=None)
        (parent / "smaps").write_text("")
        self.process(101, 100, pss=2 * GIB)
        original = Path.read_text

        def read(path, *args, **kwargs):
            if path == parent / "smaps_rollup":
                raise ProcessLookupError(3, "no mm")
            return original(path, *args, **kwargs)

        with patch.object(Path, "read_text", read):
            sample = linux_memory_snapshot([100], self.proc)
        self.assertEqual(sample.pss_bytes, 2 * GIB)
        self.assertEqual(sample.processes[0]["state"], "no_address_space")
        (parent / "smaps_rollup").write_text(f"Pss: {GIB // 1024} kB\n")
        sample = linux_memory_snapshot([100], self.proc)
        self.assertEqual(sample.pss_bytes, 3 * GIB)
        self.assertEqual(sample.processes[0]["state"], "live")

    def test_esrch_then_valid_smaps_pss_keeps_measured_memory(self):
        folder = self.process(100, 1, pss=None)
        (folder / "smaps").write_text("Pss: 50 kB\n")
        original = Path.read_text

        def read(path, *args, **kwargs):
            if path.name == "smaps_rollup":
                raise ProcessLookupError(3, "no mm at previous read")
            return original(path, *args, **kwargs)

        with patch.object(Path, "read_text", read):
            sample = linux_memory_snapshot([100], self.proc)
        self.assertEqual(sample.pss_bytes, 50 * 1024)
        self.assertEqual(sample.processes[0]["state"], "live")

    def test_empty_smaps_without_esrch_or_nonempty_invalid_smaps_are_not_excluded(self):
        folder = self.process(100, 1, rss=None, pss=None)
        original = Path.read_text
        cases = ((PermissionError(13, "denied"), ""),
                 (FileNotFoundError(2, "no rollup on this kernel"), ""),
                 (ProcessLookupError(3, "no mm"), "Rss: 100 kB\n"))
        for error, content in cases:
            with self.subTest(error=error, content=content):
                (folder / "smaps").write_text(content)

                def read(path, *args, **kwargs):
                    if path.name == "smaps_rollup":
                        raise error
                    return original(path, *args, **kwargs)

                with patch.object(Path, "read_text", read):
                    sample = linux_memory_snapshot([100], self.proc)
                self.assertTrue(sample.unknown_memory)
                self.assertEqual(sample.processes[0]["state"], "live")
                guard = _WorkflowMemoryBudget(160, sampler=lambda roots: sample)
                guard.register("test", 100)
                guard.sample()
                guard.sample()
                self.assertEqual(guard.reason, "measurement_unavailable")

    def test_no_address_space_does_not_reuse_previous_status_rss(self):
        folder = self.process(100, 1, rss=1000 * GIB, pss=None)
        (folder / "smaps").write_text("")
        original = Path.read_text

        def read(path, *args, **kwargs):
            if path.name == "smaps_rollup":
                raise ProcessLookupError(3, "mm released after status was read")
            return original(path, *args, **kwargs)

        with patch.object(Path, "read_text", read):
            sample = linux_memory_snapshot([100], self.proc)
        self.assertEqual(sample.upper_bound_bytes, 0)
        self.assertTrue(sample.upper_bound_complete)
        self.assertEqual(sample.processes[0]["state"], "no_address_space")

    def test_esrch_and_unreadable_smaps_are_not_enough_to_exclude_a_visible_pid(self):
        folder = self.process(100, 1, rss=None, pss=None)
        original_read, original_open = Path.read_text, Path.open

        def read(path, *args, **kwargs):
            if path.name == "smaps_rollup":
                raise ProcessLookupError(3, "test race")
            return original_read(path, *args, **kwargs)

        def open_file(path, *args, **kwargs):
            if path == folder / "smaps":
                raise PermissionError(13, "test denied")
            return original_open(path, *args, **kwargs)

        with patch.object(Path, "read_text", read), patch.object(Path, "open", open_file):
            sample = linux_memory_snapshot([100], self.proc)
        self.assertTrue(sample.unknown_memory)
        self.assertEqual(sample.processes[0]["state"], "live")

    def test_incomplete_measurement_is_not_reported_as_a_total_upper_bound(self):
        self.process(100, 1, pss=192413696)
        self.process(101, 100, rss=None, pss=None)
        diagnostics = self.root / "memory_diagnostics.jsonl"
        guard = _WorkflowMemoryBudget(160, sampler=lambda roots: linux_memory_snapshot(roots, self.proc),
                                      diagnostic_path=diagnostics)
        guard.register("test", 100)
        guard.sample()
        guard.sample()
        self.assertTrue(guard.event.is_set())
        self.assertEqual(guard.reason, "measurement_unavailable")
        self.assertIn("partial accounted amount is 0.18 GiB", guard.abort_detail())
        self.assertNotIn("conservative RSS-containing bound is 0.18", guard.abort_detail())
        records = [json.loads(line) for line in diagnostics.read_text().splitlines()]
        self.assertIsNone(records[-1]["upper_bound_bytes"])
        self.assertFalse(records[-1]["upper_bound_complete"])
        self.assertEqual(records[-1]["accounted_bytes"], 192413696)

    def test_rss_only_below_limit_can_continue_without_claiming_pss(self):
        self.process(100, 1, rss=10 * GIB, pss=None)
        guard = _WorkflowMemoryBudget(100, sampler=lambda roots: linux_memory_snapshot(roots, self.proc))
        guard.register("test", 100)
        self.assertEqual(guard.sample(), 0)
        guard.sample()
        self.assertFalse(guard.event.is_set())
        self.assertEqual(guard.peak_bytes, 0)
        self.assertEqual(guard.peak_upper_bound_bytes, 10 * GIB)

    def test_large_rss_is_not_mislabeled_as_a_confirmed_pss_breach(self):
        self.process(100, 1, pss=None)
        self.process(101, 100, pss=None)
        diagnostics = self.root / "memory_diagnostics.jsonl"
        guard = _WorkflowMemoryBudget(700, sampler=lambda roots: linux_memory_snapshot(roots, self.proc),
                                      diagnostic_path=diagnostics)
        guard.register("gtdbtk", 100)
        guard.sample()
        self.assertFalse(guard.event.is_set())
        guard.sample()
        self.assertTrue(guard.event.is_set())
        self.assertEqual(guard.reason, "measurement_unavailable")
        self.assertNotIn("total workflow PSS=1200", guard.abort_detail())
        self.assertIn("NOT a confirmed PSS breach", guard.abort_detail())
        records = [json.loads(line) for line in diagnostics.read_text().splitlines()]
        self.assertEqual(records[-1]["event"], "measurement_unavailable_confirmed")
        self.assertEqual(records[-1]["missing_pss_processes"], 2)
        self.assertEqual(records[-1]["root_tasks"], {"100": "gtdbtk"})
        self.assertEqual(records[-1]["processes"][0]["rss_bytes"], 600 * GIB)

    def test_single_transient_pss_spike_does_not_kill_tasks(self):
        high = MemorySnapshot(({"state": "live", "pss_bytes": 800 * GIB, "rss_bytes": 800 * GIB},))
        low = MemorySnapshot(({"state": "live", "pss_bytes": 100 * GIB, "rss_bytes": 100 * GIB},))
        samples = iter([high, low, high, low])
        guard = _WorkflowMemoryBudget(700, sampler=lambda roots: next(samples))
        guard.register("test", 100)
        for _ in range(4):
            guard.sample()
        self.assertFalse(guard.event.is_set())

    def test_confirmed_real_pss_breach_still_stops(self):
        self.process(100, 1, pss=800 * GIB)
        guard = _WorkflowMemoryBudget(700, sampler=lambda roots: linux_memory_snapshot(roots, self.proc))
        guard.register("test", 100)
        guard.sample()
        self.assertFalse(guard.event.is_set())
        guard.sample()
        self.assertTrue(guard.event.is_set())
        self.assertEqual(guard.reason, "workflow_memory")
        self.assertIn("PSS=800.00 GiB", guard.abort_detail())

    def test_live_unreadable_root_is_not_treated_as_zero_memory(self):
        self.process(100, 1)
        original = Path.read_text

        def read(path, *args, **kwargs):
            if path.name == "stat":
                raise PermissionError(13, "test stat denied")
            return original(path, *args, **kwargs)

        with patch.object(Path, "read_text", read):
            sample = linux_memory_snapshot([100], self.proc)
        self.assertTrue(sample.unknown_memory)
        guard = _WorkflowMemoryBudget(700, sampler=lambda roots: sample)
        guard.register("test", 100)
        guard.sample()
        guard.sample()
        self.assertEqual(guard.reason, "measurement_unavailable")

    def test_executor_distinguishes_confirmed_pss_from_unverifiable_rss_and_stops_process(self):
        for source in ("smaps_rollup", "rss_upper_bound", "unavailable"):
            folder = self.root / source
            folder.mkdir()
            tasks = [Task(f"task.{i}", "test", (sys.executable, "-c", "import time; time.sleep(20)"),
                          folder, sample=str(i)) for i in (1, 2)]

            def sample(roots):
                return MemorySnapshot(tuple({"pid": pid, "state": "live", "source": source,
                    "pss_bytes": 2 * GIB if source == "smaps_rollup" else None,
                    "rss_bytes": None if source == "unavailable" else 2 * GIB} for pid in roots))

            state = StateStore(folder / "state.sqlite3")
            output = io.StringIO()
            try:
                executor = Executor(state, folder / "logs", 1, 1, max_memory_gb=1,
                                    show_progress=True, memory_sampler=sample)
                with redirect_stdout(output):
                    statuses = executor.run(tasks)
            finally:
                state.close()
            self.assertEqual(set(statuses.values()), {"blocked"})
            self.assertEqual(executor.memory_upper_bound_complete, source != "unavailable")
            if source == "unavailable":
                self.assertIsNone(executor.peak_memory_upper_bound_gb)
            self.assertFalse((folder / "logs" / "task.2.log").exists())
            expected = "MEMORY LIMIT" if source == "smaps_rollup" else "MEMORY ACCOUNTING ERROR"
            self.assertIn(f"[{expected}]", output.getvalue())
            rows = [json.loads(line) for line in (folder / "logs" / "memory_diagnostics.jsonl").read_text().splitlines()]
            self.assertTrue(rows[-1]["event"].endswith("_confirmed"))
            self.assertEqual(rows[-1]["processes"][0]["source"], source)


if __name__ == "__main__":
    unittest.main()
