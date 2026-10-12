from __future__ import annotations

from argparse import Namespace
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import replace
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from metabaw.cli import _enable_bin_cuda, _configure_thread_budget
from metabaw.dependencies import CudaDevice, CupyRuntimeStatus, HostCudaStatus, cupy_runtime_status
from metabaw.direct import BINNER_ORDER, BinOptions, DirectBinBuilder
from metabaw.discovery import Analysis, ReadSample
from metabaw.executor import (Executor, RunResult, _WorkflowMemoryBudget, _execute_task,
                              _memory_limit_exceeded, _failure_diagnosis)
from metabaw.gpu_limits import install_cupy_limit, install_torch_limit
from metabaw.gpu_resources import GpuMonitor, process_tasks
from metabaw.memory import MemorySnapshot
from metabaw.model import Task
from metabaw.resources import (GIB, GpuBudget, GpuReservations, available_cpus,
                               available_memory_gib, cgroup_directories, memory_bytes,
                               plan_gpu_resources, plan_resources)
from metabaw.state import StateStore


def options(root, binner, gpu=True):
    return BinOptions(
        outdir=root / "output", workdir=root / "work", threads=32, total_threads=128,
        read_type="long" if binner == "lorbin" else "short", align_tool="bowtie2",
        flye_read_type="--nano-raw", binners=(binner,), min_contig_length=1500,
        min_fasta_kbs=200, batch_size=1024, refiner="magscot", quality_control="checkm2",
        min_completeness=50, max_contamination=10, min_quality_score=None, run_gunc=False,
        run_trna=False, trna_pass=None, run_rrna=False, rrna_pass=False, dereplicator="galah",
        environment=None, tag_contigs=False, gpu=gpu, gpu_binners=(binner,), max_gpu_memory="8G",
        assembly_strategy="existing-contigs", magscot_dir=root / "MAGScoT", checkm2_db=None,
        gunc_db=None, gtdbtk_data=None, extra_args={},
    )


def cluster_tasks(root, binner, gpu=True):
    builder = DirectBinBuilder([], options(root, binner, gpu), root)
    for i in range(4):
        name = f"S{i}"
        builder._active_sample = name
        contigs = root / f"{name}.fa"
        sample = ReadSample(name, root / f"{name}.fq", contigs=contigs)
        analysis = Analysis(name, (sample,), (contigs,), False)
        builder._binners(analysis, Task(f"prepare.{name}", "01", ("true",), root), contigs,
                         Task(f"bam.{name}", "02", ("true",), root),
                         [root / f"{name}.bam"], root / name / "bamset")
    return [task for task in builder.tasks if task.id.startswith(f"03.bin.{binner}.")]


def device(index="0", free=40, total=48):
    return CudaDevice(index, "GPU " + index, total * 1024, "580", free * 1024, "GPU-" + index)


def snapshot(pid, parent=1, pss=0, rss=None):
    return {"pid": pid, "ppid": parent, "state": "live", "pss_bytes": pss,
            "rss_bytes": pss if rss is None else rss}


class SharedPlanningTests(unittest.TestCase):
    def test_over_request_is_capped_then_divided_equally(self):
        plan = plan_resources(256, 200, 4, 16, cpu_snapshot=(tuple(range(128)), 128), memory_snapshot=100)
        self.assertEqual(plan.threads, 128)
        self.assertEqual(plan.memory_gib, 80)
        self.assertEqual(plan.memory_per_task_gib, 20)
        args = Namespace(threads=256, max_memory=200, task=4, resource_safety=True)
        with patch("metabaw.cli.plan_resources", return_value=plan), redirect_stdout(io.StringIO()):
            self.assertEqual(_configure_thread_budget(args, 16), 32)
        self.assertEqual(args.concurrent_sample_limit, 4)
        self.assertEqual(args.requested_threads, 256)
        self.assertEqual(args.max_memory, 80)

    def test_smaller_user_request_and_slot_count_are_preserved(self):
        plan = plan_resources(64, 32, 4, 16, cpu_snapshot=(tuple(range(128)), 128), memory_snapshot=100)
        self.assertEqual((plan.threads, plan.memory_gib, plan.slots, plan.memory_per_task_gib), (64, 32, 4, 8))
        self.assertEqual(plan_resources(2, 32, 4, 16, cpu_snapshot=(tuple(range(128)), 128),
                                       memory_snapshot=100).slots, 2)
        with patch("metabaw.resources.available_memory_gib", return_value=None):
            with self.assertRaisesRegex(ValueError, "Cannot measure"):
                plan_resources(8, 32, 4, 16, cpu_snapshot=(tuple(range(128)), 128))

    def test_one_cpu_is_usable_and_ram_still_has_its_80_percent_limit(self):
        plan = plan_resources(128, 200, 5, 5,
                              cpu_snapshot=((7,), 1), memory_snapshot=100)
        self.assertEqual((plan.threads, plan.slots, plan.cpu_ids), (1, 1, (7,)))
        self.assertEqual(plan.memory_gib, 80)
        with self.assertRaisesRegex(ValueError, "affinity/quota"):
            plan_resources(128, 200, 5, 5, cpu_snapshot=((7,), 0.9), memory_snapshot=100)

    def test_available_ram_uses_free_memory_and_nested_cgroup_headroom(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            meminfo = root / "meminfo"
            meminfo.write_text("MemTotal: 999999999 kB\nMemAvailable: 104857600 kB\n")
            child = root / "group" / "child"
            child.mkdir(parents=True)
            membership = root / "membership"
            membership.write_text("0::/group/child\n")
            (child / "memory.max").write_text(str(80 * GIB))
            (child / "memory.current").write_text(str(20 * GIB))
            (root / "group" / "memory.max").write_text(str(90 * GIB))
            (root / "group" / "memory.current").write_text(str(50 * GIB))
            groups = cgroup_directories(root, membership)
            self.assertEqual(groups, tuple(path.resolve() for path in (child, root / "group", root)))
            self.assertEqual(available_memory_gib(meminfo, groups), 40)
            (child / "memory.current").write_text(str(80 * GIB))
            self.assertEqual(available_memory_gib(meminfo, groups), 0)

    def test_cpu_affinity_and_quota_are_both_honored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cpu.max").write_text("650000 100000")
            with patch("metabaw.resources.os.sched_getaffinity", return_value={2, 4, 6, 8, 10, 12, 14, 16}, create=True):
                ids, capacity = available_cpus((root,))
            self.assertEqual(capacity, 6.5)
            plan = plan_resources(128, 100, 4, 16, cpu_snapshot=(ids, capacity), memory_snapshot=100)
            self.assertEqual(plan.threads, 6)
            self.assertTrue(set(plan.cpu_ids).issubset(ids))

    def test_gpu_request_is_aggregate_and_supports_shared_single_gpu(self):
        for requested, expected_share in (("100%", 8 * GIB), ("16G", 4 * GIB)):
            with self.subTest(requested=requested):
                devices, share = plan_gpu_resources((device(),), requested, 4)
                self.assertEqual(share, expected_share)
                pool = GpuReservations(devices, share)
                self.assertEqual(pool.capacity, 4)
                self.assertTrue(all(pool.acquire(f"S{i}").index == "0" for i in range(4)))
                self.assertIsNone(pool.acquire("extra"))
                pool.release("S0")
                self.assertIsNotNone(pool.acquire("extra"))
                self.assertLessEqual(sum(d.limit_bytes for d in devices), int(.8 * 40 * GIB))

    def test_gpu_shares_fit_physical_cards_even_when_heterogeneous(self):
        cases = [((device("0"), device("1")), "4G", 3),
                 ((device("0", 70, 80), device("1", 10, 16)), "100%", 4),
                 ((device("0", 10, 16), device("1", 10, 16)), "100%", 3)]
        for raw, requested, slots in cases:
            with self.subTest(requested=requested, slots=slots, free=[d.memory_free_mib for d in raw]):
                devices, share = plan_gpu_resources(raw, requested, slots)
                pool = GpuReservations(devices, share)
                self.assertEqual(pool.capacity, slots)
                self.assertTrue(all(pool.acquire(str(i)) for i in range(slots)))
                self.assertIsNone(pool.acquire("overflow"))
                self.assertLessEqual(sum(d.limit_bytes for d in devices),
                                     .8 * sum(d.memory_free_mib * 1024**2 for d in raw))
                for d in devices:
                    self.assertLessEqual(d.limit_bytes, .8 * d.available_bytes)

    def test_visible_masks_unknown_free_memory_and_byte_units(self):
        raw = (device("0"), device("1"), replace(device("2"), memory_free_mib=None))
        devices, _ = plan_gpu_resources(raw, "100%", 4, "GPU-1")
        self.assertEqual([d.index for d in devices], ["1"])
        self.assertEqual(plan_gpu_resources(raw, "100%", 4, ""), ((), 0))
        self.assertEqual(plan_gpu_resources(raw, "100%", 4, "2"), ((), 0))
        for value, expected in (("1024B", 1024), ("1GiB", GIB), ("25%", GIB), ("4096", 4096)):
            self.assertEqual(memory_bytes(value, 4 * GIB), expected)


class GlobalStageThreadTests(unittest.TestCase):
    @staticmethod
    def plan(root, opts):
        analyses = []
        for i in range(4):
            name = f"S{i}"
            contigs = root / f"{name}.fa"
            sample = ReadSample(name, root / f"{name}.R1.fq",
                                root / f"{name}.R2.fq", contigs=contigs)
            analyses.append(Analysis(name, (sample,), (contigs,), False))
        return {task.id: task for task in DirectBinBuilder(analyses, opts, root).build()}

    def assert_global_budget(self, tasks, opts, threads):
        quality = tasks[f"05.qc.{opts.quality_control}"]
        reuse_checkm = opts.refiner == "metawrap" and opts.quality_control == "checkm"
        if reuse_checkm:
            self.assertEqual(quality.cpus, 1)
            self.assertIn("metawrap-checkm-quality", quality.display_command())
            self.assertNotIn("lineage_wf", quality.display_command())
        else:
            self.assertEqual(quality.cpus, threads)
            flag = "--threads" if opts.quality_control == "checkm2" else "-t"
            self.assertRegex(quality.display_command(), rf"{flag} {threads}(?:\s|$)")
        flags = {"05.qc.gunc": "--threads", "05.qc.rna_domain": "--cpus",
                 "05.qc.rna": "--threads",
                 f"06.dereplicate.{opts.dereplicator}":
                     "--threads" if opts.dereplicator == "galah" else "-p"}
        for task_id, flag in flags.items():
            task = tasks[task_id]
            self.assertIsNone(task.sample)
            self.assertEqual(task.cpus, threads)
            self.assertRegex(task.display_command(), rf"{flag} {threads}(?:\s|$)")
        self.assertIsNone(quality.sample)
        for task_id in ("05.catalog.candidates", "05.qc.filter", "06.dereplicate.report"):
            self.assertEqual(tasks[task_id].cpus, 1)
            self.assertIsNone(tasks[task_id].sample)

    def test_all_refiner_quality_and_dereplicator_combinations_use_global_budget(self):
        for refiner in ("magscot", "das_tool", "metawrap"):
            for quality in ("checkm", "checkm2"):
                for dereplicator in ("galah", "drep"):
                    with self.subTest(refiner=refiner, quality=quality, dereplicator=dereplicator), \
                            tempfile.TemporaryDirectory() as directory:
                        root = Path(directory)
                        opts = replace(options(root, "metabat2", gpu=False), refiner=refiner,
                                       quality_control=quality, dereplicator=dereplicator,
                                       run_gunc=True, run_trna=True, run_rrna=True)
                        tasks = self.plan(root, opts)
                        self.assert_global_budget(tasks, opts, 128)
                        for i in range(4):
                            for prefix in ("02.index", "02.map", "03.bin.metabat2"):
                                task_id = f"{prefix}.S{i}" + (f".S{i}" if prefix == "02.map" else "")
                                self.assertEqual(tasks[task_id].cpus, 32)
                                self.assertEqual(tasks[task_id].sample, f"S{i}")
                            for task in tasks.values():
                                if task.stage == "04_refinement" and task.sample == f"S{i}":
                                    self.assertLessEqual(task.cpus, 32)
                            if refiner != "magscot":
                                self.assertEqual(tasks[f"04.refine.{refiner}.S{i}"].cpus, 32)

    def test_effective_cap_keeps_rounding_remainder_for_global_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = plan_resources(128, 80, 4, 4,
                                  cpu_snapshot=(tuple(range(128)), 127), memory_snapshot=100)
            self.assertEqual((plan.threads, plan.slots), (127, 4))
            opts = replace(options(root, "metabat2", gpu=False),
                           threads=plan.threads // plan.slots, total_threads=plan.threads,
                           run_gunc=True, run_trna=True, run_rrna=True)
            tasks = self.plan(root, opts)
            self.assertEqual(tasks["03.bin.metabat2.S0"].cpus, 31)
            self.assert_global_budget(tasks, opts, 127)

    def test_single_slot_and_legacy_options_without_total_budget(self):
        for total_threads in (32, None):
            with self.subTest(total_threads=total_threads), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                opts = replace(options(root, "metabat2", gpu=False), total_threads=total_threads,
                               run_gunc=True, run_trna=True, run_rrna=True)
                self.assert_global_budget(self.plan(root, opts), opts, 32)

    def test_global_assessments_reserve_all_cpus_and_ram_without_oversubscription(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            opts = replace(options(root, "metabat2", gpu=False),
                           run_gunc=True, run_trna=True, run_rrna=True, dereplicator="drep")
            tasks = self.plan(root, opts)
            # Use real generated task resource metadata with synthetic workers;
            # isolate admission accounting from biological prerequisites/outputs.
            selected = [replace(tasks[name], deps=(), wait_for=(), inputs=(), outputs=())
                        for name in ("05.qc.checkm2", "05.qc.gunc", "05.qc.rna_domain",
                                     "05.qc.rna", "06.dereplicate.drep")]
            seen = []
            active = 0
            peak = 0
            lock = threading.Lock()

            def worker(task, *args):
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                    seen.append(task)
                threading.Event().wait(0.02)
                with lock:
                    active -= 1
                return RunResult(task.id, 0, "successful global allocation fixture")

            plan = plan_resources(128, 80, 4, 4,
                                  cpu_snapshot=(tuple(range(160)), 160), memory_snapshot=100)
            state = StateStore(root / "state.sqlite3")
            try:
                executor = Executor(state, root / "logs", 128, 4, "unused", max_memory_gb=80,
                                    resource_plan=plan, parallel_ready_tasks=True,
                                    memory_sampler=lambda roots: MemorySnapshot())
                with patch("metabaw.executor._execute_task", side_effect=worker), \
                        patch("metabaw.executor.available_memory_gib", return_value=100):
                    statuses = executor.run(selected)
            finally:
                state.close()
            self.assertEqual(set(statuses.values()), {"success"})
            self.assertEqual(len(seen), 5)
            self.assertEqual(peak, 1)
            self.assertEqual({t.cpus for t in seen}, {128})
            self.assertEqual({len(t.cpu_affinity) for t in seen}, {128})
            self.assertEqual({t.memory_limit_gb for t in seen}, {80})
            self.assertEqual({t.env["CUDA_VISIBLE_DEVICES"] for t in seen}, {""})


class BinnerConcurrencyTests(unittest.TestCase):
    def test_each_of_six_real_binner_plans_can_start_four_parallel_samples(self):
        # Use the actual tool commands/metadata; replace only process execution.
        # This proves scheduler behavior, not successful biological binning.
        for binner in BINNER_ORDER:
            with self.subTest(binner=binner), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                tasks = cluster_tasks(root, binner)
                self.assertEqual(len(tasks), 4)
                # Pretend completed prerequisites have been satisfied; focus on
                # the actual cluster task's CPU/GPU/paths and concurrency.
                tasks = [replace(t, deps=(), wait_for=()) for t in tasks]
                gpu = binner != "metabat2"
                self.assertEqual({t.gpus for t in tasks}, {int(gpu)})
                self.assertEqual(len({str(t.outputs[0]) for t in tasks}), 4)
                plan = plan_resources(128, 80, 4, 4, cpu_snapshot=(tuple(range(160)), 160), memory_snapshot=100)
                budgets, share = plan_gpu_resources((device(),), "100%", 4) if gpu else ((), 0)
                barrier = threading.Barrier(4, timeout=5)
                seen = []
                lock = threading.Lock()

                def worker(task, *args):
                    with lock:
                        seen.append(task)
                    barrier.wait()
                    return RunResult(task.id, 0, "successful scheduler fixture")

                state = StateStore(root / "state.sqlite3")
                try:
                    executor = Executor(state, root / "logs", 128, 4, "unused",
                                        max_memory_gb=80, max_gpus=4 if gpu else 0,
                                        resource_plan=plan, gpu_device_budgets=budgets,
                                        gpu_share_bytes=share, parallel_ready_tasks=True,
                                        memory_sampler=lambda roots: MemorySnapshot(),
                                        gpu_sampler=lambda: ({"0": ("GPU-0", 40 * GIB)}, []))
                    with patch("metabaw.executor._execute_task", side_effect=worker), \
                            patch("metabaw.executor.available_memory_gib", return_value=100):
                        statuses = executor.run(tasks)
                finally:
                    state.close()
                self.assertEqual(set(statuses.values()), {"success"})
                self.assertEqual(len(seen), 4)
                self.assertEqual({t.memory_limit_gb for t in seen}, {20})
                self.assertEqual(sum(len(t.cpu_affinity) for t in seen), 128)
                self.assertEqual(len(set().union(*(set(t.cpu_affinity) for t in seen))), 128)
                if gpu:
                    self.assertEqual({t.env["CUDA_VISIBLE_DEVICES"] for t in seen}, {"GPU-0"})
                    self.assertEqual(sum(int(t.env["METABAW_GPU_BUDGET_BYTES"]) for t in seen), 32 * GIB)
                    self.assertTrue(all(int(t.env["METABAW_GPU_ALLOCATOR_BYTES"]) < share for t in seen))
                    self.assertTrue(all(not t.enforce_memory_limit for t in seen))
                else:
                    self.assertEqual({t.env["CUDA_VISIBLE_DEVICES"] for t in seen}, {""})

    def test_all_six_binners_also_support_four_parallel_cpu_fallbacks(self):
        for binner in BINNER_ORDER:
            with self.subTest(binner=binner), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                tasks = [replace(t, deps=(), wait_for=()) for t in cluster_tasks(root, binner, gpu=False)]
                self.assertEqual({t.gpus for t in tasks}, {0})
                barrier = threading.Barrier(4, timeout=5)
                seen = []
                def worker(task, *args):
                    seen.append(task)
                    barrier.wait()
                    return RunResult(task.id, 0, "ok")
                plan = plan_resources(128, 80, 4, 4, cpu_snapshot=(tuple(range(160)), 160), memory_snapshot=100)
                state = StateStore(root / "state.sqlite3")
                try:
                    executor = Executor(state, root / "logs", 128, 4, "unused", max_memory_gb=80,
                                        resource_plan=plan, parallel_ready_tasks=True,
                                        memory_sampler=lambda roots: MemorySnapshot())
                    with patch("metabaw.executor._execute_task", side_effect=worker), \
                            patch("metabaw.executor.available_memory_gib", return_value=100):
                        self.assertEqual(set(executor.run(tasks).values()), {"success"})
                finally:
                    state.close()
                self.assertEqual(len(seen), 4)
                self.assertEqual({t.env["CUDA_VISIBLE_DEVICES"] for t in seen}, {""})

    def test_actual_gpu_assignments_balance_four_tasks_across_two_cards(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = [replace(t, deps=(), wait_for=()) for t in cluster_tasks(root, "metadecoder")]
            budgets, share = plan_gpu_resources((device("0"), device("1")), "100%", 4)
            plan = plan_resources(128, 80, 4, 4, cpu_snapshot=(tuple(range(160)), 160), memory_snapshot=100)
            barrier = threading.Barrier(4, timeout=5)
            seen = []
            def worker(task, *args):
                seen.append(task)
                barrier.wait()
                return RunResult(task.id, 0, "ok")
            state = StateStore(root / "state.sqlite3")
            try:
                executor = Executor(state, root / "logs", 128, 4, "unused", max_memory_gb=80, max_gpus=4,
                                    resource_plan=plan, gpu_device_budgets=budgets, gpu_share_bytes=share,
                                    parallel_ready_tasks=True, memory_sampler=lambda roots: MemorySnapshot(),
                                    gpu_sampler=lambda: ({"0": ("GPU-0", 40 * GIB), "1": ("GPU-1", 40 * GIB)}, []))
                with patch("metabaw.executor._execute_task", side_effect=worker), \
                        patch("metabaw.executor.available_memory_gib", return_value=100):
                    self.assertEqual(set(executor.run(tasks).values()), {"success"})
            finally:
                state.close()
            self.assertEqual([t.env["CUDA_VISIBLE_DEVICES"] for t in seen].count("GPU-0"), 2)
            self.assertEqual([t.env["CUDA_VISIBLE_DEVICES"] for t in seen].count("GPU-1"), 2)
            self.assertEqual(sum(int(t.env["METABAW_GPU_BUDGET_BYTES"]) for t in seen), 64 * GIB)

    def test_cross_group_parallelism_preserves_data_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = [Task("02.map.A", "02_mapping", ("true",), root, sample="A"),
                     Task("03.bin.metadecoder.A", "03_binning", ("true",), root,
                          sample="A", deps=("02.map.A",)),
                     Task("03.bin.comebin.B", "03_binning", ("true",), root, sample="B")]
            completed = set()
            lock = threading.Lock()
            def worker(task, *args):
                with lock:
                    if task.deps:
                        self.assertTrue(set(task.deps).issubset(completed))
                    completed.add(task.id)
                return RunResult(task.id, 0, "ok")
            state = StateStore(root / "state.sqlite3")
            try:
                executor = Executor(state, root / "logs", 8, 2, "unused", parallel_ready_tasks=True)
                with patch("metabaw.executor._execute_task", side_effect=worker):
                    self.assertEqual(set(executor.run(tasks).values()), {"success"})
            finally:
                state.close()

    def test_ready_tasks_in_different_progress_groups_are_not_serialized(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tasks = [Task("03.bin.comebin.S0", "03_binning", ("true",), root, sample="S0", cpus=4),
                     Task("03.bin.metadecoder.S1", "03_binning", ("true",), root, sample="S1", cpus=4)]
            barrier = threading.Barrier(2, timeout=5)
            def worker(task, *args):
                barrier.wait()
                return RunResult(task.id, 0, "ok")
            state = StateStore(root / "state.sqlite3")
            try:
                executor = Executor(state, root / "logs", 8, 2, "unused", parallel_ready_tasks=True)
                with patch("metabaw.executor._execute_task", side_effect=worker):
                    self.assertEqual(set(executor.run(tasks).values()), {"success"})
            finally:
                state.close()

    def test_metadecoder_gpu_cluster_and_cpu_preparations_have_correct_reservations(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for gpu in (True, False):
                task = cluster_tasks(root, "metadecoder", gpu)[0]
                self.assertEqual(task.gpus, int(gpu))
                self.assertEqual("--disable_gpu" in task.display_command(), not gpu)
                self.assertEqual(task.env["CUDA_VISIBLE_DEVICES"] == "", not gpu)
                self.assertEqual(task.cpus, 32)

    def test_metadecoder_gpu_validation_uses_cupy_not_torch(self):
        args = Namespace(tools=["metadecoder"], max_gpu_memory="4G", resource_safety=True,
                         concurrent_sample_limit=4, task=4, gpu=None, batch_size=1024, parameter_warnings=[])
        status = CupyRuntimeStatus("MetaDecoder", "3.11", "13.0", "12000", None, 1, ("GPU",), True)
        with patch("metabaw.cli.host_cuda_status", return_value=HostCudaStatus("nvidia-smi", (device(),), "12")), \
                patch("metabaw.cli._metadecoder_cuda_status", return_value=status) as cupy, \
                patch("metabaw.cli._main_cuda_status") as torch, patch.dict(os.environ, {}, clear=True), \
                redirect_stdout(io.StringIO()):
            self.assertTrue(_enable_bin_cuda(args, {}))
        cupy.assert_called_once()
        torch.assert_not_called()
        self.assertEqual(args.gpu_binners, ("metadecoder",))
        self.assertEqual(args.cuda_task_slots, 4)
        self.assertEqual(args.gpu_share_bytes, GIB)

    def test_cupy_failure_stops_before_binning_instead_of_silently_using_cpu(self):
        args = Namespace(tools=["metadecoder"], max_gpu_memory="8G", resource_safety=True,
                         concurrent_sample_limit=4, task=4, gpu=None, batch_size=1024, parameter_warnings=[])
        status = CupyRuntimeStatus("MetaDecoder", "3.11", None, None, None, 0, (), False, "No module named cupy")
        stderr = io.StringIO()
        with patch("metabaw.cli.host_cuda_status", return_value=HostCudaStatus("nvidia-smi", (device(),), "12")), \
                patch("metabaw.cli._metadecoder_cuda_status", return_value=status), patch.dict(os.environ, {}, clear=True), \
                patch("metabaw.cli.install_main_cupy_runtime") as install, \
                redirect_stdout(io.StringIO()), redirect_stderr(stderr), \
                self.assertRaisesRegex(RuntimeError, "No module named cupy.*metabaw check --scope bin --require-cuda"):
            _enable_bin_cuda(args, {})
        install.assert_not_called()


class GpuAccountingTests(unittest.TestCase):
    def budget(self):
        return (GpuBudget("0", "GPU-0", 48 * GIB, 40 * GIB, 32 * GIB),)

    def test_descendants_are_mapped_to_tasks_once(self):
        sample = MemorySnapshot((snapshot(100), snapshot(101, 100), snapshot(102, 101), snapshot(900)))
        self.assertEqual(process_tasks(sample, {100: "A"}), {100: "A", 101: "A", 102: "A"})

    def test_parent_without_address_space_does_not_hide_gpu_workers(self):
        parent = snapshot(101, 100)
        parent.update(state="no_address_space", pss_bytes=None, rss_bytes=None)
        sample = MemorySnapshot((snapshot(100), parent, snapshot(102, 101)))
        self.assertEqual(process_tasks(sample, {100: "A"}), {100: "A", 102: "A"})

    def test_per_task_gpu_memory_is_sum_of_process_tree_and_confirmed(self):
        readings = iter([({"0": ("GPU-0", 30 * GIB)}, [(100, "GPU-0", 5 * GIB), (101, "GPU-0", 4 * GIB)]),
                         ({"0": ("GPU-0", 29 * GIB)}, [(100, "GPU-0", 6 * GIB), (101, "GPU-0", 4 * GIB)])])
        monitor = GpuMonitor(self.budget(), 8 * GIB, sampler=lambda: next(readings))
        self.assertIsNone(monitor.sample({100: "A", 101: "A"}, {"A": "0"}, now=0))
        self.assertIn("Task A", monitor.sample({100: "A", 101: "A"}, {"A": "0"}, now=2))

    def test_external_jobs_reduce_admission_ceiling_without_double_counting_owned_vram(self):
        monitor = GpuMonitor(self.budget(), 8 * GIB, sampler=lambda: (
            {"0": ("GPU-0", 12 * GIB)}, [(100, "GPU-0", 8 * GIB), (900, "GPU-0", 20 * GIB)]))
        self.assertIsNone(monitor.sample({100: "A"}, {"A": "0"}, now=0))
        self.assertEqual(monitor.current_limits["0"], 16 * GIB)
        pool = GpuReservations(self.budget(), 8 * GIB)
        self.assertIsNotNone(pool.acquire("A", monitor.current_limits))
        self.assertIsNotNone(pool.acquire("B", monitor.current_limits))
        self.assertIsNone(pool.acquire("C", monitor.current_limits))

    def test_idle_gpu_share_is_reduced_equally_instead_of_waiting_forever(self):
        pool = GpuReservations(self.budget(), 8 * GIB)
        self.assertTrue(pool.refresh_idle_share(4, {"0": 16 * GIB}))
        self.assertEqual(pool.share_bytes, 4 * GIB)
        self.assertTrue(all(pool.acquire(str(i), {"0": 16 * GIB}) for i in range(4)))
        self.assertFalse(pool.refresh_idle_share(4, {"0": 8 * GIB}))
        self.assertEqual(pool.share_bytes, 4 * GIB)
        for i in range(4):
            pool.release(str(i))
        self.assertTrue(pool.refresh_idle_share(4, {"0": 0}))
        self.assertEqual(pool.share_bytes, 0)
        self.assertIsNone(pool.acquire("none", {"0": 0}))
        self.assertTrue(pool.refresh_idle_share(4, {"0": 40 * GIB}))
        self.assertEqual(pool.share_bytes, 8 * GIB)

    def test_unreserved_or_wrong_device_cuda_is_rejected(self):
        monitor = GpuMonitor(self.budget(), 8 * GIB, sampler=lambda: (
            {"0": ("GPU-0", 40 * GIB), "1": ("GPU-1", 40 * GIB)}, [(100, "GPU-1", GIB)]))
        with self.assertRaisesRegex(ValueError, "instead of assigned"):
            monitor.sample({100: "A"}, {"A": "0"}, now=0)
        with self.assertRaisesRegex(ValueError, "Unreserved"):
            monitor.sample({100: "A"}, {}, now=2)

    def test_task_ram_share_is_guarded_even_below_workflow_limit(self):
        sample = MemorySnapshot((snapshot(100, pss=12 * GIB), snapshot(101, 100, pss=10 * GIB)))
        budget = _WorkflowMemoryBudget(80, sampler=lambda roots: sample, available_memory_provider=lambda: 100)
        budget.register("A", 100)
        budget.set_task_limit("A", 20)
        budget.sample()
        self.assertFalse(budget.event.is_set())
        budget.sample()
        self.assertEqual(budget.reason, "task_memory_share")
        self.assertIn("task=A", budget.abort_detail())

    def test_task_rss_above_share_is_unverifiable_not_confirmed_pss_breach(self):
        sample = MemorySnapshot((snapshot(100, pss=None, rss=22 * GIB),))
        budget = _WorkflowMemoryBudget(80, sampler=lambda roots: sample)
        budget.register("A", 100)
        budget.set_task_limit("A", 20)
        budget.sample()
        budget.sample()
        self.assertEqual(budget.reason, "measurement_unavailable")
        self.assertIn("NOT a confirmed", budget.abort_detail())

    def test_gpu_measurement_failure_stops_before_tasks_can_bypass_accounting(self):
        pool = GpuReservations(self.budget(), 8 * GIB)
        monitor = GpuMonitor(self.budget(), 8 * GIB, sampler=lambda: (_ for _ in ()).throw(ValueError("N/A VRAM")))
        budget = _WorkflowMemoryBudget(80, gpu_monitor=monitor, gpu_reservations=pool)
        budget.sample()
        self.assertEqual(budget.reason, "gpu_measurement_unavailable")
        self.assertIn("N/A VRAM", budget.abort_detail())

    def test_ram_ceiling_updates_for_external_use_but_never_exceeds_initial_request(self):
        sample = MemorySnapshot((snapshot(100, pss=10 * GIB),))
        available = [30]
        budget = _WorkflowMemoryBudget(80, sampler=lambda roots: sample,
                                       available_memory_provider=lambda: available[0])
        budget.register("A", 100)
        budget.sample()
        self.assertEqual(budget.limit_gb, 32)
        available[0] = 200
        budget.sample()
        self.assertEqual(budget.limit_gb, 80)

    def test_exhausted_current_ram_stops_before_starting_new_tasks(self):
        budget = _WorkflowMemoryBudget(80, available_memory_provider=lambda: 0)
        budget.sample()
        self.assertEqual(budget.reason, "ram_unavailable")
        self.assertIn("headroom is exhausted", budget.abort_detail())

    def test_shared_gpu_total_breach_is_detected_even_when_individual_shares_fit(self):
        monitor = GpuMonitor(self.budget(), 8 * GIB, sampler=lambda: (
            {"0": ("GPU-0", 0)}, [(100, "GPU-0", 8 * GIB), (101, "GPU-0", 8 * GIB)]))
        for now in (0, 2):
            breach = monitor.sample({100: "A", 101: "B"}, {"A": "0", "B": "0"}, now=now)
        self.assertIn("GPU 0", breach)

    def test_gpu_process_monitor_breach_triggers_workflow_stop(self):
        pool = GpuReservations(self.budget(), 8 * GIB)
        pool.acquire("A")
        monitor = GpuMonitor(self.budget(), 8 * GIB, sampler=lambda: (
            {"0": ("GPU-0", 30 * GIB)}, [(100, "GPU-0", 9 * GIB)]))
        sample = MemorySnapshot((snapshot(100),))
        budget = _WorkflowMemoryBudget(80, sampler=lambda roots: sample,
                                       gpu_monitor=monitor, gpu_reservations=pool)
        budget.register("A", 100)
        with patch("metabaw.gpu_resources.time.monotonic", side_effect=[0, 2]):
            budget.sample()
            self.assertTrue(budget.awaiting_confirmation)
            budget.sample()
        self.assertEqual(budget.reason, "gpu_memory")
        self.assertIn("Task A", budget.abort_detail())


class NativeAllocatorHookTests(unittest.TestCase):
    def test_runtime_resource_shares_do_not_invalidate_completed_analysis(self):
        task = Task("03.bin.metadecoder.A", "03", "metadecoder cluster --threads 32", Path("."),
                    env={"CUDA_VISIBLE_DEVICES": "0", "METABAW_MAX_GPU_MEMORY": "8G"})
        adjusted = replace(task, cpu_affinity=(2, 3), memory_limit_gb=20,
                           env={**task.env, "METABAW_MAX_GPU_MEMORY": "4G"})
        self.assertNotEqual(task.fingerprint([]), adjusted.fingerprint([]))
        self.assertEqual(task.resume_fingerprint([]), adjusted.resume_fingerprint([]))
        cpu_mode = replace(adjusted, command="metadecoder cluster --threads 32 --disable_gpu",
                           env={"CUDA_VISIBLE_DEVICES": ""})
        self.assertNotEqual(task.resume_fingerprint([]), cpu_mode.resume_fingerprint([]))

    def test_cupy_gpu_oom_is_not_misreported_as_cpu_memory_exhaustion(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "task.log"
            log.write_text("File cupy/cuda/memory.py\ncupy.cuda.memory.OutOfMemoryError: Out of memory allocating bytes\n")
            self.assertFalse(_memory_limit_exceeded(log, 1))
            self.assertIn("GPU allocation failed", _failure_diagnosis(log))
            log.write_text("File cupy/cuda/memory.py\nMemoryError: numpy cannot allocate memory\n")
            self.assertTrue(_memory_limit_exceeded(log, 1))
    def test_torch_initialization_is_lazy_and_native_override_cannot_raise_cap(self):
        calls = []
        cuda = SimpleNamespace(_lazy_init=lambda: calls.append("init"),
                               get_device_properties=lambda i: SimpleNamespace(total_memory=40 * GIB),
                               set_per_process_memory_fraction=lambda fraction, device=0: calls.append((fraction, device)),
                               memory=SimpleNamespace())
        install_torch_limit(SimpleNamespace(cuda=cuda), 8 * GIB)
        self.assertEqual(calls, [])
        cuda._lazy_init()
        self.assertIn((.2, 0), calls)
        cuda.set_per_process_memory_fraction(1.0)
        cuda.memory.set_per_process_memory_fraction(1.0)
        self.assertEqual([value for value in calls if isinstance(value, tuple)], [(.2, 0)] * 3)

    def test_cupy_default_pool_is_capped_on_assigned_visible_device(self):
        calls = []
        class Device:
            def __init__(self, i):
                calls.append(i)
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass
        cupy = SimpleNamespace(cuda=SimpleNamespace(Device=Device),
                               get_default_memory_pool=lambda: SimpleNamespace(set_limit=lambda **kw: calls.append(kw)))
        install_cupy_limit(cupy, 8 * GIB)
        self.assertEqual(calls, [0, {"size": 8 * GIB}])

    def test_missing_allocator_api_is_not_silently_accepted(self):
        with self.assertRaisesRegex(RuntimeError, "lacks"):
            install_torch_limit(SimpleNamespace(cuda=SimpleNamespace()), GIB)
        with self.assertRaisesRegex(RuntimeError, "lacks"):
            install_cupy_limit(SimpleNamespace(get_default_memory_pool=lambda: SimpleNamespace()), GIB)

    def test_failed_cupy_pool_cap_cannot_be_swallowed_as_cpu_fallback(self):
        from metabaw.gpu_limits import LimitedLoader
        cupy = SimpleNamespace(__name__="cupy", get_default_memory_pool=lambda: SimpleNamespace())
        loader = SimpleNamespace(exec_module=lambda module: None)
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as failure:
            LimitedLoader(loader, GIB).exec_module(cupy)
        self.assertEqual(failure.exception.code, 78)

    def test_sitecustomize_is_inherited_by_python_children_and_preserves_existing_setup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hook = root / "hook"
            modules = root / "modules"
            hook.mkdir()
            modules.mkdir()
            shutil.copyfile(Path(__file__).parents[1] / "src/metabaw/gpu_limits.py", hook / "sitecustomize.py")
            (modules / "sitecustomize.py").write_text("import os\nos.environ['EXISTING_SETUP'] = 'yes'\n")
            # Minimal fake CUDA library: exercise actual Python startup/import
            # and child inheritance without requiring a GPU on the test host.
            (modules / "cupy.py").write_text(
                "from contextlib import nullcontext\n"
                "from types import SimpleNamespace\n"
                "limit = 0\n"
                "def set_limit(size):\n    global limit\n    limit = size\n"
                "cuda = SimpleNamespace(Device=lambda i: nullcontext())\n"
                "def get_default_memory_pool():\n    return SimpleNamespace(set_limit=set_limit)\n")
            env = {**os.environ, "PYTHONPATH": os.pathsep.join((str(hook), str(modules))),
                   "METABAW_GPU_BUDGET_BYTES": str(8 * GIB), "METABAW_GPU_ALLOCATOR_BYTES": str(7 * GIB),
                   "PYTHONDONTWRITEBYTECODE": "1"}
            code = ("import cupy, os; print('RESULT=' + str(cupy.limit) + ',' + os.environ['EXISTING_SETUP']); "
                    "import subprocess, sys; subprocess.run([sys.executable, '-B', '-c', "
                    "\"import cupy, os; print('CHILD=' + str(cupy.limit) + ',' + os.environ['EXISTING_SETUP'])\"], check=True)")
            result = subprocess.run([sys.executable, "-B", "-c", code], env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(f"RESULT={7 * GIB},yes", result.stdout)
            self.assertIn(f"CHILD={7 * GIB},yes", result.stdout)

    def test_cupy_probe_checks_all_visible_devices_and_returns_own_version(self):
        payload = {"python": "3.11", "cupy": "13.0", "cuda": "12000", "visible": "0,1", "count": 2,
                   "devices": ["A", "B"], "allocation": True, "error": None}
        process = SimpleNamespace(returncode=0, stdout="METABAW_CUPY_STATUS=" + json.dumps(payload), stderr="")
        with patch("metabaw.dependencies.subprocess.run", return_value=process) as run:
            status = cupy_runtime_status("MetaDecoder", ("python",))
        self.assertTrue(status.available)
        self.assertEqual(status.cupy_version, "13.0")
        self.assertIn("for i in range(r['count'])", run.call_args.args[0][-1])

    def test_execute_task_injects_and_cleans_gpu_hook_without_changing_command(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = Task("gpu", "03", (sys.executable, "-B", "-c", "print('ok')"), root,
                        env={"METABAW_GPU_BUDGET_BYTES": str(GIB), "METABAW_GPU_ALLOCATOR_BYTES": str(GIB // 2)})
            result = _execute_task(task, root / "task.log", "unused", 80, root / "temp")
            self.assertEqual(result.return_code, 0)
            self.assertEqual(list((root / "temp").iterdir()), [])
            self.assertIn("gpu_python_limit_hook", (root / "task.log").read_text())


if __name__ == "__main__":
    unittest.main()
