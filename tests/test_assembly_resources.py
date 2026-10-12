"""Real assembler task plans with synthetic workers; no native MEGAHIT/Flye."""
from dataclasses import replace
import threading
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from metabaw.discovery import Analysis, ReadSample
from metabaw.direct import DirectBinBuilder
from metabaw.executor import Executor, RunResult, _execute_task
from metabaw.memory import MemorySnapshot
from metabaw.resources import GIB, plan_resources
from metabaw.resume import command_thread_counts, resume_command
from metabaw.state import StateStore
from test_parallel_resources import options
from test_resume_resources import _seed


def assembly_tasks(root, read_type, *, grouped=False, capacity=160, count=5):
    plan = plan_resources(128, 80, 5, count,
                          cpu_snapshot=(tuple(range(capacity)), capacity), memory_snapshot=100)
    opts = replace(options(root, "metabat2", gpu=False), read_type=read_type,
                   threads=plan.threads // plan.slots, total_threads=plan.threads)
    builder = DirectBinBuilder([], opts, root)
    # Multi-sample preparation creates assemblies while another sample label
    # is active. Assemblies must retain their independent analysis identities.
    builder._active_sample = "semibin_shared_preparation"
    tasks = []
    for index in range(count):
        name = f"G{index}" if grouped else f"S{index}"
        samples = []
        for number in range(2 if grouped else 1):
            sample = f"{name}_{number}"
            read1 = root / f"{sample}.R1.fq"
            read1.write_text("@read\nACGT\n+\nIIII\n", encoding="utf-8")
            read2 = root / f"{sample}.R2.fq" if read_type == "short" else None
            if read2 is not None:
                read2.write_text(read1.read_text(encoding="utf-8"), encoding="utf-8")
            samples.append(ReadSample(sample, read1, read2))
        output = root / "assemblies" / name / f"{name}.fa"
        analysis = Analysis(name, tuple(samples), (output,), grouped,
                            assembly_samples=tuple(samples), plan_id=name if grouped else None)
        tasks.append(builder._assembly_task(analysis, output))
    return plan, tasks


@pytest.mark.parametrize("read_type", ["short", "long"])
@pytest.mark.parametrize("grouped", [False, True])
@pytest.mark.parametrize("capacity", [64, 128, 160])
def test_five_assemblies_start_together_with_shared_cpu_and_ram(tmp_path, read_type, grouped, capacity):
    plan, tasks = assembly_tasks(tmp_path, read_type, grouped=grouped, capacity=capacity)
    expected_threads = 12 if capacity == 64 else 25
    assert {task.cpus for task in tasks} == {expected_threads}
    assert len({task.sample for task in tasks}) == 5
    assert all(command_thread_counts(task.display_command()) == (expected_threads,) for task in tasks)
    assert all(task.gpus == 0 for task in tasks)
    barrier = threading.Barrier(5, timeout=5)
    seen = []
    lock = threading.Lock()

    def worker(task, *_args):
        with lock:
            seen.append(task)
        barrier.wait()
        task.outputs[0].parent.mkdir(parents=True, exist_ok=True)
        task.outputs[0].write_text(">contig\nACGT\n", encoding="utf-8")
        return RunResult(task.id, 0, "successful synthetic assembly worker")

    state = StateStore(tmp_path / "state.sqlite3")
    try:
        executor = Executor(state, tmp_path / "logs", plan.threads, 5, "unused",
                            max_memory_gb=plan.memory_gib, resource_plan=plan,
                            parallel_ready_tasks=True, memory_sampler=lambda roots: MemorySnapshot())
        with patch("metabaw.executor._execute_task", side_effect=worker), \
                patch("metabaw.executor.available_memory_gib", return_value=100):
            result = executor.run(tasks)
    finally:
        state.close()
    assert set(result.values()) == {"success"}
    assert len(seen) == 5
    assert {task.memory_limit_gb for task in seen} == {16}
    assert {len(task.cpu_affinity) for task in seen} == {expected_threads}
    assigned = [cpu for task in seen for cpu in task.cpu_affinity]
    assert len(assigned) == len(set(assigned)) <= plan.threads
    assert sum(task.memory_limit_gb for task in seen) == plan.memory_gib


@pytest.mark.parametrize("read_type", ["short", "long"])
def test_one_assembly_slot_still_uses_the_full_effective_budget(tmp_path, read_type):
    plan, tasks = assembly_tasks(tmp_path, read_type, grouped=True, count=1)
    assert plan.slots == 1
    assert tasks[0].cpus == plan.threads == 128
    assert len(tasks[0].inputs) == (4 if read_type == "short" else 2)


def test_megahit_native_memory_uses_this_sample_share_instead_of_host_ram(tmp_path):
    _, tasks = assembly_tasks(tmp_path, "short")
    task = replace(tasks[0], memory_limit_gb=16)
    task.outputs[0].parent.mkdir(parents=True)
    task.outputs[0].write_text(">contig\nACGT\n", encoding="utf-8")
    with patch("metabaw.executor.subprocess.Popen", return_value=SimpleNamespace(pid=123)) as launch, \
            patch("metabaw.executor._wait_for_process", return_value=(0, False)):
        assert _execute_task(task, tmp_path / "assembly.log", "unused", 80, None).return_code == 0
    environment = launch.call_args.kwargs["env"]
    assert environment["METABAW_MAX_MEMORY_GB"] == "16"
    assert int(environment["METABAW_MEGAHIT_MEMORY_BYTES"]) == int(16 * GIB * 0.9)
    assert "megahit_graph_memory_bytes" in (tmp_path / "assembly.log").read_text(encoding="utf-8")


@pytest.mark.parametrize("read_type", ["short", "long"])
@pytest.mark.parametrize("legacy", [False, True])
def test_completed_assembly_is_reused_after_cpu_and_memory_scheduling_fix(tmp_path, read_type, legacy):
    _, tasks = assembly_tasks(tmp_path, read_type)
    new = tasks[0]
    old_command = new.display_command().replace("--num-cpu-threads 25", "--num-cpu-threads 128")
    old_command = old_command.replace("--threads 25", "--threads 128")
    old_command = old_command.replace('--memory "${METABAW_MEGAHIT_MEMORY_BYTES:-0.9}" ', "")
    old = replace(new, cpus=128, command=old_command)
    new.outputs[0].parent.mkdir(parents=True)
    new.outputs[0].write_text(">contig\nACGT\n", encoding="utf-8")
    state = StateStore(tmp_path / "state.sqlite3")
    try:
        _seed(state, [old], legacy=legacy)
        with patch("metabaw.executor._execute_task") as execute:
            result = Executor(state, tmp_path / "logs", 128, 5).run([new])
        execute.assert_not_called()
        assert result == {new.id: "skipped"}
    finally:
        state.close()


def test_resume_ignores_only_the_managed_megahit_memory_expression():
    old = "megahit --num-cpu-threads 128 --min-contig-len 1500"
    new = 'megahit --num-cpu-threads 25 --memory "${METABAW_MEGAHIT_MEMORY_BYTES:-0.9}" --min-contig-len 1500'
    assert resume_command(old) == resume_command(new)
    assert resume_command(new) != resume_command(old + " --memory 1000000")
    assert resume_command(new) != resume_command(new.replace("1500", "2000"))
