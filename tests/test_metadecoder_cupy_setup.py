"""Main-environment routing and mocked CuPy installation; no native GPU jobs."""
from __future__ import annotations

from contextlib import nullcontext, redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from metabaw.cli import _offer_cupy_repair
from metabaw.dependencies import (
    SOFTWARE, install_main_cupy_runtime, main_cupy_install_command, missing_software,
    cupy_runtime_status,
)
from metabaw.metadecoder_coverage import metadecoder_executable
from metabaw.direct import DirectBinBuilder
from metabaw.discovery import Analysis, ReadSample
from test_metadecoder_cuda_probe import cupy_status, host_status
from test_resume_resources import _bin_options


@pytest.mark.parametrize("launcher", ["absolute", "env", "env-s", "trampoline"])
def test_main_launcher_wins_over_base_on_path(tmp_path, launcher):
    prefix = tmp_path / "MetaBAW environment"
    binary = prefix / "bin"
    binary.mkdir(parents=True)
    python = binary / "python3.11"
    executable = binary / "metadecoder"
    direct = f"#!{shlex.quote(str(python))}\n"
    headers = {
        "absolute": direct,
        "env": "#!/usr/bin/env python3.11\n",
        "env-s": "#!/usr/bin/env -S python3.11 -s\n",
        "trampoline": f"#!/bin/sh\n'''exec' {shlex.quote(str(python))} \"$0\" \"$@\"\n",
    }
    executable.write_text(headers[launcher])
    with patch("sys.prefix", str(prefix)), patch("sys.executable", str(python)), \
            patch("shutil.which", side_effect=lambda name: str(python) if name == "python3.11" else "/base/bin/metadecoder"):
        assert metadecoder_executable() == str(executable)
        assert missing_software([SOFTWARE["metadecoder"]]) == []
        sample = ReadSample("S1", tmp_path / "reads.fq", contigs=tmp_path / "contigs.fa")
        tasks = DirectBinBuilder([Analysis("S1", (sample,), (sample.contigs,), False)],
                                 _bin_options(tmp_path, 32, 128), tmp_path).build()
        for name in ("03.seed.metadecoder.S1", "03.bin.metadecoder.S1"):
            command = next(task.display_command() for task in tasks if task.id == name)
            assert shlex.quote(str(executable)) in command
            assert "/base/bin/metadecoder" not in command


@pytest.mark.parametrize("kind", ["missing", "foreign", "unknown"])
def test_base_or_foreign_python_does_not_satisfy_main_requirement(tmp_path, kind):
    prefix = tmp_path / "main"
    (prefix / "bin").mkdir(parents=True)
    if kind != "missing":
        (prefix / "bin" / "metadecoder").write_text(
            "#!/base/bin/python3.12\n" if kind == "foreign" else "#!/bin/bash\n",
        )
    with patch("sys.prefix", str(prefix)), patch("sys.executable", str(prefix / "bin" / "python")), \
            patch("shutil.which", return_value="/base/bin/metadecoder"):
        assert metadecoder_executable() is None
        assert missing_software([SOFTWARE["metadecoder"]]) == [SOFTWARE["metadecoder"]]


@pytest.mark.parametrize("maximum,preferred,expected", [
    ("12.8", "11.8", "11.8"), ("12.8", None, "12.8"),
    ("12.8", "13.0", "12.8"), ("13.2", "12.8", "12.8"),
    ("12.7", None, "12.6"), (None, "11.8", "11.8"),
])
def test_cupy_plan_targets_main_prefix_and_preserves_working_cuda(tmp_path, maximum, preferred, expected):
    (tmp_path / "conda-meta").mkdir()
    with patch("sys.prefix", str(tmp_path)), patch("shutil.which", return_value="/base/bin/conda"), \
            patch("metabaw.dependencies.metadata.distributions", return_value=[]):
        command = main_cupy_install_command(maximum, preferred)
    assert command[command.index("--prefix") + 1] == str(tmp_path.resolve())
    assert "--freeze-installed" in command
    assert f"python={sys.version_info.major}.{sys.version_info.minor}" in command
    assert "cupy>=13,<14" in command
    assert f"cuda-version={expected}" in command


@pytest.mark.parametrize("maximum", [None, "unknown", "11.1"])
def test_cupy_plan_rejects_unknown_or_unsupported_driver(tmp_path, maximum):
    (tmp_path / "conda-meta").mkdir()
    with patch("sys.prefix", str(tmp_path)), patch("shutil.which", return_value="conda"), \
            patch("metabaw.dependencies.metadata.distributions", return_value=[]), \
            pytest.raises(RuntimeError, match="supported CuPy CUDA"):
        main_cupy_install_command(maximum)


def test_cupy_plan_preserves_cuda_metadata_from_main_not_base(tmp_path):
    records = tmp_path / "conda-meta"
    records.mkdir()
    (records / "cuda-version-12.6-0.json").write_text(json.dumps({"name": "cuda-version", "version": "12.6"}))
    with patch("sys.prefix", str(tmp_path)), patch("sys.executable", "/base/bin/python"), \
            patch("shutil.which", return_value="conda"), \
            patch("metabaw.dependencies.metadata.distributions", return_value=[]):
        command = main_cupy_install_command("12.8", "11.8")
    assert "cuda-version=12.6" in command


@pytest.mark.parametrize("name", ["cupy", "cupy-core", "cupy_cuda12x"])
def test_cupy_plan_never_overlays_existing_distributions(tmp_path, name):
    (tmp_path / "conda-meta").mkdir()
    with patch("sys.prefix", str(tmp_path)), patch("shutil.which", return_value="conda"), \
            patch("metabaw.dependencies.metadata.distributions", return_value=[SimpleNamespace(metadata={"Name": name})]), \
            pytest.raises(RuntimeError, match="already registered"):
        main_cupy_install_command("12.8")


@pytest.mark.parametrize("conda,has_metadata", [(None, True), ("conda", False)])
def test_cupy_plan_rejects_missing_conda_or_non_conda_prefix(tmp_path, conda, has_metadata):
    if has_metadata:
        (tmp_path / "conda-meta").mkdir()
    with patch("sys.prefix", str(tmp_path)), patch("shutil.which", return_value=conda), \
            pytest.raises(RuntimeError, match="valid MetaBAW Conda prefix"):
        main_cupy_install_command("12.8")


@pytest.mark.parametrize("code", [0, 1])
def test_cupy_install_uses_one_approved_transaction_without_fallback(code):
    command = ("conda", "install", "--prefix", "/envs/metabaw", "cupy>=13,<14")
    with patch("metabaw.dependencies.run_conda_transaction", return_value=subprocess.CompletedProcess(command, code)) as run:
        if code:
            with pytest.raises(RuntimeError, match="no destructive or different-environment fallback"):
                install_main_cupy_runtime(command)
        else:
            install_main_cupy_runtime(command)
    run.assert_called_once_with(command, environment_overrides={"CONDA_CHANNEL_PRIORITY": "flexible"})


@pytest.mark.parametrize("approved,available", [(False, False), (True, False), (True, True)])
def test_cupy_repair_requires_approval_and_retests_native_main_runtime(approved, available):
    missing, repaired = cupy_status(False), cupy_status(available)
    plan = ("conda", "install", "--prefix", "/envs/metabaw", "cupy>=13,<14")
    with patch("metabaw.cli.main_cupy_install_command", return_value=plan) as planner, \
            patch("metabaw.cli.confirm_install", return_value=approved), \
            patch("metabaw.cli.install_main_cupy_runtime") as install, \
            patch("metabaw.cli._metadecoder_cuda_status", return_value=repaired) as probe, \
            redirect_stdout(io.StringIO()):
        result = _offer_cupy_repair(missing, host_status(), "11.8")
    planner.assert_called_once_with("12", "11.8")
    assert install.call_count == probe.call_count == int(approved)
    assert result is (repaired if approved else missing)


def test_cupy_repair_failure_is_reported_without_claiming_success():
    missing = cupy_status(False)
    errors = io.StringIO()
    with patch("metabaw.cli.main_cupy_install_command", return_value=("conda",)), \
            patch("metabaw.cli.confirm_install", return_value=True), \
            patch("metabaw.cli.install_main_cupy_runtime", side_effect=RuntimeError("solver failed")), \
            patch("metabaw.cli._metadecoder_cuda_status") as probe, \
            redirect_stdout(io.StringIO()), redirect_stderr(errors):
        assert _offer_cupy_repair(missing, host_status()) is missing
    probe.assert_not_called()
    assert "CUDA REPAIR FAILED" in errors.getvalue()


def test_cupy_repair_does_not_install_for_hidden_devices_or_broken_existing_cupy():
    broken = SimpleNamespace(available=False, cupy_version="13.6", error="no visible devices")
    with patch("metabaw.cli.main_cupy_install_command") as plan, redirect_stdout(io.StringIO()):
        assert _offer_cupy_repair(broken, host_status()) is broken
        assert _offer_cupy_repair(cupy_status(False), SimpleNamespace(available=False)).available is False
    plan.assert_not_called()


@pytest.mark.parametrize("failure", [None, "kernel", "linear algebra"])
def test_generated_cupy_probe_requires_kernels_and_linear_algebra(failure):
    events = []

    class Vector:
        def __init__(self, value):
            self.value = value

        def __add__(self, number):
            events.append("kernel")
            if failure == "kernel":
                raise RuntimeError("synthetic kernel failure")
            return Vector(self.value + number)

        def sum(self):
            return SimpleNamespace(item=lambda: self.value * 2)

    def cholesky(_matrix):
        events.append("linear algebra")
        if failure == "linear algebra":
            raise RuntimeError("synthetic linear algebra failure")
        return SimpleNamespace(get=lambda: None)

    pool = SimpleNamespace(set_limit=lambda size: events.append(size), free_all_blocks=lambda: None)
    fake = SimpleNamespace(
        __version__="13.6", get_default_memory_pool=lambda: pool, ones=lambda size: Vector(1),
        eye=lambda size: None, linalg=SimpleNamespace(cholesky=cholesky),
        cuda=SimpleNamespace(Device=lambda index: nullcontext(), runtime=SimpleNamespace(
            runtimeGetVersion=lambda: 12080, getDeviceCount=lambda: 1,
            deviceSynchronize=lambda: None, getDeviceProperties=lambda index: {"name": "synthetic GPU"},
        )),
    )

    def probe(command, **_kwargs):
        output = io.StringIO()
        with patch.dict(sys.modules, {"cupy": fake}), redirect_stdout(output):
            exec(command[-1], {})
        return subprocess.CompletedProcess(command, 0, output.getvalue(), "")

    with patch("metabaw.dependencies.subprocess.run", side_effect=probe):
        status = cupy_runtime_status("MetaDecoder", ("synthetic-python",))
    assert status.available == (failure is None)
    assert 16 * 1024**2 in events
    assert "kernel" in events
    if failure != "kernel":
        assert "linear algebra" in events
    if failure:
        assert failure in status.error
