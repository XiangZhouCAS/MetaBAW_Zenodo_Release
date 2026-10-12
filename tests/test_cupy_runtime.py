"""Private CUDA library routing and mocked installs; no native Linux/GPU run."""
from contextlib import redirect_stdout, redirect_stderr
import io
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

import pytest

from metabaw.cli import _offer_cupy_repair
from metabaw.cupy_runtime import CupyInstallPlan, cupy_runtime_environment, save_cupy_runtime
from metabaw.dependencies import cupy_runtime_status, install_main_cupy_runtime
from metabaw.direct import DirectBinBuilder
from metabaw.discovery import Analysis, ReadSample
from test_metadecoder_cuda_probe import cupy_status, host_status
from test_resume_resources import _bin_options


def runtime_fixture(prefix):
    runtime = prefix / "share" / "metabaw" / "cuda" / "12.9"
    target = runtime / "targets" / "x86_64-linux"
    (target / "include").mkdir(parents=True)
    (target / "include" / "cuda_runtime.h").write_text("fixture")
    (target / "lib" / "stubs").mkdir(parents=True)
    (runtime / "bin").mkdir()
    return CupyInstallPlan("12.9", runtime, (("conda",), ("main-python",)))


def test_routing_is_only_applied_to_metadecoder_and_does_not_change_parent(tmp_path):
    plan = runtime_fixture(tmp_path)
    sample = ReadSample("S1", tmp_path / "reads.fq", contigs=tmp_path / "contigs.fa")
    with patch("sys.prefix", str(tmp_path)), patch.dict(os.environ, {"LD_LIBRARY_PATH": "/old/cuda11"}):
        save_cupy_runtime(plan)
        before = dict(os.environ)
        environment = cupy_runtime_environment()
        assert environment["METABAW_CUPY_CUDA_VERSION"] == "12.9"
        assert environment["LD_LIBRARY_PATH"].split(os.pathsep)[0].endswith("lib")
        assert "stubs" not in environment["LD_LIBRARY_PATH"]
        tasks = DirectBinBuilder([Analysis("S1", (sample,), (sample.contigs,), False)],
                                 _bin_options(tmp_path, 32, 128), tmp_path).build()
        decoder = [task for task in tasks if task.id.startswith(
            ("03.coverage.metadecoder.", "03.seed.metadecoder.", "03.bin.metadecoder."))]
        assert len(decoder) == 3
        for task in decoder:
            assert task.env["CUDA_PATH"] == environment["CUDA_PATH"]
        for task in tasks:
            if task not in decoder:
                assert "METABAW_CUPY_CUDA_VERSION" not in task.env
        assert dict(os.environ) == before


def test_main_prefix_config_does_not_leak_to_another_environment(tmp_path):
    plan = runtime_fixture(tmp_path / "main")
    with patch("sys.prefix", str(tmp_path / "main")):
        save_cupy_runtime(plan)
        assert cupy_runtime_environment()
    with patch("sys.prefix", str(tmp_path / "base")):
        assert cupy_runtime_environment() == {}


def test_missing_runtime_headers_are_not_saved(tmp_path):
    plan = CupyInstallPlan("12.9", tmp_path / "incomplete", ())
    with patch("sys.prefix", str(tmp_path)), pytest.raises(RuntimeError, match="headers"):
        save_cupy_runtime(plan)
    assert not (tmp_path / "share" / "metabaw" / "cuda" / "runtime.json").exists()


def test_failed_cupy_wheel_does_not_publish_routing(tmp_path):
    plan = runtime_fixture(tmp_path)
    with patch("metabaw.dependencies.run_conda_transaction", return_value=subprocess.CompletedProcess([], 0)), \
            patch("metabaw.dependencies.subprocess.run", return_value=subprocess.CompletedProcess([], 1)), \
            patch("metabaw.dependencies.save_cupy_runtime") as save, \
            pytest.raises(RuntimeError, match="wheel installation failed"):
        install_main_cupy_runtime(plan)
    save.assert_not_called()


def test_repair_does_not_accept_a_different_runtime_version(tmp_path):
    missing = cupy_status(False)
    plan = CupyInstallPlan("12.9", tmp_path, (("conda",), ("python",)))
    with patch("metabaw.cli.main_cupy_install_command", return_value=plan), \
            patch("metabaw.cli.confirm_install", return_value=True), \
            patch("metabaw.cli.install_main_cupy_runtime"), \
            patch("metabaw.cli._metadecoder_cuda_status", return_value=cupy_status(True)), \
            redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        assert _offer_cupy_repair(missing, host_status()) is missing


def test_probe_receives_the_same_runtime_as_native_tasks(tmp_path):
    plan = runtime_fixture(tmp_path)
    result = subprocess.CompletedProcess([], 0, 'METABAW_CUPY_STATUS={"cuda":"12090","count":1,"allocation":true}', "")
    with patch("sys.prefix", str(tmp_path)):
        save_cupy_runtime(plan)
        with patch("metabaw.dependencies.subprocess.run", return_value=result) as run:
            assert cupy_runtime_status("MetaDecoder", ("main-python",)).available
        assert run.call_args.kwargs["env"]["METABAW_CUPY_CUDA_VERSION"] == "12.9"
        assert run.call_args.kwargs["env"]["CUDA_PATH"].endswith("x86_64-linux")
