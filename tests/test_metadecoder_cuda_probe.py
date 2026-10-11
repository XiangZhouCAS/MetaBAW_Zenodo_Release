"""Synthetic CUDA probes: the native CLI's interpreter must select CuPy."""
from __future__ import annotations

from argparse import Namespace
from contextlib import redirect_stderr, redirect_stdout
import io
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from metabaw.cli import _enable_bin_cuda, _metadecoder_cuda_status, build_parser, command_check
from metabaw.dependencies import CudaDevice, CudaRuntimeStatus, CupyRuntimeStatus, HostCudaStatus


def cupy_status(available=True):
    return CupyRuntimeStatus(
        "MetaDecoder", "3.11", "13.0" if available else None, "12000" if available else None,
        None, 1 if available else 0, ("GPU",) if available else (), available,
        None if available else "ModuleNotFoundError: No module named 'cupy'",
    )


def host_status():
    return HostCudaStatus("nvidia-smi", (
        CudaDevice("0", "GPU", 48 * 1024, "driver", 40 * 1024, "GPU-0"),
    ), "12")


@pytest.mark.parametrize("shebang,launcher", [
    ("#!/native/metadecoder/bin/python3.11\n", ("/native/metadecoder/bin/python3.11",)),
    ("#!/usr/bin/env python3.12\n", ("/native/metadecoder/bin/python3.12",)),
    ("#!/usr/bin/env -S python3.12 -s\n", ("/native/metadecoder/bin/python3.12", "-s")),
    ("#!/bin/sh\n'''exec' '/native/MetaDecoder environment/bin/python3.11' \"$0\" \"$@\"\n",
     ("/native/MetaDecoder environment/bin/python3.11",)),
])
def test_probe_uses_selected_main_console_launcher(tmp_path, shebang, launcher):
    executable = tmp_path / "metadecoder"
    executable.write_text(shebang)

    def which(name):
        if name == "metadecoder":
            return str(executable)
        if name == "python3.12":
            return "/native/metadecoder/bin/python3.12"
        return None

    output = io.StringIO()
    with patch("metabaw.cli.shutil.which", side_effect=which), \
            patch("metabaw.cli.metadecoder_executable", return_value=str(executable)), \
            patch("metabaw.cli.sys.executable", "/different/metabaw/bin/python"), \
            patch("metabaw.cli.cupy_runtime_status", return_value=cupy_status()) as probe, \
            patch("metabaw.cli._main_cuda_status", side_effect=AssertionError("must not use PyTorch")), \
            redirect_stdout(output):
        result = _metadecoder_cuda_status()
    assert result.available
    probe.assert_called_once_with("MetaDecoder", launcher)
    assert str(executable) in output.getvalue() and launcher[0] in output.getvalue()


@pytest.mark.parametrize("kind", ["missing", "unreadable", "unknown"])
def test_unresolved_native_launcher_never_uses_metabaw_python_as_fallback(tmp_path, kind):
    executable = tmp_path / "metadecoder"
    if kind == "unknown":
        executable.write_text("#!/bin/bash\necho unknown wrapper\n")
    with patch("metabaw.cli.shutil.which", return_value=None if kind == "missing" else str(executable)), \
            patch("metabaw.cli.metadecoder_executable", return_value=None if kind == "missing" else str(executable)), \
            patch("metabaw.cli.cupy_runtime_status") as probe:
        result = _metadecoder_cuda_status()
    assert not result.available
    assert "MetaDecoder's own Python runtime" in result.error
    probe.assert_not_called()


def test_main_metadecoder_environment_keeps_gpu_and_four_reserved_slots(tmp_path):
    executable = tmp_path / "metadecoder"
    executable.write_text("#!/envs/metabaw/bin/python\n")
    args = Namespace(tools=["metadecoder"], max_gpu_memory="100%", resource_safety=True,
                     concurrent_sample_limit=4, task=4, gpu=None, batch_size=1024, parameter_warnings=[])
    with patch("metabaw.cli.host_cuda_status", return_value=host_status()), \
            patch("metabaw.cli.metadecoder_executable", return_value=str(executable)), \
            patch("metabaw.cli.shutil.which", return_value=str(executable)), \
            patch("metabaw.cli.sys.executable", "/envs/metabaw/bin/python"), \
            patch("metabaw.cli.cupy_runtime_status", return_value=cupy_status()) as cupy, \
            patch("metabaw.cli._main_cuda_status", side_effect=AssertionError("unrelated environment")), \
            patch.dict(os.environ, {}, clear=True), redirect_stdout(io.StringIO()):
        assert _enable_bin_cuda(args, {})
    cupy.assert_called_once_with("MetaDecoder", ("/envs/metabaw/bin/python",))
    assert args.gpu_binners == ("metadecoder",)
    assert args.cuda_device_count == 1 and args.cuda_task_slots == 4
    assert sum(d.limit_bytes for d in args.gpu_device_budgets) <= 40 * 1024**3 * 0.8
    assert not args.parameter_warnings


@pytest.mark.parametrize("require_cuda", [False, True])
@pytest.mark.parametrize("available", [False, True])
def test_check_bin_validates_cupy_independently_of_healthy_pytorch(tmp_path, require_cuda, available):
    args = build_parser().parse_args([
        "check", "--scope", "bin", *(["--require-cuda"] if require_cuda else []),
    ])
    torch = CudaRuntimeStatus("MetaBAW environment", "3.11", "2.7", "12", None, 1, ("GPU",), True)
    output, errors = io.StringIO(), io.StringIO()
    with patch("metabaw.cli._check_software_interactively", return_value=True), \
            patch("metabaw.cli._check_isolated_tools_interactively", return_value=({}, True)), \
            patch("metabaw.cli.host_cuda_status", return_value=host_status()), \
            patch("metabaw.cli._main_cuda_status", return_value=torch), \
            patch("metabaw.cli._metadecoder_cuda_status", return_value=cupy_status(available)) as probe, \
            patch("metabaw.cli._offer_cupy_repair", return_value=cupy_status(available)) as repair, \
            patch("metabaw.cli._offer_cuda_repair", side_effect=AssertionError("CuPy is not PyTorch")), \
            patch("metabaw.cli.magscot_files", return_value=()), \
            patch("metabaw.cli.save_magscot_directory"), \
            patch("metabaw.cli.configured_database_path", return_value=tmp_path), \
            patch("metabaw.cli.checkm_database_valid", return_value=True), \
            patch("metabaw.cli.diamond_database_valid", return_value=True), \
            patch("metabaw.cli.gtdbtk_database_valid", return_value=True), \
            patch("metabaw.cli.save_database_path"), \
            patch("metabaw.cli.confirm_install", side_effect=AssertionError("unexpected installation")), \
            patch.dict(os.environ, {}, clear=False), \
            redirect_stdout(output), redirect_stderr(errors):
        result = command_check(args)
    probe.assert_called_once()
    assert "MetaDecoder" in output.getvalue()
    assert result == (0 if available else 1)
    assert repair.call_count == int(not available)
    if not available:
        assert "No module named 'cupy'" in output.getvalue()
        assert "has not confirmed MetaDecoder GPU readiness" in errors.getvalue()


def test_annotation_only_check_does_not_require_metadecoder_cupy(tmp_path):
    args = build_parser().parse_args(["check", "--scope", "annotation"])
    with patch("metabaw.cli._check_software_interactively", return_value=True), \
            patch("metabaw.cli.host_cuda_status", return_value=HostCudaStatus(None, (), None, "no GPU")), \
            patch("metabaw.cli._metadecoder_cuda_status") as probe, \
            patch("metabaw.cli.configured_database_path", return_value=tmp_path), \
            patch("metabaw.cli.gtdbtk_database_valid", return_value=True), \
            patch("metabaw.cli.kofam_database_valid", return_value=True), \
            patch("metabaw.cli.dbcan_database_valid", return_value=True), \
            patch("metabaw.cli.hydrogenase_database_valid", return_value=True), \
            patch("metabaw.cli.save_database_path"), redirect_stdout(io.StringIO()):
        assert command_check(args) == 0
    probe.assert_not_called()
