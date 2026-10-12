"""R dependency repair must fix native ABI mismatches and verify loaded packages."""
import json
import subprocess
from unittest.mock import patch

import pytest

from metabaw.dependencies import RPackageRequirement, install_r_packages


@pytest.fixture
def r_environment(tmp_path):
    executable = tmp_path / "bin" / "Rscript"
    executable.parent.mkdir()
    executable.touch()
    metadata = tmp_path / "conda-meta"
    metadata.mkdir()
    (metadata / "r-base-4.4.3-test.json").write_text(
        json.dumps({"name": "r-base", "version": "4.4.3"}), encoding="utf-8",
    )
    with (
        patch("metabaw.dependencies.sys.prefix", str(tmp_path)),
        patch("metabaw.dependencies.shutil.which", return_value=str(executable)),
        patch("metabaw.dependencies.conda_frontend", return_value="mamba"),
    ):
        yield tmp_path, executable


def completed(code=0, stderr=""):
    return subprocess.CompletedProcess([], code, "", stderr)


def test_conda_repairs_r_runtime_and_rlang_with_dplyr_and_readr(r_environment):
    prefix, executable = r_environment
    with (
        patch("metabaw.dependencies.run_conda_transaction", return_value=completed()) as transaction,
        patch("metabaw.dependencies.subprocess.run", return_value=completed()) as run,
    ):
        install_r_packages([RPackageRequirement("dplyr"), RPackageRequirement("readr")])
    command = transaction.call_args.args[0]
    assert command == [
        "mamba", "install", "--yes", "--prefix", str(prefix.resolve()),
        "--override-channels", "--channel", "conda-forge", "--force-reinstall",
        "r-base=4.4.*", "r-rlang", "r-dplyr", "r-readr",
    ]
    # Only namespace validation runs in R; no CRAN source compilation is used.
    run.assert_called_once()
    assert run.call_args.args[0][0] == str(executable)
    assert "loadNamespace" in run.call_args.args[0][-1]
    assert 'c("dplyr", "readr")' in run.call_args.args[0][-1]


def test_cran_only_package_is_installed_after_conda_dependencies(r_environment):
    operations = []

    def conda(command):
        operations.append("conda")
        return completed()

    def rscript(command, **kwargs):
        operations.append("source" if "install.packages" in command[-1] else "validation")
        if "install.packages" in command[-1]:
            assert 'c("funr")' in command[-1]
            assert "dependencies=NA" in command[-1]
            assert "dependencies=TRUE" not in command[-1]
        return completed()

    with (
        patch("metabaw.dependencies.run_conda_transaction", side_effect=conda),
        patch("metabaw.dependencies.subprocess.run", side_effect=rscript),
    ):
        install_r_packages([RPackageRequirement("dplyr"), RPackageRequirement("funr")])
    assert operations == ["conda", "source", "validation"]


def test_conda_failure_does_not_reuse_the_incompatible_cran_stack(r_environment):
    with (
        patch("metabaw.dependencies.run_conda_transaction", return_value=completed(1)),
        patch("metabaw.dependencies.subprocess.run") as run,
        pytest.raises(RuntimeError, match="Conda R package repair failed"),
    ):
        install_r_packages([RPackageRequirement("readr")])
    run.assert_not_called()


@pytest.mark.parametrize("conda_succeeded", [True, False])
def test_successful_installer_exit_does_not_hide_rlang_load_failure(r_environment, conda_succeeded):
    failure = "readr: unable to load rlang.so: undefined symbol: R_mkClosure"
    with (
        patch("metabaw.dependencies.conda_frontend", return_value="mamba" if conda_succeeded else None),
        patch("metabaw.dependencies.run_conda_transaction", return_value=completed()),
        patch("metabaw.dependencies.subprocess.run", side_effect=(
            [completed(1, failure)] if conda_succeeded else [completed(), completed(1, failure)]
        )),
        pytest.raises(RuntimeError, match="not usable after installation.*R_mkClosure"),
    ):
        install_r_packages([RPackageRequirement("readr")])


@pytest.mark.parametrize("external_r", [True, False])
def test_source_fallback_rebuilds_rlang_and_omits_optional_dependencies(r_environment, external_r):
    prefix, executable = r_environment
    selected_r = str(prefix.parent / "other-r" / "Rscript") if external_r else str(executable)
    with (
        patch("metabaw.dependencies.shutil.which", return_value=selected_r),
        patch("metabaw.dependencies.conda_frontend", return_value="mamba" if external_r else None),
        patch("metabaw.dependencies.run_conda_transaction") as transaction,
        patch("metabaw.dependencies.subprocess.run", return_value=completed()) as run,
    ):
        install_r_packages([RPackageRequirement("dplyr"), RPackageRequirement("readr")])
    transaction.assert_not_called()
    install = run.call_args_list[0].args[0]
    assert install[0] == selected_r
    assert 'c("dplyr", "readr", "rlang")' in install[-1]
    assert "dependencies=NA" in install[-1]
    assert "type='source'" in install[-1]
    assert "dependencies=TRUE" not in install[-1]
    assert "loadNamespace" in run.call_args_list[1].args[0][-1]


def test_bioconductor_keeps_runtime_dependencies_and_is_validated(r_environment):
    with (
        patch("metabaw.dependencies.run_conda_transaction") as transaction,
        patch("metabaw.dependencies.subprocess.run", return_value=completed()) as run,
    ):
        install_r_packages([RPackageRequirement("example", "bioconductor")])
    transaction.assert_not_called()
    expression = run.call_args_list[0].args[0][-1]
    assert "BiocManager::install" in expression
    assert "dependencies=TRUE" not in expression
    assert "ask=FALSE, update=FALSE, dependencies=NA" in expression
    assert 'c("example")' in run.call_args_list[1].args[0][-1]


def test_runtime_version_fallback_preserves_r_minor_version(r_environment):
    prefix, _executable = r_environment
    (prefix / "conda-meta" / "r-base-4.4.3-test.json").unlink()
    with (
        patch("metabaw.dependencies.run_conda_transaction", return_value=completed()) as transaction,
        patch("metabaw.dependencies.subprocess.run", side_effect=[
            subprocess.CompletedProcess([], 0, "4.3.3", ""), completed(),
        ]),
    ):
        install_r_packages([RPackageRequirement("digest")])
    assert "r-base=4.3.*" in transaction.call_args.args[0]
    assert "r-digest" in transaction.call_args.args[0]
    assert "r-rlang" not in transaction.call_args.args[0]
