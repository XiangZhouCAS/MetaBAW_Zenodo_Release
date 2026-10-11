from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
import subprocess
import tempfile
import tomllib
import unittest
from unittest.mock import call, patch
from pathlib import Path

from metabaw.dependencies import (
    CudaDevice,
    CudaRuntimeStatus,
    HostCudaStatus,
    ISOLATED_TOOLS,
    IsolatedEnvironmentStatus,
    SoftwareRequirement,
    SoftwareRuntimeIssue,
    bin_requirements,
    configured_isolated_environment,
    install_isolated_environment,
    install_isolated_cuda_runtime,
    normalize_isolated_environment,
    install_main_cuda_runtime,
    install_software,
    persist_conda_environment_variable,
    print_isolated_status,
    run_conda_transaction,
    software_runtime_issues,
)
from metabaw.cli import (
    _check_isolated_tools_interactively,
    _check_software_interactively,
    _offer_cuda_repair,
    build_parser,
    command_check,
)


class _SuccessfulProcess:
    def __init__(self) -> None:
        self.stdout = io.BytesIO(b"")

    def wait(self) -> int:
        return 0


class CondaInstallerTests(unittest.TestCase):
    def test_gtdbtk_1_is_reported_incompatible_with_current_database(self) -> None:
        requirement = SoftwareRequirement(
            "gtdbtk",
            "gtdbtk=2.7.2",
            "GTDB-Tk taxonomy",
        )
        probe = subprocess.CompletedProcess(
            ["/env/bin/gtdbtk", "--version"],
            0,
            "GTDB-Tk v1.0.2\n",
            "",
        )
        with (
            patch("metabaw.dependencies.shutil.which", return_value="/env/bin/gtdbtk"),
            patch("metabaw.dependencies.subprocess.run", return_value=probe),
        ):
            issues = software_runtime_issues([requirement])
        self.assertEqual(len(issues), 1)
        self.assertIn("requires version 2.7.2", issues[0].detail)
        self.assertEqual(issues[0].repair_packages, ("gtdbtk=2.7.2",))

    def test_gtdbtk_272_runtime_probe_is_healthy(self) -> None:
        requirement = SoftwareRequirement(
            "gtdbtk",
            "gtdbtk=2.7.2",
            "GTDB-Tk taxonomy",
        )
        probe = subprocess.CompletedProcess(
            ["/env/bin/gtdbtk", "--version"],
            0,
            "GTDB-Tk v2.7.2\n",
            "",
        )
        with (
            patch("metabaw.dependencies.shutil.which", return_value="/env/bin/gtdbtk"),
            patch("metabaw.dependencies.subprocess.run", return_value=probe),
        ):
            issues = software_runtime_issues([requirement])
        self.assertEqual(issues, [])

    def test_gtdbtk_uses_owning_conda_prefix_version_when_cli_text_is_ambiguous(self) -> None:
        requirement = SoftwareRequirement(
            "gtdbtk",
            "gtdbtk=2.7.2",
            "GTDB-Tk taxonomy",
        )
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary)
            executable = prefix / "bin" / "gtdbtk"
            executable.parent.mkdir(parents=True)
            executable.write_text("#!/bin/sh\n", encoding="utf-8")
            metadata_directory = prefix / "conda-meta"
            metadata_directory.mkdir()
            (metadata_directory / "gtdbtk-2.7.2-test.json").write_text(
                '{"name": "gtdbtk", "version": "2.7.2"}',
                encoding="utf-8",
            )
            probe = subprocess.CompletedProcess(
                [str(executable), "--version"],
                0,
                "GTDB-Tk installed\n",
                "",
            )
            with (
                patch("metabaw.dependencies.shutil.which", return_value=str(executable)),
                patch("metabaw.dependencies.subprocess.run", return_value=probe),
            ):
                issues = software_runtime_issues([requirement])

        self.assertEqual(issues, [])

    def test_invalid_gtdbtk_database_is_not_a_software_runtime_issue(self) -> None:
        requirement = SoftwareRequirement(
            "gtdbtk",
            "gtdbtk=2.7.2",
            "GTDB-Tk taxonomy",
        )
        probe = subprocess.CompletedProcess(
            ["/env/bin/gtdbtk", "--version"],
            1,
            "",
            (
                "The GTDB-Tk reference data does not exist or is corrupted.\n"
                "GTDBTK_DATA_PATH=/env/share/gtdbtk-1.0.2/db"
            ),
        )
        with (
            patch("metabaw.dependencies.shutil.which", return_value="/env/bin/gtdbtk"),
            patch("metabaw.dependencies.configured_database_path", return_value=None),
            patch("metabaw.dependencies.subprocess.run", return_value=probe),
        ):
            issues = software_runtime_issues([requirement])

        self.assertEqual(issues, [])

    def test_gtdbtk_runtime_probe_uses_saved_valid_database(self) -> None:
        requirement = SoftwareRequirement(
            "gtdbtk",
            "gtdbtk=2.7.2",
            "GTDB-Tk taxonomy",
        )
        database = Path("/db/release232")
        probe = subprocess.CompletedProcess(
            ["/env/bin/gtdbtk", "--version"],
            0,
            "GTDB-Tk v2.7.2\n",
            "",
        )
        with (
            patch("metabaw.dependencies.shutil.which", return_value="/env/bin/gtdbtk"),
            patch(
                "metabaw.dependencies.configured_database_path",
                return_value=database,
            ),
            patch("metabaw.dependencies.gtdbtk_database_valid", return_value=True),
            patch("metabaw.dependencies.subprocess.run", return_value=probe) as run,
        ):
            issues = software_runtime_issues([requirement])

        self.assertEqual(issues, [])
        self.assertEqual(
            run.call_args.kwargs["env"]["GTDBTK_DATA_PATH"],
            str(database),
        )

    def test_gtdbtk_database_path_is_persisted_in_running_conda_prefix(self) -> None:
        completed = subprocess.CompletedProcess([], 0, "", "")
        with (
            patch.dict(os.environ, {}, clear=False),
            patch(
                "metabaw.dependencies._running_conda_prefix",
                return_value=Path("/env/metabaw"),
            ),
            patch("metabaw.dependencies.shutil.which", return_value="/base/bin/conda"),
            patch("metabaw.dependencies.subprocess.run", return_value=completed) as run,
        ):
            persisted, prefix = persist_conda_environment_variable(
                "GTDBTK_DATA_PATH",
                "/db/release232",
            )
            active_value = os.environ["GTDBTK_DATA_PATH"]

        self.assertTrue(persisted)
        self.assertEqual(prefix, str(Path("/env/metabaw")))
        self.assertEqual(active_value, "/db/release232")
        self.assertEqual(
            run.call_args.args[0],
            [
                "/base/bin/conda",
                "env",
                "config",
                "vars",
                "set",
                "--prefix",
                str(Path("/env/metabaw")),
                "GTDBTK_DATA_PATH=/db/release232",
            ],
        )

    def test_isolated_status_does_not_pad_the_status_label(self) -> None:
        status = IsolatedEnvironmentStatus(
            ISOLATED_TOOLS["comebin"],
            "metabaw-comebin-py37",
            "mamba",
            "3.7",
            "/env/bin/run_comebin.sh",
        )
        output = io.StringIO()
        with patch("sys.stdout", output):
            print_isolated_status(status)
        self.assertTrue(output.getvalue().startswith("[OK] COMEBin environment"))
        self.assertNotIn("[OK     ]", output.getvalue())

    def test_comebin_cuda_repair_uses_a_python37_compatible_profile(self) -> None:
        completed = subprocess.CompletedProcess([], 0, "", None)
        with patch(
            "metabaw.dependencies.run_conda_transaction",
            return_value=completed,
        ) as transaction:
            install_isolated_cuda_runtime(
                "comebin",
                "metabaw-comebin-py37",
                "mamba",
            )

        self.assertEqual(
            transaction.call_args.args[0],
            [
                "mamba",
                "install",
                "--yes",
                "--name",
                "metabaw-comebin-py37",
                "--channel",
                "pytorch",
                "--channel",
                "conda-forge",
                "pytorch==1.10.2",
                "cudatoolkit=11.3",
            ],
        )

    def test_main_cuda_repair_uses_the_official_pinned_profile(self) -> None:
        completed = subprocess.CompletedProcess([], 0, "", None)
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary)
            (prefix / "conda-meta").mkdir()
            with (
                patch("metabaw.dependencies.sys.prefix", str(prefix)),
                patch(
                    "metabaw.dependencies.run_conda_transaction",
                    return_value=completed,
                ) as transaction,
            ):
                install_main_cuda_runtime("mamba")

        self.assertEqual(
            transaction.call_args.args[0],
            [
                "mamba",
                "install",
                "--yes",
                "--prefix",
                str(prefix.resolve()),
                "--channel",
                "pytorch",
                "--channel",
                "nvidia",
                "pytorch==2.5.1",
                "torchvision==0.20.1",
                "torchaudio==2.5.1",
                "pytorch-cuda=11.8",
            ],
        )

    def test_comebin_cuda_repair_falls_back_to_the_cu113_wheel(self) -> None:
        failed = subprocess.CompletedProcess([], 1, "", None)
        succeeded = subprocess.CompletedProcess([], 0, "", None)
        with (
            patch(
                "metabaw.dependencies.run_conda_transaction",
                return_value=failed,
            ),
            patch(
                "metabaw.dependencies.subprocess.run",
                return_value=succeeded,
            ) as pip_run,
        ):
            install_isolated_cuda_runtime(
                "comebin",
                "metabaw-comebin-py37",
                "mamba",
            )

        self.assertEqual(
            pip_run.call_args.args[0],
            [
                "mamba",
                "run",
                "--name",
                "metabaw-comebin-py37",
                "python",
                "-m",
                "pip",
                "install",
                "--upgrade",
                "--ignore-installed",
                "--no-cache-dir",
                "torch==1.10.2+cu113",
                "--find-links",
                "https://download.pytorch.org/whl/cu113/torch_stable.html",
            ],
        )

    def test_main_cuda_repair_falls_back_to_the_official_wheel(self) -> None:
        failed = subprocess.CompletedProcess([], 1, "", None)
        succeeded = subprocess.CompletedProcess([], 0, "", None)
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary)
            (prefix / "conda-meta").mkdir()
            with (
                patch("metabaw.dependencies.sys.prefix", str(prefix)),
                patch(
                    "metabaw.dependencies.sys.executable",
                    str(prefix / "bin" / "python"),
                ),
                patch(
                    "metabaw.dependencies.run_conda_transaction",
                    return_value=failed,
                ),
                patch(
                    "metabaw.dependencies.subprocess.run",
                    return_value=succeeded,
                ) as pip_run,
            ):
                install_main_cuda_runtime("mamba")

        command = pip_run.call_args.args[0]
        self.assertIn("torch==2.5.1", command)
        self.assertIn("torchvision==0.20.1", command)
        self.assertIn("torchaudio==2.5.1", command)
        self.assertIn("https://download.pytorch.org/whl/cu118", command)

    def test_main_cuda_repair_retries_with_pip_after_false_conda_success(self) -> None:
        missing = CudaRuntimeStatus(
            "MetaBAW environment", "3.11", "2.5.1", None, None,
            0, (), False, "torch.cuda.is_available() returned False",
        )
        repaired = CudaRuntimeStatus(
            "MetaBAW environment", "3.11", "2.5.1+cu118", "11.8", None,
            1, ("NVIDIA RTX A6000",), True, None,
        )
        with (
            patch("metabaw.cli.confirm_install", return_value=True),
            patch(
                "metabaw.cli.install_main_cuda_runtime",
                side_effect=(False, True),
            ) as install,
            patch(
                "metabaw.cli._main_cuda_status",
                side_effect=(missing, repaired),
            ),
            patch("sys.stdout", io.StringIO()),
        ):
            result = _offer_cuda_repair(missing)

        self.assertIs(result, repaired)
        self.assertEqual(
            install.call_args_list,
            [call(), call(pip_only=True)],
        )

    def test_main_cuda_pip_only_repair_skips_conda_and_broken_uninstall(self) -> None:
        succeeded = subprocess.CompletedProcess([], 0, "", None)
        with tempfile.TemporaryDirectory() as temporary:
            prefix = Path(temporary)
            (prefix / "conda-meta").mkdir()
            with (
                patch("metabaw.dependencies.sys.prefix", str(prefix)),
                patch(
                    "metabaw.dependencies.sys.executable",
                    str(prefix / "bin" / "python"),
                ),
                patch("metabaw.dependencies.run_conda_transaction") as transaction,
                patch(
                    "metabaw.dependencies.subprocess.run",
                    return_value=succeeded,
                ) as pip_run,
            ):
                used_pip = install_main_cuda_runtime(pip_only=True)

        self.assertTrue(used_pip)
        transaction.assert_not_called()
        command = pip_run.call_args.args[0]
        self.assertIn("--ignore-installed", command)
        self.assertNotIn("--force-reinstall", command)
        self.assertIn("https://download.pytorch.org/whl/cu118", command)

    def test_package_metadata_pins_the_main_runtime_to_python_311(self) -> None:
        pyproject = tomllib.loads(
            (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(
            pyproject["project"]["requires-python"],
            ">=3.11,<3.12",
        )

    def test_user_group_coassembly_requires_megahit_in_main_environment(self) -> None:
        args = build_parser().parse_args(
            [
                "bin",
                "--input_reads_files",
                "reads.tsv",
                "--input_contig_files",
                "contigs.tsv",
                "--assembly-strategy",
                "coassembly",
                "--coassembly-file",
                "groups.tsv",
            ]
        )
        args.align_tool = "bowtie2"
        args.type = "short"
        args.tools = ["metabat2", "metadecoder", "vamb"]
        executables = {
            requirement.executable for requirement in bin_requirements(args)
        }
        self.assertIn("megahit", executables)
        args.assembly_strategy = "existing-contigs"
        self.assertNotIn("megahit", {item.executable for item in bin_requirements(args)})

    def test_supported_isolated_environment_registry_and_normalization(self) -> None:
        self.assertEqual(set(ISOLATED_TOOLS), {"comebin", "checkm2", "metawrap", "lorbin"})
        args = build_parser().parse_args(["check", "--all"])
        self.assertEqual({key for key in vars(args) if key.endswith("_env")},
                         {f"{key}_env" for key in ISOLATED_TOOLS})
        self.assertEqual(normalize_isolated_environment("checkm2", " /envs/checkm2 "), "/envs/checkm2")
        with self.assertRaisesRegex(ValueError, "Unknown isolated tool"):
            normalize_isolated_environment("unknown", "unused")

    def test_transaction_preserves_user_conda_configuration(self) -> None:
        captured_environment: dict[str, str] = {}

        def fake_popen(_command, **kwargs):
            captured_environment.update(kwargs["env"])
            return _SuccessfulProcess()

        user_environment = {
            "CONDA_PKGS_DIRS": "/existing/conda/pkgs",
            "CONDA_NO_PLUGINS": "false",
            "CONDA_REMOTE_READ_TIMEOUT_SECS": "777",
        }
        with patch.dict(os.environ, user_environment, clear=True):
            with patch("metabaw.dependencies.subprocess.Popen", side_effect=fake_popen):
                result = run_conda_transaction(
                    ("mamba", "install", "--yes", "metabat2"),
                    retry_count=0,
                )

        self.assertEqual(result.returncode, 0)
        for name, value in user_environment.items():
            self.assertEqual(captured_environment[name], value)

    def test_missing_packages_are_installed_as_restartable_transactions(self) -> None:
        requirements = [
            SoftwareRequirement("tool-a", "package-a", "test tool A"),
            SoftwareRequirement("tool-b", "package-b", "test tool B"),
        ]
        completed = subprocess.CompletedProcess([], 0, "", None)

        with patch(
            "metabaw.dependencies.missing_software",
            side_effect=[requirements, []],
        ):
            with patch("metabaw.dependencies.software_runtime_issues", return_value=[]):
                with patch("metabaw.dependencies.conda_frontend", return_value="mamba"):
                    with patch(
                        "metabaw.dependencies.run_conda_transaction",
                        return_value=completed,
                    ) as transaction:
                        remaining = install_software(requirements)

        self.assertEqual(remaining, [])
        self.assertEqual(transaction.call_count, 2)
        commands = [call.args[0] for call in transaction.call_args_list]
        self.assertEqual(commands[0][-1], "package-a")
        self.assertEqual(commands[1][-1], "package-b")

    def test_failed_package_does_not_stop_later_installation_transactions(self) -> None:
        requirements = [
            SoftwareRequirement("tool-a", "package-a", "test tool A"),
            SoftwareRequirement("tool-b", "package-b", "test tool B"),
        ]
        failed = subprocess.CompletedProcess([], 1, "", None)
        completed = subprocess.CompletedProcess([], 0, "", None)

        with patch(
            "metabaw.dependencies.missing_software",
            side_effect=[requirements, requirements[:1]],
        ):
            with patch("metabaw.dependencies.software_runtime_issues", return_value=[]):
                with patch("metabaw.dependencies.conda_frontend", return_value="mamba"):
                    with patch(
                        "metabaw.dependencies.run_conda_transaction",
                        side_effect=[failed, completed],
                    ) as transaction:
                        remaining = install_software(requirements)

        self.assertEqual(remaining, requirements[:1])
        self.assertEqual(transaction.call_count, 2)
        self.assertEqual(transaction.call_args_list[1].args[0][-1], "package-b")

    def test_isolated_installation_is_confirmed_once_and_attempts_every_environment(self) -> None:
        selected = ["comebin", "checkm2", "metawrap"]
        environments = {key: f"test-{key}" for key in selected}

        def unavailable(spec, environment):
            return IsolatedEnvironmentStatus(
                spec,
                environment,
                "mamba",
                None,
                None,
                "environment not found",
            )

        def install(spec, environment, frontend):
            if spec.key == "checkm2":
                raise RuntimeError("simulated failure")
            return IsolatedEnvironmentStatus(
                spec,
                environment,
                frontend,
                spec.python_version,
                f"/envs/{environment}/bin/{spec.executable}",
                None,
            )

        with (
            patch("metabaw.cli.isolated_environment_status", side_effect=unavailable),
            patch("metabaw.cli.confirm_install", return_value=True) as confirm,
            patch("metabaw.cli.install_isolated_environment", side_effect=install) as installer,
        ):
            statuses, available = _check_isolated_tools_interactively(
                selected,
                environments,
            )

        self.assertFalse(available)
        self.assertTrue(statuses["comebin"].available)
        self.assertFalse(statuses["checkm2"].available)
        self.assertTrue(statuses["metawrap"].available)
        confirm.assert_called_once()
        self.assertEqual(installer.call_count, 3)

    def test_new_runtime_issue_is_repaired_in_the_same_check_invocation(self) -> None:
        requirement = SoftwareRequirement(
            "gtdbtk",
            "gtdbtk=2.7.2",
            "GTDB-Tk taxonomy",
        )
        runtime_issue = SoftwareRuntimeIssue(
            "gtdbtk",
            "simulated post-install runtime issue",
            ("gtdbtk=2.7.2",),
        )
        with (
            patch("metabaw.cli.print_software_status"),
            patch(
                "metabaw.cli.missing_software",
                side_effect=[[requirement], [], []],
            ),
            patch(
                "metabaw.cli.software_runtime_issues",
                side_effect=[[], [runtime_issue], []],
            ),
            patch("metabaw.cli.confirm_install", return_value=True) as confirm,
            patch("metabaw.cli.install_software", return_value=[]) as installer,
        ):
            available = _check_software_interactively([requirement])

        self.assertTrue(available)
        confirm.assert_called_once()
        self.assertEqual(installer.call_count, 2)

    def test_vamb_uses_pip_instead_of_conda(self) -> None:
        requirements = [SoftwareRequirement("vamb", "vamb", "VAMB binning")]
        completed = subprocess.CompletedProcess([], 0, "", None)

        with patch(
            "metabaw.dependencies.missing_software",
            side_effect=[requirements, []],
        ):
            with patch("metabaw.dependencies.software_runtime_issues", return_value=[]):
                with patch("metabaw.dependencies.subprocess.run", return_value=completed) as run:
                    with patch("metabaw.dependencies.run_conda_transaction") as transaction:
                        remaining = install_software(requirements)

        self.assertEqual(remaining, [])
        transaction.assert_not_called()
        command = run.call_args.args[0]
        self.assertEqual(command[:4], [os.sys.executable, "-m", "pip", "install"])
        self.assertEqual(command[-1], "vamb")
        self.assertNotIn("--upgrade", command)

    def test_selected_bioconda_tools_use_the_requested_channel_order(self) -> None:
        requirements = [
            SoftwareRequirement("SemiBin2", "semibin", "SemiBin2 binning"),
            SoftwareRequirement("gunc", "gunc", "GUNC contamination checking"),
            SoftwareRequirement("gtdbtk", "gtdbtk=2.7.2", "GTDB-Tk taxonomy"),
            SoftwareRequirement("galah", "galah", "MAG dereplication"),
        ]
        completed = subprocess.CompletedProcess([], 0, "", None)

        with patch(
            "metabaw.dependencies.missing_software",
            side_effect=[requirements, []],
        ):
            with patch("metabaw.dependencies.software_runtime_issues", return_value=[]):
                with patch("metabaw.dependencies.conda_frontend", return_value="mamba"):
                    with patch(
                        "metabaw.dependencies.run_conda_transaction",
                        return_value=completed,
                    ) as transaction:
                        remaining = install_software(requirements)

        self.assertEqual(remaining, [])
        commands = [call.args[0] for call in transaction.call_args_list]
        self.assertEqual(
            commands,
            [
                [
                    "mamba",
                    "install",
                    "-y",
                    "-c",
                    "bioconda",
                    "-c",
                    "conda-forge",
                    package,
                ]
                for package in (
                    "semibin",
                    "gunc",
                    "gtdbtk=2.7.2",
                    "galah",
                )
            ],
        )


class LorBinEnvironmentRepairTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = ISOLATED_TOOLS["lorbin"]
        self.original = self.spec.default_environment
        self.replacement = self.original + "-repair"

    def status(self, environment, python="3.10", *, error=None):
        return IsolatedEnvironmentStatus(
            self.spec, environment, "conda", python,
            f"/envs/{environment}/bin/LorBin" if python is not None else None,
            error,
        )

    def inventory(self, *prefixes):
        return subprocess.CompletedProcess([], 0, json.dumps({"envs": list(prefixes)}), "")

    def test_wrong_python_creates_a_pinned_replacement_for_each_frontend(self) -> None:
        for frontend in ("conda", "mamba", "micromamba"):
            with self.subTest(frontend=frontend):
                repaired = self.status(self.replacement)
                with (
                    patch("metabaw.dependencies.isolated_environment_status", side_effect=[
                        self.status(self.original, "3.11"), repaired,
                    ]),
                    patch("metabaw.dependencies.subprocess.run", side_effect=[
                        self.inventory("/envs/" + self.original),
                        subprocess.CompletedProcess([], 0, "", ""),
                    ]) as run,
                    patch("metabaw.dependencies.run_conda_transaction", return_value=
                          subprocess.CompletedProcess([], 0, "", "")) as transaction,
                    redirect_stdout(io.StringIO()) as output,
                ):
                    result = install_isolated_environment(self.spec, self.original, frontend)
                self.assertIs(result, repaired)
                command = transaction.call_args.args[0]
                self.assertEqual(command[:5], [frontend, "create", "--yes", "--name", self.replacement])
                self.assertIn("python=3.10", command)
                self.assertNotIn(self.original, command)
                self.assertEqual(run.call_args_list[0].args[0], [frontend, "env", "list", "--json"])
                self.assertEqual(run.call_args_list[1].args[0][:4], [frontend, "run", "--name", self.replacement])
                self.assertIn("[ENV SWITCH]", output.getvalue())
                self.assertIn("preserving the original environment", output.getvalue())

    def test_occupied_unusable_replacement_names_are_not_overwritten(self) -> None:
        for python, error in (("3.11", None), (None, "cannot run Python"), ("3.10", "runtime import failed")):
            with self.subTest(python=python, error=error):
                next_name = self.replacement + "-2"
                with (
                    patch("metabaw.dependencies.isolated_environment_status", side_effect=[
                        self.status(self.original, "3.11"),
                        self.status(self.replacement, python, error=error),
                        self.status(next_name),
                    ]),
                    patch("metabaw.dependencies.subprocess.run", side_effect=[
                        self.inventory("/envs/" + self.original, "/envs/" + self.replacement),
                        subprocess.CompletedProcess([], 0, "", ""),
                    ]),
                    patch("metabaw.dependencies.run_conda_transaction", return_value=
                          subprocess.CompletedProcess([], 0, "", "")) as transaction,
                    redirect_stdout(io.StringIO()),
                ):
                    result = install_isolated_environment(self.spec, self.original, "conda")
                self.assertEqual(result.environment, next_name)
                self.assertEqual(transaction.call_args.args[0][1:5], ["create", "--yes", "--name", next_name])

    def test_a_healthy_replacement_is_reused_without_installing_or_removing_anything(self) -> None:
        repaired = self.status(self.replacement)
        with (
            patch("metabaw.dependencies.isolated_environment_status", side_effect=[
                self.status(self.original, "3.11"), repaired,
            ]),
            patch("metabaw.dependencies.subprocess.run", return_value=
                  self.inventory("/envs/" + self.replacement)) as run,
            patch("metabaw.dependencies.run_conda_transaction") as transaction,
            redirect_stdout(io.StringIO()),
        ):
            result = install_isolated_environment(self.spec, self.original, "conda")
        self.assertIs(result, repaired)
        transaction.assert_not_called()
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0], ["conda", "env", "list", "--json"])

    def test_custom_prefix_and_unregistered_sibling_data_are_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            original = Path(temporary) / "custom-lorbin"
            original.mkdir()
            sentinel = original / "user-data.txt"
            sentinel.write_text("keep original")
            occupied = Path(str(original) + "-py310")
            occupied.mkdir()
            sibling_data = occupied / "user-data.txt"
            sibling_data.write_text("keep sibling")
            replacement = str(Path(str(original) + "-py310-2").resolve())
            with (
                patch("metabaw.dependencies.isolated_environment_status", side_effect=[
                    self.status(str(original), "3.11"),
                    self.status(str(occupied), None, error="not a Python environment"),
                    self.status(replacement),
                ]),
                patch("metabaw.dependencies.subprocess.run", side_effect=[
                    self.inventory(str(original)), subprocess.CompletedProcess([], 0, "", ""),
                ]),
                patch("metabaw.dependencies.run_conda_transaction", return_value=
                      subprocess.CompletedProcess([], 0, "", "")) as transaction,
                redirect_stdout(io.StringIO()),
            ):
                result = install_isolated_environment(self.spec, str(original), "conda")
            self.assertEqual(result.environment, replacement)
            self.assertEqual(transaction.call_args.args[0][1:5], ["create", "--yes", "--prefix", replacement])
            self.assertEqual(sentinel.read_text(), "keep original")
            self.assertEqual(sibling_data.read_text(), "keep sibling")

    def test_invalid_environment_inventory_stops_before_any_installation(self) -> None:
        for result in (
            subprocess.CompletedProcess([], 1, "", "failed"),
            subprocess.CompletedProcess([], 0, "not JSON", ""),
            subprocess.CompletedProcess([], 0, '{"envs": null}', ""),
            subprocess.CompletedProcess([], 0, '{"envs": [123]}', ""),
        ):
            with self.subTest(stdout=result.stdout, returncode=result.returncode):
                with (
                    patch("metabaw.dependencies.isolated_environment_status", return_value=self.status(self.original, "3.11")),
                    patch("metabaw.dependencies.subprocess.run", return_value=result),
                    patch("metabaw.dependencies.run_conda_transaction") as transaction,
                ):
                    with self.assertRaisesRegex(RuntimeError, "original environment was preserved"):
                        install_isolated_environment(self.spec, self.original, "conda")
                transaction.assert_not_called()

    def test_other_tools_still_refuse_to_change_existing_python_versions(self) -> None:
        for key in ("comebin", "checkm2", "metawrap"):
            spec = ISOLATED_TOOLS[key]
            wrong = IsolatedEnvironmentStatus(spec, spec.default_environment, "conda", "3.11", None)
            with (
                self.subTest(key=key),
                patch("metabaw.dependencies.isolated_environment_status", return_value=wrong),
                patch("metabaw.dependencies.subprocess.run") as run,
                patch("metabaw.dependencies.run_conda_transaction") as transaction,
            ):
                with self.assertRaisesRegex(RuntimeError, "will not change Python"):
                    install_isolated_environment(spec, spec.default_environment, "conda")
                run.assert_not_called()
                transaction.assert_not_called()

    def test_replacement_is_rejected_if_post_install_validation_fails(self) -> None:
        for python, error in (("3.11", "wrong Python"), ("3.10", "cannot import LorBin")):
            with (
                self.subTest(python=python, error=error),
                patch("metabaw.dependencies.isolated_environment_status", side_effect=[
                    self.status(self.original, "3.11"), self.status(self.replacement, python, error=error),
                ]),
                patch("metabaw.dependencies.subprocess.run", side_effect=[
                    self.inventory(), subprocess.CompletedProcess([], 0, "", ""),
                ]),
                patch("metabaw.dependencies.run_conda_transaction", return_value=
                      subprocess.CompletedProcess([], 0, "", "")),
                redirect_stdout(io.StringIO()),
            ):
                with self.assertRaisesRegex(RuntimeError, "not usable after installation"):
                    install_isolated_environment(self.spec, self.original, "conda")

    def test_failed_or_declined_repairs_keep_the_selected_environment_unchanged(self) -> None:
        for approved in (False, True):
            environments = {"lorbin": self.original}
            with (
                self.subTest(approved=approved),
                patch("metabaw.cli.isolated_environment_status", return_value=self.status(self.original, "3.11")),
                patch("metabaw.cli.confirm_install", return_value=approved),
                patch("metabaw.cli.install_isolated_environment", side_effect=RuntimeError("installation failed")) as install,
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()),
            ):
                statuses, available = _check_isolated_tools_interactively(["lorbin"], environments)
            self.assertFalse(available)
            self.assertEqual(environments["lorbin"], self.original)
            self.assertEqual(statuses["lorbin"].environment, self.original)
            self.assertEqual(install.call_count, int(approved))

    def test_check_saves_and_cuda_tests_the_validated_replacement_not_the_old_name(self) -> None:
        for override in (False, True):
            with self.subTest(override=override), tempfile.TemporaryDirectory() as temporary:
                config = Path(temporary) / "config.json"
                host = HostCudaStatus("nvidia-smi", (CudaDevice("0", "test GPU", 16384, "test"),), "11.3")
                cuda = CudaRuntimeStatus("LorBin", "3.10", "1.11.0", "11.3", None, 1, ("test GPU",), True)
                stderr = io.StringIO()
                with (
                    patch.dict(os.environ, {"METABAW_CONFIG_FILE": str(config)}),
                    patch("metabaw.cli._check_software_interactively", return_value=True),
                    patch("metabaw.cli.isolated_environment_status", return_value=self.status(self.original, "3.11")),
                    patch("metabaw.cli.confirm_install", return_value=True),
                    patch("metabaw.cli.install_isolated_environment", return_value=self.status(self.replacement)),
                    patch("metabaw.cli.host_cuda_status", return_value=host),
                    patch("metabaw.cli._isolated_cuda_status", return_value=cuda) as cuda_probe,
                    redirect_stdout(io.StringIO()), redirect_stderr(stderr),
                ):
                    if override:
                        os.environ["METABAW_LORBIN_ENV"] = self.original
                    else:
                        os.environ.pop("METABAW_LORBIN_ENV", None)
                    args = build_parser().parse_args(["check", "--lorbin-env", self.original])
                    self.assertEqual(command_check(args), 0)
                    self.assertEqual(args.lorbin_env, self.replacement)
                    self.assertEqual(json.loads(config.read_text())["isolated_environments"]["lorbin"], self.replacement)
                    cuda_probe.assert_called_once_with("lorbin", self.replacement, "conda")
                    if override:
                        self.assertEqual(os.environ["METABAW_LORBIN_ENV"], self.original)
                        self.assertIn("overrides the saved LorBin environment", stderr.getvalue())
                        os.environ.pop("METABAW_LORBIN_ENV")
                    self.assertEqual(configured_isolated_environment("lorbin"), self.replacement)

    def test_failed_check_does_not_replace_saved_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = Path(temporary) / "config.json"
            content = json.dumps({"isolated_environments": {"lorbin": self.original}})
            config.write_text(content)
            with (
                patch.dict(os.environ, {"METABAW_CONFIG_FILE": str(config)}),
                patch("metabaw.cli._check_software_interactively", return_value=True),
                patch("metabaw.cli.isolated_environment_status", return_value=self.status(self.original, "3.11")),
                patch("metabaw.cli.confirm_install", return_value=True),
                patch("metabaw.cli.install_isolated_environment", side_effect=RuntimeError("installation failed")),
                patch("metabaw.cli.host_cuda_status", return_value=HostCudaStatus(None, (), None)),
                patch("metabaw.cli._isolated_cuda_status") as cuda_probe,
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()),
            ):
                args = build_parser().parse_args(["check", "--lorbin-env", self.original])
                self.assertEqual(command_check(args), 1)
            self.assertEqual(config.read_text(), content)
            cuda_probe.assert_not_called()


if __name__ == "__main__":
    unittest.main()
