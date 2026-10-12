from __future__ import annotations

import csv
from dataclasses import dataclass
import gzip
from importlib import metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
from typing import Callable, Iterable, Mapping, Sequence

from .metadecoder_coverage import metadecoder_executable
from .terminal import status_row, status_tag
from .cupy_runtime import (
    CupyInstallPlan, cupy_runtime_environment, runtime_root, save_cupy_runtime,
)
from .metadecoder_install import (
    METADECODER_RELEASE_URL,
    pip_install_environment,
    prepare_metadecoder_wheel,
)


@dataclass(frozen=True)
class SoftwareRequirement:
    executable: str
    package: str
    purpose: str


@dataclass(frozen=True)
class SoftwareRuntimeIssue:
    executable: str
    detail: str
    repair_packages: tuple[str, ...]


@dataclass(frozen=True)
class RPackageRequirement:
    name: str
    source: str = "cran"


@dataclass(frozen=True)
class IsolatedToolSpec:
    key: str
    display_name: str
    executable: str
    package: str
    python_version: str
    default_environment: str
    option: str
    purpose: str
    conda_packages: tuple[str, ...] = ()
    channels: tuple[str, ...] = ("conda-forge", "bioconda")
    pip_source: str | None = None


ISOLATED_TOOLS: dict[str, IsolatedToolSpec] = {
    "comebin": IsolatedToolSpec(
        key="comebin",
        display_name="COMEBin",
        executable="run_comebin.sh",
        package="comebin",
        python_version="3.7",
        default_environment="metabaw-comebin-py37",
        option="--comebin-env",
        purpose="COMEBin binning",
    ),
    "checkm2": IsolatedToolSpec(
        key="checkm2",
        display_name="CheckM2",
        executable="checkm2",
        package="checkm2",
        python_version="3.12",
        default_environment="metabaw-checkm2-py312",
        option="--checkm2-env",
        purpose="CheckM2 quality estimation",
    ),
    "metawrap": IsolatedToolSpec(
        key="metawrap",
        display_name="MetaWRAP",
        executable="metawrap",
        package="metawrap-refinement",
        python_version="2.7",
        default_environment="metabaw-metawrap-py27",
        option="--metawrap-env",
        purpose="MetaWRAP bin refinement",
    ),
    "lorbin": IsolatedToolSpec(
        key="lorbin",
        display_name="LorBin",
        executable="LorBin",
        package="LorBin official source ee10232",
        python_version="3.10",
        default_environment="metabaw-lorbin-py310",
        option="--lorbin-env",
        purpose="long-read metagenomic binning",
        conda_packages=(
            # LorBin documents Biopython 1.78, but Conda does not provide a
            # Python 3.10 build of that release. Biopython 1.83 retains the
            # SeqIO interfaces used by LorBin and has Python 3.10 packages.
            "biopython=1.83",
            "hmmer",
            "prodigal",
            "bedtools",
            "samtools",
            "pytorch=1.11.0",
            "torchvision=0.12.0",
            "torchaudio=0.11.0",
            "mkl=2024.0.0",
            "numpy=1.23.3",
            "pip=22.2.2",
            "setuptools=65.5.0",
            "pandas=2.2.2",
            "scikit-learn=1.1.2",
            "scipy=1.13.1",
            "joblib=1.4.2",
        ),
        channels=("bioconda", "pytorch", "conda-forge"),
        pip_source=(
            "https://github.com/LorMeBioAI/LorBin/archive/"
            "ee10232282c2b71ed3ce2a34d5dbd78af3dd0b0a.tar.gz"
        ),
    ),
}

COMEBIN_ENV_DEFAULT = ISOLATED_TOOLS["comebin"].default_environment
COMEBIN_PYTHON_VERSION = ISOLATED_TOOLS["comebin"].python_version
CHECKM2_ENV_DEFAULT = ISOLATED_TOOLS["checkm2"].default_environment
METAWRAP_ENV_DEFAULT = ISOLATED_TOOLS["metawrap"].default_environment
LORBIN_ENV_DEFAULT = ISOLATED_TOOLS["lorbin"].default_environment
COMEBIN_REQUIREMENT = SoftwareRequirement(
    ISOLATED_TOOLS["comebin"].executable,
    ISOLATED_TOOLS["comebin"].package,
    ISOLATED_TOOLS["comebin"].purpose,
)

ISOLATED_CUDA_PACKAGES: dict[
    str,
    tuple[tuple[str, ...], tuple[str, ...]],
] = {
    "comebin": (
        ("pytorch", "conda-forge"),
        (
            "pytorch==1.10.2",
            "cudatoolkit=11.3",
        ),
    ),
    "lorbin": (
        ("pytorch", "conda-forge", "bioconda"),
        (
            "pytorch=1.11.0",
            "torchvision=0.12.0",
            "torchaudio=0.11.0",
            "cudatoolkit=11.3",
            "pytorch-mutex=1.0=cuda",
            "mkl=2024.0.0",
        ),
    ),
}

ISOLATED_CUDA_PIP_FALLBACKS: dict[str, tuple[str, str, str]] = {
    "comebin": (
        "torch==1.10.2+cu113",
        "--find-links",
        "https://download.pytorch.org/whl/cu113/torch_stable.html",
    ),
}

ISOLATED_RUNTIME_PROBE_ENV = {
    "BLIS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}


@dataclass(frozen=True)
class IsolatedEnvironmentStatus:
    spec: IsolatedToolSpec
    environment: str
    frontend: str | None
    python_version: str | None
    executable: str | None
    error: str | None = None

    @property
    def available(self) -> bool:
        return (
            self.frontend is not None
            and self.python_version == self.spec.python_version
            and self.executable is not None
            and self.error is None
        )


COMEBinEnvironmentStatus = IsolatedEnvironmentStatus


@dataclass(frozen=True)
class CudaDevice:
    index: str
    name: str
    memory_total_mib: int
    driver_version: str
    memory_free_mib: int | None = None
    uuid: str | None = None


@dataclass(frozen=True)
class HostCudaStatus:
    executable: str | None
    devices: tuple[CudaDevice, ...]
    advertised_cuda_version: str | None
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.executable is not None and bool(self.devices) and self.error is None


@dataclass(frozen=True)
class CudaRuntimeStatus:
    label: str
    python_version: str | None
    torch_version: str | None
    torch_cuda_version: str | None
    cuda_visible_devices: str | None
    device_count: int
    devices: tuple[str, ...]
    allocation_test: bool
    error: str | None = None

    @property
    def available(self) -> bool:
        return (
            self.error is None
            and self.device_count > 0
            and self.allocation_test
        )


@dataclass(frozen=True)
class CupyRuntimeStatus:
    label: str
    python_version: str | None
    cupy_version: str | None
    cuda_version: str | None
    cuda_visible_devices: str | None
    device_count: int
    devices: tuple[str, ...]
    allocation_test: bool
    error: str | None = None
    compiled_cuda_version: str | None = None

    @property
    def available(self) -> bool:
        return self.error is None and self.device_count > 0 and self.allocation_test


SOFTWARE: dict[str, SoftwareRequirement] = {
    "bash": SoftwareRequirement("bash", "bash", "task shell"),
    "bowtie2": SoftwareRequirement("bowtie2", "bowtie2", "short-read mapping"),
    "bowtie2-build": SoftwareRequirement("bowtie2-build", "bowtie2", "Bowtie2 indexing"),
    "minimap2": SoftwareRequirement("minimap2", "minimap2", "long-read or CoverM mapping"),
    "minibwa": SoftwareRequirement(
        "minibwa",
        "lh3/minibwa source",
        "accurate long-read mapping",
    ),
    "samtools": SoftwareRequirement("samtools", "samtools", "BAM filtering, sorting, and indexing"),
    "jgi_summarize_bam_contig_depths": SoftwareRequirement(
        "jgi_summarize_bam_contig_depths", "metabat2", "MetaBAT2 depth calculation"
    ),
    "metabat2": SoftwareRequirement("metabat2", "metabat2", "MetaBAT2 binning"),
    "strobealign": SoftwareRequirement("strobealign", "strobealign", "short-read VAMB abundance"),
    "vamb": SoftwareRequirement("vamb", "vamb", "VAMB binning"),
    "metadecoder": SoftwareRequirement("metadecoder", "metadecoder", "MetaDecoder binning"),
    "SemiBin2": SoftwareRequirement("SemiBin2", "semibin", "SemiBin2 binning"),
    "Rscript": SoftwareRequirement("Rscript", "r-base", "MAGScoT refinement"),
    "Rscript_das_tool": SoftwareRequirement(
        "Rscript", "r-base", "DAS Tool refinement"
    ),
    "prodigal": SoftwareRequirement("prodigal", "prodigal", "gene prediction"),
    "parallel": SoftwareRequirement("parallel", "parallel", "parallel single-threaded jobs"),
    "hmmsearch": SoftwareRequirement("hmmsearch", "hmmer", "marker HMM search"),
    "hmmalign": SoftwareRequirement("hmmalign", "hmmer", "marker alignment"),
    "pplacer": SoftwareRequirement("pplacer", "pplacer", "reference-tree placement"),
    "guppy": SoftwareRequirement("guppy", "pplacer", "reference-tree conversion"),
    "skani": SoftwareRequirement("skani", "skani", "GTDB-Tk ANI screening or Galah dereplication"),
    "DAS_Tool": SoftwareRequirement("DAS_Tool", "das_tool", "DAS Tool refinement"),
    "diamond": SoftwareRequirement("diamond", "diamond", "DAS Tool search engine"),
    "pullseq": SoftwareRequirement("pullseq", "pullseq", "DAS Tool bin extraction"),
    "ruby": SoftwareRequirement("ruby", "ruby", "DAS Tool marker parsing"),
    "checkm": SoftwareRequirement("checkm", "checkm-genome", "CheckM quality estimation"),
    "gunc": SoftwareRequirement("gunc", "gunc", "GUNC contamination checking"),
    "gtdbtk": SoftwareRequirement(
        "gtdbtk",
        "gtdbtk=2.7.2",
        "GTDB-Tk taxonomy or RNA domain detection",
    ),
    "tRNAscan-SE": SoftwareRequirement("tRNAscan-SE", "trnascan-se", "tRNA quality control"),
    "barrnap": SoftwareRequirement("barrnap", "barrnap", "rRNA quality control"),
    "galah": SoftwareRequirement("galah", "galah", "MAG dereplication"),
    "dRep": SoftwareRequirement("dRep", "drep", "MAG dereplication"),
    "coverm": SoftwareRequirement("coverm", "coverm", "MAG abundance profiling"),
    "exec_annotation": SoftwareRequirement(
        "exec_annotation", "kofamscan", "KEGG Orthology annotation with KofamScan"
    ),
    "run_dbcan": SoftwareRequirement(
        "run_dbcan", "dbcan", "CAZy annotation with run_dbCAN"
    ),
    "blastp": SoftwareRequirement(
        "blastp", "blast", "HydDB hydrogenase homology search"
    ),
    "makeblastdb": SoftwareRequirement(
        "makeblastdb", "blast", "HydDB protein database indexing"
    ),
    "git": SoftwareRequirement("git", "git", "MAGScoT source installation"),
    "megahit": SoftwareRequirement(
        "megahit",
        "megahit",
        "MetaBAW paired short-read assembly",
    ),
    "flye": SoftwareRequirement(
        "flye",
        "flye",
        "MetaBAW long-read metagenome assembly",
    ),
}

R_PACKAGE_PROFILES: dict[str, tuple[RPackageRequirement, ...]] = {
    "MAGScoT": (
        RPackageRequirement("optparse"),
        RPackageRequirement("dplyr"),
        RPackageRequirement("readr"),
        RPackageRequirement("funr"),
        RPackageRequirement("digest"),
    ),
    "DAS Tool": (
        RPackageRequirement("data.table"),
        RPackageRequirement("magrittr"),
        RPackageRequirement("docopt"),
    ),
}
MAGSCOT_R_NAMES = tuple(
    requirement.name for requirement in R_PACKAGE_PROFILES["MAGScoT"]
)
DAS_TOOL_R_NAMES = tuple(
    requirement.name for requirement in R_PACKAGE_PROFILES["DAS Tool"]
)
CONDA_R_PACKAGES = {
    name: f"r-{name.lower()}"
    for name in ("optparse", "dplyr", "readr", "digest", "data.table", "magrittr", "docopt")
}
PIP_INSTALL_URLS = {
    # VAMB recommends installation from PyPI for supported Python versions.
    "vamb": "vamb",
    "metadecoder": METADECODER_RELEASE_URL,
}

# These packages are distributed through Bioconda and should be resolved with
# Bioconda before conda-forge, matching their documented installation commands.
BIOCONDA_FIRST_PACKAGES = {"semibin", "gunc", "gtdbtk", "galah"}

SOURCE_INSTALL_EXECUTABLES = {"minibwa"}
MINIBWA_SOURCE_URL = "https://github.com/lh3/minibwa.git"
MINIBWA_SOURCE_REVISION = "f0e117436c28addc359b67123d2353f0d4a1f9e8"


def _minibwa_source_directory() -> Path:
    return (
        Path.home()
        / ".cache"
        / "metabaw"
        / f"minibwa-{MINIBWA_SOURCE_REVISION[:12]}"
    )


def unique_requirements(names: Iterable[str]) -> list[SoftwareRequirement]:
    seen: set[str] = set()
    requirements: list[SoftwareRequirement] = []
    for name in names:
        requirement = SOFTWARE[name]
        if requirement.executable not in seen:
            requirements.append(requirement)
            seen.add(requirement.executable)
    return requirements


def bin_requirements(args: object) -> list[SoftwareRequirement]:
    aligners = getattr(args, "required_align_tools", None) or [str(getattr(args, "align_tool"))]
    names = ["bash", *aligners, "samtools"]
    if "bowtie2" in aligners:
        names.append("bowtie2-build")
    for binner in getattr(args, "tools"):
        if binner == "metabat2":
            names.extend(("jgi_summarize_bam_contig_depths", "metabat2"))
        elif binner == "vamb":
            names.append("vamb")
            if getattr(args, "requires_vamb_aemb", getattr(args, "type") == "short"):
                names.append("strobealign")
        elif binner == "metadecoder":
            names.append("metadecoder")
        elif binner == "comebin":
            # COMEBin is checked separately because its Bioconda package pins
            # Python 3.7 and must not be installed into MetaBAW's Python 3.11
            # environment.
            continue
        elif binner == "semibin2":
            names.append("SemiBin2")
        elif binner == "lorbin":
            # LorBin 0.1.0 uses a pinned Python 3.10 and PyTorch 1.11 stack.
            continue
    refiner = getattr(args, "refinement")
    if refiner == "magscot":
        # Git is an installation dependency, not a workflow executable.
        names.extend(("Rscript", "prodigal", "parallel", "hmmsearch"))
    elif refiner == "das_tool":
        names.extend(
            (
                "DAS_Tool",
                "Rscript_das_tool",
                "diamond",
                "prodigal",
                "pullseq",
                "ruby",
            )
        )
    else:
        # MetaWRAP is checked separately because its Bioconda package pins
        # Python 2.7.
        pass
    if (
        getattr(args, "quality_control") == "checkm"
        and refiner != "metawrap"
    ):
        names.extend(("checkm", "prodigal", "hmmsearch", "hmmalign", "pplacer"))
    if getattr(args, "gunc"):
        names.extend(("gunc", "prodigal", "diamond"))
    if getattr(args, "trna") or getattr(args, "rrna"):
        # RNA domain detection runs GTDB-Tk identify, not classify_wf.
        names.extend(("gtdbtk", "prodigal", "hmmsearch"))
    if getattr(args, "trna"):
        names.append("tRNAscan-SE")
    if getattr(args, "rrna"):
        names.append("barrnap")
    if getattr(args, "dereplication_tool") == "galah":
        # The generated Galah command explicitly selects both skani methods.
        names.extend(("galah", "skani"))
    else:
        names.append("dRep")
    assemblers = getattr(args, "required_assemblers", None)
    if assemblers is None:
        # Compatibility for callers inspecting parser output before manifests
        # have been resolved; command_bin supplies the exact MEGAHIT/Flye set.
        assemblers = (
            ("megahit",)
            if getattr(args, "assembly_strategy", "existing-contigs") == "coassembly"
            else ()
        )
    names.extend(assemblers)
    return unique_requirements(names)


def annotation_requirements(args: object | None = None) -> list[SoftwareRequirement]:
    names = ["bash", "coverm", "minimap2", "samtools"]
    if args is None or not getattr(args, "gtdbtk_res", None):
        # Match GTDB-Tk 2.7.2 classify_wf's external dependency checks.
        names.extend((
            "gtdbtk", "prodigal", "hmmsearch", "hmmalign", "pplacer", "guppy", "skani",
        ))
    run_kegg = args is None or bool(getattr(args, "kegg", True))
    run_cazy = args is None or bool(getattr(args, "cazy", True))
    run_hydrogenase = args is None or bool(getattr(args, "hydrogenase", True))
    if run_kegg or run_cazy or run_hydrogenase:
        names.append("prodigal")
    if run_kegg:
        names.extend(("exec_annotation", "ruby", "parallel", "hmmsearch"))
    if run_cazy:
        names.extend(("run_dbcan", "diamond", "hmmsearch"))
    if run_hydrogenase:
        names.extend(("blastp", "makeblastdb", "diamond"))
    return unique_requirements(names)


def all_requirements() -> list[SoftwareRequirement]:
    return unique_requirements(SOFTWARE)


def missing_software(
    requirements: Sequence[SoftwareRequirement],
) -> list[SoftwareRequirement]:
    return [
        requirement
        for requirement in requirements
        if (metadecoder_executable() if requirement.executable == "metadecoder"
            else shutil.which(requirement.executable)) is None
    ]


def _checkm_help_exit_is_healthy(process: subprocess.CompletedProcess[str]) -> bool:
    """Accept CheckM's argparse usage exit while preserving real import failures."""
    if process.returncode != 2:
        return False
    output = "\n".join(
        part for part in (process.stdout, process.stderr) if part
    ).lower()
    fatal_markers = (
        "traceback (most recent call last)",
        "modulenotfounderror",
        "importerror",
        "no module named",
        "cannot import name",
        "error while loading shared libraries",
    )
    return "usage: checkm" in output and not any(
        marker in output for marker in fatal_markers
    )


def _gtdbtk_reference_data_error(
    process: subprocess.CompletedProcess[str],
) -> bool:
    """Distinguish an invalid GTDB data path from a broken executable."""
    if process.returncode == 0:
        return False
    output = "\n".join(
        part for part in (process.stdout, process.stderr) if part
    ).lower()
    if "reference data does not exist or is corrupted" in output:
        return True
    return "gtdbtk_data_path" in output and any(
        marker in output
        for marker in (
            "does not exist",
            "not defined",
            "not set",
            "reference data",
            "database",
        )
    )


def _conda_package_version(
    executable: str | Path,
    package: str,
) -> str | None:
    """Read a package version from the Conda prefix owning an executable."""
    executable_path = Path(executable).expanduser()
    prefixes: list[Path] = []
    for candidate in (
        executable_path.parent.parent,
        executable_path.resolve().parent.parent,
        Path(os.environ["CONDA_PREFIX"]) if os.environ.get("CONDA_PREFIX") else None,
        Path(sys.prefix),
    ):
        if candidate is None:
            continue
        normalized = candidate.expanduser().resolve()
        if normalized not in prefixes:
            prefixes.append(normalized)
    for prefix in prefixes:
        metadata_directory = prefix / "conda-meta"
        if not metadata_directory.is_dir():
            continue
        for record in sorted(metadata_directory.glob(f"{package}-*.json"), reverse=True):
            try:
                payload = json.loads(record.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError):
                continue
            if str(payload.get("name", "")).lower() != package.lower():
                continue
            version = payload.get("version")
            if version:
                return str(version)
    return None


def software_runtime_issues(
    requirements: Sequence[SoftwareRequirement],
) -> list[SoftwareRuntimeIssue]:
    executables = {requirement.executable for requirement in requirements}
    probes = (
        (
            "barrnap",
            ("--version",),
            ("barrnap", "perl-path-tiny"),
        ),
        (
            "checkm",
            ("--help",),
            ("checkm-genome", "setuptools<82"),
        ),
        (
            "gtdbtk",
            ("--version",),
            ("gtdbtk=2.7.2",),
        ),
    )
    issues: list[SoftwareRuntimeIssue] = []
    for executable, arguments, repair_packages in probes:
        if executable not in executables:
            continue
        path = shutil.which(executable)
        if path is None:
            continue
        probe_environment: dict[str, str] | None = None
        if executable == "gtdbtk":
            configured_data = configured_database_path(
                None,
                "GTDBTK_DATA_PATH",
                "gtdbtk",
                validator=gtdbtk_database_valid,
            )
            if configured_data is not None and gtdbtk_database_valid(
                configured_data
            ):
                probe_environment = os.environ.copy()
                probe_environment["GTDBTK_DATA_PATH"] = str(configured_data)
        try:
            run_options: dict[str, object] = {
                "stdout": subprocess.PIPE,
                "stderr": subprocess.PIPE,
                "text": True,
                "check": False,
                "timeout": 30,
            }
            if probe_environment is not None:
                run_options["env"] = probe_environment
            process = subprocess.run([path, *arguments], **run_options)
        except (OSError, subprocess.TimeoutExpired) as exc:
            issues.append(
                SoftwareRuntimeIssue(
                    executable,
                    f"runtime probe could not start: {exc}",
                    repair_packages,
                )
            )
            continue
        if executable == "gtdbtk" and _gtdbtk_reference_data_error(process):
            # Database validation and interactive path repair belong to the
            # database-checking phase. Reinstalling GTDB-Tk cannot repair an
            # invalid GTDBTK_DATA_PATH value.
            continue
        if executable == "gtdbtk" and process.returncode == 0:
            conda_version = _conda_package_version(path, "gtdbtk")
            version_output = "\n".join(
                part for part in (process.stdout, process.stderr) if part
            )
            match = re.search(
                r"(?i)\bgtdb-?tk\b[^0-9\r\n]*([0-9]+)\.([0-9]+)(?:\.([0-9]+))?",
                version_output,
            )
            parsed_version = (
                ".".join(part or "0" for part in match.groups())
                if match is not None
                else None
            )
            found_version = conda_version or parsed_version
            if found_version == "2.7.2":
                continue
            issues.append(
                SoftwareRuntimeIssue(
                    executable,
                    f"GTDB-Tk {found_version or 'unknown'} is incompatible with the current R232 "
                    "reference package; MetaBAW requires version 2.7.2",
                    repair_packages,
                )
            )
            continue
        if process.returncode == 0 or (
            executable == "checkm" and _checkm_help_exit_is_healthy(process)
        ):
            continue
        diagnostic_output = "\n".join(
            part for part in (process.stderr, process.stdout) if part
        )
        detail = " | ".join(
            line.strip()
            for line in diagnostic_output.splitlines()
            if line.strip()
        )
        if len(detail) > 500:
            detail = detail[:497] + "..."
        if executable == "checkm" and (
            "No module named 'pkg_resources'" in detail
            or 'No module named "pkg_resources"' in detail
        ):
            repair_packages = ("setuptools<82",)
            detail = (
                f"{detail}; CheckM requires pkg_resources, which was removed "
                "from setuptools 82"
            )
        command = " ".join((path, *arguments))
        issues.append(
            SoftwareRuntimeIssue(
                executable,
                (
                    f"`{command}` exited with code "
                    f"{process.returncode}: {detail or 'no diagnostic text'}"
                ),
                repair_packages,
            )
        )
    return issues


def conda_frontend() -> str | None:
    for executable in ("mamba", "micromamba", "conda"):
        if shutil.which(executable):
            return executable
    return None


_TRANSIENT_CONDA_ERRORS = (
    "timeout was reached",
    "operation timed out",
    "download error",
    "connection reset",
    "temporary failure",
    "could not resolve host",
    "failed to connect",
    "http 500",
    "http 502",
    "http 503",
    "http 504",
    "cache file",
    "modified by another program",
    "could not set lock",
    "cannot lock",
    "resource temporarily unavailable",
)

_CORRUPT_CONDA_CACHE_ERRORS = (
    "error when extracting package",
    "extraction failed",
    "invalid utf-8 byte",
    "invalid utf8 byte",
    "incorrect checksum",
    "checksum mismatch",
    "std::bad_alloc",
)


def run_conda_transaction(
    command: Sequence[str],
    *,
    retry_count: int = 2,
    environment_overrides: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a Conda/Mamba transaction with streamed output and network retries."""
    environment = os.environ.copy()
    # Respect explicit user settings while making large biological packages
    # practical on slow or high-latency connections.
    environment.setdefault("CONDA_REMOTE_CONNECT_TIMEOUT_SECS", "60")
    environment.setdefault("CONDA_REMOTE_READ_TIMEOUT_SECS", "300")
    environment.setdefault("CONDA_REMOTE_MAX_RETRIES", "5")
    if environment_overrides:
        environment.update(environment_overrides)
    attempts = retry_count + 1
    last_output = ""
    last_returncode = 1
    cache_cleaned = False
    for attempt in range(1, attempts + 1):
        process = subprocess.Popen(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=0,
            env=environment,
        )
        recent = bytearray()
        assert process.stdout is not None
        read_chunk = getattr(process.stdout, "read1", process.stdout.read)
        while True:
            chunk = read_chunk(4096)
            if not chunk:
                break
            binary_stdout = getattr(sys.stdout, "buffer", None)
            if binary_stdout is not None:
                binary_stdout.write(chunk)
                binary_stdout.flush()
            else:
                sys.stdout.write(chunk.decode("utf-8", errors="replace"))
                sys.stdout.flush()
            recent.extend(chunk)
            if len(recent) > 262_144:
                del recent[:-131_072]
        last_returncode = process.wait()
        last_output = recent.decode("utf-8", errors="replace")
        if last_returncode == 0:
            return subprocess.CompletedProcess(
                list(command), last_returncode, last_output, None
            )
        lowered_output = last_output.lower()
        transient = any(
            marker in lowered_output for marker in _TRANSIENT_CONDA_ERRORS
        )
        corrupt_cache = any(
            marker in lowered_output for marker in _CORRUPT_CONDA_CACHE_ERRORS
        )
        if corrupt_cache and not cache_cleaned:
            clean_command = [
                shutil.which("conda") or str(command[0]),
                "clean",
                "--yes",
                "--tarballs",
            ]
            print(
                "[WARNING] Conda/Mamba found a corrupted package archive or "
                "extraction cache. Removing cached package archives before retry.",
                file=sys.stderr,
                flush=True,
            )
            clean_process = subprocess.run(
                clean_command,
                check=False,
                env=environment,
            )
            cache_cleaned = clean_process.returncode == 0
            if not cache_cleaned:
                print(
                    "[WARNING] Automatic Conda/Mamba cache cleanup failed; "
                    "the installation retry may encounter the same package.",
                    file=sys.stderr,
                    flush=True,
                )
        if (not transient and not corrupt_cache) or attempt == attempts:
            break
        delay = min(5 * attempt, 15)
        if transient:
            detail = "temporary download failure; completed cache entries will be reused"
        else:
            detail = "corrupted package cache; affected packages will be downloaded again"
        print(
            f"[WARNING] Conda/Mamba {detail}. Retrying transaction "
            f"{attempt + 1}/{attempts} in {delay}s.",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(delay)
    return subprocess.CompletedProcess(
        list(command), last_returncode, last_output, None
    )


def conda_environment_selector(environment: str) -> tuple[str, str]:
    value = environment.strip()
    if not value:
        raise ValueError("Isolated environment cannot be empty")
    is_prefix = (
        Path(value).is_absolute()
        or value.startswith((".", "~"))
        or "/" in value
        or "\\" in value
    )
    if is_prefix:
        return "--prefix", str(Path(value).expanduser().resolve())
    return "--name", value


def isolated_environment_config_path() -> Path:
    explicit = os.environ.get("METABAW_CONFIG_FILE")
    if explicit:
        return Path(explicit).expanduser().resolve()
    config_home = os.environ.get("XDG_CONFIG_HOME")
    root = Path(config_home).expanduser() if config_home else Path.home() / ".config"
    return root / "metabaw" / "config.json"


def _runtime_config() -> dict[str, object]:
    path = isolated_environment_config_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def saved_isolated_environment(key: str) -> str | None:
    if key not in ISOLATED_TOOLS:
        raise ValueError(f"Unknown isolated tool: {key}")
    environments = _runtime_config().get("isolated_environments")
    if not isinstance(environments, dict):
        return None
    value = environments.get(key)
    return str(value).strip() if value else None


def configured_isolated_environment(key: str) -> str:
    if key not in ISOLATED_TOOLS:
        raise ValueError(f"Unknown isolated tool: {key}")
    environment_name = f"METABAW_{key.upper()}_ENV"
    configured = (
        os.environ.get(environment_name)
        or saved_isolated_environment(key)
        or ISOLATED_TOOLS[key].default_environment
    )
    return normalize_isolated_environment(key, configured)


def normalize_isolated_environment(key: str, environment: str) -> str:
    """Validate a supported tool and normalize its environment name or prefix."""
    if key not in ISOLATED_TOOLS:
        raise ValueError(f"Unknown isolated tool: {key}")
    return environment.strip()


def save_isolated_environment(key: str, environment: str) -> str:
    if key not in ISOLATED_TOOLS:
        raise ValueError(f"Unknown isolated tool: {key}")
    environment = normalize_isolated_environment(key, environment)
    selector, normalized = conda_environment_selector(environment)
    saved_value = normalized if selector == "--prefix" else environment.strip()
    path = isolated_environment_config_path()
    config = _runtime_config()
    environments = config.get("isolated_environments")
    if not isinstance(environments, dict):
        environments = {}
        config["isolated_environments"] = environments
    environments[key] = saved_value
    config["version"] = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return saved_value


def configured_magscot_directory() -> Path:
    configured = (
        os.environ.get("MAGSCOT_DIR")
        or _runtime_config().get("magscot_directory")
        or "~/.cache/metabaw/MAGScoT"
    )
    return Path(str(configured)).expanduser().resolve()


def save_magscot_directory(directory: Path) -> Path:
    normalized = directory.expanduser().resolve()
    path = isolated_environment_config_path()
    config = _runtime_config()
    config["magscot_directory"] = str(normalized)
    config["version"] = 1
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return normalized


def isolated_run_prefix(
    environment: str,
    frontend: str | None = None,
    prefer_lock_free_runner: bool = False,
    live_output: bool = False,
) -> tuple[str, ...]:
    selected_frontend = frontend or conda_frontend() or "conda"
    if (
        prefer_lock_free_runner
        and Path(selected_frontend).name.lower() in {"mamba", "micromamba"}
    ):
        conda = shutil.which("conda")
        if conda is not None:
            selected_frontend = conda
    frontend_name = Path(selected_frontend).name.lower().removesuffix(".exe")
    output_options = (
        ("--no-capture-output",)
        if live_output and frontend_name == "conda"
        else ()
    )
    return (
        selected_frontend,
        "run",
        *output_options,
        *conda_environment_selector(environment),
    )


def host_cuda_status() -> HostCudaStatus:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return HostCudaStatus(
            None,
            (),
            None,
            "nvidia-smi is not available",
        )
    query = subprocess.run(
        [
            executable,
            "--query-gpu=index,name,memory.total,driver_version,memory.free,uuid",
            "--format=csv,noheader,nounits",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if query.returncode != 0:
        detail = (query.stderr or query.stdout).strip()
        return HostCudaStatus(
            executable,
            (),
            None,
            detail or f"nvidia-smi exited with code {query.returncode}",
        )
    devices: list[CudaDevice] = []
    try:
        for row in csv.reader(query.stdout.splitlines()):
            values = [value.strip() for value in row]
            if len(values) < 4:
                continue
            devices.append(
                CudaDevice(
                    index=values[0],
                    name=values[1],
                    memory_total_mib=max(0, int(float(values[2]))),
                    driver_version=values[3],
                    memory_free_mib=(
                        max(0, int(float(values[4])))
                        if len(values) >= 5 and values[4]
                        else None
                    ),
                    uuid=values[5] if len(values) >= 6 else None,
                )
            )
    except ValueError as exc:
        return HostCudaStatus(
            executable,
            (),
            None,
            f"cannot parse nvidia-smi GPU information: {exc}",
        )
    banner = subprocess.run(
        [executable],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    advertised = None
    if banner.returncode == 0:
        match = re.search(
            r"CUDA(?: UMD)? Version:\s*([0-9.]+)",
            banner.stdout,
        )
        if match:
            advertised = match.group(1)
    if not devices:
        return HostCudaStatus(
            executable,
            (),
            advertised,
            "nvidia-smi reported no NVIDIA GPUs",
        )
    return HostCudaStatus(executable, tuple(devices), advertised)


def cuda_runtime_status(
    label: str,
    python_command: Sequence[str],
) -> CudaRuntimeStatus:
    marker = "METABAW_CUDA_STATUS="
    probe = (
        "from __future__ import print_function\n"
        "import json, os, sys\n"
        "result = {"
        "'python': '%s.%s' % (sys.version_info[0], sys.version_info[1]), "
        "'torch': None, 'torch_cuda': None, "
        "'visible': os.environ.get('CUDA_VISIBLE_DEVICES'), "
        "'available': False, 'count': 0, 'devices': [], "
        "'allocation': False, 'error': None}\n"
        "try:\n"
        "    import torch\n"
        "    result['torch'] = str(torch.__version__)\n"
        "    result['torch_cuda'] = str(torch.version.cuda) if torch.version.cuda else None\n"
        "    result['available'] = bool(torch.cuda.is_available())\n"
        "    if result['available']:\n"
        "        if not hasattr(torch.cuda, 'set_per_process_memory_fraction'):\n"
        "            raise RuntimeError('PyTorch lacks allocator memory-budget control')\n"
        "        result['count'] = int(torch.cuda.device_count())\n"
        "        result['devices'] = [str(torch.cuda.get_device_name(i)) "
        "for i in range(result['count'])]\n"
        "        for i in range(result['count']):\n"
        "            test = torch.zeros(1, device='cuda:%d' % i)\n"
        "            torch.cuda.synchronize(i)\n"
        "            del test\n"
        "        result['allocation'] = True\n"
        "    else:\n"
        "        result['error'] = 'torch.cuda.is_available() returned False'\n"
        "except Exception as exc:\n"
        "    result['error'] = '%s: %s' % (type(exc).__name__, exc)\n"
        f"print({marker!r} + json.dumps(result, sort_keys=True))\n"
    )
    try:
        process = subprocess.run(
            [*python_command, "-c", probe],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except OSError as exc:
        return CudaRuntimeStatus(
            label,
            None,
            None,
            None,
            os.environ.get("CUDA_VISIBLE_DEVICES"),
            0,
            (),
            False,
            f"cannot start CUDA probe: {exc}",
        )
    payload: dict[str, object] | None = None
    for line in reversed(process.stdout.splitlines()):
        if not line.startswith(marker):
            continue
        try:
            candidate = json.loads(line[len(marker) :])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict):
            payload = candidate
            break
    if payload is None:
        detail = (process.stderr or process.stdout).strip()
        if detail:
            detail = detail.splitlines()[-1]
        return CudaRuntimeStatus(
            label,
            None,
            None,
            None,
            os.environ.get("CUDA_VISIBLE_DEVICES"),
            0,
            (),
            False,
            detail or f"CUDA probe exited with code {process.returncode}",
        )
    error_value = payload.get("error")
    error = str(error_value) if error_value else None
    if process.returncode != 0 and error is None:
        error = f"CUDA probe exited with code {process.returncode}"
    devices_value = payload.get("devices")
    devices = (
        tuple(str(value) for value in devices_value)
        if isinstance(devices_value, list)
        else ()
    )
    try:
        device_count = int(payload.get("count", 0))
    except (TypeError, ValueError):
        device_count = 0
    return CudaRuntimeStatus(
        label=label,
        python_version=str(payload.get("python") or "") or None,
        torch_version=str(payload.get("torch") or "") or None,
        torch_cuda_version=str(payload.get("torch_cuda") or "") or None,
        cuda_visible_devices=(
            str(payload["visible"]) if payload.get("visible") is not None else None
        ),
        device_count=device_count,
        devices=devices,
        allocation_test=bool(payload.get("allocation")),
        error=error,
    )


def cupy_runtime_status(label: str, python_command: Sequence[str]) -> CupyRuntimeStatus:
    """Validate MetaDecoder's own CuPy path, without requiring PyTorch."""
    marker = "METABAW_CUPY_STATUS="
    probe = (
        "import json, os, sys\n"
        "r = {'python': '%s.%s' % sys.version_info[:2], 'cupy': None, 'cuda': None, "
        "'visible': os.environ.get('CUDA_VISIBLE_DEVICES'), 'count': 0, "
        "'devices': [], 'allocation': False, 'error': None}\n"
        "try:\n"
        "    import cupy as cp\n"
        "    r['cupy'] = str(cp.__version__)\n"
        "    r['cuda_build'] = str(cp.cuda.runtime.runtimeGetVersion())\n"
        "    r['cuda'] = str(cp.cuda.get_local_runtime_version())\n"
        "    target = os.environ.get('METABAW_CUPY_CUDA_VERSION')\n"
        "    if target:\n"
        "        major, minor = map(int, target.split('.'))\n"
        "        if int(r['cuda']) != major * 1000 + minor * 10:\n"
        "            raise RuntimeError('CuPy runtime does not match configured nvidia-smi CUDA ' + target)\n"
        "    r['count'] = int(cp.cuda.runtime.getDeviceCount())\n"
        "    if not r['count']: raise RuntimeError('CuPy sees no CUDA devices')\n"
        "    pool = cp.get_default_memory_pool()\n"
        "    if not hasattr(pool, 'set_limit'): raise RuntimeError('CuPy lacks memory-pool limit control')\n"
        "    for i in range(r['count']):\n"
        "        with cp.cuda.Device(i):\n"
        "            pool.set_limit(size=16 * 1024**2)\n"
        "            test = cp.ones(2)\n"
        "            if float((test + 1).sum().item()) != 4.0: raise RuntimeError('CuPy kernel test failed')\n"
        "            factor = cp.linalg.cholesky(cp.eye(2))\n"
        "            factor.get()\n"
        "            cp.cuda.runtime.deviceSynchronize()\n"
        "            r['devices'].append(str(cp.cuda.runtime.getDeviceProperties(i)['name']))\n"
        "            del test, factor\n"
        "            pool.free_all_blocks()\n"
        "    r['allocation'] = True\n"
        "except Exception as exc:\n"
        "    r['error'] = '%s: %s' % (type(exc).__name__, exc)\n"
        f"print({marker!r} + json.dumps(r, sort_keys=True))\n"
    )
    try:
        result = subprocess.run([*python_command, "-c", probe], capture_output=True,
                                text=True, check=False, timeout=60,
                                env={**os.environ, **cupy_runtime_environment()})
        payload = next(json.loads(line[len(marker):]) for line in reversed(result.stdout.splitlines())
                       if line.startswith(marker))
        error = payload.get("error") or (f"CuPy probe exited with code {result.returncode}"
                                         if result.returncode else None)
        return CupyRuntimeStatus(label, payload.get("python"), payload.get("cupy"),
                                 payload.get("cuda"), payload.get("visible"),
                                 int(payload.get("count", 0)), tuple(payload.get("devices", ())),
                                 bool(payload.get("allocation")), error, payload.get("cuda_build"))
    except (OSError, RuntimeError, subprocess.SubprocessError, StopIteration, ValueError, TypeError) as exc:
        return CupyRuntimeStatus(label, None, None, None, os.environ.get("CUDA_VISIBLE_DEVICES"),
                                 0, (), False, f"Cannot validate CuPy CUDA runtime: {exc}")


def print_cupy_runtime_status(status: CupyRuntimeStatus) -> None:
    state = "CUDA OK" if status.available else "CUDA MISSING"
    print(f"{status_tag(state)} {status.label}: Python {status.python_version or 'unknown'}; "
          f"CuPy {status.cupy_version or 'not installed'}; CUDA runtime {status.cuda_version or 'none'}; "
          f"CuPy build CUDA {status.compiled_cuda_version or 'unknown'}; "
          f"visible devices={status.device_count}; "
          f"{status.error or 'allocation, kernel, linear-algebra and pool-limit tests passed'}", flush=True)


def print_host_cuda_status(status: HostCudaStatus) -> None:
    if not status.available:
        print(
            f"{status_tag('CUDA MISSING')} NVIDIA driver: {status.error or 'unavailable'}",
            flush=True,
        )
        return
    drivers = ", ".join(sorted({device.driver_version for device in status.devices}))
    cuda = status.advertised_cuda_version or "unknown"
    devices = "; ".join(
        (
            f"{device.index}:{device.name} "
            f"({device.memory_total_mib} MiB total, "
            f"{device.memory_free_mib} MiB free)"
            if device.memory_free_mib is not None
            else f"{device.index}:{device.name} ({device.memory_total_mib} MiB)"
        )
        for device in status.devices
    )
    print(
        f"{status_tag('CUDA OK')} NVIDIA driver {drivers}; advertised CUDA {cuda}; GPUs: {devices}",
        flush=True,
    )


def print_cuda_runtime_status(status: CudaRuntimeStatus) -> None:
    state = "CUDA OK" if status.available else "CUDA MISSING"
    visible = (
        status.cuda_visible_devices
        if status.cuda_visible_devices is not None
        else "not set"
    )
    detail = (
        f"Python {status.python_version or 'unknown'}; "
        f"PyTorch {status.torch_version or 'not installed'}; "
        f"PyTorch CUDA {status.torch_cuda_version or 'none'}; "
        f"CUDA_VISIBLE_DEVICES={visible}"
    )
    if status.available:
        detail += f"; devices={', '.join(status.devices)}; allocation test passed"
    else:
        detail += f"; {status.error or 'CUDA allocation test failed'}"
    print(f"{status_tag(state)} {status.label}: {detail}", flush=True)


def isolated_environment_status(
    spec: IsolatedToolSpec,
    environment: str,
    frontend: str | None = None,
) -> IsolatedEnvironmentStatus:
    selected_frontend = frontend or conda_frontend()
    if selected_frontend is None:
        return IsolatedEnvironmentStatus(
            spec,
            environment,
            None,
            None,
            None,
            "mamba, micromamba, or conda is not available",
        )
    probe = (
        "import json, sys\n"
        "try:\n"
        "    from shutil import which\n"
        "except ImportError:\n"
        "    from distutils.spawn import find_executable as which\n"
        "runtime_error = None\n"
        f"if {spec.key!r} == 'lorbin':\n"
        "    try:\n"
        "        import torch\n"
        "        torch.zeros(1)\n"
        "        __import__('lorbin.lorbin')\n"
        "    except BaseException as exc:\n"
        "        runtime_error = '%s: %s' % (type(exc).__name__, exc)\n"
        "print(json.dumps({"
        "'python': '%s.%s' % (sys.version_info[0], sys.version_info[1]), "
        f"'executable': which({spec.executable!r}), "
        "'runtime_error': runtime_error"
        "}))\n"
    )
    command = [
        *isolated_run_prefix(environment, selected_frontend),
        "python",
        "-c",
        probe,
    ]
    process = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
        env={**os.environ, **ISOLATED_RUNTIME_PROBE_ENV},
    )
    if process.returncode != 0:
        detail = (process.stderr or process.stdout).strip()
        if detail:
            detail = detail.splitlines()[-1]
        return IsolatedEnvironmentStatus(
            spec,
            environment,
            selected_frontend,
            None,
            None,
            detail or f"environment probe exited with code {process.returncode}",
        )
    payload: dict[str, object] | None = None
    for line in reversed(process.stdout.splitlines()):
        try:
            candidate = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and "python" in candidate:
            payload = candidate
            break
    if payload is None:
        return IsolatedEnvironmentStatus(
            spec,
            environment,
            selected_frontend,
            None,
            None,
            "environment probe returned no readable status",
        )
    python_version = str(payload["python"])
    executable_value = payload.get("executable")
    executable = str(executable_value) if executable_value else None
    runtime_error_value = payload.get("runtime_error")
    runtime_error = str(runtime_error_value) if runtime_error_value else None
    error = None
    if python_version != spec.python_version:
        error = (
            f"Python {python_version} is incompatible; "
            f"{spec.display_name} requires Python {spec.python_version}"
        )
    elif executable is None:
        error = f"{spec.executable} is not installed"
    elif runtime_error is not None:
        error = (
            f"{spec.display_name} runtime import failed: {runtime_error}. "
            "The environment must successfully import PyTorch and LorBin."
        )
    return IsolatedEnvironmentStatus(
        spec,
        environment,
        selected_frontend,
        python_version,
        executable,
        error,
    )


def _lorbin_replacement_environment(
    spec: IsolatedToolSpec,
    environment: str,
    frontend: str,
) -> IsolatedEnvironmentStatus:
    """Find a healthy or unused sibling without modifying an existing env."""
    command = [frontend, "env", "list", "--json"]
    try:
        listed = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, check=False, timeout=30,
        )
        inventory = json.loads(listed.stdout) if listed.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "Cannot safely select a replacement LorBin environment: "
            f"environment inventory failed ({exc}). The original environment was preserved."
        ) from exc
    prefixes = inventory.get("envs") if isinstance(inventory, dict) else None
    if not isinstance(prefixes, list) or not all(isinstance(item, str) and item.strip() for item in prefixes):
        raise RuntimeError(
            "Cannot safely select a replacement LorBin environment: "
            f"{' '.join(command)} returned no valid environment inventory. "
            "The original environment was preserved."
        )
    known_prefixes = {Path(item).expanduser().resolve() for item in prefixes}
    known_names = {os.path.normcase(path.name) for path in known_prefixes}
    selector, original = conda_environment_selector(environment)
    version_suffix = f"-py{spec.python_version.replace('.', '')}"
    stem = original + ("-repair" if original.endswith(version_suffix) else version_suffix)
    for number in range(1, 101):
        candidate = stem if number == 1 else f"{stem}-{number}"
        if selector == "--prefix":
            path = Path(candidate)
            occupied = path.exists() or path.is_symlink() or path in known_prefixes
        else:
            occupied = os.path.normcase(candidate) in known_names
        if occupied:
            status = isolated_environment_status(spec, candidate, frontend)
            if status.available:
                return status
            # Even an environment with no usable Python may contain user data.
            # Do not install into it, downgrade it or remove it automatically.
            continue
        return IsolatedEnvironmentStatus(
            spec, candidate, frontend, None, None, "environment not found",
        )
    raise RuntimeError(
        "Cannot find a healthy or unused replacement LorBin environment after "
        "100 candidates. Select a new name or prefix with --lorbin-env."
    )


def install_isolated_environment(
    spec: IsolatedToolSpec,
    environment: str,
    frontend: str | None = None,
) -> IsolatedEnvironmentStatus:
    selected_frontend = frontend or conda_frontend()
    if selected_frontend is None:
        raise RuntimeError(
            f"Cannot install {spec.display_name} because mamba, micromamba, "
            "or conda is not available"
        )
    status = isolated_environment_status(spec, environment, selected_frontend)
    if (
        status.python_version is not None
        and status.python_version != spec.python_version
    ):
        if spec.key != "lorbin":
            raise RuntimeError(
                f"{spec.display_name} environment {environment!r} uses Python "
                f"{status.python_version}. Select a different environment with "
                f"{spec.option}; MetaBAW will not change Python in an existing environment."
            )
        replacement = _lorbin_replacement_environment(spec, environment, selected_frontend)
        print(
            f"[ENV SWITCH] LorBin environment {environment!r} uses Python "
            f"{status.python_version}, but requires Python {spec.python_version}; "
            f"preserving the original environment and using {replacement.environment!r}.",
            flush=True,
        )
        environment = replacement.environment
        status = replacement
        if status.available:
            return status
    managed_environment = (
        environment == spec.default_environment
        or Path(environment.rstrip("/\\")).name == spec.default_environment
    )
    recreate_broken_lorbin = (
        spec.key == "lorbin"
        and managed_environment
        and status.error is not None
        and "iJIT_NotifyEvent" in status.error
    )
    if recreate_broken_lorbin:
        remove_command = [
            selected_frontend,
            "env",
            "remove",
            "--yes",
            *conda_environment_selector(environment),
        ]
        print(
            f"Recreating MetaBAW-managed LorBin environment after a broken "
            f"PyTorch/MKL runtime was detected: {environment}",
            flush=True,
        )
        removed = subprocess.run(remove_command, check=False)
        if removed.returncode != 0:
            raise RuntimeError(
                f"Could not remove the broken LorBin environment (exit code "
                f"{removed.returncode}): {' '.join(remove_command)}"
            )
    action = (
        "create"
        if recreate_broken_lorbin or status.python_version is None
        else "install"
    )
    channel_arguments = [
        argument
        for channel in spec.channels
        for argument in ("--channel", channel)
    ]
    command = [
        selected_frontend,
        action,
        "--yes",
        *conda_environment_selector(environment),
        *channel_arguments,
        f"python={spec.python_version}",
        *(spec.conda_packages or (spec.package,)),
    ]
    print(
        f"Installing {spec.display_name} in isolated Python {spec.python_version} environment: "
        f"{environment}",
        flush=True,
    )
    # Some Mamba frontends reject the command-line form
    # ``--channel-priority flexible`` and then misparse package specifications
    # as unknown arguments. The environment setting is transaction-local and
    # compatible with Conda's Mamba-backed frontend.
    process = run_conda_transaction(
        command,
        environment_overrides={"CONDA_CHANNEL_PRIORITY": "flexible"},
    )
    parser_rejected = (
        process.returncode == 2
        and "unrecognized arguments" in (process.stdout or "").lower()
    )
    frontend_name = Path(selected_frontend).name.lower().removesuffix(".exe")
    conda_fallback = shutil.which("conda")
    if parser_rejected and frontend_name in {"mamba", "micromamba"} and conda_fallback:
        command = [conda_fallback, *command[1:]]
        print(
            f"[WARNING] {selected_frontend} rejected standard environment package "
            "arguments; retrying the same transaction with Conda.",
            file=sys.stderr,
            flush=True,
        )
        process = run_conda_transaction(
            command,
            environment_overrides={"CONDA_CHANNEL_PRIORITY": "flexible"},
        )
    if process.returncode != 0:
        raise RuntimeError(
            f"{spec.display_name} environment installation failed with exit code "
            f"{process.returncode}: {' '.join(command)}"
        )
    if spec.pip_source:
        pip_command = [
            *isolated_run_prefix(environment, selected_frontend),
            "python",
            "-m",
            "pip",
            "install",
            "--no-deps",
            spec.pip_source,
        ]
        print(
            f"Installing {spec.display_name} from its pinned official source",
            flush=True,
        )
        pip_process = subprocess.run(pip_command, check=False)
        if pip_process.returncode != 0:
            raise RuntimeError(
                f"{spec.display_name} source installation failed with exit code "
                f"{pip_process.returncode}: {' '.join(pip_command)}"
            )
    installed = isolated_environment_status(spec, environment, selected_frontend)
    if not installed.available:
        raise RuntimeError(
            f"{spec.display_name} environment {environment!r} is not usable after installation: "
            f"{installed.error or 'unknown error'}"
        )
    return installed


def install_isolated_cuda_runtime(
    key: str,
    environment: str,
    frontend: str | None = None,
    *,
    pip_only: bool = False,
) -> bool:
    """Install an isolated CUDA runtime; return whether pip fallback was used."""
    if key not in ISOLATED_CUDA_PACKAGES:
        raise RuntimeError(f"No isolated CUDA repair profile is available for {key}")
    selected_frontend = frontend or conda_frontend()
    if selected_frontend is None:
        raise RuntimeError(
            "Cannot install CUDA-enabled PyTorch because mamba, micromamba, "
            "or conda is not available"
        )
    process: subprocess.CompletedProcess[str] | None = None
    command: list[str] = []
    if not pip_only:
        channels, packages = ISOLATED_CUDA_PACKAGES[key]
        channel_arguments = [
            argument
            for channel in channels
            for argument in ("--channel", channel)
        ]
        command = [
            selected_frontend,
            "install",
            "--yes",
            *conda_environment_selector(environment),
            *channel_arguments,
            *packages,
        ]
        print(
            f"Installing the CUDA-enabled PyTorch profile for "
            f"{ISOLATED_TOOLS[key].display_name}: {environment}",
            flush=True,
        )
        process = run_conda_transaction(
            command,
            environment_overrides={
                "CONDA_CHANNEL_PRIORITY": "strict" if key == "comebin" else "flexible"
            },
        )
    use_pip = pip_only or (process is not None and process.returncode != 0)
    if use_pip:
        fallback = ISOLATED_CUDA_PIP_FALLBACKS.get(key)
        if fallback is None:
            raise RuntimeError(
                f"No official pip CUDA fallback is available for "
                f"{ISOLATED_TOOLS[key].display_name}"
            )
        requirement, index_option, index_url = fallback
        pip_command = [
            *isolated_run_prefix(environment, selected_frontend),
            "python",
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--ignore-installed",
            "--no-cache-dir",
            requirement,
            index_option,
            index_url,
        ]
        reason = (
            "Conda completed but its runtime still failed the CUDA allocation test"
            if pip_only
            else "Conda could not solve the CUDA profile"
        )
        print(
            f"[WARNING] {reason}; overlay-installing the official PyTorch CUDA "
            "wheel without uninstalling the existing package.",
            flush=True,
        )
        pip_process = subprocess.run(pip_command, check=False)
        if pip_process.returncode != 0:
            raise RuntimeError(
                "CUDA-enabled PyTorch repair failed after Conda and pip; "
                f"pip exit code {pip_process.returncode}: {' '.join(pip_command)}"
            )
    return use_pip


def install_main_cuda_runtime(
    frontend: str | None = None,
    *,
    pip_only: bool = False,
) -> bool:
    """Install main-environment CUDA PyTorch; return whether pip was used."""
    selected_frontend = frontend or conda_frontend()
    if selected_frontend is None and not pip_only:
        raise RuntimeError(
            "Cannot install CUDA-enabled PyTorch because mamba, micromamba, "
            "or conda is not available"
        )
    prefix = Path(sys.prefix).resolve()
    if not (prefix / "conda-meta").is_dir():
        raise RuntimeError(
            f"The MetaBAW Python environment is not a Conda environment: {prefix}. "
            "Install a CUDA-enabled PyTorch build in this environment manually."
        )
    process: subprocess.CompletedProcess[str] | None = None
    command: list[str] = []
    if not pip_only:
        command = [
            selected_frontend,
            "install",
            "--yes",
            "--prefix",
            str(prefix),
            "--channel",
            "pytorch",
            "--channel",
            "nvidia",
            "pytorch==2.5.1",
            "torchvision==0.20.1",
            "torchaudio==2.5.1",
            "pytorch-cuda=11.8",
        ]
        print(
            f"Installing CUDA-enabled PyTorch in the MetaBAW environment: {prefix}",
            flush=True,
        )
        process = run_conda_transaction(
            command,
            environment_overrides={"CONDA_CHANNEL_PRIORITY": "strict"},
        )
    use_pip = pip_only or (process is not None and process.returncode != 0)
    if use_pip:
        pip_command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--ignore-installed",
            "--no-cache-dir",
            "torch==2.5.1",
            "torchvision==0.20.1",
            "torchaudio==2.5.1",
            "--index-url",
            "https://download.pytorch.org/whl/cu118",
        ]
        reason = (
            "Conda completed but its runtime still failed the CUDA allocation test"
            if pip_only
            else "Conda could not solve the CUDA profile"
        )
        print(
            f"[WARNING] {reason}; overlay-installing the official PyTorch "
            "CUDA 11.8 wheel without uninstalling the existing package.",
            flush=True,
        )
        pip_process = subprocess.run(pip_command, check=False)
        if pip_process.returncode != 0:
            raise RuntimeError(
                "CUDA-enabled PyTorch repair failed after Conda and pip; "
                f"pip exit code {pip_process.returncode}: {' '.join(pip_command)}"
            )
    return use_pip


def main_cupy_install_command(
    advertised_cuda: str | None,
    preferred_cuda: str | None = None,
) -> CupyInstallPlan:
    """Match nvidia-smi exactly, with CUDA libraries separate from PyTorch's."""
    frontend = shutil.which("conda")
    prefix = Path(sys.prefix).resolve()
    if frontend is None or not (prefix / "conda-meta").is_dir():
        raise RuntimeError(
            f"CuPy repair requires Conda and a valid MetaBAW Conda prefix: {prefix}. "
            "Install a single compatible CuPy distribution in this environment manually."
        )
    existing = sorted({
        name for distribution in metadata.distributions()
        if (name := (distribution.metadata.get("Name") or "").lower().replace("_", "-"))
        and (name in {"cupy", "cupy-core"} or name.startswith("cupy-cuda"))
    })
    if existing:
        raise RuntimeError(
            "CuPy distributions are already registered in the MetaBAW environment: "
            + ", ".join(existing)
            + ". Repair them with their original package manager; do not overlay another CuPy package."
        )

    supported = ((11, 2), (11, 3), (11, 4), (11, 5), (11, 6), (11, 7), (11, 8),
                 (12, 0), (12, 1), (12, 2), (12, 3), (12, 4), (12, 5), (12, 6),
                 (12, 8), (12, 9), (13, 0))
    match = re.fullmatch(r"(\d+)(?:\.(\d+))?(?:\.\d+)?", advertised_cuda or "")
    selected = (int(match[1]), int(match[2] or 0)) if match else None
    if selected not in supported:
        raise RuntimeError(
            f"Cannot select a supported CuPy CUDA runtime matching nvidia-smi {advertised_cuda!r}; "
            "no older CUDA version will be substituted."
        )
    cuda = f"{selected[0]}.{selected[1]}"
    # This library-only prefix has no Python and no main-environment
    # pytorch-cuda constraints. CuPy itself still uses MetaBAW's interpreter.
    runtime_prefix = runtime_root() / cuda
    action = "install" if (runtime_prefix / "conda-meta").is_dir() else "create"
    libraries = ((f"cudatoolkit={cuda}",) if selected[0] == 11 else (
        f"cuda-version={cuda}", "cuda-cudart", "cuda-cudart-dev", "cuda-nvrtc",
        "cuda-nvrtc-dev", "libcublas", "libcusolver", "libcusparse", "libcurand",
        "libcufft", "libnvjitlink",
    ))
    runtime_command = (
        frontend, action, "--yes", "--prefix", str(runtime_prefix),
        *(() if action == "install" else ("--no-default-packages",)),
        "--override-channels", "--channel", "conda-forge", *libraries,
    )
    # --no-deps keeps existing numerical packages, including NumPy, unchanged.
    # The pinned wheel requires NumPy >=1.22,<2.6 and fastrlock >=0.5.
    try:
        numpy_version = metadata.version("numpy")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError("Install MetaDecoder's NumPy dependency in the main environment first") from exc
    numbers = re.match(r"(\d+)\.(\d+)", numpy_version)
    if not numbers or not (1, 22) <= (int(numbers[1]), int(numbers[2])) < (2, 6):
        raise RuntimeError(f"CuPy 13.6 requires NumPy >=1.22,<2.6; found {numpy_version}")
    requirements = [f"cupy-cuda{selected[0]}x==13.6.0"]
    try:
        lock_version = metadata.version("fastrlock")
    except metadata.PackageNotFoundError:
        requirements.append("fastrlock>=0.5")
    else:
        numbers = re.match(r"(\d+)\.(\d+)", lock_version)
        if not numbers or (int(numbers[1]), int(numbers[2])) < (0, 5):
            raise RuntimeError(f"CuPy requires fastrlock >=0.5; found {lock_version}")
    pip_command = (sys.executable, "-m", "pip", "install", "--no-deps",
                   "--only-binary=:all:", *requirements)
    return CupyInstallPlan(cuda, runtime_prefix, (runtime_command, pip_command))


def install_main_cupy_runtime(plan: CupyInstallPlan) -> None:
    """Install library-only CUDA and main-Python CuPy, then publish routing."""
    process = run_conda_transaction(
        plan.commands[0], environment_overrides={"CONDA_CHANNEL_PRIORITY": "strict"},
    )
    if process.returncode != 0:
        raise RuntimeError(
            f"CuPy CUDA {plan.cuda_version} libraries failed to install (exit code {process.returncode}); "
            "the MetaBAW environment was not changed."
        )
    process = subprocess.run(plan.commands[1], check=False, env=pip_install_environment())
    if process.returncode != 0:
        raise RuntimeError(f"Main-environment CuPy wheel installation failed (exit code {process.returncode})")
    save_cupy_runtime(plan)


def print_isolated_status(status: IsolatedEnvironmentStatus) -> None:
    if status.available:
        state = "OK"
    elif status.python_version is not None or status.executable is not None:
        state = "BROKEN"
    else:
        state = "MISSING"
    python_version = status.python_version or "unknown"
    detail = status.error or status.executable or f"{status.spec.executable} not found"
    print(
        f"{status_tag(state)} {status.spec.display_name} environment {status.environment}; "
        f"Python {python_version}; {detail}",
        flush=True,
    )


def comebin_run_prefix(
    environment: str,
    frontend: str | None = None,
) -> tuple[str, ...]:
    return isolated_run_prefix(environment, frontend)


def comebin_environment_status(
    environment: str,
    frontend: str | None = None,
) -> IsolatedEnvironmentStatus:
    return isolated_environment_status(ISOLATED_TOOLS["comebin"], environment, frontend)


def install_comebin_environment(
    environment: str,
    frontend: str | None = None,
) -> IsolatedEnvironmentStatus:
    return install_isolated_environment(ISOLATED_TOOLS["comebin"], environment, frontend)


def print_comebin_status(status: IsolatedEnvironmentStatus) -> None:
    print_isolated_status(status)


def _minibwa_compiler() -> str | None:
    for executable in ("gcc", "cc", "clang"):
        candidate = shutil.which(executable)
        if candidate:
            return candidate
    bin_directory = Path(sys.prefix) / "bin"
    for pattern in ("*-conda-linux-gnu-cc", "*-conda-linux-gnu-gcc"):
        candidates = sorted(bin_directory.glob(pattern))
        if candidates:
            return str(candidates[0])
    return None


def _install_minibwa_build_dependencies() -> None:
    packages: list[str] = []
    if shutil.which("git") is None:
        packages.append("git")
    if shutil.which("make") is None:
        packages.append("make")
    if _minibwa_compiler() is None:
        packages.append("c-compiler")
    zlib_headers = (
        Path(sys.prefix) / "include" / "zlib.h",
        Path("/usr/include/zlib.h"),
        Path("/usr/local/include/zlib.h"),
    )
    if not any(path.is_file() for path in zlib_headers):
        packages.append("zlib")
    if not packages:
        return
    frontend = conda_frontend()
    if frontend is None:
        raise RuntimeError(
            "Minibwa source compilation requires Git, Make, a C compiler, "
            "and zlib development files. Mamba, Micromamba, or Conda is "
            "required to install the missing build dependencies."
        )
    command = [
        frontend,
        "install",
        "--yes",
        "--prefix",
        str(Path(sys.prefix).resolve()),
        "--channel",
        "conda-forge",
        *packages,
    ]
    print(
        "Installing minibwa build dependencies: " + " ".join(packages),
        flush=True,
    )
    process = run_conda_transaction(command)
    if process.returncode != 0:
        raise RuntimeError(
            "Minibwa build dependency installation failed with exit code "
            f"{process.returncode}: {' '.join(command)}"
        )


def install_minibwa() -> Path:
    """Build the pinned official minibwa source and install its binary."""
    _install_minibwa_build_dependencies()
    git = shutil.which("git")
    make = shutil.which("make")
    compiler = _minibwa_compiler()
    if git is None or make is None or compiler is None:
        raise RuntimeError(
            "Minibwa build tools remain unavailable after dependency "
            "installation; required executables are git, make, and a C compiler"
        )
    source = _minibwa_source_directory()
    if source.exists() and not (source / ".git").is_dir():
        raise RuntimeError(
            f"Minibwa source cache exists but is not a Git checkout: {source}. "
            "Move or remove that directory, then rerun `metabaw check`."
        )
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        clone = subprocess.run(
            [git, "clone", MINIBWA_SOURCE_URL, str(source)],
            check=False,
        )
        if clone.returncode != 0:
            raise RuntimeError(
                f"Could not clone official minibwa source (exit code "
                f"{clone.returncode}): {MINIBWA_SOURCE_URL}"
            )
        checkout = subprocess.run(
            [git, "-C", str(source), "checkout", "--detach", MINIBWA_SOURCE_REVISION],
            check=False,
        )
        if checkout.returncode != 0:
            raise RuntimeError(
                "Could not select the pinned minibwa source revision "
                f"{MINIBWA_SOURCE_REVISION}"
            )
    revision = subprocess.run(
        [git, "-C", str(source), "rev-parse", "HEAD"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if revision.returncode != 0 or revision.stdout.strip() != MINIBWA_SOURCE_REVISION:
        raise RuntimeError(
            f"Minibwa source cache is not at the required revision "
            f"{MINIBWA_SOURCE_REVISION}: {source}"
        )
    build_environment = os.environ.copy()
    include_directory = Path(sys.prefix) / "include"
    library_directory = Path(sys.prefix) / "lib"
    build_environment["CPPFLAGS"] = (
        f"-I{include_directory} " + build_environment.get("CPPFLAGS", "")
    ).strip()
    build_environment["LDFLAGS"] = (
        f"-L{library_directory} " + build_environment.get("LDFLAGS", "")
    ).strip()
    print(
        f"Building minibwa revision {MINIBWA_SOURCE_REVISION[:12]} from "
        f"{MINIBWA_SOURCE_URL}",
        flush=True,
    )
    build = subprocess.run(
        [make, "-C", str(source), f"CC={compiler}"],
        check=False,
        env=build_environment,
    )
    binary = source / "minibwa"
    if build.returncode != 0 or not binary.is_file():
        raise RuntimeError(
            f"Minibwa source build failed with exit code {build.returncode}: {source}"
        )
    destination_directory = (
        Path(sys.prefix) / "Scripts" if os.name == "nt" else Path(sys.prefix) / "bin"
    )
    destination_directory.mkdir(parents=True, exist_ok=True)
    destination = destination_directory / ("minibwa.exe" if os.name == "nt" else "minibwa")
    shutil.copy2(binary, destination)
    destination.chmod(destination.stat().st_mode | 0o111)
    path_entries = [
        Path(path) for path in os.environ.get("PATH", "").split(os.pathsep) if path
    ]
    if destination.parent not in path_entries:
        os.environ["PATH"] = (
            str(destination.parent)
            + os.pathsep
            + os.environ.get("PATH", "")
        )
    return destination


def install_software(
    requirements: Sequence[SoftwareRequirement],
    include_magscot_r_packages: bool = False,
) -> list[SoftwareRequirement]:
    missing = missing_software(requirements)
    profiles = required_r_package_profiles(requirements)
    if include_magscot_r_packages and "MAGScoT" not in profiles:
        profiles.append("MAGScoT")
    missing_r = {
        profile: missing_r_packages(profile)
        for profile in profiles
    }
    runtime_issues = software_runtime_issues(requirements)
    if (
        not missing
        and not any(missing_r.values())
        and not runtime_issues
    ):
        return []
    pip_requirements = [
        requirement
        for requirement in missing
        if requirement.executable in PIP_INSTALL_URLS
    ]
    source_requirements = [
        requirement
        for requirement in missing
        if requirement.executable in SOURCE_INSTALL_EXECUTABLES
    ]
    conda_requirements = [
        requirement
        for requirement in missing
        if requirement.executable not in PIP_INSTALL_URLS
        and requirement.executable not in SOURCE_INSTALL_EXECUTABLES
    ]
    packages = list(
        dict.fromkeys(requirement.package for requirement in conda_requirements)
    )
    for issue in runtime_issues:
        packages.extend(
            package
            for package in issue.repair_packages
            if package not in packages
        )
    installation_failures: list[str] = []
    if packages:
        frontend = conda_frontend()
        if frontend is None:
            installation_failures.extend(packages)
            print(
                "[INSTALL FAILED] Conda packages cannot be installed because "
                "mamba, micromamba, or conda is not available. Other installation "
                "methods will still be attempted.",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                f"Installing {len(packages)} missing software package(s) "
                "as independent restartable transactions.",
                flush=True,
            )
            for number, package in enumerate(packages, start=1):
                base_package = re.split(r"[<>=!~\s]", package, maxsplit=1)[0]
                bioconda_first = base_package in BIOCONDA_FIRST_PACKAGES
                channels = (
                    ("bioconda", "conda-forge")
                    if bioconda_first
                    else ("conda-forge", "bioconda")
                )
                command = [frontend, "install", "-y" if bioconda_first else "--yes"]
                channel_option = "-c" if bioconda_first else "--channel"
                for channel in channels:
                    command.extend((channel_option, channel))
                command.append(package)
                print(
                    f"[INSTALL {number}/{len(packages)}] {package}",
                    flush=True,
                )
                try:
                    process = run_conda_transaction(command)
                except Exception as exc:
                    installation_failures.append(package)
                    print(
                        f"[INSTALL FAILED {number}/{len(packages)}] {package}: {exc}. "
                        "Continuing with the remaining software.",
                        file=sys.stderr,
                        flush=True,
                    )
                    continue
                if process.returncode != 0:
                    installation_failures.append(package)
                    print(
                        f"[INSTALL FAILED {number}/{len(packages)}] {package}: "
                        f"exit code {process.returncode}. Continuing with the "
                        "remaining software.",
                        file=sys.stderr,
                        flush=True,
                    )
    for requirement in pip_requirements:
        print(
            f"Installing {requirement.executable} from its upstream Python package",
            flush=True,
        )
        try:
            source = (str(prepare_metadecoder_wheel())
                      if requirement.executable == "metadecoder"
                      else PIP_INSTALL_URLS[requirement.executable])
            environment = pip_install_environment()
            command = [sys.executable, "-m", "pip", "install", source]
            process = subprocess.run(command, check=False, env=environment)
            if (process.returncode == 0 and requirement.executable == "metadecoder"
                    and metadecoder_executable() is None):
                # Restore a missing/stale console script without replacing the
                # numerical dependencies that the preceding install validated.
                process = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", source],
                    check=False,
                    env=environment,
                )
        except Exception as exc:
            installation_failures.append(requirement.executable)
            print(
                f"[INSTALL FAILED] {requirement.executable}: {exc}. Continuing "
                "with the remaining software.",
                file=sys.stderr,
                flush=True,
            )
            continue
        if process.returncode != 0:
            installation_failures.append(requirement.executable)
            print(
                f"[INSTALL FAILED] {requirement.executable}: exit code "
                f"{process.returncode}. Continuing with the remaining software.",
                file=sys.stderr,
                flush=True,
            )
    for requirement in source_requirements:
        try:
            if requirement.executable == "minibwa":
                install_minibwa()
        except Exception as exc:
            installation_failures.append(requirement.executable)
            print(
                f"[INSTALL FAILED] {requirement.executable}: {exc}. Continuing "
                "with the remaining software.",
                file=sys.stderr,
                flush=True,
            )
    r_requirements = [
        requirement
        for profile, names in missing_r.items()
        for requirement in R_PACKAGE_PROFILES[profile]
        if requirement.name in names
    ]
    if r_requirements:
        try:
            install_r_packages(r_requirements)
        except Exception as exc:
            installation_failures.append("R packages")
            print(
                f"[INSTALL FAILED] R packages: {exc}.",
                file=sys.stderr,
                flush=True,
            )
    if installation_failures:
        print(
            "[INSTALL SUMMARY] Failed installation attempt(s): "
            + ", ".join(installation_failures)
            + ". All other selected software was still attempted.",
            file=sys.stderr,
            flush=True,
        )
    return missing_software(requirements)


def required_r_package_profiles(
    requirements: Sequence[SoftwareRequirement],
) -> list[str]:
    profiles: list[str] = []
    if any(
        requirement.executable == "Rscript" and "MAGScoT" in requirement.purpose
        for requirement in requirements
    ):
        profiles.append("MAGScoT")
    if any(
        requirement.executable == "DAS_Tool"
        or (requirement.executable == "Rscript" and "DAS Tool" in requirement.purpose)
        for requirement in requirements
    ):
        profiles.append("DAS Tool")
    return profiles


def missing_r_packages(profile: str) -> list[str]:
    if profile not in R_PACKAGE_PROFILES:
        raise ValueError(f"Unknown R package profile: {profile}")
    rscript = shutil.which("Rscript")
    names = tuple(requirement.name for requirement in R_PACKAGE_PROFILES[profile])
    if rscript is None:
        return list(names)
    missing: list[str] = []
    for package in names:
        process = subprocess.run(
            [
                rscript,
                "-e",
                (
                    f"quit(status=ifelse(requireNamespace('{package}', quietly=TRUE),"
                    "0,1))"
                ),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if process.returncode != 0:
            missing.append(package)
    return missing


def _r_character_vector(values: Sequence[str]) -> str:
    quoted = ", ".join(
        '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
        for value in values
    )
    return f"c({quoted})"


def install_r_packages(
    requirements: Sequence[RPackageRequirement],
) -> None:
    rscript = shutil.which("Rscript")
    if rscript is None:
        raise RuntimeError(
            "Rscript is unavailable after installing external software; "
            "cannot install required R packages"
        )
    unique = {
        (requirement.name, requirement.source.lower()): requirement
        for requirement in requirements
    }
    unsupported = sorted(
        source
        for _name, source in unique
        if source not in {"cran", "bioconductor"}
    )
    if unsupported:
        raise ValueError(
            "Unsupported R package source(s): " + ", ".join(unsupported)
        )
    cran = sorted(name for name, source in unique if source == "cran")
    bioconductor = sorted(
        name for name, source in unique if source == "bioconductor"
    )
    if not unique:
        return
    prefix = Path(sys.prefix).resolve()
    conda_names = sorted(set(cran) & CONDA_R_PACKAGES.keys())
    frontend = (
        conda_frontend()
        if conda_names and (prefix / "conda-meta").is_dir()
        and Path(rscript).resolve().is_relative_to(prefix)
        else None
    )
    if frontend is not None:
        # Restore both the R headers/runtime and compiled packages together.
        # A recorded Conda package can have been overwritten by CRAN, so an
        # ordinary install may incorrectly report that the broken ABI is fine.
        version = _conda_package_version(rscript, "r-base")
        if version is None:
            probe = subprocess.run(
                [rscript, "--vanilla", "-e", "cat(as.character(getRversion()))"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                check=False, timeout=30,
            )
            version = probe.stdout.strip() if probe.returncode == 0 else ""
        match = re.match(r"^(\d+\.\d+)(?:\.|$)", version)
        if match is None:
            raise RuntimeError("Cannot identify the current R version for compatible Conda R packages")
        packages = [f"r-base={match.group(1)}.*"]
        if {"dplyr", "readr"}.intersection(conda_names):
            packages.append("r-rlang")
        packages.extend(CONDA_R_PACKAGES[name] for name in conda_names)
        command = [
            frontend, "install", "--yes", "--prefix", str(prefix),
            "--override-channels", "--channel", "conda-forge",
            "--force-reinstall", *packages,
        ]
        print(
            f"Installing compatible R {match.group(1)} packages with Conda: "
            + ", ".join(packages), flush=True,
        )
        process = run_conda_transaction(command)
        if process.returncode != 0:
            raise RuntimeError(
                "Conda R package repair failed with exit code "
                f"{process.returncode}; rerun `metabaw check` after resolving the "
                "Conda error. The compiled R packages must match the R runtime."
            )
        cran = [name for name in cran if name not in conda_names]
    elif {"dplyr", "readr"}.intersection(cran):
        # On unmanaged R installations, rebuild rlang instead of reusing an
        # installed library that may target a different R binary interface.
        cran = sorted(set(cran) | {"rlang"})
    commands: list[tuple[str, list[str]]] = []
    if cran:
        expression = (
            "options(repos=c(CRAN='https://cloud.r-project.org')); "
            f"install.packages({_r_character_vector(cran)}, "
            "dependencies=NA, type='source')"
        )
        commands.append(("CRAN", [rscript, "-e", expression]))
    if bioconductor:
        expression = (
            "options(repos=c(CRAN='https://cloud.r-project.org')); "
            "if (!requireNamespace('BiocManager', quietly=TRUE)) "
            "install.packages('BiocManager', dependencies=NA); "
            "if (!requireNamespace('BiocManager', quietly=TRUE)) "
            "quit(status=1); "
            f"BiocManager::install({_r_character_vector(bioconductor)}, "
            "ask=FALSE, update=FALSE, dependencies=NA)"
        )
        commands.append(("Bioconductor", [rscript, "-e", expression]))
    for source, command in commands:
        print(
            f"Installing missing {source} R packages with Rscript",
            flush=True,
        )
        process = subprocess.run(command, check=False)
        if process.returncode != 0:
            raise RuntimeError(
                f"{source} R package installation failed with exit code "
                f"{process.returncode}"
            )
    # install.packages() can exit successfully while reporting package failures
    # as warnings. Validate the namespaces, including their native libraries.
    validation = subprocess.run(
        [rscript, "-e", (
            f"for (package in {_r_character_vector(sorted({name for name, _source in unique}))}) "
            "tryCatch(loadNamespace(package), error=function(e) { "
            "message(package, ': ', conditionMessage(e)); quit(status=1) })"
        )],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        check=False, timeout=60,
    )
    if validation.returncode != 0:
        detail = (validation.stderr or validation.stdout or "namespace load failed").strip()
        raise RuntimeError(f"Required R packages are not usable after installation: {detail}")


def missing_magscot_r_packages() -> list[str]:
    return missing_r_packages("MAGScoT")


def missing_dastool_r_packages() -> list[str]:
    return missing_r_packages("DAS Tool")


def magscot_files(directory: Path) -> tuple[Path, Path, Path]:
    return (
        directory / "MAGScoT.R",
        directory / "hmm" / "gtdbtk_rel207_Pfam-A.hmm",
        directory / "hmm" / "gtdbtk_rel207_tigrfam.hmm",
    )


def install_magscot(directory: Path) -> None:
    missing = [path for path in magscot_files(directory) if not path.is_file()]
    if not missing:
        return
    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError(
            f"MAGScoT directory exists but is incomplete: {directory}; missing={missing}"
        )
    if directory.is_symlink():
        raise RuntimeError(f"Incomplete MAGScoT directory is a symlink: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 4):
        # Only this private attempt directory can be removed after failure.
        with tempfile.TemporaryDirectory(prefix=".magscot-", dir=directory.parent) as temporary:
            source = Path(temporary) / "source"
            command = ["git", *(["-c", "http.version=HTTP/1.1"] if attempt > 1 else []),
                       "clone", "--depth", "1", "https://github.com/ikmb/MAGScoT.git", str(source)]
            print(f"[DOWNLOAD] MAGScoT attempt {attempt}/3"
                  + (" using HTTP/1.1" if attempt > 1 else ""), flush=True)
            try:
                process = subprocess.run(command, check=False, timeout=180)
            except subprocess.TimeoutExpired:
                process = None
            if process is not None and process.returncode == 0:
                if any(not path.is_file() for path in magscot_files(source)):
                    raise RuntimeError("Official MAGScoT checkout is missing its script or marker HMMs")
                if directory.exists():
                    directory.rmdir()  # Only an empty destination can be replaced.
                source.replace(directory)
                return
    raise RuntimeError(
        f"MAGScoT installation failed after 3 attempts: {directory}. "
        "Check this server's HTTPS/proxy connection to GitHub, or transfer the official "
        "ikmb/MAGScoT repository from a machine with GitHub access and rerun "
        "metabaw check --scope bin --magscot-dir /path/to/MAGScoT."
    )


def _run_database_command(command: list[str]) -> None:
    print("Installing database:", " ".join(command), flush=True)
    process = subprocess.run(command, check=False)
    if process.returncode != 0:
        raise RuntimeError(
            f"Database installation failed with exit code {process.returncode}: {' '.join(command)}"
        )


def install_checkm2_database(
    directory: Path,
    run_prefix: Sequence[str] = (),
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    _run_database_command(
        [
            *run_prefix,
            "checkm2",
            "database",
            "--download",
            "--path",
            str(directory),
        ]
    )
    matches = sorted(directory.rglob("*.dmnd"))
    if not matches:
        raise RuntimeError(f"CheckM2 database download created no .dmnd file under {directory}")
    return matches[0]


def install_gunc_database(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    _run_database_command(["gunc", "download_db", str(directory)])
    matches = sorted(directory.rglob("*.dmnd"))
    if not matches:
        raise RuntimeError(f"GUNC database download created no .dmnd file under {directory}")
    return matches[0]


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    downloaded = 0
    with urllib.request.urlopen(url) as response, partial.open("wb") as output:
        while True:
            block = response.read(8 * 1024 * 1024)
            if not block:
                break
            output.write(block)
            downloaded += len(block)
            if downloaded % (1024 * 1024 * 1024) < len(block):
                print(f"Downloaded {downloaded / (1024 ** 3):.1f} GiB", flush=True)
    partial.replace(destination)


def _safe_extract_tar(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as handle:
        for member in handle:
            if member.issym() or member.islnk():
                raise RuntimeError(f"Refusing link in database archive: {member.name}")
            target = (root / member.name).resolve()
            try:
                target.relative_to(root)
            except ValueError as exc:
                raise RuntimeError(
                    f"Refusing unsafe database archive member: {member.name}"
                ) from exc
            handle.extract(member, root)


def install_gtdbtk_database(directory: Path) -> Path:
    try:
        directory.mkdir(parents=True, exist_ok=True)
        archive = directory / "gtdbtk_data.tar.gz"
        url = (
            "https://data.gtdb.ecogenomic.org/releases/latest/auxillary_files/"
            "gtdbtk_package/full_package/gtdbtk_data.tar.gz"
        )
        _download_file(url, archive)
        extracted = directory / "data"
        _safe_extract_tar(archive, extracted)
        candidates = [
            path
            for path in (extracted, *extracted.iterdir())
            if path.is_dir() and gtdbtk_database_valid(path)
        ]
    except (OSError, tarfile.TarError) as exc:
        raise RuntimeError(
            f"GTDB-Tk database installation failed under {directory}: {exc}"
        ) from exc
    if not candidates:
        raise RuntimeError(f"Cannot locate extracted GTDB-Tk data under {extracted}")
    return candidates[0]


def _nonempty_file(path: Path) -> bool:
    try:
        if not path.is_file():
            return False
        with path.open("rb") as handle:
            return bool(handle.read(1))
    except OSError:
        return False


def _directory_has_nonempty_file(path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        return any(_nonempty_file(candidate) for candidate in path.rglob("*"))
    except OSError:
        return False


def diamond_database_missing(path: Path) -> tuple[str, ...]:
    """Return reasons why a CheckM2 or GUNC DIAMOND database is unusable."""
    missing: list[str] = []
    if not path.is_file():
        missing.append("database file does not exist")
    elif path.suffix.lower() != ".dmnd":
        missing.append("database file must end with .dmnd")
    elif not _nonempty_file(path):
        missing.append("database file is empty or unreadable")
    return tuple(missing)


def diamond_database_valid(path: Path) -> bool:
    return not diamond_database_missing(path)


def hydrogenase_database_missing(directory: Path) -> tuple[str, ...]:
    """Return missing or malformed assets from the hydrogenase database."""
    if not directory.is_dir():
        return ("database root is not a directory",)
    fasta = directory / "hyddb.all.fa"
    fefe = directory / "FeFe.dmnd"
    terminal = directory / "Terminal.dmnd"
    mapping = directory / "hyd_id-name.script.txt"
    issues: list[str] = []
    for path in (fasta, fefe, terminal, mapping):
        if not _nonempty_file(path):
            issues.append(path.name)
    if _nonempty_file(fasta):
        try:
            with fasta.open("r", encoding="utf-8-sig") as handle:
                first = next((line.strip() for line in handle if line.strip()), "")
            if not first.startswith(">"):
                issues.append("hyddb.all.fa is not a protein FASTA")
        except (OSError, UnicodeError):
            issues.append("hyddb.all.fa is unreadable")
    if _nonempty_file(mapping):
        has_mapping = False
        try:
            with mapping.open("r", encoding="utf-8-sig") as handle:
                for raw in handle:
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    fields = line.split("\t") if "\t" in line else line.split()
                    if len(fields) < 2:
                        continue
                    if fields[0].lower() == "id" and fields[1].lower() in {
                        "gene", "type", "class"
                    }:
                        continue
                    value = "".join(character for character in fields[1].lower() if character.isalpha())
                    if value == "fe" or "nife" in value or "fefe" in value:
                        has_mapping = True
                        break
            if not has_mapping:
                issues.append(
                    "hyd_id-name.script.txt has no Fe, NiFe, or FeFe mapping rows"
                )
        except (OSError, UnicodeError):
            issues.append("hyd_id-name.script.txt is unreadable")
    return tuple(issues)


def hydrogenase_database_valid(directory: Path) -> bool:
    return not hydrogenase_database_missing(directory)


def checkm_database_missing(directory: Path) -> tuple[str, ...]:
    """Return missing assets from a legacy CheckM data root."""
    missing: list[str] = []
    if not directory.is_dir():
        return ("database root is not a directory",)
    required_files = (
        directory / ".dmanifest",
        directory / "hmms" / "phylo.hmm",
        directory / "hmms" / "checkm.hmm",
        directory / "pfam" / "Pfam-A.hmm.dat",
        directory / "selected_marker_sets.tsv",
        directory / "taxon_marker_sets.tsv",
    )
    for path in required_files:
        if not _nonempty_file(path):
            missing.append(str(path.relative_to(directory)))
    # CheckM calls its reference-tree directory ``genome_tree``.  ``phylo``
    # is the name of an HMM file, not a top-level database directory.
    for name in ("genome_tree", "distributions"):
        if not _directory_has_nonempty_file(directory / name):
            missing.append(f"{name}/")
    return tuple(missing)


def checkm_database_valid(directory: Path) -> bool:
    return not checkm_database_missing(directory)


def gtdbtk_database_missing(directory: Path) -> tuple[str, ...]:
    """Return missing assets from an unpacked GTDB-Tk reference-data root."""
    if not directory.is_dir():
        return ("database root is not a directory",)
    missing: list[str] = []
    required_files = (
        directory / "metadata" / "metadata.txt",
        directory / "taxonomy" / "gtdb_taxonomy.tsv",
    )
    for path in required_files:
        if not _nonempty_file(path):
            missing.append(str(path.relative_to(directory)))
    for name in ("markers", "masks", "msa", "pplacer", "radii"):
        if not _directory_has_nonempty_file(directory / name):
            missing.append(f"{name}/")
    if not any(
        _directory_has_nonempty_file(directory / name)
        for name in ("skani", "fastani", "mash")
    ):
        missing.append("skani/, fastani/, or mash/")
    return tuple(missing)


def gtdbtk_database_valid(directory: Path) -> bool:
    return not gtdbtk_database_missing(directory)


def gtdbtk_classification_database_missing(directory: Path) -> tuple[str, ...]:
    """Classification additionally needs ANI sketches and tree RED values."""
    missing = list(gtdbtk_database_missing(directory))
    if directory.is_dir():
        for name in ("skani", "mrca_red"):
            if not _directory_has_nonempty_file(directory / name):
                reason = f"{name}/"
                if reason not in missing:
                    missing.append(reason)
    return tuple(missing)


def gtdbtk_classification_database_valid(directory: Path) -> bool:
    return not gtdbtk_classification_database_missing(directory)


def kofam_database_missing(directory: Path) -> tuple[Path, ...]:
    """Return required KOfam assets that are absent or empty."""
    required = (directory / "ko_list", directory / "profiles")
    missing: list[Path] = []
    if not _nonempty_file(required[0]):
        missing.append(required[0])
    profiles = required[1]
    if not _directory_has_nonempty_file(profiles):
        missing.append(profiles)
    return tuple(missing)


def kofam_database_valid(directory: Path) -> bool:
    return directory.is_dir() and not kofam_database_missing(directory)


def ensure_kofam_description_file(directory: Path) -> Path:
    """Return K.descript.txt, deriving it from ko_list for older databases."""
    description_path = directory / "K.descript.txt"
    if description_path.is_file() and description_path.stat().st_size > 0:
        return description_path
    ko_list = directory / "ko_list"
    if not ko_list.is_file() or ko_list.stat().st_size == 0:
        raise RuntimeError(
            "KOfam KO descriptions are unavailable: K.descript.txt and a "
            f"usable ko_list are both missing under {directory}"
        )
    with ko_list.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration as error:
            raise RuntimeError(f"KOfam ko_list is empty: {ko_list}") from error
        normalized = [field.strip().lstrip("#").lower() for field in header]
        try:
            ko_index = normalized.index("knum")
            description_index = normalized.index("definition")
        except ValueError as error:
            raise RuntimeError(
                "K.descript.txt is missing and KOfam ko_list has no knum/definition "
                f"columns from which it can be generated: {ko_list}"
            ) from error
        descriptions: list[tuple[str, str]] = []
        for line_number, fields in enumerate(reader, start=2):
            if len(fields) <= max(ko_index, description_index):
                continue
            ko = fields[ko_index].strip()
            description = fields[description_index].strip()
            if not re.fullmatch(r"K\d{5}", ko) or not description:
                continue
            descriptions.append((ko, description))
    if not descriptions:
        raise RuntimeError(
            "K.descript.txt is missing and no KO descriptions could be derived "
            f"from {ko_list}"
        )
    description_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = description_path.with_suffix(description_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("Kid", "descript"))
        writer.writerows(descriptions)
    temporary.replace(description_path)
    return description_path


def dbcan_database_missing(directory: Path) -> tuple[str, ...]:
    """Return missing run_dbCAN assets for the default CAZyme methods.

    run_dbCAN releases have used both ``dbCAN.hmm``/``dbCAN-sub.hmm`` and
    ``dbCAN.txt``/``dbCAN_sub.hmm`` for the same family and subfamily HMM
    assets.  Accept either complete naming layout; the annotation workflow
    creates canonical aliases in its temporary work directory.
    """
    required_groups = (
        ("CAZy.dmnd",),
        ("dbCAN.hmm", "dbCAN.txt"),
        ("dbCAN-sub.hmm", "dbCAN_sub.hmm"),
        ("fam-substrate-mapping.tsv",),
    )
    missing: list[str] = []
    for alternatives in required_groups:
        if not any(
            _nonempty_file(directory / name)
            for name in alternatives
        ):
            missing.append(" or ".join(alternatives))
    return tuple(missing)


def dbcan_database_valid(directory: Path) -> bool:
    return directory.is_dir() and not dbcan_database_missing(directory)


def install_kofam_database(directory: Path) -> Path:
    """Download and unpack the official KOfam profiles and KO threshold list."""
    directory.mkdir(parents=True, exist_ok=True)
    profiles_archive = directory / "profiles.tar.gz"
    ko_list_archive = directory / "ko_list.gz"
    profiles = directory / "profiles"
    ko_list = directory / "ko_list"
    if not profiles.is_dir() or not any(profiles.rglob("*.hmm")):
        _download_file(
            "https://www.genome.jp/ftp/db/kofam/profiles.tar.gz",
            profiles_archive,
        )
        _safe_extract_tar(profiles_archive, directory)
    if not ko_list.is_file() or ko_list.stat().st_size == 0:
        _download_file(
            "https://www.genome.jp/ftp/db/kofam/ko_list.gz",
            ko_list_archive,
        )
        with gzip.open(ko_list_archive, "rb") as source, ko_list.open("wb") as output:
            shutil.copyfileobj(source, output)
    ensure_kofam_description_file(directory)
    missing = kofam_database_missing(directory)
    if missing:
        raise RuntimeError(
            "KOfam database installation is incomplete; missing: "
            + ", ".join(map(str, missing))
        )
    return directory


def install_dbcan_database(directory: Path) -> Path:
    """Download the official CAZyme-only run_dbCAN database bundle."""
    directory.mkdir(parents=True, exist_ok=True)
    _run_database_command(
        [
            "run_dbcan",
            "database",
            "--db_dir",
            str(directory),
            "--aws_s3",
            "--no-cgc",
            "--timeout",
            "60",
            "--retries",
            "5",
            "--resume",
        ]
    )
    missing = dbcan_database_missing(directory)
    if missing:
        raise RuntimeError(
            "dbCAN database installation is incomplete; missing: "
            + ", ".join(missing)
        )
    return directory


def _database_config_path() -> Path:
    return Path.home() / ".config" / "metabaw" / "databases.json"


def _database_config() -> dict[str, str]:
    path = _database_config_path()
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {str(key): str(value) for key, value in data.items()}


def _running_conda_prefix() -> Path | None:
    """Return the Conda prefix that owns the running MetaBAW interpreter."""
    candidates = [Path(sys.prefix)]
    environment_prefix = os.environ.get("CONDA_PREFIX")
    if environment_prefix:
        candidates.append(Path(environment_prefix).expanduser())
    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "conda-meta").is_dir():
            return resolved
    return None


def persist_conda_environment_variable(name: str, value: str) -> tuple[bool, str]:
    """Persist a variable in the Conda environment that contains MetaBAW."""
    os.environ[name] = value
    prefix = _running_conda_prefix()
    if prefix is None:
        return False, "the running MetaBAW interpreter is not inside a Conda environment"
    conda = shutil.which("conda")
    if conda is None:
        return False, "the conda executable is not available on PATH"
    try:
        process = subprocess.run(
            [
                conda,
                "env",
                "config",
                "vars",
                "set",
                "--prefix",
                str(prefix),
                f"{name}={value}",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, f"could not run conda: {exc}"
    if process.returncode != 0:
        diagnostic = " | ".join(
            line.strip()
            for output in (process.stderr, process.stdout)
            for line in output.splitlines()
            if line.strip()
        )
        return False, diagnostic or f"conda exited with code {process.returncode}"
    return True, str(prefix)


def save_database_path(name: str, path: Path) -> None:
    config_path = _database_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config = _database_config()
    resolved = str(path.resolve())
    config[name] = resolved
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    if name == "gtdbtk":
        persisted, detail = persist_conda_environment_variable(
            "GTDBTK_DATA_PATH",
            resolved,
        )
        if persisted:
            print(
                f"[CONFIG] Saved GTDBTK_DATA_PATH in Conda environment {detail}; "
                "reactivate the environment before running gtdbtk directly",
                flush=True,
            )
        else:
            print(
                "[WARNING] Saved the GTDB-Tk path for MetaBAW, but could not "
                f"persist GTDBTK_DATA_PATH in Conda: {detail}. MetaBAW commands "
                "will still use the saved path.",
                file=sys.stderr,
                flush=True,
            )


def configured_database_path(
    explicit: str | None,
    environment_name: str,
    config_name: str | None = None,
    artifact_suffix: str | None = None,
    validator: Callable[[Path], bool] | None = None,
) -> Path | None:
    def resolve_candidate(value: str) -> Path:
        path = Path(value).expanduser().resolve()
        if artifact_suffix and path.is_dir():
            matches = sorted(path.rglob(f"*{artifact_suffix}"))
            if matches:
                return next(
                    (candidate for candidate in matches if _nonempty_file(candidate)),
                    matches[0],
                )
        return path

    def is_valid_candidate(path: Path) -> bool:
        if artifact_suffix:
            return path.is_file() and path.name.endswith(artifact_suffix)
        if validator is not None:
            return validator(path)
        return path.exists()

    if explicit:
        return resolve_candidate(explicit)

    environment_value = os.environ.get(environment_name)
    saved_value = _database_config().get(config_name) if config_name else None
    environment_path = (
        resolve_candidate(environment_value) if environment_value else None
    )
    saved_path = resolve_candidate(saved_value) if saved_value else None

    if environment_path is not None and is_valid_candidate(environment_path):
        return environment_path
    if saved_path is not None and is_valid_candidate(saved_path):
        if environment_path is not None:
            print(
                f"[WARNING] Ignoring invalid {environment_name} path "
                f"{environment_path}; using saved database path {saved_path}",
                file=sys.stderr,
                flush=True,
            )
        return saved_path
    return environment_path or saved_path


def print_software_status(requirements: Sequence[SoftwareRequirement]) -> None:
    name_width = max(34, max((len(item.executable) for item in requirements), default=0))
    for requirement in requirements:
        location = (metadecoder_executable() if requirement.executable == "metadecoder"
                    else shutil.which(requirement.executable))
        state = "OK" if location else "MISSING"
        if location:
            detail = location
        elif requirement.executable == "metadecoder":
            detail = f"upstream Python package required in the MetaBAW environment: {sys.prefix}"
        elif requirement.executable in PIP_INSTALL_URLS:
            detail = "upstream Python package"
        elif requirement.executable in SOURCE_INSTALL_EXECUTABLES:
            detail = f"official source: {MINIBWA_SOURCE_URL}"
        else:
            detail = f"Conda package: {requirement.package}"
        print(
            status_row(state, requirement.executable, f"{requirement.purpose}; {detail}", name_width),
            flush=True,
        )


def confirm_install(prompt: str) -> bool:
    if not sys.stdin.isatty():
        return False
    response = input(f"{prompt} [y/N]: ").strip().lower()
    return response in {"y", "yes"}
