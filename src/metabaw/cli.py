from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
from typing import Callable, Sequence, TypeVar

from . import __version__
from .terminal import status_row, status_tag
from .relocation import ResumeLayout
from .direct import (
    BINNER_ORDER,
    AnnotationBuilder,
    AnnotationOptions,
    BinOptions,
    DirectBinBuilder,
    NICHE_MINIMUM_INPUTS,
    annotation_niche_eligible,
    parse_choices,
)
from .dependencies import (
    CudaRuntimeStatus,
    HostCudaStatus,
    ISOLATED_TOOLS,
    IsolatedEnvironmentStatus,
    IsolatedToolSpec,
    SOFTWARE,
    SoftwareRequirement,
    SoftwareRuntimeIssue,
    all_requirements,
    annotation_requirements,
    bin_requirements,
    configured_database_path,
    configured_isolated_environment,
    configured_magscot_directory,
    confirm_install,
    checkm_database_missing,
    checkm_database_valid,
    cuda_runtime_status,
    CupyRuntimeStatus,
    cupy_runtime_status,
    dbcan_database_missing,
    dbcan_database_valid,
    diamond_database_missing,
    diamond_database_valid,
    ensure_kofam_description_file,
    gtdbtk_classification_database_missing,
    gtdbtk_classification_database_valid,
    gtdbtk_database_missing,
    gtdbtk_database_valid,
    hydrogenase_database_missing,
    hydrogenase_database_valid,
    host_cuda_status,
    install_checkm2_database,
    install_dbcan_database,
    install_gtdbtk_database,
    install_gunc_database,
    install_kofam_database,
    install_isolated_cuda_runtime,
    install_isolated_environment,
    install_magscot,
    install_main_cuda_runtime,
    main_cupy_install_command,
    install_main_cupy_runtime,
    install_software,
    isolated_environment_status,
    isolated_run_prefix,
    kofam_database_missing,
    kofam_database_valid,
    magscot_files,
    missing_dastool_r_packages,
    missing_magscot_r_packages,
    missing_software,
    normalize_isolated_environment,
    print_cuda_runtime_status,
    print_cupy_runtime_status,
    print_host_cuda_status,
    print_isolated_status,
    print_software_status,
    required_r_package_profiles,
    save_database_path,
    save_isolated_environment,
    save_magscot_directory,
    isolated_environment_config_path,
    software_runtime_issues,
)
from .discovery import (
    ReadSample,
    attach_contigs,
    build_analyses,
    build_cohort_analyses,
    discover_contigs,
    discover_contigs_from_file,
    discover_mag_files,
    discover_reads,
    discover_reads_from_file,
    public_sample_name,
    read_multi_files,
)
from .memory import detected_total_memory_gib
from .metadecoder_coverage import _python_launcher as metadecoder_python_launcher, metadecoder_executable
from .executor import Executor
from .internal import gtdbtk_result_missing, read_kegg_descriptions
from .model import Task
from .resources import GpuReservations, plan_gpu_resources, plan_resources
from .state import StateStore
from .startup_log import capture_startup, finish_startup
from .workflow_report import write_workflow_details
from .named_inputs import (
    BinningProfile, NamedBinBuilder, eligible_binners, load_named_samples,
    attach_named_contigs,
    named_analyses, named_coassembly_analyses, read_coassembly_groups,
    read_multi_names, sample_read_type,
    read_named_reads, read_genome_files, genome_name,
)
class MetaBAWHelpFormatter(argparse.HelpFormatter):
    """Show a requirement marker or an explicit default for every option."""

    def _format_action(self, action: argparse.Action) -> str:
        if getattr(action, "_metabaw_module_list_only", False):
            # Show each module's help directly under its section title, without
            # the redundant parent metavar/choice-list row. Parsing is intact.
            return "".join(self._format_action(child) for child in action._get_subactions())
        return super()._format_action(action)

    def _get_help_string(self, action: argparse.Action) -> str:
        text = action.help or "No additional description."
        if text == argparse.SUPPRESS:
            return text
        if not action.option_strings:
            return text
        if action.dest == "flye_read_type":
            return text + (" (default)" if "--nano-raw" in action.option_strings else "")
        if action.required:
            return text if "required" in text.lower() else f"{text} (required)"
        if "default:" in text.lower() or action.default is argparse.SUPPRESS:
            return text
        if action.default is None:
            default = "not set"
        elif action.default is True:
            default = "enabled"
        elif action.default is False:
            default = "disabled"
        elif isinstance(action.default, list):
            default = ", ".join(map(str, action.default)) or "empty"
        else:
            default = str(action.default)
        return f"{text} (default: {default})"


class _ExplicitOptionParser(argparse.ArgumentParser):
    """Preserve which CLI options were explicitly supplied by the user."""

    def parse_args(
        self,
        args: Sequence[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        raw_args = list(sys.argv[1:] if args is None else args)
        bare_command = raw_args[0] if len(raw_args) == 1 else None
        try:
            parsed = super().parse_args(raw_args, namespace)
        except SystemExit as stopped:
            if stopped.code and bare_command in {"bin", "annotation"}:
                help_flag = "-h" if bare_command == "annotation" else "--full-help"
                print(
                    f"Hint: run 'metabaw {bare_command} {help_flag}' to show "
                    "detailed help for all parameters.",
                    file=sys.stderr,
                )
            raise
        parsed._provided_options = {
            token.split("=", 1)[0]
            for token in raw_args
            if token.startswith("-")
        }
        return parsed


CORE_HELP_DESTS = {
    "bin": (
        "input_reads_files", "input_contig_files", "assembly_strategy",
        "coassembly_file", "align_tool",
        "output", "tmp_files", "threads", "tools",
    ),
}


DEFAULT_COMEBIN_BATCH_SIZE = 1024
DEFAULT_GPU_MEMORY_BUDGET = "100%"


# Conservative per-process CUDA budgets used to prevent GPU-capable binners
# from starting configurations that are known to be too small for reliable
# training. These are MetaBAW scheduling minima rather than hardware claims
# made by the upstream projects.
GPU_MINIMUM_MEMORY_GIB = {
    "vamb": 4.0,
    "semibin2": 4.0,
    # COMEBin is calculated dynamically from --batch-size.
    "comebin": 8.0,
    "lorbin": 4.0,
    # CuPy DPGMM has no fixed upstream VRAM minimum; honor any positive
    # assigned share and let the bounded allocator report input-specific OOM.
    "metadecoder": 0.0,
}
GPU_BINNER_LABELS = {
    "vamb": "VAMB",
    "semibin2": "SemiBin2",
    "comebin": "COMEBin",
    "lorbin": "LorBin",
    "metadecoder": "MetaDecoder",
}


def _page_text(text: str) -> None:
    """Show TEXT in a terminal pager, falling back to plain output."""
    if not sys.stdout.isatty():
        print(text, end="")
        return
    command = shlex.split(os.environ.get("PAGER", "less"))
    if Path(command[0]).name == "less":
        command.append("-R")
    try:
        process = subprocess.Popen(command, stdin=subprocess.PIPE, text=True)
        process.communicate(text)
    except (OSError, ValueError):
        print(text, end="")


class _ShortHelpAction(argparse.Action):
    """Print usage and core options only, without paging."""

    def __init__(self, option_strings, dest="help", default=argparse.SUPPRESS, help=None):
        super().__init__(option_strings, dest, default=default, nargs=0, help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        command_name = parser.prog.split()[-1]
        core = set(CORE_HELP_DESTS.get(command_name, ())) | {"help", "full_help"}
        core_actions = [
            action
            for action in parser._actions
            if action.option_strings
            and action.help is not argparse.SUPPRESS
            and not getattr(action, "_metabaw_full_help_only", False)
            and action.dest in core
        ]
        formatter = parser._get_formatter()
        if parser.description:
            formatter.add_text(parser.description)
        formatter.start_section("core options")
        for action in core_actions:
            formatter.add_argument(action)
        formatter.end_section()
        formatter.add_text("Use --full-help to page through every option.")
        print(formatter.format_help(), end="")
        parser.exit()


class _FullHelpAction(argparse.Action):
    """Page through the complete help like coverm does."""

    def __init__(self, option_strings, dest="full_help", default=argparse.SUPPRESS, help=None):
        super().__init__(option_strings, dest, default=default, nargs=0, help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        original_usage = parser.usage
        try:
            if getattr(parser, "_metabaw_hide_full_help_usage", False):
                parser.usage = argparse.SUPPRESS
            help_text = parser.format_help()
        finally:
            parser.usage = original_usage
        _page_text(help_text)
        parser.exit()


_PROCESS_STARTED_AT = time.monotonic()


def _elapsed_clock(started_at: float | None = None) -> str:
    origin = _PROCESS_STARTED_AT if started_at is None else started_at
    total = max(0, int(time.monotonic() - origin))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _log_failure_excerpt(
    path: Path, maximum_lines: int = 8, maximum_chars: int = 1200
) -> str:
    """Return the useful tail of a subprocess log for a top-level error."""
    try:
        lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip() and not line.lstrip().startswith("# command:")
        ]
    except OSError:
        return ""
    excerpt = " | ".join(lines[-maximum_lines:])
    if len(excerpt) > maximum_chars:
        excerpt = "..." + excerpt[-(maximum_chars - 3) :]
    return excerpt


def _task_id_summary(task_ids: Sequence[str], maximum: int = 5) -> str:
    preview = ", ".join(task_ids[:maximum])
    remaining = len(task_ids) - maximum
    return preview + (f", +{remaining} more" if remaining > 0 else "")


def _main_cuda_status() -> CudaRuntimeStatus:
    return cuda_runtime_status("MetaBAW environment", (sys.executable,))


def _metadecoder_cuda_status() -> CupyRuntimeStatus:
    """Probe the same main-environment MetaDecoder used by native tasks."""
    executable = metadecoder_executable()
    try:
        if executable is None:
            raise ValueError(
                f"Install MetaDecoder in the MetaBAW environment {sys.prefix}; "
                "executables from Conda base or another Python environment are not used"
            )
        command = tuple(metadecoder_python_launcher(executable))
    except (OSError, ValueError) as exc:
        return CupyRuntimeStatus(
            "MetaDecoder", None, None, None, os.environ.get("CUDA_VISIBLE_DEVICES"),
            0, (), False, f"Cannot identify MetaDecoder's own Python runtime: {exc}",
        )
    print(
        f"[GPU PROBE] MetaDecoder executable={executable}; "
        f"Python launcher={shlex.join(command)}.",
        flush=True,
    )
    return cupy_runtime_status("MetaDecoder", command)


def _isolated_cuda_status(
    key: str,
    environment: str,
    frontend: str | None,
) -> CudaRuntimeStatus:
    spec = ISOLATED_TOOLS[key]
    command = (*isolated_run_prefix(environment, frontend), "python")
    return cuda_runtime_status(
        f"{spec.display_name} environment {environment}",
        command,
    )


def _offer_cuda_repair(
    status: CudaRuntimeStatus,
    key: str | None = None,
    environment: str | None = None,
    frontend: str | None = None,
) -> CudaRuntimeStatus:
    if status.available:
        return status
    target = (
        ISOLATED_TOOLS[key].display_name
        if key is not None
        else "the MetaBAW environment"
    )
    if not confirm_install(f"Install a CUDA-enabled PyTorch build for {target}"):
        return status
    if key is None:
        used_pip = install_main_cuda_runtime()
        repaired = _main_cuda_status()
        if not repaired.available and not used_pip:
            install_main_cuda_runtime(pip_only=True)
            repaired = _main_cuda_status()
    else:
        if environment is None:
            raise RuntimeError(f"No environment was supplied for CUDA repair: {key}")
        used_pip = install_isolated_cuda_runtime(key, environment, frontend)
        repaired = _isolated_cuda_status(key, environment, frontend)
        if key == "comebin" and not repaired.available and not used_pip:
            install_isolated_cuda_runtime(
                key,
                environment,
                frontend,
                pip_only=True,
            )
            repaired = _isolated_cuda_status(key, environment, frontend)
    print("[CUDA RETEST] Verifying the repaired runtime.", flush=True)
    print_cuda_runtime_status(repaired)
    return repaired


def _offer_cupy_repair(
    status: CupyRuntimeStatus,
    host: HostCudaStatus,
    preferred_cuda: str | None = None,
) -> CupyRuntimeStatus:
    """Offer an explicit, main-environment-only CuPy install during `check`."""
    if status.available or not host.available:
        return status
    if "Cannot identify MetaDecoder's own Python runtime" in (status.error or ""):
        print("[CUDA REPAIR] Install or repair the main-environment MetaDecoder launcher first.", flush=True)
        return status
    if status.cupy_version is not None:
        print(
            "[CUDA REPAIR] MetaDecoder CuPy is installed but its runtime failed; "
            "inspect the reported CUDA error before replacing any packages.",
            flush=True,
        )
        return status
    try:
        plan = main_cupy_install_command(host.advertised_cuda_version, preferred_cuda)
        print(f"[CUPY CUDA TARGET] nvidia-smi CUDA {plan.cuda_version}; "
              f"library-only prefix {plan.runtime_prefix}; Python {sys.executable}", flush=True)
        for command in plan.commands:
            print(f"[CUPY INSTALL PLAN] {shlex.join(command)}", flush=True)
        if not confirm_install(f"Install CuPy and its CUDA libraries in the MetaBAW environment {sys.prefix}"):
            return status
        install_main_cupy_runtime(plan)
        repaired = _metadecoder_cuda_status()
        expected = sum(int(part) * multiplier for part, multiplier
                       in zip(plan.cuda_version.split("."), (1000, 10)))
        if repaired.available and repaired.cuda_version != str(expected):
            raise RuntimeError(f"CuPy loaded CUDA runtime {repaired.cuda_version}, "
                               f"but nvidia-smi target {plan.cuda_version} requires {expected}")
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        print(f"[CUDA REPAIR FAILED] MetaDecoder: {exc}", file=sys.stderr, flush=True)
        return status
    print("[CUDA RETEST] Verifying MetaDecoder CuPy in the MetaBAW environment.", flush=True)
    print_cupy_runtime_status(repaired)
    return repaired


def _selected_gpu_memory_mib(status: HostCudaStatus) -> int | None:
    if not status.devices:
        return None
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        first = visible.split(",", 1)[0].strip()
        for device in status.devices:
            if device.index == first:
                return device.memory_total_mib
    return status.devices[0].memory_total_mib


def _gpu_task_slots(
    status: HostCudaStatus,
    visible_device_count: int,
    memory_budget: str,
    maximum_tasks: int,
) -> int:
    """GPU concurrency follows equal shares within the shared 80% budget."""
    if visible_device_count < 1 or maximum_tasks < 1:
        return 0
    devices, share = plan_gpu_resources(status.devices[:visible_device_count], memory_budget,
                                        maximum_tasks, os.environ.get("CUDA_VISIBLE_DEVICES"))
    return min(maximum_tasks, GpuReservations(devices, share).capacity)


def _gpu_memory_gib(value: str, total_memory_mib: int | None = None) -> float | None:
    match = re.fullmatch(
        r"(\d+(?:\.\d+)?)([KMGTP]?i?B?|%)?",
        value,
        re.IGNORECASE,
    )
    if match is None:
        return None
    amount = float(match.group(1))
    unit = (match.group(2) or "").upper()
    if unit == "%":
        if total_memory_mib is None:
            return None
        return total_memory_mib / 1024 * amount / 100
    if not unit:
        return amount / 1024**3
    prefix = unit[0]
    power = {"K": 1, "M": 2, "G": 3, "T": 4, "P": 5}[prefix]
    return amount * 1024**power / 1024**3


def _option_was_provided(args: argparse.Namespace, option: str) -> bool:
    return option in getattr(args, "_provided_options", set())


def _thread_budget(
    total_threads: int,
    maximum_parallel_samples: int,
    workload_count: int,
) -> tuple[int, int]:
    """Return per-sample threads and the effective concurrent sample count."""
    concurrent = max(
        1,
        min(maximum_parallel_samples, max(1, workload_count), total_threads),
    )
    return max(1, total_threads // concurrent), concurrent


def _configure_thread_budget(
    args: argparse.Namespace,
    workload_count: int,
) -> int:
    if getattr(args, "resource_safety", False):
        plan = plan_resources(args.threads, args.max_memory, args.task, workload_count)
        args.resource_plan = plan
        args.requested_threads, args.requested_max_memory = args.threads, args.max_memory
        args.threads, args.max_memory = plan.threads, plan.memory_gib
        print(f"[RESOURCE CAP] CPU capacity={plan.available_cpu_capacity:g}; "
              f"RAM available={plan.memory_available_gib:.3f} GiB; RAM cap=80%; "
              f"requested CPU/RAM={plan.threads_requested}/{plan.memory_requested_gib:g} GiB; "
              f"effective CPU/RAM={plan.threads}/{plan.memory_gib:.3f} GiB; "
              f"RAM share={plan.memory_per_task_gib:.3f} GiB per sample slot.", flush=True)
    per_sample, concurrent = _thread_budget(
        args.threads,
        args.task,
        workload_count,
    )
    args.threads_per_task = per_sample
    args.concurrent_sample_limit = concurrent
    allocated = per_sample * concurrent
    remainder = max(0, args.threads - allocated)
    remainder_text = (
        f"; {remainder} thread(s) remain available for scheduler overhead"
        if remainder
        else ""
    )
    print(
        f"[RESOURCES] -t/--threads={args.threads} is the total CPU budget; "
        f"up to {concurrent} concurrent sample task(s) receive "
        f"{per_sample} thread(s) each{remainder_text}.",
        flush=True,
    )
    return per_sample


def _record_parameter_warning(args: argparse.Namespace, message: str) -> None:
    warnings = getattr(args, "parameter_warnings", None)
    if warnings is None:
        warnings = []
        args.parameter_warnings = warnings
    if message not in warnings:
        warnings.append(message)
        print(f"[WARNING] {message}", file=sys.stderr, flush=True)


def _resolve_bin_option_priorities(args: argparse.Namespace) -> None:
    """Apply documented soft-option priorities before workflow planning."""
    args.parameter_warnings = []
    if args.gpu is False and _option_was_provided(args, "--max-gpu-memory"):
        _record_parameter_warning(
            args,
            "--no-gpu takes precedence over --max-gpu-memory; the explicitly "
            f"supplied GPU memory budget ({args.max_gpu_memory}) is ignored.",
        )
    if args.environment and "semibin2" not in args.tools:
        _record_parameter_warning(
            args,
            f"--environment {args.environment!r} is effective only when "
            "SemiBin2 is selected with --tools; it is ignored for the current "
            "binner selection.",
        )
        args.environment = None
    elif args.environment:
        print(
            f"[CONFIG] --environment {args.environment!r} will be passed to "
            "SemiBin2 for compatible per-sample runs.",
            flush=True,
        )


def _comebin_minimum_memory_gib(batch_size: int) -> float:
    """Return MetaBAW's conservative COMEBin VRAM estimate for a batch."""
    return max(1.0, batch_size / 128.0)


def _gpu_minimum_memory_gib(tool: str, args: argparse.Namespace) -> float:
    if tool == "comebin":
        requested = getattr(args, "requested_batch_size", None)
        if requested is None:
            requested = args.batch_size
        return _comebin_minimum_memory_gib(requested)
    return GPU_MINIMUM_MEMORY_GIB[tool]


def _apply_comebin_gpu_budget(
    args: argparse.Namespace,
    total_memory_mib: int | None = None,
    emit: bool = True,
) -> None:
    requested = getattr(args, "requested_batch_size", None)
    if requested is None:
        requested = args.batch_size
    args.requested_batch_size = requested
    args.batch_size = requested
    gpu_binners = set(getattr(args, "gpu_binners", args.tools))
    if not args.gpu or "comebin" not in gpu_binners:
        return
    if emit:
        print(
            f"[GPU] COMEBin batch size {requested} meets the conservative "
            f"{_comebin_minimum_memory_gib(requested):g} GiB minimum for the "
            f"{args.max_gpu_memory} per-task memory budget.",
            flush=True,
        )


COMEBIN_CPU_DEFAULT_BATCH_SIZE = 128


def _comebin_script_metadata(executable: str | None) -> tuple[str | None, bool]:
    """Return the launcher version and whether it accepts ``-d``."""
    if not executable:
        return None, False
    try:
        launcher = Path(executable).resolve().read_text(
            encoding="utf-8", errors="replace"
        )
    except OSError:
        return None, False
    match = re.search(r'^VERSION=["\']([^"\']+)["\']', launcher, re.MULTILINE)
    version = match.group(1) if match else None
    supports_device = bool(
        re.search(r"(?:^|\s)-d\s+STR\b", launcher, re.MULTILINE)
        and re.search(r"getopts\s+[^\n]*d:", launcher)
    )
    return version, supports_device


def _configure_comebin_training(args: argparse.Namespace) -> None:
    """Apply safe, visible defaults after the final CPU/GPU choice is known."""
    if "comebin" not in args.tools:
        return
    requested = getattr(args, "requested_batch_size", args.batch_size)
    args.requested_batch_size = requested
    uses_gpu = bool(
        args.gpu and "comebin" in set(getattr(args, "gpu_binners", ()))
    )
    if uses_gpu:
        args.batch_size = requested
        return

    explicitly_requested = _option_was_provided(args, "--batch-size")
    if not explicitly_requested and requested > COMEBIN_CPU_DEFAULT_BATCH_SIZE:
        args.batch_size = COMEBIN_CPU_DEFAULT_BATCH_SIZE
        print(
            f"[COMEBIN CPU] COMEBin will train on CPU; its implicit batch size "
            f"is reduced from {requested} to {args.batch_size}. Contrastive "
            "similarity work grows quadratically with batch size, so the "
            "upstream default can make the first epoch take hours on CPU. "
            "Use --batch-size to override this safeguard explicitly.",
            flush=True,
        )
    else:
        args.batch_size = requested
        if requested > COMEBIN_CPU_DEFAULT_BATCH_SIZE:
            _record_parameter_warning(
                args,
                f"COMEBin will train on CPU with explicitly requested "
                f"--batch-size {requested}; training.log is written only after "
                "an epoch completes in COMEBin 1.0.x, so its first update can "
                "take several hours. Use --batch-size "
                f"{COMEBIN_CPU_DEFAULT_BATCH_SIZE} for the CPU-safe profile.",
            )

    version = getattr(args, "comebin_version", None)
    if version and version.startswith("1.0."):
        _record_parameter_warning(
            args,
            f"COMEBin {version} was detected. Its training.log does not report "
            "intra-epoch progress; follow the MetaBAW task log for tqdm output. "
            "An unchanged training.log alone does not prove that the process "
            "has stopped.",
        )


def _apply_gpu_memory_minima(
    args: argparse.Namespace,
    total_memory_mib: int | None,
) -> tuple[str, ...]:
    """Select GPU binners whose requested VRAM budget meets safe minima."""
    selected = [tool for tool in args.tools if tool in GPU_MINIMUM_MEMORY_GIB]
    memory_gib = _gpu_memory_gib(args.max_gpu_memory, total_memory_mib)
    if memory_gib is None:
        args.gpu_binners = tuple(selected)
        return tuple(selected)
    enabled: list[str] = []
    for tool in selected:
        minimum = _gpu_minimum_memory_gib(tool, args)
        if memory_gib + 1e-9 < minimum:
            requested_batch = getattr(args, "requested_batch_size", None)
            if requested_batch is None:
                requested_batch = args.batch_size
            batch_detail = (
                f" at --batch-size {requested_batch}" if tool == "comebin" else ""
            )
            _record_parameter_warning(
                args,
                f"--max-gpu-memory {args.max_gpu_memory} resolves to "
                f"{memory_gib:g} GiB, below the MetaBAW minimum of "
                f"{minimum:g} GiB for {GPU_BINNER_LABELS[tool]}"
                f"{batch_detail}; "
                f"{GPU_BINNER_LABELS[tool]} will use CPU for this run.",
            )
            continue
        enabled.append(tool)
    args.gpu_binners = tuple(enabled)
    return tuple(enabled)


def _enable_bin_cuda(
    args: argparse.Namespace,
    isolated_status: dict[str, IsolatedEnvironmentStatus],
) -> bool:
    """Enable CUDA independently for each eligible selected binner."""
    host = host_cuda_status()
    print_host_cuda_status(host)
    if not host.available:
        args.gpu = False
        args.cuda_device_count = 0
        args.cuda_task_slots = 0
        detail = host.error or "unknown error"
        print(
            f"[GPU AUTO] CUDA is unavailable ({detail}); using CPU execution.",
            flush=True,
        )
        return False
    selected_gpu_tools = [
        tool for tool in args.tools if tool in GPU_MINIMUM_MEMORY_GIB
    ]
    if not selected_gpu_tools:
        print(
            "[GPU] None of the selected binners has a GPU execution path; "
            "using CPU execution.",
            flush=True,
        )
        args.gpu = False
        args.cuda_device_count = 0
        args.cuda_task_slots = 0
        return False
    if getattr(args, "resource_safety", False):
        args.requested_max_gpu_memory = args.max_gpu_memory
        devices, share = plan_gpu_resources(host.devices, args.max_gpu_memory,
                                            args.concurrent_sample_limit,
                                            os.environ.get("CUDA_VISIBLE_DEVICES"))
        args.gpu_device_budgets, args.gpu_share_bytes = devices, share
        if not devices or share <= 0:
            args.gpu = False
            args.cuda_device_count = args.cuda_task_slots = 0
            _record_parameter_warning(args, "No visible GPU has measurable free VRAM within the 80% budget; selected binners use CPU")
            return False
        args.max_gpu_memory = str(share)
        print(f"[GPU RESOURCE CAP] requested aggregate budget={args.requested_max_gpu_memory}; "
              f"available VRAM={sum(device.available_bytes for device in devices) / 1024**3:.3f} GiB; "
              f"aggregate reserved budget={sum(device.limit_bytes for device in devices) / 1024**3:.3f} GiB; "
              f"equal task share={share / 1024**3:.3f} GiB; "
              f"physical GPUs={','.join(device.index for device in devices)}.", flush=True)
    gpu_binners = _apply_gpu_memory_minima(
        args,
        _selected_gpu_memory_mib(host),
    )
    if not gpu_binners:
        args.gpu = False
        args.cuda_device_count = 0
        args.cuda_task_slots = 0
        print(
            "[GPU AUTO] The configured GPU memory budget is below the "
            "minimum for every selected GPU-capable binner; using CPU "
            "execution.",
            flush=True,
        )
        return False
    runtime_entries: list[
        tuple[CudaRuntimeStatus | CupyRuntimeStatus, tuple[str, ...]]
    ] = []
    main_tools = tuple(
        tool for tool in ("vamb", "semibin2") if tool in gpu_binners
    )
    if main_tools:
        runtime_entries.append((_main_cuda_status(), main_tools))
    if "metadecoder" in gpu_binners:
        runtime_entries.append((_metadecoder_cuda_status(), ("metadecoder",)))
    for key in ("comebin", "lorbin"):
        if key not in gpu_binners:
            continue
        environment_status = isolated_status.get(key)
        if environment_status is None or not environment_status.available:
            continue
        runtime_entries.append(
            (
                _isolated_cuda_status(
                    key,
                    getattr(args, f"{key}_env"),
                    environment_status.frontend,
                ),
                (key,),
            )
        )
    if not runtime_entries:
        print(
            "[GPU] None of the selected binners has a GPU execution path; "
            "using CPU execution.",
            flush=True,
        )
        args.gpu = False
        args.cuda_device_count = 0
        args.cuda_task_slots = 0
        return False
    enabled = set(gpu_binners)
    runtime_statuses: list[CudaRuntimeStatus | CupyRuntimeStatus] = []
    for status, tools in runtime_entries:
        if isinstance(status, CupyRuntimeStatus):
            print_cupy_runtime_status(status)
        else:
            print_cuda_runtime_status(status)
        if status.available:
            runtime_statuses.append(status)
            continue
        if isinstance(status, CupyRuntimeStatus):
            raise RuntimeError(
                f"MetaDecoder GPU runtime is unavailable in the MetaBAW environment "
                f"({status.error or 'CuPy CUDA validation failed'}). Run "
                "`metabaw check --scope bin --require-cuda` to configure MetaDecoder "
                "and CuPy before binning, or explicitly use --no-gpu for a CPU run."
            )
        enabled.difference_update(tools)
        labels = ", ".join(GPU_BINNER_LABELS[tool] for tool in tools)
        _record_parameter_warning(
            args,
            f"CUDA runtime is unavailable for {labels} "
            f"({status.error or 'CUDA allocation failed'}); "
            f"{labels} will use CPU for this run.",
        )
    gpu_binners = tuple(tool for tool in args.tools if tool in enabled)
    args.gpu_binners = gpu_binners
    if not gpu_binners:
        args.gpu = False
        args.cuda_device_count = 0
        args.cuda_task_slots = 0
        print(
            "[GPU AUTO] No selected binner has both sufficient GPU memory "
            "and a usable CUDA runtime; using CPU execution.",
            flush=True,
        )
        return False
    args.gpu = True
    args.cuda_device_count = min(
        len(host.devices),
        *(status.device_count for status in runtime_statuses),
    )
    if getattr(args, "resource_safety", False):
        args.gpu_device_budgets = args.gpu_device_budgets[:args.cuda_device_count]
        args.cuda_device_count = len(args.gpu_device_budgets)
        args.cuda_task_slots = min(args.concurrent_sample_limit,
                                  GpuReservations(args.gpu_device_budgets, args.gpu_share_bytes).capacity)
    else:
        args.cuda_task_slots = _gpu_task_slots(host, args.cuda_device_count, args.max_gpu_memory, args.task)
    print(
        f"[GPU] {args.cuda_device_count} visible device(s) provide "
        f"{args.cuda_task_slots} reserved concurrent GPU task slot(s); "
        f"eligibility budget={args.max_gpu_memory}; GPU binners="
        f"{','.join(gpu_binners)}.",
        flush=True,
    )
    _apply_comebin_gpu_budget(args, _selected_gpu_memory_mib(host))
    return True


def _task_dict(task: Task) -> dict[str, object]:
    return {
        "id": task.id,
        "stage": task.stage,
        "description": task.description,
        "cpus": task.cpus,
        "gpus": task.gpus,
        "priority": task.priority,
        "allow_failed_dependencies": task.allow_failed_deps,
        "failure_tolerated": task.failure_tolerated,
        "memory_limit_enforced": task.enforce_memory_limit,
        "per_process_address_limit_enforced": task.enforce_memory_limit,
        "minimum_memory_gb": task.minimum_memory_gb,
        "memory_requirement_hint": task.memory_requirement_hint,
        "environment": dict(sorted(task.env.items())),
        "dependencies": list(task.deps),
        "wait_for": list(task.wait_for),
        "inputs": [str(path) for path in task.inputs],
        "outputs": [str(path) for path in task.outputs],
        "output_alternatives": [
            [str(path) for path in output_set]
            for output_set in task.output_alternatives
        ],
        "fasta_output_directories": [
            str(path) for path in task.fasta_output_dirs
        ],
        "allow_empty_refinement": task.allow_empty_refinement,
        "sample": task.sample,
        "automatic_retries": task.automatic_retries,
        "command": task.display_command(),
        "cwd": str(task.cwd),
    }


def _stage_task_ancestors(
    start_id: str,
    by_id: dict[str, Task],
    stage: str,
) -> set[str]:
    """Return START_ID and same-stage data ancestors for one binner chain."""
    selected: set[str] = set()
    pending = [start_id]
    while pending:
        task_id = pending.pop()
        if task_id in selected:
            continue
        task = by_id.get(task_id)
        if task is None or task.stage != stage:
            continue
        selected.add(task_id)
        pending.extend(task.deps)
    return selected


def _record_detail(record: object | None, field: str) -> object | None:
    return getattr(record, field, None) if record is not None else None


def _tsv_text(value: object | None, maximum: int = 1200) -> str:
    if value is None:
        return ""
    rendered = " ".join(str(value).replace("\t", " ").splitlines()).strip()
    if len(rendered) > maximum:
        return rendered[: maximum - 3] + "..."
    return rendered


def _write_binner_summary(
    path: Path,
    tasks: Sequence[Task],
    statuses: dict[str, str],
    records: dict[str, object],
    payload: dict[str, object],
    log_dir: Path,
) -> None:
    """Write one policy-aware result row for every requested analysis/binner."""
    by_id = {task.id: task for task in tasks}
    task_order = {task.id: index for index, task in enumerate(tasks)}
    global_failures = [
        task.id for task in tasks if statuses.get(task.id) == "failed"
    ]
    binners = [
        str(name)
        for name in payload.get("binners", [])
        if str(name) in BINNER_ORDER
    ]
    analyses = payload.get("analyses", [])
    rows: list[dict[str, object]] = []
    if not isinstance(analyses, list):
        analyses = []
    for raw_analysis in analyses:
        if not isinstance(raw_analysis, dict):
            continue
        analysis = str(raw_analysis.get("name", ""))
        public_name = str(raw_analysis.get("public_name", analysis))
        source_samples = raw_analysis.get("samples", [])
        source_sample_text = (
            ",".join(map(str, source_samples))
            if isinstance(source_samples, list)
            else str(source_samples)
        )
        refinement_tasks = [
            task
            for task in tasks
            if task.stage == "04_refinement" and task.sample == analysis
        ]
        refinement_states = [
            statuses[task.id]
            for task in refinement_tasks
            if task.id in statuses
        ]
        if any(state == "failed" for state in refinement_states):
            refinement_status = "failed"
        elif any(state in {"success", "skipped"} for state in refinement_states):
            refinement_status = "completed"
        elif any(state == "blocked" for state in refinement_states):
            refinement_status = "blocked"
        else:
            refinement_status = "not_started"

        for binner in binners:
            publish_id = f"03.publish.{binner}.{analysis}"
            map_id = f"03.map.{binner}.{analysis}"
            publish_task = by_id.get(publish_id)
            map_task = by_id.get(map_id)
            family_ids = (
                _stage_task_ancestors(publish_id, by_id, "03_binning")
                if publish_task is not None
                else set()
            )
            failed_family = sorted(
                (
                    task_id
                    for task_id in family_ids
                    if statuses.get(task_id) == "failed"
                ),
                key=lambda task_id: task_order.get(task_id, len(task_order)),
            )
            publish_status = statuses.get(publish_id)
            if publish_task is None:
                status = "not_applicable"
            elif failed_family:
                status = "failed"
            elif publish_status == "success":
                status = "success"
            elif publish_status == "skipped":
                status = "reused"
            elif publish_status == "blocked":
                status = "blocked"
            elif publish_status == "running":
                status = "running"
            else:
                status = "not_started"

            failed_step = failed_family[0] if failed_family else ""
            if not failed_step and status == "blocked" and global_failures:
                failed_step = global_failures[0]
            record = records.get(failed_step) if failed_step else None
            reason = _record_detail(record, "message")
            if status == "not_applicable":
                reason = raw_analysis.get("excluded_binners", {}).get(
                    binner, "selected binner is not applicable to this analysis"
                )
            elif status == "blocked" and failed_step and not reason:
                reason = f"blocked after task failure: {failed_step}"
            log_path = _record_detail(record, "log_path")
            if not log_path and failed_step:
                candidate = log_dir / (
                    "".join(
                        char if char.isalnum() or char in "._-" else "_"
                        for char in failed_step
                    )
                    + ".log"
                )
                if candidate.exists():
                    log_path = str(candidate)
            map_status = statuses.get(map_id)
            entered_refinement = (
                map_status in {"success", "skipped"}
                and refinement_status in {"completed", "failed"}
            )
            mapping_path = (
                str(map_task.outputs[0])
                if map_task is not None and map_task.outputs
                else ""
            )
            published_directory = (
                str(publish_task.outputs[0])
                if publish_task is not None and publish_task.outputs
                else ""
            )
            rows.append(
                {
                    "analysis": analysis,
                    "public_name": public_name,
                    "source_samples": source_sample_text,
                    "read_type": raw_analysis.get("read_type", ""),
                    "binner": binner,
                    "failure_policy": payload.get("failure_policy", ""),
                    "status": status,
                    "failed_step": failed_step,
                    "exit_code": _record_detail(record, "exit_code"),
                    "reason": _tsv_text(reason),
                    "log_path": _tsv_text(log_path),
                    "mapping_file": mapping_path,
                    "published_bin_directory": published_directory,
                    "entered_refinement": "yes" if entered_refinement else "no",
                    "refinement_status": refinement_status,
                }
            )

    fieldnames = (
        "analysis",
        "public_name",
        "source_samples",
        "read_type",
        "binner",
        "failure_policy",
        "status",
        "failed_step",
        "exit_code",
        "reason",
        "log_path",
        "mapping_file",
        "published_bin_directory",
        "entered_refinement",
        "refinement_status",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_workflow_summary(
    path: Path,
    output: Path,
    payload: dict[str, object],
    final_status: str,
    statuses: dict[str, str],
    failures: Sequence[str] = (),
    tolerated_failures: Sequence[str] = (),
    required_blocked: Sequence[str] = (),
    reason: str | None = None,
) -> None:
    document = {
        "software": "metaBAW",
        "version": __version__,
        "module": payload.get("module", "workflow"),
        "failure_policy": payload.get("failure_policy"),
        "final_status": final_status,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "task_status_counts": dict(sorted(Counter(statuses.values()).items())),
        "failed_tasks": list(failures),
        "tolerated_failed_tasks": list(tolerated_failures),
        "required_blocked_tasks": list(required_blocked),
        "reason": reason,
        "binner_summary": (
            str(output / "binner_summary.tsv")
            if payload.get("module") == "binning"
            else None
        ),
        "run_manifest": str(output / "run_manifest.json"),
    }
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _parse_advanced(values: list[str]) -> dict[str, str]:
    parsed: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--advanced-arg must be TOOL=ARGUMENTS")
        tool, arguments = value.split("=", 1)
        tool = tool.strip().lower()
        if not tool or tool in parsed:
            raise ValueError(f"Empty or duplicate --advanced-arg tool: {tool!r}")
        parsed[tool] = arguments.strip()
    return parsed


def _temporary_path(value: str | None, output: Path) -> Path:
    if value is None:
        return output / "tmp"
    requested = Path(value).expanduser()
    return (requested if requested.is_absolute() else output / requested).resolve()


def _safe_to_delete_temp(temp_files: Path, output: Path) -> bool:
    try:
        output.relative_to(temp_files)
    except ValueError:
        return temp_files != output
    return False


def _option_was_provided(args: argparse.Namespace, *option_names: str) -> bool:
    provided = getattr(args, "_provided_options", None)
    return provided is not None and any(name in provided for name in option_names)


def _validate_metawrap_binner_limit(
    refinement: str,
    binners: Sequence[str],
) -> None:
    """Reject MetaWRAP input sets that exceed its three-binner limit."""
    if refinement != "metawrap" or len(binners) <= 3:
        return
    selected = ", ".join(binners)
    raise ValueError(
        "--refinement metawrap accepts results from at most 3 binning tools, "
        f"but {len(binners)} were selected: {selected}. Reduce --tools to 3 "
        "or fewer, or select --refinement magscot/das_tool."
    )


def _ensure_software(
    requirements: list[SoftwareRequirement],
    workflow: str,
) -> None:
    """Inspect workflow requirements without installing or prompting."""
    missing = missing_software(requirements)
    missing_r = _missing_r_dependency_groups(requirements)
    runtime_issues = software_runtime_issues(requirements)
    if not missing and not missing_r and not runtime_issues:
        print_software_status(requirements)
        return
    problems = [
        f"{item.executable} (package: {item.package}; {item.purpose})"
        for item in missing
    ]
    problems.extend(
        f"{profile} R package: {package}"
        for profile, packages in missing_r.items()
        for package in packages
    )
    problems.extend(
        f"{issue.executable} runtime: {issue.detail}" for issue in runtime_issues
    )
    raise RuntimeError(
        f"Required software for {workflow} is missing or unusable: "
        + "; ".join(problems)
    )


def _ensure_isolated_tool(
    spec: IsolatedToolSpec,
    environment: str,
) -> IsolatedEnvironmentStatus:
    """Check a selected isolated environment; repairs belong to `check`."""
    status = isolated_environment_status(spec, environment)
    if status.available:
        print_isolated_status(status)
        return status
    raise RuntimeError(
        f"{spec.display_name} environment {environment!r} is unavailable: "
        f"{status.error or 'unknown error'}. Run `metabaw check "
        f"{spec.option} {environment}` in an interactive terminal."
    )


def _check_isolated_tools_interactively(
    selected: Sequence[str],
    environments: dict[str, str],
) -> tuple[dict[str, IsolatedEnvironmentStatus], bool]:
    """Inspect and, when approved, install every selected isolated environment."""
    statuses: dict[str, IsolatedEnvironmentStatus] = {}
    for key in selected:
        status = isolated_environment_status(ISOLATED_TOOLS[key], environments[key])
        statuses[key] = status
        print_isolated_status(status)

    unavailable = [key for key in selected if not statuses[key].available]
    if not unavailable:
        return statuses, True

    names = ", ".join(
        f"{ISOLATED_TOOLS[key].display_name} ({environments[key]})"
        for key in unavailable
    )
    print(
        "Missing selected isolated software: " + names,
        file=sys.stderr,
        flush=True,
    )
    if not confirm_install(
        "Install or repair all missing selected isolated environments now"
    ):
        print(
            "Installation declined; the missing isolated software remains unavailable.",
            file=sys.stderr,
            flush=True,
        )
        return statuses, False

    total = len(unavailable)
    for number, key in enumerate(unavailable, start=1):
        spec = ISOLATED_TOOLS[key]
        environment = environments[key]
        print(
            f"[ENV INSTALL {number}/{total}] {spec.display_name}: {environment}",
            flush=True,
        )
        try:
            statuses[key] = install_isolated_environment(
                spec,
                environment,
                statuses[key].frontend,
            )
            # LorBin may have been installed in a new Python 3.10 environment
            # instead of downgrading the originally selected environment.
            # Save and CUDA-test the actual validated environment, not the old name.
            if statuses[key].available:
                environments[key] = statuses[key].environment
        except Exception as exc:
            print(
                f"[ENV INSTALL FAILED {number}/{total}] {spec.display_name}: {exc}. "
                "Continuing with the remaining isolated environments.",
                file=sys.stderr,
                flush=True,
            )
            statuses[key] = isolated_environment_status(spec, environment)

    remaining = [key for key in selected if not statuses[key].available]
    if remaining:
        print(
            "Isolated software remains missing after all installation attempts: "
            + ", ".join(ISOLATED_TOOLS[key].display_name for key in remaining),
            file=sys.stderr,
            flush=True,
        )
        return statuses, False
    return statuses, True


def _require_database(
    name: str,
    current: Path | None,
    validator: Callable[[Path], bool],
    missing_reasons: Callable[[Path], Sequence[object]],
    config_hint: str,
) -> Path:
    """Validate existing reference assets; never download during a run."""
    if current is not None and validator(current):
        print(f"{status_tag('OK')} {name} database: {current}", flush=True)
        return current
    detail = str(current) if current is not None else "not configured"
    if current is not None:
        reasons = tuple(missing_reasons(current))
        if reasons:
            detail += "; missing/invalid: " + ", ".join(map(str, reasons))
    raise RuntimeError(
        f"{name} database: {detail}. Configure it with {config_hint}."
    )


def _prompt_database_path(
    name: str,
    current: Path | None,
    validator: Callable[[Path], bool],
    config_name: str,
    *,
    artifact_suffix: str | None = None,
    missing_reasons: Callable[[Path], Sequence[object]] | None = None,
) -> Path | None:
    """Prompt for a reusable database path when an interactive check finds none."""
    if current is not None and validator(current):
        return current
    if not sys.stdin.isatty():
        return current
    while True:
        try:
            value = input(
                f"{name} database path (press Enter to leave it unset): "
            ).strip()
        except EOFError:
            return current
        if not value:
            return current
        candidate = configured_database_path(
            value,
            "",
            artifact_suffix=artifact_suffix,
            validator=validator,
        )
        if candidate is not None and validator(candidate):
            save_database_path(config_name, candidate)
            print(f"[CONFIG] Saved {name} database: {candidate}", flush=True)
            return candidate
        checked_path = candidate or Path(value).expanduser()
        reasons = tuple(missing_reasons(checked_path)) if missing_reasons else ()
        detail = (
            f" Missing/invalid: {', '.join(map(str, reasons))}."
            if reasons
            else ""
        )
        print(
            f"[INVALID] {name} database path: {checked_path}.{detail} "
            "Enter the database root/file again, or press Enter to skip.",
            file=sys.stderr,
            flush=True,
        )


_PreflightResult = TypeVar("_PreflightResult")


class _WorkflowPreflight:
    """Collect every selected dependency failure before stopping the workflow."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.problems: list[str] = []

    def check(
        self, label: str, validate: Callable[[], _PreflightResult],
    ) -> _PreflightResult | None:
        try:
            return validate()
        except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
            problem = f"{label}: {exc}"
            self.problems.append(problem)
            print(f"{status_tag('MISSING', stream=sys.stderr)} {problem}", file=sys.stderr, flush=True)
            return None

    def database(
        self,
        name: str,
        argument: str,
        environment: str,
        config: str,
        validator: Callable[[Path], bool],
        missing_reasons: Callable[[Path], Sequence[object]],
        option: str,
        artifact_suffix: str | None = None,
    ) -> Path | None:
        def validate() -> Path:
            current = configured_database_path(
                getattr(self.args, argument, None), environment, config,
                artifact_suffix=artifact_suffix, validator=validator,
            )
            return _require_database(
                name, current, validator, missing_reasons,
                f"`metabaw check --all {option} PATH` or {environment}",
            )

        database = self.check(name, validate)
        if database is not None:
            setattr(self.args, argument, str(database))
        return database

    def finish(self) -> None:
        if self.problems:
            output = getattr(self.args, "output", None)
            log = (
                str(Path(output).expanduser().resolve() / "start_info.txt")
                if output is not None else "start_info.txt"
            )
            raise RuntimeError(
                f"Dependency preflight failed ({len(self.problems)} check(s)); "
                "no workflow tasks were started. Install/configure the [MISSING] "
                f"items with `metabaw check` and rerun. Details: {log}"
            )
        print(
            "[PREFLIGHT] All required software and reference assets are configured.",
            flush=True,
        )


def _require_magscot_files(directory: Path) -> Path:
    missing: list[str] = []
    for path in magscot_files(directory):
        try:
            with path.open("rb") as handle:
                if not handle.read(1):
                    missing.append(f"{path} (empty)")
        except OSError as exc:
            missing.append(f"{path} ({exc})")
    if missing:
        raise RuntimeError("missing/invalid files: " + ", ".join(missing))
    print(f"{status_tag('OK')} MAGScoT script and marker HMMs: {directory}", flush=True)
    return directory


def _preflight_bin(args: argparse.Namespace) -> None:
    checks = _WorkflowPreflight(args)
    args._software_executables = {}
    checks.check("Software", lambda: _ensure_software(bin_requirements(args), "bin"))
    isolated_status: dict[str, IsolatedEnvironmentStatus] = {}
    selected = [key for key in ("comebin", "lorbin") if key in args.tools]
    if args.refinement == "metawrap":
        selected.append("metawrap")
    if args.quality_control == "checkm2":
        selected.append("checkm2")
    for key in selected:
        spec = ISOLATED_TOOLS[key]
        status = checks.check(
            spec.display_name,
            lambda key=key, spec=spec: _ensure_isolated_tool(
                spec, getattr(args, f"{key}_env"),
            ),
        )
        if status is not None:
            isolated_status[key] = status
            args._software_executables[spec.executable] = status.executable
            setattr(args, f"{key}_frontend", status.frontend)
            if key == "comebin":
                (
                    args.comebin_version, args.comebin_explicit_device,
                ) = _comebin_script_metadata(status.executable)
    if args.refinement == "magscot":
        directory = checks.check(
            "MAGScoT", lambda: _require_magscot_files(
                Path(args.magscot_dir).expanduser().resolve(),
            ),
        )
        if directory is not None:
            args.magscot_dir = str(directory)
    # dRep receives --genomeInfo from the existing quality report and does not
    # run CheckM. Only CheckM QC and MetaWRAP need the legacy reference data.
    if args.quality_control == "checkm" or args.refinement == "metawrap":
        checkm = checks.database(
            "CheckM", "checkm_db", "CHECKM_DATA_PATH", "checkm",
            checkm_database_valid, checkm_database_missing, "--checkm-db",
        )
        if checkm is not None:
            os.environ["CHECKM_DATA_PATH"] = str(checkm)
    if args.quality_control == "checkm2":
        checks.database(
            "CheckM2", "checkm2_db", "CHECKM2DB", "checkm2",
            diamond_database_valid, diamond_database_missing, "--checkm2-db", ".dmnd",
        )
    if args.gunc:
        checks.database(
            "GUNC", "gunc_db", "GUNC_DB", "gunc",
            diamond_database_valid, diamond_database_missing, "--gunc-db", ".dmnd",
        )
    if args.trna or args.rrna:
        checks.database(
            "GTDB-Tk", "gtdbtk_data", "GTDBTK_DATA_PATH", "gtdbtk",
            gtdbtk_database_valid, gtdbtk_database_missing, "--gtdbtk-db",
        )
    checks.finish()
    if args.gpu is not False:
        _enable_bin_cuda(args, isolated_status)


def _preflight_annotation(args: argparse.Namespace) -> None:
    checks = _WorkflowPreflight(args)
    checks.check(
        "Software", lambda: _ensure_software(annotation_requirements(args), "annotation"),
    )
    if args.gtdbtk_res:
        args.gtdbtk_data = None
    else:
        checks.database(
            "GTDB-Tk", "gtdbtk_data", "GTDBTK_DATA_PATH", "gtdbtk",
            gtdbtk_classification_database_valid,
            gtdbtk_classification_database_missing, "--gtdbtk-db",
        )
    if args.kegg:
        database = checks.database(
            "KOfam/KEGG", "kegg_db", "KOFAM_DB", "kofam",
            kofam_database_valid, kofam_database_missing, "--kegg-db",
        )
        if database is not None:
            def validate_descriptions() -> None:
                description_path = ensure_kofam_description_file(database)
                read_kegg_descriptions(description_path)
                print(
                    f"[KEGG] KO gene names and descriptions: {description_path}",
                    flush=True,
                )
            checks.check("KEGG description table", validate_descriptions)
    if args.cazy:
        checks.database(
            "dbCAN/CAZy", "dbcan_db", "DBCAN_DB", "dbcan",
            dbcan_database_valid, dbcan_database_missing, "--dbcan-db",
        )
    if args.hydrogenase:
        checks.database(
            "Hydrogenase", "hydrogenase_db", "HYDROGENASE_DB", "hydrogenase",
            hydrogenase_database_valid, hydrogenase_database_missing, "--hydrogenase-db",
        )
    checks.finish()


def _default_bin_dependency_profile() -> argparse.Namespace:
    return argparse.Namespace(
        align_tool="bowtie2",
        tools=["metabat2", "metadecoder", "vamb"],
        type="short",
        refinement="magscot",
        quality_control="checkm2",
        gunc=False,
        trna=False,
        rrna=False,
        dereplication_tool="galah",
    )


def _merge_requirements(
    *groups: list[SoftwareRequirement],
) -> list[SoftwareRequirement]:
    merged: list[SoftwareRequirement] = []
    seen: set[str] = set()
    for group in groups:
        for requirement in group:
            if requirement.executable not in seen:
                merged.append(requirement)
                seen.add(requirement.executable)
    return merged


def _missing_r_dependency_groups(
    requirements: list[SoftwareRequirement],
) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    profiles = required_r_package_profiles(requirements)
    if "MAGScoT" in profiles:
        missing = missing_magscot_r_packages()
        if missing:
            groups["MAGScoT"] = missing
    if "DAS Tool" in profiles:
        missing = missing_dastool_r_packages()
        if missing:
            groups["DAS Tool"] = missing
    return groups


def _software_problem_names(
    missing: Sequence[SoftwareRequirement],
    missing_r: dict[str, list[str]],
    runtime_issues: Sequence[SoftwareRuntimeIssue],
) -> list[str]:
    return [
        *(requirement.executable for requirement in missing),
        *(
            f"{profile}:R:{package}"
            for profile, packages in missing_r.items()
            for package in packages
        ),
        *(f"{issue.executable}:runtime" for issue in runtime_issues),
    ]


def _check_software_interactively(
    requirements: list[SoftwareRequirement],
) -> bool:
    print_software_status(requirements)
    missing = missing_software(requirements)
    missing_r = _missing_r_dependency_groups(requirements)
    runtime_issues = software_runtime_issues(requirements)
    for profile, packages in missing_r.items():
        for package in packages:
            print(
                status_row("MISSING", f"R package {package}", f"required by {profile}"),
                flush=True,
            )
    for issue in runtime_issues:
        print(
            status_row("BROKEN", issue.executable, issue.detail),
            flush=True,
        )
    if not missing and not missing_r and not runtime_issues:
        return True
    names = _software_problem_names(missing, missing_r, runtime_issues)
    print(
        "Missing selected software: " + ", ".join(names),
        file=sys.stderr,
        flush=True,
    )
    if not confirm_install("Install the missing selected software now"):
        print(
            "Installation declined; the missing software remains unavailable.",
            file=sys.stderr,
            flush=True,
        )
        return False
    previous = tuple(sorted(names))
    maximum_passes = 4
    for pass_number in range(1, maximum_passes + 1):
        if pass_number > 1:
            print(
                f"[INSTALL PASS {pass_number}/{maximum_passes}] Repairing "
                "dependencies detected after earlier installations.",
                flush=True,
            )
        install_software(requirements)
        remaining = missing_software(requirements)
        remaining_r = _missing_r_dependency_groups(requirements)
        remaining_runtime = software_runtime_issues(requirements)
        if not remaining and not remaining_r and not remaining_runtime:
            return True
        names = _software_problem_names(
            remaining,
            remaining_r,
            remaining_runtime,
        )
        current = tuple(sorted(names))
        if current == previous:
            break
        newly_detected = sorted(set(current) - set(previous))
        if newly_detected:
            print(
                "[RECHECK] Newly detectable dependencies: "
                + ", ".join(newly_detected),
                flush=True,
            )
        previous = current
    print(
        "Software remains missing after all installation passes: "
        + ", ".join(names),
        file=sys.stderr,
        flush=True,
    )
    return False


def command_check(args: argparse.Namespace) -> int:
    explicitly_selected_isolated = [
        key
        for key in ISOLATED_TOOLS
        if _option_was_provided(
            args,
            f"--{key}-env",
            f"--{key}-environment",
        )
    ]
    selected_all = args.all
    binning_scope = args.scope == "bin"
    explicit_essential = _option_was_provided(args, "-e", "--essential")
    inferred_isolated_scope = (
        args.scope is None
        and not selected_all
        and bool(explicitly_selected_isolated)
        and not explicit_essential
    )
    selected_essential = (
        not selected_all
        and args.scope is None
        and args.essential
        and not inferred_isolated_scope
    )
    if selected_all:
        requirements = all_requirements()
    elif args.scope == "annotation":
        requirements = annotation_requirements()
    elif inferred_isolated_scope:
        requirements = []
    elif selected_essential:
        requirements = _merge_requirements(
            bin_requirements(_default_bin_dependency_profile()),
            annotation_requirements(),
        )
    else:
        requirements = bin_requirements(_default_bin_dependency_profile())
    if any(
        requirement.executable == "gtdbtk" for requirement in requirements
    ) and args.gtdbtk_data:
        requested_gtdbtk = Path(args.gtdbtk_data).expanduser().resolve()
        if gtdbtk_database_valid(requested_gtdbtk):
            # Ensure the runtime probe uses the path supplied to this check,
            # rather than a stale Conda environment variable.
            os.environ["GTDBTK_DATA_PATH"] = str(requested_gtdbtk)
    all_available = _check_software_interactively(requirements)

    selected_isolated: list[str] = []
    if selected_all:
        selected_isolated = list(ISOLATED_TOOLS)
    elif selected_essential or binning_scope:
        selected_isolated = ["checkm2"]
    for key in explicitly_selected_isolated:
        if key not in selected_isolated:
            selected_isolated.append(key)
    isolated_environments: dict[str, str] = {}
    for key in selected_isolated:
        requested_environment = getattr(args, f"{key}_env")
        environment = normalize_isolated_environment(
            key,
            requested_environment,
        )
        if environment != requested_environment:
            print(
                f"[CONFIG] Replacing obsolete {ISOLATED_TOOLS[key].display_name} "
                f"environment {requested_environment!r} with {environment!r}.",
                flush=True,
            )
        setattr(args, f"{key}_env", environment)
        isolated_environments[key] = environment
    isolated_status, isolated_available = _check_isolated_tools_interactively(
        selected_isolated,
        isolated_environments,
    )
    all_available = all_available and isolated_available
    for key, status in isolated_status.items():
        if not status.available:
            continue
        environment = isolated_environments[key]
        saved = save_isolated_environment(key, environment)
        setattr(args, f"{key}_env", saved)
        print(
            f"[CONFIG] Saved {ISOLATED_TOOLS[key].display_name} environment: {saved}",
            flush=True,
        )
        override = os.environ.get(f"METABAW_{key.upper()}_ENV")
        if override and normalize_isolated_environment(key, override) != saved:
            print(
                f"[WARNING] METABAW_{key.upper()}_ENV={override!r} overrides the "
                f"saved {ISOLATED_TOOLS[key].display_name} environment in future runs. "
                f"Update or unset that environment variable to use {saved!r}.",
                file=sys.stderr,
                flush=True,
            )
    if selected_isolated:
        print(
            f"[CONFIG] Reusable environment settings: "
            f"{isolated_environment_config_path()}",
            flush=True,
        )

    cuda_host = host_cuda_status()
    print_host_cuda_status(cuda_host)
    cuda_entries: list[
        tuple[CudaRuntimeStatus, str | None, str | None, str | None]
    ] = []
    if selected_all or selected_essential or binning_scope:
        cuda_entries.append((_main_cuda_status(), None, None, None))
    for key in ("comebin", "lorbin"):
        status = isolated_status.get(key)
        if status is None or not status.available:
            continue
        cuda_entries.append(
            (
                _isolated_cuda_status(
                    key,
                    getattr(args, f"{key}_env"),
                    status.frontend,
                ),
                key,
                getattr(args, f"{key}_env"),
                status.frontend,
            )
        )
    cuda_runtimes: list[CudaRuntimeStatus | CupyRuntimeStatus] = []
    main_cuda_version: str | None = None
    for status, key, environment, frontend in cuda_entries:
        print_cuda_runtime_status(status)
        if cuda_host.available and not status.available:
            status = _offer_cuda_repair(
                status,
                key,
                environment,
                frontend,
            )
        if key is None and status.available:
            main_cuda_version = status.torch_cuda_version
        cuda_runtimes.append(status)
    if any(requirement.executable == "metadecoder" for requirement in requirements):
        # A healthy main-environment PyTorch does not validate CuPy. MetaDecoder
        # and all three native stages now use the same main-environment launcher.
        cupy_status = _metadecoder_cuda_status()
        print_cupy_runtime_status(cupy_status)
        if cuda_host.available and not cupy_status.available:
            cupy_status = _offer_cupy_repair(cupy_status, cuda_host, main_cuda_version)
            if not cupy_status.available:
                print(
                    "[CUDA REQUIRED] A visible NVIDIA GPU was found, but MetaDecoder's "
                    "main-environment CuPy runtime is not usable. This check has not "
                    "confirmed MetaDecoder GPU readiness; use --no-gpu only for an "
                    "intentional CPU binning run.",
                    file=sys.stderr, flush=True,
                )
                all_available = False
        cuda_runtimes.append(cupy_status)
    cuda_available = cuda_host.available and all(
        status.available for status in cuda_runtimes
    )
    if args.require_cuda and not cuda_available:
        print(
            "CUDA was required, but the driver or a selected runtime environment "
            "failed its allocation test.",
            file=sys.stderr,
            flush=True,
        )
        all_available = False
    elif not cuda_available:
        print(
            "[CUDA INFO] CUDA is optional for this check; CPU execution remains available.",
            flush=True,
        )

    checks_default_bin = selected_all or selected_essential or binning_scope
    if checks_default_bin:
        magscot_dir = Path(args.magscot_dir).expanduser().resolve()
        missing_magscot = [
            path for path in magscot_files(magscot_dir) if not path.is_file()
        ]
        if missing_magscot:
            print(
                f"{status_tag('MISSING')} MAGScoT files: "
                + ", ".join(map(str, missing_magscot)),
                file=sys.stderr,
                flush=True,
            )
            if confirm_install(f"Install MAGScoT under {magscot_dir}"):
                if _check_software_interactively([SOFTWARE["git"]]):
                    install_magscot(magscot_dir)
                else:
                    all_available = False
            else:
                all_available = False
        remaining_magscot = [
            path for path in magscot_files(magscot_dir) if not path.is_file()
        ]
        if not remaining_magscot:
            saved_magscot = save_magscot_directory(magscot_dir)
            args.magscot_dir = str(saved_magscot)
            print(
                f"[CONFIG] Saved MAGScoT directory: {saved_magscot}",
                flush=True,
            )

    database_root = Path(args.database_dir).expanduser().resolve()
    database_status: list[
        tuple[str, Path | None, Callable[[Path], bool]]
    ] = []
    checks_checkm = selected_all
    checks_checkm2 = (
        selected_all
        or selected_essential
        or binning_scope
    )
    checks_gunc = selected_all
    checks_gtdbtk = selected_all or selected_essential or args.scope == "annotation"
    if checks_checkm:
        checkm = configured_database_path(
            args.checkm_db,
            "CHECKM_DATA_PATH",
            "checkm",
            validator=checkm_database_valid,
        )
        if selected_all:
            checkm = _prompt_database_path(
                "CheckM",
                checkm,
                checkm_database_valid,
                "checkm",
                missing_reasons=checkm_database_missing,
            )
        if args.checkm_db and checkm is not None and checkm_database_valid(checkm):
            save_database_path("checkm", checkm)
        database_status.append(("CheckM", checkm, checkm_database_valid))
    if checks_checkm2:
        checkm2 = configured_database_path(
            args.checkm2_db,
            "CHECKM2DB",
            "checkm2",
            artifact_suffix=".dmnd",
        )
        if selected_all:
            checkm2 = _prompt_database_path(
                "CheckM2",
                checkm2,
                diamond_database_valid,
                "checkm2",
                artifact_suffix=".dmnd",
            )
        installed = False
        if args.install_databases and (
            checkm2 is None or not diamond_database_valid(checkm2)
        ):
            status = isolated_status.get("checkm2")
            if status is None:
                raise RuntimeError(
                    "CheckM2 must be installed before its database can be downloaded."
                )
            checkm2 = install_checkm2_database(
                database_root / "checkm2",
                isolated_run_prefix(args.checkm2_env, status.frontend),
            )
            installed = True
        if (
            (args.checkm2_db or installed)
            and checkm2 is not None
            and diamond_database_valid(checkm2)
        ):
            save_database_path("checkm2", checkm2)
        database_status.append(("CheckM2", checkm2, diamond_database_valid))
    if checks_gunc:
        gunc = configured_database_path(
            args.gunc_db,
            "GUNC_DB",
            "gunc",
            artifact_suffix=".dmnd",
        )
        if selected_all:
            gunc = _prompt_database_path(
                "GUNC",
                gunc,
                diamond_database_valid,
                "gunc",
                artifact_suffix=".dmnd",
            )
        installed = False
        if args.install_databases and (
            gunc is None or not diamond_database_valid(gunc)
        ):
            gunc = install_gunc_database(database_root / "gunc")
            installed = True
        if (
            (args.gunc_db or installed)
            and gunc is not None
            and diamond_database_valid(gunc)
        ):
            save_database_path("gunc", gunc)
        database_status.append(("GUNC", gunc, diamond_database_valid))
    if checks_gtdbtk:
        gtdbtk = configured_database_path(
            args.gtdbtk_data,
            "GTDBTK_DATA_PATH",
            "gtdbtk",
            validator=gtdbtk_database_valid,
        )
        gtdbtk = _prompt_database_path(
            "GTDB-Tk",
            gtdbtk,
            gtdbtk_database_valid,
            "gtdbtk",
            missing_reasons=gtdbtk_database_missing,
        )
        installed = False
        if args.install_databases and (
            gtdbtk is None or not gtdbtk_database_valid(gtdbtk)
        ):
            gtdbtk = install_gtdbtk_database(database_root / "gtdbtk")
            installed = True
        if gtdbtk is not None and gtdbtk_database_valid(gtdbtk):
            save_database_path("gtdbtk", gtdbtk)
        database_status.append(("GTDB-Tk", gtdbtk, gtdbtk_database_valid))
    checks_kofam = (
        selected_all
        or selected_essential
        or args.scope == "annotation"
    )
    if checks_kofam:
        kofam = configured_database_path(
            args.kegg_db,
            "KOFAM_DB",
            "kofam",
            validator=kofam_database_valid,
        )
        if selected_all:
            kofam = _prompt_database_path(
                "KOfam/KEGG",
                kofam,
                kofam_database_valid,
                "kofam",
            )
        installed = False
        if args.install_databases and (
            kofam is None or not kofam_database_valid(kofam)
        ):
            kofam = install_kofam_database(database_root / "kofam")
            installed = True
        if (
            (args.kegg_db or installed)
            and kofam is not None
            and kofam_database_valid(kofam)
        ):
            save_database_path("kofam", kofam)
        database_status.append(("KOfam/KEGG", kofam, kofam_database_valid))
    checks_dbcan = (
        selected_all
        or selected_essential
        or args.scope == "annotation"
    )
    if checks_dbcan:
        dbcan = configured_database_path(
            args.dbcan_db,
            "DBCAN_DB",
            "dbcan",
            validator=dbcan_database_valid,
        )
        if selected_all:
            dbcan = _prompt_database_path(
                "dbCAN/CAZy",
                dbcan,
                dbcan_database_valid,
                "dbcan",
                missing_reasons=dbcan_database_missing,
            )
        installed = False
        if args.install_databases and (
            dbcan is None or not dbcan_database_valid(dbcan)
        ):
            dbcan = install_dbcan_database(database_root / "dbcan")
            installed = True
        if (
            (args.dbcan_db or installed)
            and dbcan is not None
            and dbcan_database_valid(dbcan)
        ):
            save_database_path("dbcan", dbcan)
        database_status.append(("dbCAN/CAZy", dbcan, dbcan_database_valid))
    checks_hydrogenase = (
        selected_all
        or selected_essential
        or args.scope == "annotation"
    )
    if checks_hydrogenase:
        hydrogenase = configured_database_path(
            args.hydrogenase_db,
            "HYDROGENASE_DB",
            "hydrogenase",
            validator=hydrogenase_database_valid,
        )
        if selected_all:
            hydrogenase = _prompt_database_path(
                "Hydrogenase",
                hydrogenase,
                hydrogenase_database_valid,
                "hydrogenase",
                missing_reasons=hydrogenase_database_missing,
            )
        if (
            args.hydrogenase_db
            and hydrogenase is not None
            and hydrogenase_database_valid(hydrogenase)
        ):
            save_database_path("hydrogenase", hydrogenase)
        database_status.append(
            ("Hydrogenase", hydrogenase, hydrogenase_database_valid)
        )
    missing_databases = []
    for name, path, validator in database_status:
        valid = path is not None and validator(path)
        state = "OK" if valid else "MISSING"
        detail = ""
        if path is not None and not valid and name == "CheckM":
            detail = "; missing=" + ", ".join(checkm_database_missing(path))
        elif path is not None and not valid and name in {"CheckM2", "GUNC"}:
            detail = "; invalid=" + ", ".join(diamond_database_missing(path))
        elif path is not None and not valid and name == "GTDB-Tk":
            detail = "; missing=" + ", ".join(gtdbtk_database_missing(path))
        elif path is not None and not valid and name == "KOfam/KEGG":
            detail = "; missing=" + ", ".join(
                str(item) for item in kofam_database_missing(path)
            )
        elif path is not None and not valid and name == "dbCAN/CAZy":
            detail = "; missing=" + ", ".join(dbcan_database_missing(path))
        elif path is not None and not valid and name == "Hydrogenase":
            detail = "; missing=" + ", ".join(
                hydrogenase_database_missing(path)
            )
        print(f"[{state}] {name} database: {path or 'not configured'}{detail}")
        if not valid:
            missing_databases.append(name)
    if missing_databases:
        print(
            "Missing databases were not downloaded. Re-run with --install-databases "
            "or configure their paths.",
            file=sys.stderr,
        )
        return 1
    if not all_available:
        print(
            "One or more selected software dependencies are still missing.",
            file=sys.stderr,
            flush=True,
        )
        return 1
    print("All selected software and databases are available.", flush=True)
    return 0


@capture_startup(output_position=2)
def _run_direct(
    args: argparse.Namespace,
    tasks: list[Task],
    output: Path,
    temp_files: Path,
    payload: dict[str, object],
) -> int:
    workflow_started_at = time.monotonic()
    # Public workflow commands expose neither --dry-run nor --force. Keep safe
    # internal defaults for direct callers of the shared executor.
    dry_run = bool(getattr(args, "dry_run", False))
    force = bool(getattr(args, "force", False))
    runtime_dir = temp_files / "runtime"
    log_dir = runtime_dir / "logs"
    system_temp_dir = temp_files / "system_tmp"
    manifest = output / "run_manifest.json"
    previous_run = None
    if manifest.is_file():
        try:
            previous_run = json.loads(manifest.read_text(encoding="utf-8"))
            if not isinstance(previous_run, dict):
                previous_run = None
        except (OSError, ValueError) as exc:
            print(f"[RESUME] Previous manifest could not be read: {exc}; using SQLite checkpoints.", flush=True)
    resume_layout = ResumeLayout(output, temp_files, previous_run)
    # Retain old absolute-path task recipes before overwriting run_manifest.
    # Portable records can relocate without it; old releases need this proof.
    if not dry_run and (runtime_dir / "state.sqlite3").is_file():
        previous_state = StateStore(runtime_dir / "state.sqlite3")
        try:
            resume_layout.remember(previous_state)
        finally:
            previous_state.close()
    workflow_details = output / "workflow_details.md"
    workflow_summary = runtime_dir / "workflow_summary.json"
    memory_diagnostics = runtime_dir / "memory_diagnostics.jsonl"
    binner_summary = output / "binner_summary.tsv"
    obsolete_resolved_parameters = output / "resolved_parameters.tsv"
    failure_policy = str(payload.get("failure_policy", ""))
    strict_failure_policy = (
        payload.get("module") == "binning" and failure_policy == "strict"
    )
    total_threads = int(getattr(args, "threads", 1))
    task_limit = int(getattr(args, "task", 1))
    threads_per_task = int(getattr(args, "threads_per_task", total_threads))
    concurrent_sample_limit = int(
        getattr(args, "concurrent_sample_limit", task_limit)
    )
    delete_tmp_files = bool(getattr(args, "delete_tmp_files", False))
    known_memory_tasks = [task for task in tasks if task.minimum_memory_gb > 0]
    estimated_minimum_gb = max(
        (task.minimum_memory_gb for task in known_memory_tasks),
        default=0.0,
    )
    if known_memory_tasks:
        estimate_sources = sorted(
            {
                task.description or task.id
                for task in known_memory_tasks
                if task.minimum_memory_gb == estimated_minimum_gb
            }
        )
        difference = args.max_memory - estimated_minimum_gb
        balance = (
            f"headroom={difference:g} GiB"
            if difference >= 0
            else f"shortfall={-difference:g} GiB"
        )
        estimate_detail = (
            f"estimated minimum total workflow memory="
            f"{estimated_minimum_gb:g} GiB; configured "
            f"--max-memory={args.max_memory:g} GiB; {balance}; basis="
            f"{', '.join(estimate_sources)}. This is a conservative known-task "
            "floor; data-dependent additional usage remains monitored at runtime."
        )
    else:
        estimate_detail = (
            "estimated minimum total workflow memory has no fixed declared "
            f"high-memory floor for the selected tasks; configured --max-memory="
            f"{args.max_memory:g} GiB. Data-dependent usage remains monitored "
            "at runtime."
        )
    print(f"[00:00:00] [MEMORY ESTIMATE] {estimate_detail}", flush=True)
    if not dry_run:
        output.mkdir(parents=True, exist_ok=True)
        # This report was removed in favor of workflow_details.md. Clear a
        # stale copy when an existing output directory is resumed so it cannot
        # be mistaken for a file produced by the current run.
        if obsolete_resolved_parameters.is_file():
            obsolete_resolved_parameters.unlink()
        log_dir.mkdir(parents=True, exist_ok=True)
        # Runtime reports are not final results. Migrate copies left by older
        # releases on resume, preserving diagnostic history if a destination
        # already exists (including when --tmp-files is on another filesystem).
        for destination in (workflow_summary, memory_diagnostics):
            previous = output / destination.name
            if previous.is_file():
                if destination.exists():
                    destination = runtime_dir / (
                        f"{destination.stem}.previous-{time.time_ns()}{destination.suffix}"
                    )
                shutil.move(str(previous), str(destination))
        system_temp_dir.mkdir(parents=True, exist_ok=True)
        niche_decision = payload.get("niche")
        if payload.get("module") == "annotation" and isinstance(niche_decision, dict):
            niche_log = output / "niche_status.log"
            with niche_log.open("a", encoding="utf-8") as handle:
                handle.write(
                    f"{datetime.now(timezone.utc).isoformat()} {niche_decision['message']}\n"
                )
        payload.update(
            {
                "software": "metaBAW",
                "version": __version__,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "resources": {
                    "total_threads": total_threads,
                    "threads_per_sample_task": threads_per_task,
                    "threads_per_task": threads_per_task,
                    "concurrent_samples": concurrent_sample_limit,
                    "maximum_cpu_threads": total_threads,
                    "max_memory_gb": args.max_memory,
                    "total_memory_gb": args.max_memory,
                    "memory_scope": "combined_pss_of_workflow_task_process_trees",
                    "memory_measurement": "linux_smaps_rollup_or_smaps_pss; rss_is_upper_bound_only",
                    "estimated_minimum_total_memory_gb": estimated_minimum_gb,
                    "memory_estimate_scope": "known_high_memory_task_floor",
                    "memory_estimate_sources": sorted(
                        task.id
                        for task in known_memory_tasks
                        if task.minimum_memory_gb == estimated_minimum_gb
                    ),
                    "gpu_execution": bool(getattr(args, "gpu", False)),
                    "visible_gpu_devices": int(
                        getattr(args, "cuda_device_count", 0)
                    ),
                    "gpu_task_slots": int(
                        getattr(args, "cuda_task_slots", 0)
                    ),
                    "cpu_capacity_fraction": 1.0 if getattr(args, "resource_safety", False) else None,
                    "memory_safety_fraction": 0.8 if getattr(args, "resource_safety", False) else None,
                    "gpu_safety_fraction": 0.8 if getattr(args, "resource_safety", False) else None,
                    "threads_requested": getattr(args, "requested_threads", total_threads),
                    "memory_requested_gib": getattr(args, "requested_max_memory", args.max_memory),
                    "memory_per_sample_slot_gib": args.max_memory / concurrent_sample_limit,
                },
                "temporary_files": {
                    "workspace": str(temp_files),
                    "delete_after_success": delete_tmp_files,
                    "system_temp": str(system_temp_dir),
                    "state": str(runtime_dir / "state.sqlite3"),
                    "logs": str(log_dir),
                    "memory_diagnostics": str(memory_diagnostics),
                    "workflow_summary": str(workflow_summary),
                },
                "result_files": {
                    "start_info": str(output / "start_info.txt"),
                    "workflow_details": str(workflow_details),
                    "niche_status_log": (
                        str(output / "niche_status.log")
                        if payload.get("module") == "annotation" and isinstance(niche_decision, dict)
                        else None
                    ),
                    "binner_summary": (
                        str(binner_summary)
                        if payload.get("module") == "binning"
                        else None
                    ),
                },
                "tasks": [_task_dict(task) for task in tasks],
            }
        )
        payload["workflow_details"] = write_workflow_details(
            workflow_details, args, tasks, payload
        )
        print(f"[00:00:00] [WORKFLOW DETAILS] {workflow_details}", flush=True)
        manifest.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    memory_requirement = max(
        (task for task in tasks if task.minimum_memory_gb > args.max_memory),
        key=lambda task: task.minimum_memory_gb,
        default=None,
    )
    if memory_requirement is not None:
        hint = memory_requirement.memory_requirement_hint or (
            f"increase --max-memory to at least "
            f"{memory_requirement.minimum_memory_gb:g} GiB"
        )
        detail = (
            f"configured --max-memory={args.max_memory:g} GiB is below the known "
            f"minimum of {memory_requirement.minimum_memory_gb:g} GiB required by "
            f"task {memory_requirement.id} "
            f"({memory_requirement.description or memory_requirement.stage}); "
            f"{hint}. No workflow task process was started."
        )
        log_suffix = ""
        if not dry_run:
            preflight_log = log_dir / "memory_preflight.log"
            preflight_log.parent.mkdir(parents=True, exist_ok=True)
            preflight_log.write_text(
                f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] "
                f"[MEMORY ESTIMATE] {estimate_detail}\n"
                f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] "
                f"[MEMORY CONFIG ERROR] {detail}\n",
                encoding="utf-8",
            )
            if payload.get("module") == "binning":
                _write_binner_summary(
                    binner_summary,
                    tasks,
                    {},
                    {},
                    payload,
                    log_dir,
                )
            payload["execution"] = {
                "final_status": "failed_preflight",
                "reason": detail,
                "task_status_counts": {},
            }
            manifest.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            _write_workflow_summary(
                workflow_summary,
                output,
                payload,
                "failed_preflight",
                {},
                reason=detail,
            )
            log_suffix = f" log={preflight_log}"
        print(
            f"[00:00:00] [MEMORY CONFIG ERROR] {detail}{log_suffix}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    if dry_run:
        print(f"metaBAW plan: {len(tasks)} tasks")
        print(f"metaBAW temporary workspace: {temp_files}")
        current_stage = None
        for task in tasks:
            if task.stage != current_stage:
                current_stage = task.stage
                print(f"\n[{current_stage}]")
            print(f"  {task.id}  cpus={task.cpus}  {task.description}")
            print(f"    $ {task.display_command()}")
        return 0
    print(
        f"[{_elapsed_clock(workflow_started_at)}] [RESOURCES] "
        f"--max-memory={args.max_memory:g} GiB is the total workflow PSS "
        "budget shared by all concurrent task process trees; PSS uses smaps_rollup "
        "or smaps. RSS is an upper bound only, never reported as PSS.",
        flush=True,
    )
    if payload.get("module") == "binning":
        print(
            f"[{_elapsed_clock(workflow_started_at)}] [FAILURE POLICY] "
            f"{failure_policy}: "
            + (
                "the first selected binner-chain failure stops new workflow tasks."
                if strict_failure_policy
                else "failed binners are excluded and available binners continue."
            ),
            flush=True,
        )
    finish_startup(args)
    print(f"[{_elapsed_clock(workflow_started_at)}] [START INFO] {output / 'start_info.txt'}", flush=True)
    if payload.get("module") == "binning":
        print(
            f"[{_elapsed_clock(workflow_started_at)}] [RESULT FILES] "
            f"details={workflow_details}; "
            f"binners={binner_summary}",
            flush=True,
        )
    print(
        f"[{_elapsed_clock(workflow_started_at)}] [PIPELINE] "
        f"Running {payload.get('module', 'workflow')} "
        f"with {len(tasks)} tasks (total_threads={args.threads}, "
        f"threads_per_sample={getattr(args, 'threads_per_task', args.threads)}, "
        f"concurrent_samples={getattr(args, 'concurrent_sample_limit', args.task)}, "
        f"total_memory_limit={args.max_memory:g} GiB); logs={log_dir}",
        flush=True,
    )
    state = StateStore(runtime_dir / "state.sqlite3")
    memory_abort_reason: str | None = None
    records: dict[str, object] = {}
    try:
        executor = Executor(
            state=state,
            log_dir=log_dir,
            max_cpus=args.threads,
            max_parallel=getattr(
                args, "concurrent_sample_limit", args.task
            ),
            shell_executable=os.environ.get("SHELL", "/bin/bash"),
            retries=args.retries,
            fail_fast=strict_failure_policy,
            max_memory_gb=args.max_memory,
            show_progress=True,
            temp_dir=system_temp_dir,
            max_gpus=getattr(
                args,
                "cuda_task_slots",
                getattr(args, "cuda_device_count", 0),
            ),
            started_at=workflow_started_at,
            memory_diagnostic_path=memory_diagnostics,
            resource_plan=getattr(args, "resource_plan", None),
            gpu_device_budgets=getattr(args, "gpu_device_budgets", ()) if getattr(args, "gpu", False) else (),
            gpu_share_bytes=getattr(args, "gpu_share_bytes", 0) if getattr(args, "gpu", False) else 0,
            parallel_ready_tasks=bool(getattr(args, "resource_safety", False)),
            resume_layout=resume_layout,
        )
        statuses = executor.run(tasks, force=force)
        peak_memory = getattr(executor, "peak_memory_gb", None)
        reported_abort_reason = getattr(executor, "memory_abort_reason", None)
        if isinstance(reported_abort_reason, str) and reported_abort_reason:
            memory_abort_reason = reported_abort_reason
        resources = payload.get("resources")
        if isinstance(resources, dict):
            if isinstance(peak_memory, (int, float)):
                resources["observed_peak_workflow_pss_gb"] = round(peak_memory, 3)
            upper_bound = getattr(executor, "peak_memory_upper_bound_gb", None)
            upper_bound_complete = getattr(executor, "memory_upper_bound_complete", None)
            if isinstance(upper_bound, (int, float)):
                resources["observed_peak_workflow_memory_upper_bound_gb"] = round(upper_bound, 3)
            elif upper_bound_complete is False:
                resources["observed_peak_workflow_memory_upper_bound_gb"] = None
            if isinstance(upper_bound_complete, bool):
                resources["observed_memory_upper_bound_complete"] = upper_bound_complete
            pss_complete = getattr(executor, "memory_pss_complete", None)
            if isinstance(pss_complete, bool):
                resources["observed_pss_measurements_complete"] = pss_complete
            if memory_abort_reason is not None:
                resources["memory_abort_reason"] = memory_abort_reason
            manifest.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
    finally:
        records = {record.task_id: record for record in state.records()}
        state.close()
    counts = Counter(statuses.values())
    summary = (
        ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
        or "no tasks"
    )
    by_id = {task.id: task for task in tasks}
    tolerated_failures = [
        task_id
        for task_id, status in statuses.items()
        if status == "failed" and by_id[task_id].failure_tolerated
    ]
    if tolerated_failures:
        print(
            f"[{_elapsed_clock(workflow_started_at)}] [WARNING] "
            f"{len(tolerated_failures)} optional "
            f"task(s) failed; workflow continued: "
            f"{_task_id_summary(tolerated_failures)}",
            file=sys.stderr,
            flush=True,
        )
    failures = [
        task_id
        for task_id, status in statuses.items()
        if status == "failed" and not by_id[task_id].failure_tolerated
    ]
    required_blocked = [
        task_id
        for task_id, status in statuses.items()
        if status == "blocked" and not by_id[task_id].failure_tolerated
    ]
    final_status = "failed" if failures or required_blocked else "success"
    if memory_abort_reason is not None:
        execution_reason = memory_abort_reason
    elif strict_failure_policy and failures:
        execution_reason = (
            "--failure-policy strict stopped scheduling after required task "
            f"failure: {failures[0]}"
        )
    elif failures:
        execution_reason = f"required task failure: {failures[0]}"
    elif tolerated_failures:
        execution_reason = (
            "workflow completed with binner failures tolerated by "
            "--failure-policy best-effort"
        )
    else:
        execution_reason = None
    payload["execution"] = {
        "final_status": final_status,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "task_status_counts": dict(sorted(counts.items())),
        "failed_tasks": failures,
        "tolerated_failed_tasks": tolerated_failures,
        "required_blocked_tasks": required_blocked,
        "reason": execution_reason,
    }
    manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if payload.get("module") == "binning":
        _write_binner_summary(
            binner_summary,
            tasks,
            statuses,
            records,
            payload,
            log_dir,
        )
    _write_workflow_summary(
        workflow_summary,
        output,
        payload,
        final_status,
        statuses,
        failures=failures,
        tolerated_failures=tolerated_failures,
        required_blocked=required_blocked,
        reason=execution_reason,
    )
    if failures or required_blocked:
        details = []
        if failures:
            details.append(
                f"failed={len(failures)} [{_task_id_summary(failures)}]"
            )
        if required_blocked:
            details.append(
                "required_blocked="
                f"{len(required_blocked)} [{_task_id_summary(required_blocked)}]"
            )
        if memory_abort_reason is not None:
            resource_flag = "GPU resource control" if memory_abort_reason.startswith("GPU resource control:") else "--max-memory"
            print(
                f"[{_elapsed_clock(workflow_started_at)}] [ERROR] "
                f"Workflow stopped by {resource_flag}: {memory_abort_reason}; "
                f"{'; '.join(details)}; status={summary}; logs={log_dir}",
                file=sys.stderr,
                flush=True,
            )
        elif strict_failure_policy and failures:
            print(
                f"[{_elapsed_clock(workflow_started_at)}] [ERROR] "
                "Workflow stopped by --failure-policy strict after required "
                f"task failure: {failures[0]}; {'; '.join(details)}; "
                f"status={summary}; logs={log_dir}; "
                f"binner_summary={binner_summary}",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                f"[{_elapsed_clock(workflow_started_at)}] [ERROR] "
                f"Workflow did not complete: {'; '.join(details)}; "
                f"status={summary}; logs={log_dir}",
                file=sys.stderr,
                flush=True,
            )
        return 1
    if args.delete_tmp_files:
        if not _safe_to_delete_temp(temp_files, output):
            raise ValueError(
                "--delete-tmp-files cannot remove a temporary directory that contains --output"
            )
        shutil.rmtree(temp_files)
        temp_detail = "temporary files deleted"
    else:
        temp_detail = f"temporary files={temp_files}"
    print(
        f"[{_elapsed_clock(workflow_started_at)}] [COMPLETE] "
        f"Workflow completed successfully; status={summary}; "
        f"results={output}; {temp_detail}",
        flush=True,
    )
    return 0


@capture_startup()
def command_bin(args: argparse.Namespace) -> int:
    """Bin named existing assemblies or explicitly grouped coassemblies."""
    args.comebin_env = configured_isolated_environment("comebin")
    args.lorbin_env = configured_isolated_environment("lorbin")
    args.metawrap_env = configured_isolated_environment("metawrap")
    args.checkm2_env = configured_isolated_environment("checkm2")
    args.magscot_dir = str(configured_magscot_directory())
    args.checkm2_db = None
    args.gunc_db = None
    args.gtdbtk_data = None
    if args.trna_pass is not None:
        args.trna = True
    if args.rrna_pass:
        args.rrna = True
    args.min_contig_length = (
        1500 if args.min_contig_length is None else args.min_contig_length
    )
    if args.threads < 1 or args.task < 1:
        raise ValueError("--threads and --task must be positive")
    if args.max_memory <= 0:
        raise ValueError("--max-memory must be positive")
    if (
        (args.min_contig_length is not None and args.min_contig_length < 1)
        or args.minfasta_kbs < 1
        or args.batch_size < 1
    ):
        raise ValueError("contig length, minimum bin size and batch size must be positive")
    if not 0 <= args.con <= 100 or not 0 <= args.com <= 100:
        raise ValueError("--con and --com must be in [0, 100]")
    if args.trna_pass is not None and args.trna_pass < 1:
        raise ValueError("--trna-pass must be positive")
    if not re.fullmatch(
        r"\d+(?:\.\d+)?(?:[KMGTP]i?B?|%)?",
        args.max_gpu_memory,
        re.IGNORECASE,
    ):
        raise ValueError(
            "--max-gpu-memory must look like 4G, 4096M, 50%, or a byte count"
        )

    explicit_tools = parse_choices(args.tools) if args.tools is not None else None
    if explicit_tools is not None:
        unknown = sorted(set(explicit_tools) - set(BINNER_ORDER))
        if not explicit_tools or unknown:
            raise ValueError(f"--tools must select one or more of {list(BINNER_ORDER)}; unknown={unknown}")

    if args.assembly_strategy == "coassembly":
        if not args.coassembly_file:
            raise ValueError(
                "--assembly-strategy coassembly requires --coassembly-file"
            )
        if args.multi_files:
            raise ValueError(
                "--assembly-strategy coassembly cannot be combined with --multi-files; "
                "define coassembly membership only in --coassembly-file"
            )
    elif args.coassembly_file:
        raise ValueError(
            "--coassembly-file requires --assembly-strategy coassembly"
        )

    coassembly_groups = {}
    if args.assembly_strategy == "coassembly":
        reads = read_named_reads(args.input_reads_files)
        coassembly_groups = read_coassembly_groups(args.coassembly_file, reads)
        reads = attach_named_contigs(reads, args.input_contig_files)
    else:
        reads = load_named_samples(args.input_reads_files, args.input_contig_files)
    output = Path(args.output).expanduser().resolve()
    tmp = _temporary_path(args.tmp_files, output)
    group = read_multi_names(args.multi_files, reads) if args.multi_files else []
    args.mode = (
        "coassembly" if coassembly_groups else "multi" if group else "single"
    )
    analyses = (
        named_coassembly_analyses(
            reads,
            coassembly_groups,
            output / "coassembly" / "assemblies",
            output / "assemblies",
        )
        if coassembly_groups
        else named_analyses(reads, group, output / "assemblies")
    )
    profiles: dict[str, BinningProfile] = {}
    excluded: dict[str, dict[str, str]] = {}
    for sample in reads:
        read_type = sample_read_type(sample)
        aligner = args.align_tool or ("bowtie2" if read_type == "short" else "minimap2")
        if (read_type == "long" and aligner == "bowtie2") or (read_type == "short" and aligner == "minibwa"):
            raise ValueError(f"--align-tool {aligner} cannot process {read_type} sample {sample.name}; omit it for per-sample defaults or use minimap2")
        tools = eligible_binners(read_type, explicit_tools)
        if not tools:
            raise ValueError(f"No eligible binner for {read_type} sample {sample.name}; --tools {explicit_tools} excludes every usable tool")
        _validate_metawrap_binner_limit(args.refinement, tools)
        profiles[sample.name] = BinningProfile(read_type, aligner, tools)
        excluded[sample.name] = {
            tool: f"{tool} is excluded for {'paired short-read' if read_type == 'short' else 'single-file long-read'} samples"
            for tool in (explicit_tools or []) if tool not in tools
        }
    for analysis in analyses:
        if not analysis.combined:
            continue
        member_types = {
            sample_read_type(sample) for sample in analysis.assembly_samples
        }
        if len(member_types) != 1:
            raise ValueError(
                f"Coassembly group {analysis.plan_id!r} cannot mix paired short "
                "and single-file long-read samples"
            )
        representative = analysis.assembly_samples[0].name
        profiles[analysis.name] = profiles[representative]
        excluded[analysis.name] = dict(excluded[representative])
    read_types = {profile.read_type for profile in profiles.values()}
    args.type = next(iter(read_types)) if len(read_types) == 1 else "mixed"
    args.required_align_tools = sorted({profile.align_tool for profile in profiles.values()})
    args.align_tool = args.align_tool or (
        args.required_align_tools[0] if len(args.required_align_tools) == 1 else "mixed"
    )
    binners = [tool for tool in BINNER_ORDER if any(tool in p.binners for p in profiles.values())]
    requested_binners = [tool for tool in BINNER_ORDER if tool in (explicit_tools or binners)]
    args.tools = binners
    args.required_assemblers = sorted({
        "megahit" if profiles[analysis.name].read_type == "short" else "flye"
        for analysis in analyses
        if analysis.assembly_samples
    })
    args.requires_vamb_aemb = any(
        "vamb" in profiles[analysis.name].binners
        and all(sample.read2 is not None for sample in analysis.samples)
        for analysis in analyses
    )
    _resolve_bin_option_priorities(args)
    print(f"[INPUT] Named manifests: {len(reads)} samples; type={args.type}; mode={args.mode}", flush=True)
    if group:
        print(f"[MULTI] Cross-map all {len(group)} selected samples to every selected sample assembly: {','.join(group)}", flush=True)
    coassembly_by_sample = {
        sample: group_name
        for group_name, members in coassembly_groups.items()
        for sample in members
    }
    for group_name, members in coassembly_groups.items():
        group_profile = profiles[f"coassembly_{group_name}"]
        assembler = "MEGAHIT" if group_profile.read_type == "short" else "Flye"
        print(
            f"[COASSEMBLY] {group_name}: {','.join(members)}; "
            f"assembler={assembler}; contigs={group_name}.contigs.ok.fa; "
            f"bin-prefix={group_name}_{'-'.join(members)}",
            flush=True,
        )
    for sample in reads:
        profile = profiles[sample.name]
        if sample.name in coassembly_by_sample:
            assembly_source = f"coassembly:{coassembly_by_sample[sample.name]}"
        elif sample.contigs is not None:
            assembly_source = "existing-contigs"
        else:
            assembly_source = "MEGAHIT" if profile.read_type == "short" else "Flye"
        print(
            f"[INPUT] {sample.name}: {profile.read_type}; assembly={assembly_source}; "
            f"aligner={profile.align_tool}; binners={','.join(profile.binners)}",
            flush=True,
        )
        for reason in excluded[sample.name].values():
            _record_parameter_warning(args, f"{sample.name}: {reason}")

    args.requested_batch_size = args.batch_size
    args.cuda_device_count = 0
    args.cuda_task_slots = 0
    args.gpu_binners = ()
    args.resource_safety = True
    threads_per_sample = _configure_thread_budget(args, len(analyses))
    mag_suffix = args.mag_suffix.lstrip(".")
    if not mag_suffix or any(char in mag_suffix for char in "/\\"):
        raise ValueError(f"Invalid MAG suffix: {args.mag_suffix!r}")
    selected_environments = []
    if "comebin" in binners:
        selected_environments.append(("comebin", args.comebin_env))
    if "lorbin" in binners:
        selected_environments.append(("lorbin", args.lorbin_env))
    if args.refinement == "metawrap":
        selected_environments.append(("metawrap", args.metawrap_env))
    if args.quality_control == "checkm2":
        selected_environments.append(("checkm2", args.checkm2_env))
    for key, environment in selected_environments:
        if not environment.strip():
            raise ValueError(
                f"{ISOLATED_TOOLS[key].option} cannot be empty when "
                f"{ISOLATED_TOOLS[key].display_name} is selected"
            )

    _preflight_bin(args)
    _configure_comebin_training(args)
    if "semibin2" in binners and args.environment and group:
        _record_parameter_warning(args, "SemiBin2 uses self-supervised multi-sample training for the selected group; the environment model is used only for independent samples.")
    options = BinOptions(
        outdir=output,
        workdir=tmp / "work",
        threads=threads_per_sample,
        total_threads=args.threads,
        read_type=args.type,
        align_tool=args.align_tool,
        flye_read_type=args.flye_read_type,
        binners=tuple(binners),
        min_contig_length=args.min_contig_length,
        min_fasta_kbs=args.minfasta_kbs,
        batch_size=args.batch_size,
        comebin_explicit_device=getattr(args, "comebin_explicit_device", False),
        refiner=args.refinement,
        quality_control=args.quality_control,
        min_completeness=args.com,
        max_contamination=args.con,
        min_quality_score=args.quality_score,
        run_gunc=args.gunc,
        run_trna=args.trna,
        trna_pass=args.trna_pass,
        run_rrna=args.rrna,
        rrna_pass=args.rrna_pass,
        dereplicator=args.dereplication_tool,
        ani=args.ani,
        min_aligned_fraction=args.min_aligned_fraction,
        mag_suffix=mag_suffix,
        environment=args.environment,
        tag_contigs=args.tag_contigs,
        gpu=args.gpu,
        gpu_binners=tuple(args.gpu_binners),
        max_gpu_memory=args.max_gpu_memory,
        assembly_strategy=args.assembly_strategy,
        magscot_dir=Path(args.magscot_dir).expanduser().resolve(),
        checkm2_db=Path(args.checkm2_db).expanduser().resolve() if args.checkm2_db else None,
        gunc_db=Path(args.gunc_db).expanduser().resolve() if args.gunc_db else None,
        gtdbtk_data=Path(args.gtdbtk_data).expanduser().resolve() if args.gtdbtk_data else None,
        extra_args=_parse_advanced(args.advanced_arg),
        failure_policy=args.failure_policy,
        comebin_run_prefix=isolated_run_prefix(
            args.comebin_env,
            getattr(args, "comebin_frontend", None),
            prefer_lock_free_runner=True,
            live_output=True,
        ),
        checkm2_run_prefix=isolated_run_prefix(
            args.checkm2_env,
            getattr(args, "checkm2_frontend", None),
            prefer_lock_free_runner=True,
        ),
        metawrap_run_prefix=isolated_run_prefix(
            args.metawrap_env,
            getattr(args, "metawrap_frontend", None),
            prefer_lock_free_runner=True,
        ),
        lorbin_run_prefix=isolated_run_prefix(
            args.lorbin_env,
            getattr(args, "lorbin_frontend", None),
            prefer_lock_free_runner=True,
        ),
    )

    tasks = NamedBinBuilder(analyses, options, options.workdir, profiles).build()
    refinement_completeness, refinement_contamination = options.refinement_thresholds
    if options.refiner in {"magscot", "metawrap"}:
        print(
            f"[REFINEMENT THRESHOLDS] {options.refiner}: "
            f"completeness >= {refinement_completeness:g}%; "
            f"contamination <= {refinement_contamination:g}%; "
            f"relative relaxation={'20%' if options.refinement_thresholds_relaxed else 'none'}. "
            f"Final {options.quality_control} filter remains: "
            f"completeness >= {options.min_completeness:g}%; "
            f"contamination <= {options.max_contamination:g}%.",
            flush=True,
        )
    payload = {
        "module": "binning",
        "edition": "named-inputs",
        "mode": args.mode,
        "assembly_strategy": args.assembly_strategy,
        "effective_assembly": (
            "coassembly" if coassembly_groups else
            "individual" if args.required_assemblers and not args.input_contig_files else
            "mixed-existing-and-individual" if args.required_assemblers else
            "existing-contigs"
        ),
        "flye_read_type": args.flye_read_type,
        "required_assemblers": list(args.required_assemblers),
        "read_type": args.type,
        "input_sources": {
            "reads": {"kind": "named_tsv", "path": str(Path(args.input_reads_files).expanduser().resolve())},
            "contigs": (
                {"kind": "named_tsv", "path": str(Path(args.input_contig_files).expanduser().resolve())}
                if args.input_contig_files else None
            ),
            "multi_group": str(Path(args.multi_files).expanduser().resolve()) if args.multi_files else None,
            "coassembly_groups": str(Path(args.coassembly_file).expanduser().resolve()) if args.coassembly_file else None,
        },
        "multi_samples": group,
        "coassembly_groups": {
            name: list(members) for name, members in coassembly_groups.items()
        },
        "input_samples": [
            {"sample": sample.name, "read_type": sample_read_type(sample),
             "reads": [str(path) for path in sample.reads],
             "contigs": str(sample.contigs) if sample.contigs is not None else None,
             "align_tool": profiles[sample.name].align_tool,
             "binners": list(profiles[sample.name].binners)}
            for sample in reads
        ],
        "align_tool": args.align_tool,
        "binners": requested_binners,
        "effective_binners": binners,
        "failure_policy": args.failure_policy,
        "refinement": args.refinement,
        "quality_control": args.quality_control,
        "quality_thresholds": {
            "minimum_completeness": options.min_completeness,
            "maximum_contamination": options.max_contamination,
            "minimum_quality_score": options.min_quality_score,
        },
        "refinement_thresholds": (
            {
                "minimum_completeness": refinement_completeness,
                "maximum_contamination": refinement_contamination,
                "relative_relaxation": 0.2 if options.refinement_thresholds_relaxed else 0.0,
            }
            if options.refiner in {"magscot", "metawrap"} else None
        ),
        "dereplication_tool": args.dereplication_tool,
        "gpu": {
            "enabled": args.gpu,
            "enabled_binners": list(args.gpu_binners),
            "cpu_fallback_binners": [
                tool
                for tool in binners
                if tool in GPU_MINIMUM_MEMORY_GIB
                and tool not in args.gpu_binners
            ],
            "visible_device_count": args.cuda_device_count,
            "concurrent_task_slots": args.cuda_task_slots,
            "memory_budget_per_task": args.max_gpu_memory if args.gpu else None,
            "requested_memory_budget_aggregate": getattr(args, "requested_max_gpu_memory", args.max_gpu_memory),
            "aggregate_memory_budget_bytes": (
                sum(device.limit_bytes for device in getattr(args, "gpu_device_budgets", ())) if args.gpu else 0
            ),
            "device_memory_budgets": [
                {"index": device.index, "uuid": device.uuid, "available_bytes": device.available_bytes,
                 "limit_bytes": device.limit_bytes}
                for device in getattr(args, "gpu_device_budgets", ()) if args.gpu
            ],
            "comebin_batch_size_requested": args.requested_batch_size,
            "comebin_batch_size_effective": args.batch_size,
            "comebin_minimum_gpu_memory_gib": (
                _comebin_minimum_memory_gib(args.requested_batch_size)
                if "comebin" in binners
                else None
            ),
        },

        "parameter_warnings": list(args.parameter_warnings),
        "semibin2_environment": args.environment,
        "isolated_environments": dict(selected_environments),
        "analyses": [
            {"name": analysis.name, "public_name": analysis.public_name,
             "explicit_mapping": analysis.explicit_mapping,
             "combined": analysis.combined,
             "cross_mapped": analysis.cross_mapped,
             "read_type": profiles[analysis.name].read_type,
             "align_tool": profiles[analysis.name].align_tool,
             "binners": list(profiles[analysis.name].binners),
             "excluded_binners": excluded[analysis.name],
             "samples": [sample.name for sample in analysis.samples],
             "assembly_samples": [sample.name for sample in analysis.assembly_samples],
             "assembly_software": (
                 "MEGAHIT" if analysis.assembly_samples and profiles[analysis.name].read_type == "short"
                 else "Flye" if analysis.assembly_samples else None
             ),
             "coassembly_group": analysis.plan_id,
             "contigs": [str(path) for path in analysis.contigs]}
            for analysis in analyses
        ],
    }
    return _run_direct(args, tasks, output, tmp, payload)


@capture_startup()
def command_annotation(args: argparse.Namespace) -> int:
    if args.threads < 1 or args.task < 1:
        raise ValueError("--threads and --task must be positive")
    if args.max_memory <= 0:
        raise ValueError("--max-memory must be positive")
    if args.cv_correct and args.niche_classify_method != "cv":
        raise ValueError("--cv-correct requires --niche_classify_method cv")
    mag_sources = read_genome_files(args.input_genome_files)
    output = Path(args.output).expanduser().resolve()
    tmp = _temporary_path(args.tmp_files, output)
    mag_dir = tmp / "work" / "annotation" / "genomes"
    mags = [mag_dir / f"{genome_name(path)}.fa" for path in mag_sources]
    if any(path.resolve().is_relative_to(mag_dir.resolve()) for path in mag_sources):
        raise ValueError("Genome inputs must be outside the annotation genome staging directory")
    gtdbtk_result = None
    if args.gtdbtk_res:
        gtdbtk_result = Path(args.gtdbtk_res).expanduser().resolve()
        missing = gtdbtk_result_missing(gtdbtk_result, mags)
        if missing:
            raise ValueError(
                "Provided GTDB-Tk result path contains incomplete species "
                "annotation information; GTDB-Tk must be rerun: "
                f"{gtdbtk_result}; missing={'; '.join(missing)}"
            )
        args.gtdbtk_res = str(gtdbtk_result)
        print(
            f"[REUSE] Validated GTDB-Tk species annotations for {len(mags)} "
            f"MAG(s) under {gtdbtk_result}; GTDB-Tk classification will be skipped.",
            flush=True,
        )
    reads = read_named_reads(args.input_reads_files)
    read_types = {sample_read_type(sample) for sample in reads}
    read_type = next(iter(read_types)) if len(read_types) == 1 else "mixed"
    print(f"[INPUT] Annotation: {len(mags)} listed MAGs; {len(reads)} named read samples; type={read_type}", flush=True)
    methods = parse_choices(args.methods)
    allowed = {
        "relative_abundance",
        "rpkm",
        "tpm",
        "mean",
    }
    unknown = sorted(set(methods) - allowed)
    if not methods or unknown:
        raise ValueError(
            f"Unsupported CoverM methods: {unknown}; --methods accepts only "
            "relative_abundance, rpkm, tpm, mean"
        )
    niche_requested = not args.no_niche
    niche_eligible = annotation_niche_eligible(len(mags), len(reads))
    niche_enabled = niche_requested and niche_eligible
    if not niche_requested:
        niche_reason = "disabled explicitly by --no-niche"
    elif not niche_eligible:
        niche_reason = "requires genomes > 3 AND read samples > 3 (at least 4 each)"
    else:
        niche_reason = "both input counts exceed 3"
    niche_message = (
        f"[NICHE] {'Enabled' if niche_enabled else 'Skipped'}: "
        f"genomes={len(mags)}; read_samples={len(reads)}; "
        f"method={args.niche_classify_method}; abundance={args.niche_abundance}; "
        f"cv_correct={args.cv_correct}; {niche_reason}."
        + ("" if niche_enabled else " Niche classification will not run; other annotation stages continue.")
    )
    niche_decision = {
        "requested": niche_requested, "enabled": niche_enabled,
        "genome_count": len(mags), "read_sample_count": len(reads),
        "minimum_genomes": NICHE_MINIMUM_INPUTS,
        "minimum_read_samples": NICHE_MINIMUM_INPUTS,
        "classify_method": args.niche_classify_method,
        "abundance": args.niche_abundance,
        "cv_correct": args.cv_correct,
        "internal_read_counts": niche_enabled,
        "reason": niche_reason, "message": niche_message,
    }
    args.no_niche = not niche_enabled
    print(niche_message, flush=True)
    if niche_enabled:
        missing_niche_methods = sorted({args.niche_abundance} - set(methods))
        if missing_niche_methods:
            raise ValueError(
                "Ecological niche classification requires CoverM methods "
                "matching --niche_abundance; missing: "
                + ", ".join(missing_niche_methods)
            )
    # Outputs-first resume must not recycle old niche labels after a criterion
    # or abundance change. Users can retain prior results in their original dir.
    previous_manifest = output / "run_manifest.json"
    if previous_manifest.is_file() and any((output / folder).exists() for folder in ("coverm", "niche")):
        previous = json.loads(previous_manifest.read_text(encoding="utf-8"))
        old_criterion = previous.get("niche_classify_method", previous.get("niche_method", "cv"))
        old_criterion = "occupancy" if old_criterion == "prevalence" else old_criterion
        old_settings = (
            previous.get("coverm_methods"), old_criterion,
            previous.get("niche_abundance", "relative_abundance"),
            previous.get("niche", {}).get("enabled"),
            previous.get("cv_correct", previous.get("niche", {}).get("cv_correct", False)),
        )
        current_settings = (methods, args.niche_classify_method, args.niche_abundance, niche_enabled, args.cv_correct)
        if old_settings != current_settings:
            raise ValueError(
                "Annotation abundance/niche settings differ from existing outputs; "
                "use a new -o output directory to avoid reusing stale abundance or niche results."
            )
    args.kegg_db = None
    args.dbcan_db = None
    args.hydrogenase_db = None
    _preflight_annotation(args)
    suffix = args.output_file_suffix
    if not suffix.startswith("."):
        suffix = "." + suffix
    annotation_workloads = max(len(reads), len(mags))
    threads_per_sample = _configure_thread_budget(args, annotation_workloads)
    if gtdbtk_result is None:
        print(f"[RESOURCES] GTDB-Tk cpus={args.threads} from -t/--threads; "
              f"pplacer_cpus=1; local subprocess temporary parent={args.gtdbtk_tmpdir}", flush=True)
    options = AnnotationOptions(
        mag_dir=mag_dir,
        mags=tuple(mags),
        mag_suffix="fa",
        reads=tuple(reads),
        output=output,
        output_suffix=suffix,
        threads=threads_per_sample,
        read_type="auto",
        methods=tuple(methods),
        place_species=args.place_species,
        niche_rank=args.niche_rank,
        niche_method=args.niche_classify_method,
        niche_abundance=args.niche_abundance,
        cv_correct=args.cv_correct,
        no_niche=args.no_niche,
        gtdbtk_data=Path(args.gtdbtk_data).expanduser().resolve() if args.gtdbtk_data else None,
        run_kegg=args.kegg,
        run_cazy=args.cazy,
        run_hydrogenase=args.hydrogenase,
        kegg_db=Path(args.kegg_db).expanduser().resolve() if args.kegg_db else None,
        dbcan_db=Path(args.dbcan_db).expanduser().resolve() if args.dbcan_db else None,
        hydrogenase_db=(
            Path(args.hydrogenase_db).expanduser().resolve()
            if args.hydrogenase_db
            else None
        ),
        gtdbtk_result=gtdbtk_result,
        total_threads=args.threads,
        mag_sources=tuple(mag_sources),
        gtdbtk_tmpdir=Path(args.gtdbtk_tmpdir).expanduser().resolve(),
    )
    tasks = AnnotationBuilder(options, tmp / "work").build()
    payload = {
        "module": "annotation",
        "edition": "named-inputs",
        "read_type": read_type,
        "input_sources": {
            "reads": {"kind": "named_tsv", "path": str(Path(args.input_reads_files).expanduser().resolve())},
            "genomes": {"kind": "path_list", "path": str(Path(args.input_genome_files).expanduser().resolve())},
        },
        "mags": [str(path) for path in mag_sources],
        "genomes": [
            {"genome": genome_name(source), "source": str(source), "staged": str(staged)}
            for source, staged in zip(mag_sources, mags, strict=True)
        ],
        "samples": [sample.name for sample in reads],
        "input_samples": [
            {"sample": sample.name, "reads": [str(path) for path in sample.reads],
             "read_type": sample_read_type(sample),
             "mapper": "minimap2-sr" if sample.read2 is not None else "minimap2-ont"}
            for sample in reads
        ],
        "coverm_methods": methods,
        "niche": niche_decision,
        "niche_rank": args.niche_rank,
        "niche_classify_method": args.niche_classify_method,
        "niche_abundance": args.niche_abundance,
        "cv_correct": args.cv_correct,
        "kegg": args.kegg,
        "cazy": args.cazy,
        "hydrogenase": args.hydrogenase,
        "gtdbtk_result_source": str(gtdbtk_result) if gtdbtk_result else None,
        "gtdbtk_classification_skipped": gtdbtk_result is not None,
        "gtdbtk_parameters": {"cpus": args.threads, "thread_source": "--threads",
                              "pplacer_cpus": 1, "subprocess_tmp_parent": str(options.gtdbtk_tmpdir)},
    }
    return _run_direct(args, tasks, output, tmp, payload)


def build_parser() -> argparse.ArgumentParser:
    detected_memory_gib = detected_total_memory_gib()
    default_max_memory_gib = (
        detected_memory_gib if detected_memory_gib is not None else 160.0
    )
    parser = _ExplicitOptionParser(
        prog="metabaw",
        description="MetaBAW: Metagenome Binning Automated Workflow",
        formatter_class=MetaBAWHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"metaBAW {__version__}",
    )
    parser._positionals.title = "Module options"
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="MODULE")
    subparsers._metabaw_module_list_only = True

    command = subparsers.add_parser(
        "bin",
        help="run sequence mapping, binning, refinement, quality control and dereplication",
        description=(
            "Assemble missing contigs from named short/long reads, or bin named "
            "existing and explicitly grouped assemblies."
        ),
        allow_abbrev=False,
        formatter_class=MetaBAWHelpFormatter,
        add_help=False,
    )
    command._metabaw_hide_full_help_usage = True
    command.add_argument(
        "-h", "--help", action=_ShortHelpAction, help="show core options and exit"
    )
    command.add_argument(
        "--full-help", action=_FullHelpAction, help="page through every option and exit"
    )
    command.add_argument(
        "--input_reads_files", required=True, metavar="TSV",
        help="SAMPLE<TAB>READ1,READ2 (paired short reads) or SAMPLE<TAB>READ (long reads); mixed types are supported",
    )
    command.add_argument(
        "--input_contig_files", metavar="TSV",
        help=(
            "optional SAMPLE<TAB>CONTIG_FASTA; samples without supplied contigs "
            "are assembled from reads with MEGAHIT (paired short) or Flye (long)"
        ),
    )
    command.add_argument(
        "--assembly-strategy",
        choices=("existing-contigs", "coassembly"),
        default="existing-contigs",
        help=(
            "existing-contigs uses supplied contigs and individually assembles "
            "missing ones (default); coassembly assembles groups from --coassembly-file"
        ),
    )
    command.add_argument(
        "--coassembly-file",
        metavar="TSV",
        help=(
            "SAMPLE<TAB>GROUP; required with --assembly-strategy coassembly; "
            "each group needs at least two samples of one read type"
        ),
    )
    command.add_argument(
        "--align-tool",
        choices=("bowtie2", "minimap2", "minibwa"),
        help="default: bowtie2 for short reads, minimap2 for long reads",
    )
    command.add_argument(
        "--align_tool",
        dest="align_tool",
        choices=("bowtie2", "minimap2", "minibwa"),
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    flye_type = command.add_mutually_exclusive_group()
    flye_type.add_argument(
        "--pacbio-raw", dest="flye_read_type", action="store_const",
        const="--pacbio-raw",
        help="Flye PacBio CLR mode; read paths come from --input_reads_files",
    )
    flye_type.add_argument(
        "--pacbio-corr", dest="flye_read_type", action="store_const",
        const="--pacbio-corr",
        help="Flye corrected PacBio mode; read paths come from --input_reads_files",
    )
    flye_type.add_argument(
        "--pacbio-hifi", dest="flye_read_type", action="store_const",
        const="--pacbio-hifi",
        help="Flye PacBio HiFi mode; read paths come from --input_reads_files",
    )
    flye_type.add_argument(
        "--nano-raw", dest="flye_read_type", action="store_const",
        const="--nano-raw",
        help="Flye ONT raw-read mode (default); paths come from --input_reads_files",
    )
    flye_type.add_argument(
        "--nano-corr", dest="flye_read_type", action="store_const",
        const="--nano-corr",
        help="Flye corrected ONT mode; read paths come from --input_reads_files",
    )
    flye_type.add_argument(
        "--nano-hq", dest="flye_read_type", action="store_const",
        const="--nano-hq",
        help="Flye ONT Q20/high-quality mode; paths come from --input_reads_files",
    )
    command.set_defaults(flye_read_type="--nano-raw")
    command.add_argument("-o", "--output", default="metabaw_result", help="result directory")
    command.add_argument(
        "--tmp-files",
        help="temporary directory; relative names are created under --output (default: <output>/tmp)",
    )
    command.add_argument(
        "--delete-tmp-files",
        action="store_true",
        help="delete the temporary directory after a successful run",
    )
    command.add_argument(
        "-t",
        "--threads",
        type=int,
        default=1,
        help=(
            "total CPU thread budget shared by concurrently running sample "
            "tasks; limited to permitted logical CPU capacity (affinity/cgroup quota)"
        ),
    )
    command.add_argument(
        "--max-memory",
        type=float,
        default=default_max_memory_gib,
        help=(
            "maximum combined resident memory in GiB for all workflow task "
            "process trees; defaults to the detected machine/cgroup memory "
            "total; binning caps this at 80%% of currently available RAM and "
            "divides it among concurrent sample slots; exceeding a share terminates the workflow"
        ),
    )
    command.add_argument(
        "--task",
        type=int,
        default=1,
        help="maximum number of samples processed concurrently",
    )
    command.add_argument("--retries", type=int, default=0, help="retries after a task failure")
    command.add_argument(
        "--failure-policy",
        choices=("strict", "best-effort"),
        default="strict",
        help=(
            "handling of a selected binner-chain failure: strict stops new "
            "workflow tasks and returns failure; best-effort continues with "
            "the remaining successful binners"
        ),
    )
    command.add_argument(
        "--no-gpu",
        dest="gpu",
        action="store_false",
        help="disable CUDA probing and GPU execution (default: automatic CUDA detection)",
    )
    command.set_defaults(gpu=None)
    command.add_argument(
        "--max-gpu-memory",
        default=DEFAULT_GPU_MEMORY_BUDGET,
        help=(
            "aggregate GPU memory budget shared across parallel sample tasks; ignored with "
            "--no-gpu; below a binner minimum it forces that binner to CPU "
            "(VAMB/SemiBin2/LorBin: 4G; COMEBin: depends on --batch-size); "
            "accepts bytes, K/M/G/T/P units, or a percentage "
            "(default: request 100%% of visible GPU total, capped to 80%% of free VRAM and divided among --task slots)"
        ),
    )
    command.add_argument(
        "--multi-files", metavar="FILE",
        help="one sample name per line; cross-map all listed samples to each listed existing assembly; unlisted samples remain independent",
    )
    command.add_argument(
        "--tools",
        nargs="+",
        default=None,
        metavar="BINNER",
        help=(
            "metabat2/metadecoder/vamb/comebin/semibin2/lorbin; comma or space "
            "separated (default: short=metabat2,metadecoder,vamb; "
            "long=metadecoder,vamb,lorbin)"
        ),
    )
    command.add_argument(
        "--min-contig-length",
        type=int,
        default=None,
        help="minimum contig length in base pairs (default: 1500)",
    )
    command.add_argument(
        "--minfasta-kbs",
        type=int,
        default=200,
        help="minimum output bin size in kilobases",
    )
    command.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_COMEBIN_BATCH_SIZE,
        help=(
            "COMEBin training batch size; used to calculate its conservative "
            "GPU-memory minimum (upstream/GPU default: 1024; effective CPU "
            "default: 128 unless this option is supplied explicitly)"
        ),
    )
    command.add_argument(
        "--refinement",
        type=str.lower,
        choices=("magscot", "das_tool", "metawrap"),
        default="magscot",
        help="bin refinement program",
    )
    command.add_argument(
        "--quality-control",
        type=str.lower,
        choices=("checkm2", "checkm"),
        default="checkm2",
        help="MAG quality estimation program",
    )
    command.add_argument("--con", type=float, default=10.0, help="final QC maximum contamination (%%); refinement may use a relaxed prefilter")
    command.add_argument("--com", type=float, default=50.0, help="final QC minimum completeness (%%); refinement may use a relaxed prefilter")
    command.add_argument(
        "--quality-score",
        type=float,
        help="optional minimum completeness - 5 * contamination score",
    )
    command.add_argument("--gunc", action="store_true", help="run GUNC contamination checks")
    command.add_argument(
        "--trna",
        action="store_true",
        help="predict and report tRNAs without filtering MAGs",
    )
    command.add_argument(
        "--trna-pass",
        type=int,
        help=(
            "require at least this many distinct tRNA types; "
            "also enables tRNA prediction"
        ),
    )
    command.add_argument(
        "--rrna",
        action="store_true",
        help="predict and report 5S, 16S, and 23S rRNAs without filtering MAGs",
    )
    command.add_argument(
        "--rrna-pass",
        action="store_true",
        help=(
            "require 5S, 16S, and 23S rRNAs; "
            "also enables rRNA prediction"
        ),
    )
    command.add_argument(
        "--dereplication-tool",
        type=str.lower,
        choices=("galah", "drep"),
        default="galah",
        help="MAG dereplication program",
    )
    command.add_argument(
        "--ani",
        type=float,
        default=99.0,
        help="dereplication ANI threshold in percent (default: 99)",
    )
    command.add_argument(
        "--min-aligned-fraction",
        type=float,
        default=30.0,
        help="minimum aligned fraction in percent for dereplication (default: 30)",
    )
    command.add_argument(
        "-x",
        "--mag-suffix",
        default="fa",
        help=(
            "MAG file extension; propagated to CheckM2/CheckM, GUNC, GTDB-Tk, "
            "and dereplication tools (default: fa)"
        ),
    )
    command.add_argument(
        "--dereplication_tool",
        dest="dereplication_tool",
        type=str.lower,
        choices=("galah", "drep"),
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    command.add_argument(
        "--environment",
        choices=(
            "human_gut",
            "dog_gut",
            "ocean",
            "soil",
            "cat_gut",
            "human_oral",
            "mouse_gut",
            "pig_gut",
            "built_environment",
            "wastewater",
            "chicken_caecum",
            "global",
        ),
        help=(
            "SemiBin2 pretrained environment model; effective only when "
            "SemiBin2 is selected with --tools, then passed to SemiBin2; "
            "otherwise ignored"
        ),
    )
    command.add_argument(
        "--tag-contigs",
        action="store_true",
        help="rename final contigs using the bin_name_number convention",
    )
    command.add_argument(
        "--advanced-arg",
        action="append",
        default=[],
        metavar="TOOL=ARGS",
        help="expert escape hatch; repeat for multiple tools",
    )
    command.set_defaults(func=command_bin)

    command = subparsers.add_parser(
        "annotation",
        help=(
            "run taxonomic assignment, abundance estimation, niche classification, "
            "KEGG Orthologues (KOs), Carbohydrate-Active enZYmes (CAZymes) and "
            "hydrogen metabolism annotations"
        ),
        description=(
            "Classify MAGs, quantify abundance, classify ecological niches, and "
            "annotate KEGG orthologs, CAZymes, hydrogenases, and hydrogen-"
            "metabolism terminal enzymes."
        ),
        formatter_class=MetaBAWHelpFormatter,
        add_help=False,
        allow_abbrev=False,
    )
    command.add_argument(
        "-h", "--help", action="help", help="show all options and exit"
    )
    command.add_argument(
        "--input_reads_files", required=True, metavar="TSV",
        help="SAMPLE<TAB>READ1,READ2 (paired short reads) or SAMPLE<TAB>READ (long reads); same format as binning",
    )
    command.add_argument(
        "--input_genome_files", required=True, metavar="FILE",
        help="one genome FASTA path per line, without a sample-name column; .fa/.fna/.fasta and gzip variants are supported",
    )
    command.add_argument(
        "--no-place-species",
        dest="place_species",
        action="store_false",
        default=True,
        help="disable GTDB-Tk species placement; placement is enabled when this option is omitted "
             "(default: not selected)",
    )
    command.add_argument(
        "--gtdbtk-tmpdir", default="/tmp", metavar="DIR",
        help="local filesystem parent for a private GTDB-Tk subprocess/IPC temporary directory; "
             "must not be NFS; pplacer scratch remains in the workflow workspace (default: /tmp)",
    )._metabaw_full_help_only = True
    command.add_argument(
        "--methods",
        nargs="+",
        default=["relative_abundance", "rpkm", "tpm", "mean"],
        metavar="METHOD",
        help="abundance outputs: relative_abundance, rpkm, tpm, mean; comma or space separated",
    )
    command.add_argument(
        "-o", "--output", default="metabaw_annotation_result", help="result directory"
    )
    command.add_argument(
        "--tmp-files",
        help="temporary directory; relative names are created under --output (default: <output>/tmp)",
    )
    command.add_argument(
        "--delete-tmp-files",
        action="store_true",
        help="delete the temporary directory after a successful run",
    )
    command.add_argument(
        "--output-file-suffix", default=".tsv", help="suffix for generated abundance tables"
    )
    command.add_argument(
        "-t",
        "--threads",
        type=int,
        default=1,
        help=(
            "total CPU thread budget shared by concurrently running sample "
            "tasks; also passed directly to GTDB-Tk --cpus"
        ),
    )
    command.add_argument(
        "--max-memory",
        type=float,
        default=default_max_memory_gib,
        help=(
            "maximum combined resident memory in GiB for all workflow task "
            "process trees; defaults to the detected machine/cgroup memory "
            "total; generated GTDB-Tk classification uses one pplacer "
            "CPU plus disk scratch and a 160 GiB preflight minimum (not a peak prediction), while a complete "
            "--gtdbtk_res skips that requirement; "
            "an undersized known configuration is rejected before tasks start"
        ),
    )
    command.add_argument(
        "--task",
        type=int,
        default=1,
        help=(
            "maximum number of read samples or MAGs processed concurrently; "
            "the total -t/--threads budget is divided evenly across these slots"
        ),
    )
    command.add_argument("--retries", type=int, default=0, help="retries after a task failure")
    command.add_argument(
        "--niche-rank",
        choices=("domain", "phylum", "class", "order", "family", "genus", "species", "strain"),
        default="family",
        help="ecological-niche level (default: family)",
    )
    command.add_argument(
        "--niche_classify_method", "--niche-classify-method",
        choices=("cv", "occupancy"),
        default="cv",
        help=(
            "ecological-niche classification criterion: CV rank quartiles "
            "(generalist/intermediate/specialist) or occupancy (default: cv)"
        ),
    )
    command.add_argument(
        "--cv-correct", action="store_true",
        help=(
            "rank taxa by corrected CV (CV - sqrt(sample_count/total_taxon_reads)) "
            "instead of raw CV; requires --niche_classify_method cv"
        ),
    )
    command.add_argument(
        "--niche_abundance", "--niche-abundance",
        choices=("relative_abundance", "rpkm", "tpm", "mean"),
        default="relative_abundance",
        help="abundance metric used for niche classification; must be included in --methods when niche is enabled",
    )
    # Retain old commands as a hidden alias, with one canonical recorded value.
    command.add_argument(
        "--niche-method", dest="niche_classify_method",
        type=lambda value: "occupancy" if value == "prevalence" else value,
        choices=("cv", "occupancy"), default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    command.add_argument(
        "--no-niche", action="store_true",
        help="disable ecological niche classification; otherwise enabled only with at least 4 genomes AND 4 read samples",
    )
    command.add_argument(
        "--no-kegg",
        dest="kegg",
        action="store_false",
        help="disable the default KEGG Orthology annotation (default: not selected)",
    )
    command.add_argument(
        "--no-cazy",
        dest="cazy",
        action="store_false",
        help="disable the default CAZy annotation (default: not selected)",
    )
    command.add_argument(
        "--no-hyd",
        dest="hydrogenase",
        action="store_false",
        help=(
            "disable the default hydrogenase and hydrogen-metabolism "
            "terminal-enzyme annotation "
            "(default: not selected)"
        ),
    )
    command.set_defaults(kegg=True, cazy=True, hydrogenase=True)
    gtdbtk_input = command.add_mutually_exclusive_group()
    gtdbtk_input.add_argument(
        "--gtdbtk-db",
        dest="gtdbtk_data",
        help="GTDB-Tk database path (default: GTDBTK_DATA_PATH environment variable)",
    )
    gtdbtk_input.add_argument(
        "--gtdbtk_res",
        help=(
            "existing GTDB-Tk output directory; validate complete taxonomy for "
            "every input MAG and skip GTDB-Tk classification"
        ),
    )
    command.set_defaults(func=command_annotation)

    command = subparsers.add_parser(
        "check",
        help="check software, runtime environments and databases",
        description=(
            "Check selected dependencies, validate database paths, and ask before "
            "installing missing software or saving replacement database locations."
        ),
        formatter_class=MetaBAWHelpFormatter,
    )
    selection = command.add_mutually_exclusive_group()
    selection.add_argument(
        "--all",
        action="store_true",
        help=(
            "check every supported program, environment, database, and reference "
            "asset; prompt for invalid or unset database paths"
        ),
    )
    selection.add_argument(
        "-e",
        "--essential",
        action="store_true",
        help="check only dependencies used by the default binning and annotation workflows",
    )
    selection.add_argument(
        "--scope",
        choices=("bin", "annotation"),
        help="check dependencies for one workflow module",
    )
    command.set_defaults(essential=True)
    command.add_argument(
        "--magscot-dir",
        default=str(configured_magscot_directory()),
        help="directory containing MAGScoT.R and hmm/ to validate and save",
    )
    command.add_argument(
        "--database-dir",
        default="~/.cache/metabaw/databases",
        help="parent directory used when downloading databases",
    )
    command.add_argument(
        "--comebin-env",
        "--comebin-environment",
        dest="comebin_env",
        default=configured_isolated_environment("comebin"),
        help=(
            "COMEBin Python 3.7 Conda environment name or prefix to validate, "
            "install, and save for later commands"
        ),
    )
    command.add_argument(
        "--checkm2-env",
        "--checkm2-environment",
        dest="checkm2_env",
        default=configured_isolated_environment("checkm2"),
        help=(
            "CheckM2 Python 3.12 Conda environment name or prefix to validate, "
            "install, and save for later commands"
        ),
    )
    command.add_argument(
        "--metawrap-env",
        "--metawrap-environment",
        dest="metawrap_env",
        default=configured_isolated_environment("metawrap"),
        help=(
            "MetaWRAP Python 2.7 Conda environment name or prefix to validate, "
            "install, and save for later commands"
        ),
    )
    command.add_argument(
        "--lorbin-env",
        "--lorbin-environment",
        dest="lorbin_env",
        default=configured_isolated_environment("lorbin"),
        help=(
            "LorBin Python 3.10 Conda environment name or prefix to validate, "
            "install from pinned official source, and save for later commands; "
            "wrong-Python environments are preserved and replaced after installation approval; "
            "when explicitly supplied without --scope, select the LorBin check"
        ),
    )
    command.add_argument(
        "--install-databases",
        action="store_true",
        help=(
            "download supported missing databases; legacy CheckM data must be "
            "supplied as an existing path"
        ),
    )
    command.add_argument(
        "--require-cuda",
        action="store_true",
        help=(
            "offer to repair CUDA PyTorch and MetaDecoder CuPy runtimes, then return "
            "an error unless the driver and each selected environment pass GPU tests"
        ),
    )
    command.add_argument(
        "--checkm-db",
        help=(
            "existing legacy CheckM data root containing .dmanifest, hmms/, "
            "genome_tree/, distributions/, pfam/, selected_marker_sets.tsv, and "
            "taxon_marker_sets.tsv (default: CHECKM_DATA_PATH or saved configuration)"
        ),
    )
    command.add_argument(
        "--checkm2-db",
        help=(
            "existing non-empty CheckM2 .dmnd database file or parent directory "
            "(default: CHECKM2DB or saved configuration)"
        ),
    )
    command.add_argument(
        "--gunc-db",
        help=(
            "existing non-empty GUNC .dmnd database file or parent directory "
            "(default: GUNC_DB or saved configuration)"
        ),
    )
    command.add_argument(
        "--gtdbtk-db",
        dest="gtdbtk_data",
        help=(
            "existing unpacked GTDB-Tk reference-data root (default: "
            "GTDBTK_DATA_PATH or saved configuration)"
        ),
    )
    command.add_argument(
        "--kegg-db",
        help=(
            "existing KOfam database root containing ko_list and profiles/ "
            "(default: KOFAM_DB or saved check configuration)"
        ),
    )
    command.add_argument(
        "--dbcan-db",
        help=(
            "existing run_dbCAN database root containing CAZy.dmnd, "
            "dbCAN.hmm or dbCAN.txt, dbCAN-sub.hmm or dbCAN_sub.hmm, and "
            "fam-substrate-mapping.tsv "
            "(default: DBCAN_DB or saved check configuration)"
        ),
    )
    command.add_argument(
        "--hydrogenase-db",
        help=(
            "existing hydrogenase database root containing hyddb.all.fa, "
            "FeFe.dmnd, Terminal.dmnd, and hyd_id-name.script.txt "
            "(default: HYDROGENASE_DB or saved check configuration)"
        ),
    )
    command.set_defaults(func=command_check)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    args._invocation = ["metabaw", *raw_argv]
    args._invocation_cwd = str(Path.cwd())
    print("MetaBAW is Running.", flush=True)
    print(f"Command: {shlex.join(['metabaw', *raw_argv])}", flush=True)
    try:
        code = args.func(args)
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[{_elapsed_clock()}] [ERROR] {exc}", file=sys.stderr, flush=True)
        code = 2
    except KeyboardInterrupt:
        print(
            f"[{_elapsed_clock()}] [ABORTED] Interrupted by user",
            file=sys.stderr,
            flush=True,
        )
        code = 130
    if code == 0:
        print("ALL DONE.", flush=True)
    raise SystemExit(code)
