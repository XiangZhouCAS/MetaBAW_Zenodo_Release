"""Recognize CPU-only command differences without hiding analysis options."""
from __future__ import annotations

import json
import re
import shlex
from typing import Mapping


def checkpoint_task_id(task_id: str) -> str:
    """Keep the historical checkpoint key for a presentation-only task rename."""
    prefix = "04.export_refined_bins."
    if task_id.startswith(prefix) and len(task_id) > len(prefix):
        return "04.bins." + task_id[len(prefix):]
    return task_id


def current_task_id(task_id: str) -> str:
    """Upgrade only the former refined-MAG export task prefix."""
    prefix = "04.bins."
    if task_id.startswith(prefix) and len(task_id) > len(prefix):
        return "04.export_refined_bins." + task_id[len(prefix):]
    return task_id


# Short flags are deliberately tool-specific: minimap2 -p is an alignment
# threshold, COMEBin -p is a BAM directory, and Prodigal -p is a gene-call mode.
THREAD_OPTIONS = {
    "megahit": {"--num-cpu-threads"},
    "flye": {"--threads", "-t"},
    "bowtie2-build": {"--threads"},
    "bowtie2": {"--threads", "-p"},
    "minimap2": {"-t"},
    "minibwa": {"-t"},
    "samtools": {"-@", "--threads"},
    "metabat2": {"-t", "--numThreads"},
    "vamb": {"-p", "--threads"},
    "metadecoder": {"--threads"},
    "run_comebin.sh": {"-t"},
    "SemiBin2": {"--threads", "-p"},
    "LorBin": {"--num_process"},
    "lorbin_compat.py": {"--num_process"},
    "parallel": {"--jobs", "-j"},
    "hmmsearch": {"--cpu"},
    "DAS_Tool": {"--threads", "-t"},
    "metawrap": {"-t"},
    "checkm": {"-t", "--threads"},
    "checkm2": {"--threads", "-t"},
    "gunc": {"--threads"},
    "tRNAscan-SE": {"--thread", "--threads"},
    "barrnap": {"--threads"},
    "galah": {"--threads", "-t"},
    "dRep": {"-p"},
    "gtdbtk": {"--cpus", "--pplacer_cpus"},
    "coverm": {"--threads", "-t"},
    "exec_annotation": {"--cpu"},
    "run_dbcan": {"--threads"},
    "blastp": {"-num_threads"},
    "diamond": {"--threads", "-p"},
    "metabaw.internal": {"--threads", "--pplacer-threads"},
}
THREAD_ENVIRONMENT = frozenset({
    "BLIS_NUM_THREADS", "MKL_NUM_THREADS", "MKL_DYNAMIC",
    "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS", "OMP_THREAD_LIMIT", "OMP_DYNAMIC",
    "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
    "TF_NUM_INTEROP_THREADS", "TF_NUM_INTRAOP_THREADS",
    "KMP_DEVICE_THREAD_LIMIT", "KMP_ALL_THREADS", "KMP_TEAMS_THREAD_LIMIT",
    "METABAW_MAX_GPU_MEMORY",
})


def resume_environment(environment: Mapping[str, str]) -> dict[str, str]:
    return {key: value for key, value in sorted(environment.items())
            if key not in THREAD_ENVIRONMENT}


def _name(value: str) -> str:
    return value.replace("\\", "/").rsplit("/", 1)[-1]


def _program(tokens: list[str]) -> tuple[str, int]:
    index = 0
    while index < len(tokens) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tokens[index]):
        index += 1
    if index >= len(tokens):
        return "", index
    if _name(tokens[index]) in {"conda", "mamba", "micromamba"}:
        index += 1
        if index >= len(tokens) or tokens[index] != "run":
            return "", index
        index += 1
        while index < len(tokens) and tokens[index].startswith("-"):
            option = tokens[index]
            index += 1
            if option in {"-n", "--name", "-p", "--prefix", "--cwd"}:
                index += 1
            elif option not in {"--no-capture-output", "--live-stream"} and "=" not in option:
                return "", index
    if index >= len(tokens):
        return "", index
    name = _name(tokens[index])
    index += 1
    if re.fullmatch(r"python(?:[0-9.]+)?(?:\.exe)?", name):
        while index < len(tokens) and tokens[index] in {"-B", "-u", "-s", "-E"}:
            index += 1
        if index < len(tokens) and tokens[index] == "-m":
            index += 1
            if index < len(tokens):
                name = tokens[index]
                index += 1
        elif index < len(tokens) and not tokens[index].startswith("-"):
            name = _name(tokens[index])
            index += 1
        else:
            return "", index
    return name, index


def _helper_prefix(tokens: list[str], start: int, program: str) -> list[str]:
    """Canonicalize MetaBAW's own installation, not upstream environments.

    Moving/reinstalling MetaBAW can change sys.executable from python to
    python3.11. Its location is a launcher detail, not a contig-filter option.
    Conda run's frontend location is likewise not an environment selection.
    Preserve Conda environment names/prefixes, all analysis arguments, and every
    external executable/script path. Strict execution hashes remain unchanged.
    """
    prefix = tokens[:start]
    frontend = 0
    while frontend < len(prefix) and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", prefix[frontend]):
        frontend += 1
    if (frontend + 1 < len(prefix) and _name(prefix[frontend]) in {"conda", "mamba", "micromamba"}
            and prefix[frontend + 1] == "run"):
        prefix[frontend] = _name(prefix[frontend])
    module = program == "metabaw.internal" and start >= 3 and tokens[start - 2] == "-m"
    script = (program == "lorbin_compat.py" and start >= 2
              and tokens[start - 1].replace("\\", "/").endswith("/metabaw/lorbin_compat.py"))
    if not (module or script):
        return prefix
    index = start - (3 if module else 2)
    while index >= 0 and tokens[index] in {"-B", "-u", "-s", "-E"}:
        index -= 1
    if index < 0 or not re.fullmatch(r"python(?:[0-9.]+)?(?:\.exe)?", _name(tokens[index])):
        return prefix
    prefix[index] = "@METABAW_PYTHON@"
    if script:
        prefix[-1] = "@METABAW_LORBIN_COMPAT@"
    return prefix


def _normalize_segment(tokens: list[str], depth: int, launcher_independent: bool = False) -> tuple[list[str], list[int]]:
    program, start = _program(tokens)
    options = THREAD_OPTIONS.get(program, set())
    normalized = _helper_prefix(tokens, start, program) if launcher_independent else tokens[:start]
    counts: list[int] = []
    index = start
    while index < len(tokens):
        token = tokens[index]
        option, separator, value = token.partition("=")
        consumed = 1
        if program == "megahit" and option == "--memory":
            memory_value = value if separator else (tokens[index + 1] if index + 1 < len(tokens) else "")
            if memory_value == "${METABAW_MEGAHIT_MEMORY_BYTES:-0.9}":
                # This exact managed expression depends on the runtime RAM
                # allocation. Preserve user-supplied memory/analysis options.
                index += 1 if separator else 2
                continue
        if not separator and option in options and index + 1 < len(tokens):
            value = tokens[index + 1]
            consumed = 2
        if option not in options:
            for short in options:
                if short.startswith("--"):
                    continue
                if token.startswith(short) and re.fullmatch(r"\d+", token[len(short):]):
                    option, value = short, token[len(short):]
                    break
        if option in options and re.fullmatch(r"\d+", value):
            counts.append(int(value))
            index += consumed
            continue
        # COMEBin's isolated command is a single quoted argument, not a shell
        # command at the outer level. Normalize it separately, preserving all
        # non-thread arguments (batch size, device, data paths, etc.).
        if program == "metabaw.internal" and option == "--command" and depth < 3:
            if separator:
                nested, nested_counts = _normalize_command(value, depth + 1, launcher_independent=launcher_independent)
                normalized.append("--command=" + json.dumps(nested))
                counts.extend(nested_counts)
                index += 1
                continue
            if index + 1 < len(tokens):
                nested, nested_counts = _normalize_command(tokens[index + 1], depth + 1, launcher_independent=launcher_independent)
                normalized.extend((token, json.dumps(nested)))
                counts.extend(nested_counts)
                index += 2
                continue
        normalized.append(token)
        index += 1
    return normalized, counts


def _normalize_command(command: str, depth: int = 0, *, launcher_independent: bool = False) -> tuple[tuple[str, ...], tuple[int, ...]]:
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=";&|<>()")
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        return ("unparsed-command", command), ()
    normalized: list[str] = []
    counts: list[int] = []
    segment: list[str] = []
    for token in tokens:
        if re.fullmatch(r"[;&|()]+", token):
            values, threads = _normalize_segment(segment, depth, launcher_independent)
            normalized.extend(values)
            counts.extend(threads)
            normalized.append(token)
            segment = []
        else:
            segment.append(token)
    values, threads = _normalize_segment(segment, depth, launcher_independent)
    normalized.extend(values)
    counts.extend(threads)
    return tuple(normalized), tuple(counts)


def resume_command(command: str, *, launcher_independent: bool = False) -> tuple[str, ...]:
    return _normalize_command(command, launcher_independent=launcher_independent)[0]


def command_thread_counts(command: str) -> tuple[int, ...]:
    """CPU candidates for validating hashes produced by older executors."""
    return _normalize_command(command)[1]
