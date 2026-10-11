from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Iterable, Iterator, TextIO

from .discovery import normalize_sample_name
from .checkm_compat import run_checkm
from .metadecoder_coverage import run_coverage as run_metadecoder_coverage
from .refinement import read_empty_refinement, write_empty_refinement


FASTA_SUFFIXES = (".fa", ".fna", ".fasta", ".fa.gz", ".fna.gz", ".fasta.gz")


def _multiprocessing_socket_path_too_long(configured: Path) -> bool:
    """Return whether Python's expected multiprocessing socket can exceed POSIX limits."""
    socket_probe = configured / "pymp-12345678" / "listener-12345678"
    return len(os.fsencode(str(socket_probe))) >= 100


def _filesystem_type(path: Path) -> str | None:
    """Read Linux mount metadata; use the deepest mount containing this path."""
    try:
        lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    matches = []
    for line in lines:
        fields = line.split()
        try:
            separator = fields.index("-")
            mount = Path(re.sub(r"\\([0-7]{3})", lambda m: chr(int(m[1], 8)), fields[4]))
            if path.is_relative_to(mount):
                matches.append((len(mount.parts), fields[separator + 1]))
        except (ValueError, IndexError):
            continue
    return max(matches)[1] if matches else None


def _gtdbtk_temp_environment(local_parent: Path | None = None) -> tuple[dict[str, str], Path | None]:
    """Use a real private local directory, never a symlink back onto NFS.

    Only short-lived subprocess/IPC files go here. Large pplacer mmap scratch
    files keep their explicitly configured workflow scratch directory.
    """
    environment = os.environ.copy()
    if sys.platform == "win32":
        return environment, None
    parent = Path(local_parent or "/tmp").expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)
    filesystem = _filesystem_type(parent)
    if filesystem in {"nfs", "nfs4", "cifs", "smb3", "fuse.sshfs"}:
        raise ValueError(f"--gtdbtk-tmpdir {parent} is on {filesystem}; choose a short local filesystem path such as /tmp")
    # Validate before creating anything to keep ownership/cleanup unambiguous.
    if _multiprocessing_socket_path_too_long(parent / "mbw-gtdb-12345678"):
        raise ValueError("--gtdbtk-tmpdir is too long for multiprocessing sockets; use a short local path such as /tmp")
    temporary = Path(tempfile.mkdtemp(prefix="mbw-gtdb-", dir=parent))
    environment.update({"TMPDIR": str(temporary), "TMP": str(temporary), "TEMP": str(temporary)})
    print(
        f"[INFO] GTDB-Tk subprocess/IPC files use private directory {temporary} "
        f"(filesystem={filesystem or 'not detected; verify local storage'}); "
        "no symlink to the workflow/NFS temporary directory.",
        flush=True,
    )
    return environment, temporary


def _samtools_version(output: str) -> str | None:
    """Extract the version from a real samtools version line."""
    match = re.search(
        r"(?im)^\s*samtools(?:\s+version)?\s+v?([0-9]+(?:\.[0-9]+){1,2}(?:[-+._a-z0-9]*)?)\s*$",
        output,
    )
    return match.group(1) if match else None


def prepare_samtools_compat(output: Path) -> None:
    """Write a PATH shim that hides warnings before `samtools --version`."""
    executable = shutil.which("samtools")
    if executable is None:
        raise ValueError("samtools was not found in PATH")
    probe = subprocess.run(
        [executable, "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    combined = "\n".join(part for part in (probe.stdout, probe.stderr) if part)
    version = _samtools_version(combined)
    if version is None:
        preview = " | ".join(line.strip() for line in combined.splitlines()[:4])
        raise ValueError(
            "samtools --version did not contain a parseable 'samtools X.Y' line; "
            f"output={preview or '<empty>'}. Reinstall a complete samtools >=1.9 package."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "#!/usr/bin/env sh\n"
        "if [ \"${1:-}\" = \"--version\" ]; then\n"
        f"    printf '%s\\n' {shlex.quote(f'samtools {version}')}\n"
        "    exit 0\n"
        "fi\n"
        f"exec {shlex.quote(executable)} \"$@\"\n",
        encoding="utf-8",
    )
    output.chmod(output.stat().st_mode | 0o111)


def run_gtdbtk_classify(
    genome_dir: Path,
    output_dir: Path,
    extension: str,
    threads: int,
    place_species: bool,
    pplacer_threads: int = 1,
    scratch_dir: Path | None = None,
    ipc_tmpdir: Path | None = None,
) -> None:
    """Run GTDB-Tk while adapting to version-specific placement options."""
    executable = shutil.which("gtdbtk")
    if executable is None:
        raise ValueError("gtdbtk was not found in PATH")
    environment, private_temp = _gtdbtk_temp_environment(ipc_tmpdir)
    try:
        help_probe = subprocess.run(
            [executable, "classify_wf", "--help"],
            capture_output=True,
            text=True,
            check=False,
            env=environment,
        )
        help_text = f"{help_probe.stdout}\n{help_probe.stderr}"
        command = [
            executable,
            "classify_wf",
            "--genome_dir",
            str(genome_dir),
            "--out_dir",
            str(output_dir),
            "--extension",
            extension,
            "--cpus",
            str(threads),
        ]
        if place_species:
            if "--place_species" in help_text:
                command.append("--place_species")
            else:
                print(
                    "[WARNING] This GTDB-Tk version does not support --place_species; "
                    "classification will continue without that optional flag. Upgrade "
                    "GTDB-Tk to enable species placement.",
                    flush=True,
                )
        if "--pplacer_cpus" not in help_text:
            raise RuntimeError(
                "This GTDB-Tk version does not support --pplacer_cpus. "
                "MetaBAW requires this option to keep pplacer within the total "
                "workflow memory budget; install the configured GTDB-Tk 2.7.2 "
                "environment."
            )
        command.extend(["--pplacer_cpus", str(max(1, pplacer_threads))])
        if scratch_dir is not None:
            if "--scratch_dir" not in help_text:
                raise RuntimeError(
                    "This GTDB-Tk version does not support --scratch_dir. "
                    "MetaBAW requires disk-backed pplacer memory for predictable "
                    "workflow memory use; install the configured GTDB-Tk 2.7.2 "
                    "environment."
                )
            shutil.rmtree(scratch_dir, ignore_errors=True)
            scratch_dir.mkdir(parents=True, exist_ok=True)
            command.extend(["--scratch_dir", str(scratch_dir)])
        shutil.rmtree(output_dir, ignore_errors=True)
        result = subprocess.run(command, check=False, env=environment)
    finally:
        if private_temp is not None:
            try:
                # This is only the unique directory allocated above, never its parent.
                shutil.rmtree(private_temp)
            except OSError as exc:
                print(
                    f"[WARNING] Could not remove private GTDB-Tk temporary directory "
                    f"{private_temp}: {exc}",
                    flush=True,
                )
    if result.returncode != 0:
        raise RuntimeError(f"GTDB-Tk classify_wf exited with code {result.returncode}")
    normalized_extension = extension.lstrip(".").lower()
    genomes = sorted(
        path
        for path in genome_dir.iterdir()
        if path.is_file()
        and path.name.lower().endswith(f".{normalized_extension}")
    )
    missing = gtdbtk_result_missing(output_dir, genomes)
    if missing:
        raise RuntimeError(
            "GTDB-Tk classify_wf exited successfully but produced incomplete "
            f"species annotation information: {'; '.join(missing)}"
        )
    (output_dir / ".metabaw.complete").write_text(
        "GTDB-Tk species annotations validated by metaBAW\n",
        encoding="utf-8",
    )


def apply_memory_limit_from_environment() -> None:
    """Apply the workflow ceiling as a secondary per-process safety limit."""
    raw = os.environ.get("METABAW_MAX_MEMORY_GB", "").strip()
    if not raw or os.name != "posix":
        return
    try:
        import resource

        requested = max(1, int(float(raw) * 1024**3))
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        target = requested
        if soft != resource.RLIM_INFINITY:
            target = min(target, soft)
        if hard != resource.RLIM_INFINITY:
            target = min(target, hard)
        resource.setrlimit(resource.RLIMIT_AS, (target, hard))
    except (ImportError, OSError, ValueError):
        return


def open_text(path: Path, mode: str = "rt") -> TextIO:
    if path.name.lower().endswith(".gz"):
        return gzip.open(path, mode, encoding="utf-8")  # type: ignore[return-value]
    return path.open(mode.replace("t", ""), encoding="utf-8")


def iter_fasta(path: Path) -> Iterator[tuple[str, str, str]]:
    header: str | None = None
    chunks: list[str] = []
    with open_text(path) as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.rstrip("\r\n")
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header.split()[0], header, "".join(chunks)
                header = line[1:].strip()
                if not header:
                    raise ValueError(f"Empty FASTA header in {path} at line {line_number}")
                chunks = []
            else:
                if header is None:
                    raise ValueError(f"Sequence found before first header in {path} at line {line_number}")
                chunks.append(line.strip())
    if header is not None:
        yield header.split()[0], header, "".join(chunks)


def write_record(handle: TextIO, identifier: str, sequence: str, width: int = 80) -> None:
    handle.write(f">{identifier}\n")
    for start in range(0, len(sequence), width):
        handle.write(sequence[start : start + width] + "\n")


def fasta_filter(input_path: Path, output_path: Path, minimum_length: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    kept = 0
    with output_path.open("w", encoding="utf-8") as output:
        for identifier, _header, sequence in iter_fasta(input_path):
            if identifier in seen:
                raise ValueError(f"Duplicate contig identifier {identifier!r} in {input_path}")
            seen.add(identifier)
            if len(sequence) >= minimum_length:
                write_record(output, identifier, sequence)
                kept += 1
    if kept == 0:
        raise ValueError(f"No contigs >= {minimum_length} bp in {input_path}")


def rename_fasta_records(input_path: Path, output_path: Path, prefix: str) -> None:
    """Rename FASTA records to PREFIX_1, PREFIX_2, ... in input order."""
    safe_prefix = _safe_filename(prefix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as output:
        for count, (_identifier, _header, sequence) in enumerate(
            iter_fasta(input_path), start=1
        ):
            write_record(output, f"{safe_prefix}_{count}", sequence)
    if count == 0:
        output_path.unlink(missing_ok=True)
        raise ValueError(f"No FASTA records found in {input_path}")


def concatenate_fastas(
    items: list[str],
    output_path: Path,
    minimum_length: int,
    separator: str = "__",
) -> None:
    """Concatenate assemblies while making contig identifiers globally unique."""
    if not separator:
        raise ValueError("FASTA identifier separator cannot be empty")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    kept = 0
    with output_path.open("w", encoding="utf-8") as output:
        for item in items:
            label, raw_path = item.split("=", 1)
            path = Path(raw_path)
            safe_label = _safe_filename(label)
            if separator in safe_label:
                raise ValueError(
                    f"Sample label {safe_label!r} contains the FASTA separator {separator!r}"
                )
            for identifier, _header, sequence in iter_fasta(path):
                if len(items) > 1 and separator in identifier:
                    raise ValueError(
                        f"Contig identifier {identifier!r} contains the FASTA separator "
                        f"{separator!r}"
                    )
                renamed = (
                    f"{safe_label}{separator}{identifier}"
                    if len(items) > 1
                    else identifier
                )
                if renamed in seen:
                    raise ValueError(f"Duplicate contig identifier after concatenation: {renamed!r}")
                seen.add(renamed)
                if len(sequence) >= minimum_length:
                    write_record(output, renamed, sequence)
                    kept += 1
    if kept == 0:
        raise ValueError(f"No contigs >= {minimum_length} bp in the selected assemblies")


def _fasta_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and any(path.name.lower().endswith(suffix) for suffix in FASTA_SUFFIXES)
    )


def _fasta_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".gz"):
        name = name[:-3]
    for suffix in (".fasta", ".fna", ".fa"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def bins_to_map(
    bins_dir: Path,
    output: Path,
    label: str,
    minimum_bin_bp: int = 0,
) -> None:
    files = _fasta_files(bins_dir)
    if not files:
        raise ValueError(f"No FASTA bins found in {bins_dir}")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        for fasta in files:
            records = list(iter_fasta(fasta))
            if sum(len(sequence) for _contig, _header, sequence in records) < minimum_bin_bp:
                continue
            bin_id = f"{label}__{_fasta_stem(fasta)}"
            for contig, _header, _sequence in records:
                writer.writerow((bin_id, contig, label))
                rows_written += 1
    if rows_written == 0:
        raise ValueError(
            f"No bins under {bins_dir} passed the minimum size of "
            f"{minimum_bin_bp} bp"
        )


def normalize_map(input_path: Path, output: Path, label: str, order: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with input_path.open("r", encoding="utf-8-sig", newline="") as source, output.open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        writer = csv.writer(destination, delimiter="\t", lineterminator="\n")
        for row_number, row in enumerate(csv.reader(source, delimiter="\t"), start=1):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) < 2:
                raise ValueError(f"Expected at least two columns in {input_path}, row {row_number}")
            lowered = {row[0].strip().lower(), row[1].strip().lower()}
            if row_number == 1 and lowered & {"clustername", "contigname", "bin", "contig"}:
                continue
            if order == "bin-contig":
                bin_id, contig = row[0].strip(), row[1].strip()
            else:
                contig, bin_id = row[0].strip(), row[1].strip()
            writer.writerow((f"{label}__{bin_id}", contig, label))
            rows_written += 1
    if rows_written == 0:
        raise ValueError(f"No assignments found in {input_path}")


def combine_maps(inputs: list[Path], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8") as destination:
        for input_path in inputs:
            if not input_path.is_file():
                print(f"Skipping unavailable binner assignment map: {input_path}", flush=True)
                continue
            with input_path.open("r", encoding="utf-8") as source:
                for line in source:
                    if line.strip():
                        destination.write(line if line.endswith("\n") else line + "\n")
                        count += 1
    if count == 0:
        raise ValueError("No bin assignments were available to combine")


def _validate_magscot_scores(mapping: Path) -> None:
    """Require the native scoring output before adopting a missing mapping."""
    suffix = ".refined.contig_to_bin.out"
    if not mapping.name.endswith(suffix):
        raise ValueError(f"Cannot identify the MAGScoT score table for {mapping}")
    scores = mapping.with_name(mapping.name.removesuffix(suffix) + ".scores.out")
    if not scores.is_file() or scores.stat().st_size == 0:
        raise ValueError(f"MAGScoT did not produce a current score table: {scores}")
    with scores.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"bin", "Completeness", "Contamination", "score"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Invalid MAGScoT score table columns: {scores}")
        for row in reader:
            if None in row or any(row.get(key) is None for key in required):
                raise ValueError(f"Malformed MAGScoT score table: {scores}")


def finalize_magscot_mapping(
    combined: Path, mapping: Path, run_log: Path | None = None,
) -> None:
    """Validate a refined mapping, a confirmed empty result, or score-only mode.

    MAGScoT intentionally switches to score-only mode when its input contains
    assignments from one binner. In that case it exits successfully without
    writing a refined mapping. MetaBAW keeps the original assignments because
    refinement is impossible but the binner result remains valid.

    With multiple binners, the native no-bin message and a current scoring table
    authorize a header-only mapping, never a fallback to rejected input bins.
    The managed runner calls this helper only after a zero native exit status.
    """
    if mapping.is_file() and mapping.stat().st_size > 0:
        if _bin_assignments(mapping, allow_empty=True):
            return

    assignments: list[tuple[str, str]] = []
    binners: set[str] = set()
    with combined.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle, delimiter="\t"):
            if len(row) < 3:
                continue
            bin_id, contig, binner = (value.strip() for value in row[:3])
            if not bin_id or not contig or not binner:
                continue
            assignments.append((bin_id, contig))
            binners.add(binner)

    if not assignments:
        raise ValueError(f"No valid binner assignments found in {combined}")
    native_output = run_log.read_text(encoding="utf-8", errors="replace") if run_log else ""
    empty_result = re.search(
        r"^\s*No bins with score\s*>=\s*"
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
        r"\s+were found in the dataset\.\s*$",
        native_output,
        re.MULTILINE,
    )
    score_only = len(binners) == 1 and (
        run_log is None
        or "Scoring bins only." in native_output
    )
    if not empty_result and not score_only:
        raise ValueError(
            "MAGScoT completed without a valid refined mapping across "
            f"{len(binners)} binners; native output did not confirm an empty "
            "result or single-binner score-only mode"
        )
    if run_log is not None:
        _validate_magscot_scores(mapping)

    mapping.parent.mkdir(parents=True, exist_ok=True)
    with mapping.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("binnew", "contig"))
        if not empty_result:
            writer.writerows(assignments)
    if empty_result:
        print(
            "[MAGScoT EMPTY] 0 refined bins: no bins met the native refinement "
            "thresholds. Recording an empty result for this sample; other "
            "samples continue and original binner outputs remain unchanged.",
            flush=True,
        )
        return
    print(
        "MAGScoT received one available binner and entered score-only mode; "
        f"preserving assignments from {next(iter(binners))}.",
        flush=True,
    )


def run_magscot_refinement(
    combined: Path, mapping: Path, run_log: Path, native_command_json: str,
) -> None:
    """Stream native output and distinguish a valid no-bin result from failure."""
    command = json.loads(native_command_json)
    if not isinstance(command, list) or not command or not all(
        isinstance(argument, str) and argument for argument in command
    ):
        raise ValueError("MAGScoT native command must be a nonempty JSON list of arguments")
    suffix = ".refined.contig_to_bin.out"
    if not mapping.name.endswith(suffix):
        raise ValueError(f"Unexpected MAGScoT mapping filename: {mapping}")
    prefix = mapping.name.removesuffix(suffix)
    # These are exactly this sample's owned native outputs, not source bins.
    # Removing old scoring files prevents adoption of a zero-exit early abort.
    for path in (
        mapping, mapping.with_name(prefix + ".scores.out"),
        mapping.with_name(prefix + ".refined.out"),
    ):
        path.unlink(missing_ok=True)
    mapping.parent.mkdir(parents=True, exist_ok=True)
    run_log.parent.mkdir(parents=True, exist_ok=True)
    with run_log.open("w", encoding="utf-8") as log:
        with subprocess.Popen(
            command, cwd=mapping.parent, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace", bufsize=1,
        ) as process:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)
    finalize_magscot_mapping(combined, mapping, run_log=run_log)


def map_for_dastool(input_path: Path, output: Path) -> None:
    """Convert metaBAW's bin-contig-label map to DAS Tool's contig-bin table."""
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with input_path.open("r", encoding="utf-8-sig", newline="") as source, output.open(
        "w", encoding="utf-8", newline=""
    ) as destination:
        writer = csv.writer(destination, delimiter="\t", lineterminator="\n")
        for row in csv.reader(source, delimiter="\t"):
            if len(row) >= 2 and row[0].strip() and row[1].strip():
                writer.writerow((row[1].strip(), row[0].strip()))
                count += 1
    if count == 0:
        raise ValueError(f"No assignments found in {input_path}")


def _available_fasta_directory(candidates: Iterable[Path]) -> Path | None:
    for directory in candidates:
        if directory.is_dir() and _fasta_files(directory):
            return directory
    return None


def resolve_semibin_multisample(
    tool_root: Path,
    samples: list[str],
    output_root: Path,
    completion_marker: Path,
) -> None:
    """Normalize version-dependent SemiBin2 multi-sample output directories."""
    selected: list[tuple[str, Path]] = []
    for sample in samples:
        sample_root = tool_root / "samples" / sample
        source = _available_fasta_directory(
            (
                sample_root / "output_bins",
                sample_root / "output_recluster_bins",
                sample_root / "output_prerecluster_bins",
            )
        )
        if source is None:
            raise ValueError(
                "SemiBin2 generated no FASTA bins for sample "
                f"{sample!r}; checked output_bins, output_recluster_bins, "
                "and output_prerecluster_bins"
            )
        selected.append((sample, source))

    staging = output_root.with_name(f".{output_root.name}.metabaw-staging")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        for sample, source in selected:
            destination = staging / sample
            destination.mkdir()
            files = _fasta_files(source)
            for fasta in files:
                _link_or_copy(fasta, destination / fasta.name)
            with (destination / "manifest.tsv").open(
                "w",
                encoding="utf-8",
                newline="",
            ) as handle:
                writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
                writer.writerow(("source_directory", "file"))
                writer.writerows((str(source), fasta.name) for fasta in files)
        if output_root.exists():
            shutil.rmtree(output_root)
        staging.replace(output_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    completion_marker.parent.mkdir(parents=True, exist_ok=True)
    completion_marker.write_text(
        "\n".join(f"{sample}\t{source}" for sample, source in selected) + "\n",
        encoding="utf-8",
    )


def _parse_binner_spec(value: str) -> tuple[str, Path, Path]:
    try:
        payload = json.loads(value)
        name = str(payload["name"])
        mapping = Path(payload["mapping"])
        bins = Path(payload["bins"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid binner specification: {value!r}") from exc
    return name, mapping, bins


def run_dastool_refinement(
    specs: list[str],
    assembly: Path,
    output_prefix: Path,
    output_bins: Path,
    completion_marker: Path,
    threads: int,
    extra_args: str,
) -> None:
    """Run DAS Tool with every complete binner result that is available."""
    available: list[tuple[str, Path]] = []
    input_root = output_prefix.parent / "dastool_inputs"
    if input_root.exists():
        shutil.rmtree(input_root)
    input_root.mkdir(parents=True)
    for value in specs:
        name, mapping, bins = _parse_binner_spec(value)
        if (
            not mapping.is_file()
            or not bins.is_dir()
            or not _fasta_files(bins)
        ):
            print(f"Skipping unavailable DAS Tool input: {name}", flush=True)
            continue
        converted = input_root / f"{_safe_filename(name)}.contig2bin.tsv"
        map_for_dastool(mapping, converted)
        available.append((name, converted))
    if not available:
        raise ValueError(
            "DAS Tool cannot run because none of the selected binners "
            "generated complete bins and assignment maps"
        )

    if output_bins.exists():
        shutil.rmtree(output_bins)
    if completion_marker.exists():
        completion_marker.unlink()
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    # These native result files are owned by this sample. Never accept stale
    # evidence from an earlier run (including a stopped no-bin run).
    for suffix in ("_DASTool.log", "_allBins.eval", "_DASTool_summary.tsv",
                   "_DASTool_contig2bin.tsv"):
        Path(str(output_prefix) + suffix).unlink(missing_ok=True)
    command = [
        "DAS_Tool",
        "-i",
        ",".join(str(path) for _name, path in available),
        "-l",
        ",".join(name for name, _path in available),
        "-c",
        str(assembly),
        "-o",
        str(output_prefix),
        "--search_engine",
        "diamond",
        "--write_bins",
        "--write_bin_evals",
        "-t",
        str(threads),
        *shlex.split(extra_args),
    ]
    print(
        "DAS Tool inputs: " + ", ".join(name for name, _path in available),
        flush=True,
    )
    result = subprocess.run(command, check=False)
    if _available_fasta_directory((output_bins,)) is None:
        # DAS Tool 1.1.7 intentionally aborts before exporting bins when no
        # score reaches the cutoff. This is the only tolerated native abort:
        # require its final diagnostic plus a fresh, agreeing evaluation.
        if result.returncode not in (0, 1) or not _dastool_no_bins(output_prefix):
            if result.returncode:
                raise subprocess.CalledProcessError(result.returncode, command)
            raise ValueError(f"DAS Tool generated no FASTA bins in {output_bins}")
        write_empty_refinement(output_bins, "das_tool")
        print("[DAS TOOL EMPTY] 0 refined bins: none met the native score threshold.", flush=True)
    elif result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)
    completion_marker.write_text(
        "\n".join(name for name, _path in available) + "\n",
        encoding="utf-8",
    )


def _dastool_no_bins(output_prefix: Path) -> bool:
    log = Path(str(output_prefix) + "_DASTool.log")
    evaluation = Path(str(output_prefix) + "_allBins.eval")
    if not log.is_file() or not evaluation.is_file():
        return False
    lines = [line.strip() for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
    match = re.fullmatch(
        r"No bins with bin-score >([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?) "
        r"found\. Adjust score_threshold to report bins with lower quality\.",
        lines[-1] if lines else "",
    )
    if match is None:
        return False
    cutoff = float(match[1])
    if not math.isfinite(cutoff) or not 0 <= cutoff <= 1:
        return False
    # Selected-bin exports must not contradict the no-bin diagnostic.
    for suffix in ("_DASTool_contig2bin.tsv", "_DASTool_summary.tsv"):
        path = Path(str(output_prefix) + suffix)
        if path.exists() and path.stat().st_size:
            return False
    with evaluation.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"bin", "bin_set", "bin_score", "SCG_completeness", "SCG_redundancy"}
        if not required.issubset(reader.fieldnames or []):
            return False
        for row in reader:
            if None in row or any(not row.get(key) for key in required):
                return False
            try:
                values = [float(row[key]) for key in
                          ("bin_score", "SCG_completeness", "SCG_redundancy")]
            except (TypeError, ValueError):
                return False
            if not all(math.isfinite(value) for value in values) or values[0] >= cutoff:
                return False
    return True


def run_metawrap_refinement(
    items: list[str],
    runner_json: str,
    output_root: Path,
    output_bins: Path,
    stats: Path,
    completion_marker: Path,
    threads: int,
    minimum_completeness: float,
    maximum_contamination: float,
    extra_args: str,
) -> None:
    """Run MetaWRAP with up to three FASTA-only binner directories."""
    try:
        decoded_runner = json.loads(runner_json)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid MetaWRAP runner JSON") from exc
    if not isinstance(decoded_runner, list) or not all(
        isinstance(argument, str) for argument in decoded_runner
    ):
        raise ValueError("MetaWRAP runner JSON must be a list of strings")
    available: list[tuple[str, Path]] = []
    for item in items:
        name, raw_path = item.split("=", 1)
        path = Path(raw_path)
        if path.is_dir() and _fasta_files(path):
            available.append((name, path))
        else:
            print(f"Skipping unavailable MetaWRAP input: {name}", flush=True)
    if not available:
        raise ValueError(
            "MetaWRAP cannot run because none of the selected binners "
            "generated complete FASTA bins"
        )
    if len(available) > 3:
        raise ValueError("MetaWRAP bin_refinement accepts at most three binner directories")

    staged_root = output_root.with_name(f"{output_root.name}_inputs")
    if staged_root.exists():
        shutil.rmtree(staged_root)
    staged_root.mkdir(parents=True)
    staged: list[tuple[str, Path]] = []
    staged_names: set[str] = set()
    staged_rows: list[tuple[str, str, str, str]] = []
    for name, path in available:
        staged_name = _safe_filename(name)
        if staged_name in staged_names:
            raise ValueError(f"Duplicate MetaWRAP input name after normalization: {name}")
        staged_names.add(staged_name)
        destination = staged_root / staged_name
        destination.mkdir()
        for fasta in _fasta_files(path):
            _link_or_copy(fasta, destination / fasta.name)
            staged_rows.append((name, str(path), str(destination), fasta.name))
        staged.append((name, destination))
    with (staged_root / "manifest.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("binner", "source_directory", "staged_directory", "fasta"))
        writer.writerows(staged_rows)

    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.parent.mkdir(parents=True, exist_ok=True)
    command = [
        *decoded_runner,
        "metawrap",
        "bin_refinement",
        "-o",
        str(output_root),
        "-t",
        str(threads),
    ]
    for letter, (_name, path) in zip(("A", "B", "C"), staged, strict=False):
        command.extend((f"-{letter}", str(path)))
    command.extend(
        (
            "-c",
            f"{minimum_completeness:g}",
            "-x",
            f"{maximum_contamination:g}",
            *shlex.split(extra_args),
        )
    )
    print(
        "MetaWRAP FASTA-only inputs: "
        + ", ".join(f"{name}={path}" for name, path in staged),
        flush=True,
    )
    subprocess.run(command, check=True)
    if not stats.is_file() or stats.stat().st_size == 0:
        raise ValueError(f"MetaWRAP did not generate the expected quality table: {stats}")
    if _available_fasta_directory((output_bins,)) is None:
        # The root was cleared above, so the directory and stats are current.
        # A missing directory or a nonempty/malformed quality table is not a
        # completed zero-bin result, even if the upstream shell returned zero.
        rows, _name, _completeness, _contamination = _read_checkm2(stats)
        if not output_bins.is_dir() or rows:
            raise ValueError(f"MetaWRAP generated no FASTA bins in {output_bins}")
        write_empty_refinement(output_bins, "metawrap")
        print("[METAWRAP EMPTY] 0 refined bins passed the final quality filter.", flush=True)
    completion_marker.write_text(
        "\n".join(name for name, _path in available) + "\n",
        encoding="utf-8",
    )


def merge_aemb(items: list[str], output: Path) -> None:
    """Merge strobealign AEMB files into VAMB's contigname abundance matrix."""
    samples: list[str] = []
    values: dict[str, dict[str, str]] = defaultdict(dict)
    contig_order: list[str] = []
    for item in items:
        sample, raw_path = item.split("=", 1)
        if sample in samples:
            raise ValueError(f"Duplicate AEMB sample: {sample}")
        samples.append(sample)
        with Path(raw_path).open("r", encoding="utf-8-sig", newline="") as handle:
            for row_number, row in enumerate(csv.reader(handle, delimiter="\t"), start=1):
                if len(row) < 2:
                    continue
                contig, abundance = row[0].strip(), row[1].strip()
                if row_number == 1 and contig.lower() in {"contigname", "contig", "reference"}:
                    continue
                if contig not in values:
                    contig_order.append(contig)
                values[contig][sample] = abundance
    if not contig_order:
        raise ValueError("No contig abundance rows found in strobealign AEMB files")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("contigname", *samples))
        for contig in contig_order:
            writer.writerow((contig, *(values[contig].get(sample, "0") for sample in samples)))


def run_strobealign_aemb(
    contigs: Path,
    reads: list[Path],
    output: Path,
    threads: int,
) -> None:
    """Run strobealign AEMB with adaptive recovery from thread exhaustion."""
    if threads < 1:
        raise ValueError("strobealign threads must be positive")
    if not reads:
        raise ValueError("At least one read file is required for strobealign AEMB")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix + ".partial")
    partial.unlink(missing_ok=True)
    current_threads = threads
    environment = os.environ.copy()
    environment.update(
        {
            "BLIS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "VECLIB_MAXIMUM_THREADS": "1",
        }
    )
    while True:
        command = [
            "strobealign",
            "--aemb",
            "-t",
            str(current_threads),
            str(contigs),
            *(str(path) for path in reads),
        ]
        with partial.open("w", encoding="utf-8") as handle:
            result = subprocess.run(
                command,
                stdout=handle,
                stderr=subprocess.PIPE,
                text=True,
                env=environment,
                check=False,
            )
        stderr = result.stderr or ""
        resource_failure = (
            result.returncode in {134, -6}
            or "resource temporarily unavailable" in stderr.lower()
            or "unable to allocate necessary resources" in stderr.lower()
            or "pthread_create" in stderr.lower()
        )
        if result.returncode == 0:
            if partial.stat().st_size == 0:
                partial.unlink(missing_ok=True)
                raise RuntimeError("strobealign AEMB completed without producing output")
            partial.replace(output)
            return
        partial.unlink(missing_ok=True)
        if resource_failure and current_threads > 1:
            reduced_threads = max(1, current_threads // 2)
            print(
                "strobealign AEMB exhausted a thread or process resource at "
                f"{current_threads} threads; retrying with {reduced_threads} threads",
                file=sys.stderr,
                flush=True,
            )
            current_threads = reduced_threads
            continue
        detail = " | ".join(line.strip() for line in stderr.splitlines()[-8:] if line.strip())
        raise RuntimeError(
            f"strobealign AEMB failed with exit code {result.returncode}"
            + (f": {detail}" if detail else "")
        )


def parse_magscot_hmm(pfam: Path, tigr: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.writer(destination, delimiter="\t", lineterminator="\n")
        for path, use_accession in ((pfam, True), (tigr, False)):
            with path.open("r", encoding="utf-8", errors="replace") as source:
                for line in source:
                    if not line.strip() or line.startswith("#"):
                        continue
                    fields = line.split()
                    if len(fields) < 5:
                        continue
                    protein = fields[0]
                    marker = fields[3] if use_accession and fields[3] != "-" else fields[2]
                    writer.writerow((protein, marker, fields[4]))
                    count += 1
    if count == 0:
        raise ValueError("No MAGScoT marker hits were parsed from HMMER output")


def _safe_filename(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    return cleaned.strip("._") or "bin"


def _natural_sort_key(value: str) -> tuple[tuple[object, ...], ...]:
    """Return a deterministic human/numeric ordering key for public MAG IDs."""
    return tuple(
        (0, int(part), len(part), part)
        if part.isdigit()
        else (1, part.casefold(), part)
        for part in re.split(r"(\d+)", value)
        if part
    )


def _natural_sorted(values: Iterable[str]) -> list[str]:
    return sorted(values, key=_natural_sort_key)


def _clear_completion_marker(completion_marker: Path | None) -> None:
    """Remove a stale marker before starting a materialization operation."""
    if completion_marker is None:
        return
    if completion_marker.exists() and completion_marker.is_dir():
        raise ValueError(
            f"Completion marker must be a file, not a directory: {completion_marker}"
        )
    if completion_marker.exists() or completion_marker.is_symlink():
        completion_marker.unlink()


def _write_completion_marker(completion_marker: Path | None) -> None:
    """Atomically publish a marker only after all named outputs are complete."""
    if completion_marker is None:
        return
    completion_marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = completion_marker.with_name(
        f".{completion_marker.name}.tmp-{os.getpid()}"
    )
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    try:
        temporary.write_text("complete\n", encoding="utf-8")
        temporary.replace(completion_marker)
    finally:
        if temporary.exists() or temporary.is_symlink():
            temporary.unlink()


def materialize_bins(
    assembly: Path,
    mapping: Path,
    output_dir: Path,
    prefix: str,
    suffix: str = "fa",
    completion_marker: Path | None = None,
) -> None:
    _clear_completion_marker(completion_marker)
    contig_to_bin: dict[str, str] = {}
    with mapping.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.reader(handle, delimiter="\t"), start=1):
            if not row or len(row) < 2:
                continue
            bin_id, contig = row[0].strip(), row[1].strip()
            if row_number == 1 and bin_id.lower() in {"bin", "binnew", "clustername"}:
                continue
            previous = contig_to_bin.setdefault(contig, bin_id)
            if previous != bin_id:
                raise ValueError(f"Contig {contig!r} is assigned to multiple refined bins")
    if not contig_to_bin:
        raise ValueError(f"No assignments found in {mapping}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    handles: OrderedDict[str, TextIO] = OrderedDict()
    filenames = {
        bin_id: f"{_safe_filename(prefix)}_{number}.{suffix}"
        for number, bin_id in enumerate(
            _natural_sorted(set(contig_to_bin.values())), start=1
        )
    }
    counts: defaultdict[str, int] = defaultdict(int)

    def get_handle(bin_id: str) -> TextIO:
        if bin_id in handles:
            handle = handles.pop(bin_id)
            handles[bin_id] = handle
            return handle
        if len(handles) >= 128:
            _, old_handle = handles.popitem(last=False)
            old_handle.close()
        filename = filenames[bin_id]
        handle = (output_dir / filename).open("a", encoding="utf-8")
        handles[bin_id] = handle
        return handle

    try:
        for contig, _header, sequence in iter_fasta(assembly):
            bin_id = contig_to_bin.get(contig)
            if bin_id is None:
                continue
            write_record(get_handle(bin_id), contig, sequence)
            counts[bin_id] += 1
    finally:
        for handle in handles.values():
            handle.close()
    missing = _natural_sorted(set(contig_to_bin.values()) - set(counts))
    if missing:
        raise ValueError(f"Assignments refer to bins with no matching contigs: {missing[:10]}")
    with (output_dir / "manifest.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("bin", "file", "contigs"))
        for bin_id in _natural_sorted(counts):
            writer.writerow((bin_id, filenames[bin_id], counts[bin_id]))
    _write_completion_marker(completion_marker)


def _bin_assignments(mapping: Path, *, allow_empty: bool = False) -> dict[str, str]:
    contig_to_bin: dict[str, str] = {}
    explicit_empty_header = False
    with mapping.open("r", encoding="utf-8-sig", newline="") as handle:
        for row_number, row in enumerate(csv.reader(handle, delimiter="\t"), start=1):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) < 2 or not row[0].strip() or not row[1].strip():
                raise ValueError(f"Invalid bin assignment in {mapping}, row {row_number}")
            bin_id, contig = row[0].strip(), row[1].strip()
            if row_number == 1 and bin_id.lower() in {"bin", "binnew", "clustername"}:
                explicit_empty_header = row == ["binnew", "contig"]
                continue
            previous = contig_to_bin.setdefault(contig, bin_id)
            if previous != bin_id:
                raise ValueError(f"Contig {contig!r} is assigned to multiple refined bins")
    if not contig_to_bin and not (allow_empty and explicit_empty_header):
        raise ValueError(f"No assignments found in {mapping}")
    return contig_to_bin


def _write_materialized_bins(
    assembly: Path,
    contig_to_bin: dict[str, str],
    filenames: dict[str, str],
    output_dir: Path,
) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    handles: OrderedDict[str, TextIO] = OrderedDict()

    def get_handle(bin_id: str) -> TextIO:
        if bin_id in handles:
            handle = handles.pop(bin_id)
            handles[bin_id] = handle
            return handle
        if len(handles) >= 128:
            _, old_handle = handles.popitem(last=False)
            old_handle.close()
        handle = (output_dir / filenames[bin_id]).open("a", encoding="utf-8")
        handles[bin_id] = handle
        return handle

    counts: defaultdict[str, int] = defaultdict(int)
    try:
        for contig, _header, sequence in (iter_fasta(assembly) if contig_to_bin else ()):
            bin_id = contig_to_bin.get(contig)
            if bin_id is None:
                continue
            write_record(get_handle(bin_id), contig, sequence)
            counts[bin_id] += 1
    finally:
        for handle in handles.values():
            handle.close()
    missing = _natural_sorted(set(contig_to_bin.values()) - set(counts))
    if missing:
        raise ValueError(f"Assignments refer to bins with no matching contigs: {missing[:10]}")
    with (output_dir / "manifest.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("bin", "file", "contigs"))
        for bin_id in _natural_sorted(counts):
            writer.writerow((bin_id, filenames[bin_id], counts[bin_id]))


def materialize_refined(
    assembly: Path,
    mapping: Path,
    output_dir: Path,
    prefix: str,
    sources: list[Path],
    suffix: str = "fa",
    completion_marker: Path | None = None,
) -> None:
    """Write refined bins while keeping original binner names for unchanged bins.

    A refined bin whose contig set is identical to a bin already published by an
    individual binner keeps that bin's sample_tool_sequence filename. Only bins
    created or modified by the refinement step receive new prefix_sequence names.
    """
    _clear_completion_marker(completion_marker)
    contig_to_bin = _bin_assignments(mapping, allow_empty=True)
    if not contig_to_bin:
        _write_materialized_bins(assembly, {}, {}, output_dir)
        _write_completion_marker(completion_marker)
        print(f"[MAGScoT EMPTY] Published 0 refined MAGs for {prefix}.", flush=True)
        return
    source_names: dict[frozenset[str], str] = {}
    for directory in sources:
        if not directory.is_dir():
            print(f"Skipping unavailable binner bins directory: {directory}", flush=True)
            continue
        for fasta in _fasta_files(directory):
            contigs = frozenset(contig for contig, _header, _sequence in iter_fasta(fasta))
            if contigs:
                source_names.setdefault(contigs, _fasta_stem(fasta))
    bin_to_contigs: dict[str, set[str]] = defaultdict(set)
    for contig, bin_id in contig_to_bin.items():
        bin_to_contigs[bin_id].add(contig)
    filenames: dict[str, str] = {}
    refined_number = 0
    for bin_id in _natural_sorted(bin_to_contigs):
        kept = source_names.get(frozenset(bin_to_contigs[bin_id]))
        if kept is not None:
            filenames[bin_id] = f"{kept}.{suffix}"
        else:
            refined_number += 1
            filenames[bin_id] = f"{_safe_filename(prefix)}_{refined_number}.{suffix}"
    _write_materialized_bins(assembly, contig_to_bin, filenames, output_dir)
    _write_completion_marker(completion_marker)


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _stage_bam_link(source: Path, destination: Path) -> None:
    """Stage a BAM asset without copying it unless links are unavailable."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        destination.symlink_to(source.resolve())
    except OSError:
        _link_or_copy(source, destination)


def stage_annotation_genomes(items: list[str], output_dir: Path) -> None:
    """Validate and stage only selected MAGs; never change source FASTA files."""
    from .named_inputs import genome_name

    destination = output_dir.resolve()
    if output_dir.is_symlink():
        raise ValueError(f"Genome staging directory must not be a symlink: {output_dir}")
    sources: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for item in items:
        name, raw_path = item.split("=", 1)
        source = Path(raw_path).resolve()
        if source.is_relative_to(destination):
            raise ValueError(f"Genome input is inside the staging output: {source}")
        if name != genome_name(source) or name.casefold() in seen:
            raise ValueError(f"Invalid or duplicate staged genome name: {name}")
        seen.add(name.casefold())
        sources.append((name, source))
    if not sources:
        raise ValueError("No genome files selected for annotation")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    try:
        for name, source in sources:
            seen_contigs: set[str] = set()
            with (staging / f"{name}.fa").open("w", encoding="utf-8") as handle:
                for identifier, header, sequence in iter_fasta(source):
                    if not sequence or identifier in seen_contigs:
                        raise ValueError(f"Empty sequence or duplicate contig {identifier!r} in {source}")
                    seen_contigs.add(identifier)
                    write_record(handle, header, sequence)
            if not seen_contigs:
                raise ValueError(f"Genome FASTA contains no records: {source}")
        with (staging / "manifest.tsv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(("genome", "source", "staged_file"))
            writer.writerows((name, str(source), f"{name}.fa") for name, source in sources)
        (staging / ".metabaw.complete").write_text("complete\n", encoding="utf-8")
        if destination.exists():
            shutil.rmtree(destination)
        staging.replace(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def stage_bams(items: list[str], output_dir: Path) -> None:
    staging_dir = output_dir.with_name(f".{output_dir.name}.staging-{os.getpid()}")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True)
    try:
        for item in items:
            sample, raw_path = item.split("=", 1)
            source = Path(raw_path)
            if not source.is_file():
                raise FileNotFoundError(f"BAM input does not exist: {source}")
            destination = staging_dir / f"{_safe_filename(sample)}.bam"
            _stage_bam_link(source, destination)
            bai_candidates = [Path(str(source) + ".bai"), source.with_suffix(".bai")]
            bai = next((path for path in bai_candidates if path.is_file()), None)
            if bai is None:
                raise FileNotFoundError(f"BAM index does not exist for {source}")
            _stage_bam_link(bai, Path(str(destination) + ".bai"))
        staging_dir.replace(output_dir)
    except BaseException:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise


def collect_fasta(sources: list[Path], output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    seen: set[str] = set()
    collected: list[tuple[str, str]] = []
    for directory in sources:
        for source in _fasta_files(directory):
            if source.name in seen:
                raise ValueError(f"Duplicate candidate bin filename: {source.name}")
            seen.add(source.name)
            _link_or_copy(source, output_dir / source.name)
            collected.append((str(directory), source.name))
    if not seen:
        raise ValueError("No refined FASTA bins were collected")
    with (output_dir / "manifest.tsv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as table:
        writer = csv.writer(table, delimiter="\t", lineterminator="\n")
        writer.writerow(("source_directory", "file"))
        writer.writerows(collected)


def publish_fasta(source_dir: Path, output_dir: Path, prefix: str, suffix: str = "fa") -> None:
    """Publish bins under stable sample_tool_sequence .fa filenames."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    files = _fasta_files(source_dir)
    if not files:
        raise ValueError(f"No FASTA bins found in {source_dir}")
    manifest = output_dir / "manifest.tsv"
    with manifest.open("w", encoding="utf-8", newline="") as table:
        writer = csv.writer(table, delimiter="\t", lineterminator="\n")
        writer.writerow(("source", "published"))
        for number, source in enumerate(files, start=1):
            name = f"{_safe_filename(prefix)}_{number}.{suffix}"
            with (output_dir / name).open("w", encoding="utf-8") as output:
                for identifier, _header, sequence in iter_fasta(source):
                    write_record(output, identifier, sequence)
            writer.writerow((source.name, name))


def normalize_refined_fasta(
    source_dir: Path,
    output_dir: Path,
    prefix: str,
    sources: list[Path],
    suffix: str = "fa",
    completion_marker: Path | None = None,
) -> None:
    """Normalize refined FASTA names while retaining unchanged-bin provenance."""
    _clear_completion_marker(completion_marker)
    refined_files = sorted(
        _fasta_files(source_dir),
        key=lambda path: _natural_sort_key(path.name),
    )
    empty = read_empty_refinement(source_dir)
    if empty is not None and refined_files:
        raise ValueError(f"Empty-refinement evidence conflicts with FASTA bins: {source_dir}")
    if not refined_files and empty is None:
        raise ValueError(f"No refined FASTA bins found in {source_dir}")

    source_names: dict[frozenset[str], str] = {}
    for directory in sources if refined_files else ():
        if not directory.is_dir():
            print(f"Skipping unavailable binner bins directory: {directory}", flush=True)
            continue
        for fasta in _fasta_files(directory):
            contigs = frozenset(
                identifier
                for identifier, _header, _sequence in iter_fasta(fasta)
            )
            if contigs:
                source_names.setdefault(contigs, _fasta_stem(fasta))

    planned: list[tuple[Path, str, str]] = []
    used_names: set[str] = set()
    refined_number = 0
    for fasta in refined_files:
        contigs = frozenset(
            identifier
            for identifier, _header, _sequence in iter_fasta(fasta)
        )
        if not contigs:
            raise ValueError(f"Refined FASTA contains no sequences: {fasta}")
        kept = source_names.get(contigs)
        if kept is None:
            refined_number += 1
            published = f"{_safe_filename(prefix)}_{refined_number}.{suffix}"
            provenance = "refined"
        else:
            published = f"{kept}.{suffix}"
            provenance = "unchanged"
        if published in used_names:
            raise ValueError(
                f"Duplicate normalized refined MAG filename: {published}"
            )
        used_names.add(published)
        planned.append((fasta, published, provenance))

    staging = output_dir.with_name(f".{output_dir.name}.metabaw-normalizing")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        manifest = staging / "manifest.tsv"
        with manifest.open("w", encoding="utf-8", newline="") as table:
            writer = csv.writer(table, delimiter="\t", lineterminator="\n")
            writer.writerow(("source", "published", "provenance"))
            for source, published, provenance in planned:
                with (staging / published).open("w", encoding="utf-8") as output:
                    for identifier, _header, sequence in iter_fasta(source):
                        write_record(output, identifier, sequence)
                writer.writerow((source.name, published, provenance))
        if empty is not None:
            write_empty_refinement(staging, str(empty["tool"]))
        if output_dir.exists():
            shutil.rmtree(output_dir)
        staging.replace(output_dir)
        _write_completion_marker(completion_marker)
        if empty is not None:
            print(f"[REFINEMENT EMPTY] Published 0 refined MAGs for {prefix}.", flush=True)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def tag_contigs(input_dir: Path, output_dir: Path, suffix: str = "fa") -> None:
    """Write bins with contig identifiers renamed to BIN_1, BIN_2, ..."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    count = 0
    for source in _fasta_files(input_dir):
        bin_name = _safe_filename(_fasta_stem(source))
        destination = output_dir / f"{bin_name}.{suffix}"
        with destination.open("w", encoding="utf-8") as handle:
            for number, (_identifier, _header, sequence) in enumerate(iter_fasta(source), start=1):
                write_record(handle, f"{bin_name}_{number}", sequence)
                count += 1
    if count == 0:
        raise ValueError(f"No FASTA records found in {input_dir}")


def report_coassembly_provenance(
    bins_dir: Path,
    output: Path,
    suffix: str,
    coassemblies: list[str],
    individual: list[str],
) -> None:
    """Write one explicit provenance row for every final representative MAG."""
    definitions: dict[str, tuple[str, str, str]] = {}
    for value in coassemblies:
        try:
            identifier, raw_sets = value.split("=", 1)
            fields = raw_sets.split("|")
            if len(fields) == 2:  # Backward-compatible internal format.
                assembly_id = identifier
                assembly_samples, recovery_samples = fields
            elif len(fields) == 3:
                assembly_id, assembly_samples, recovery_samples = fields
            else:
                raise ValueError
        except ValueError as exc:
            raise ValueError(f"Invalid coassembly provenance value: {value!r}") from exc
        identifier = identifier.strip()
        assembly_id = assembly_id.strip()
        if not identifier or not assembly_id:
            raise ValueError(f"Coassembly provenance ID is empty: {value!r}")
        definitions[identifier] = (assembly_id, assembly_samples, recovery_samples)

    output.parent.mkdir(parents=True, exist_ok=True)
    files = _fasta_files(bins_dir)
    if not files:
        raise ValueError(f"No final MAG FASTA files found in {bins_dir}")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (
                "mag",
                "assembly_origin",
                "assembly_id",
                "assembly_samples",
                "recovery_samples",
            )
        )
        for fasta in files:
            stem = _fasta_stem(fasta)
            matching = [
                identifier
                for identifier in definitions
                if stem == identifier or stem.startswith(identifier + "_")
            ]
            if matching:
                identifier = max(matching, key=len)
                assembly_id, assembly_samples, recovery_samples = definitions[identifier]
                writer.writerow(
                    (
                        fasta.name,
                        "coassembly",
                        assembly_id,
                        assembly_samples,
                        recovery_samples,
                    )
                )
            else:
                writer.writerow(
                    (
                        fasta.name,
                        "individual_assembly",
                        "",
                        "",
                        "",
                    )
                )


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"true", "t", "1", "yes", "pass"}


def _read_checkm2(path: Path) -> tuple[dict[str, dict[str, str]], str, str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = reader.fieldnames or []
        normalized = {field.lower().replace(" ", "_"): field for field in fields}
        name_col = (
            normalized.get("name")
            or normalized.get("genome")
            or normalized.get("bin_id")
            or normalized.get("bin")
        )
        completeness_col = normalized.get("completeness")
        contamination_col = normalized.get("contamination")
        if not name_col or not completeness_col or not contamination_col:
            raise ValueError(f"Unrecognized CheckM2 report columns: {fields}")
        rows = {_canonical_genome(row[name_col]): row for row in reader}
    return rows, name_col, completeness_col, contamination_col


def write_drep_genome_info(
    bins_dir: Path,
    quality_report: Path,
    output: Path,
    strip_extension: bool = False,
) -> None:
    """Convert a supported quality table into dRep genomeInformation CSV."""
    quality_rows, _name_col, completeness_col, contamination_col = _read_checkm2(
        quality_report
    )
    fasta_files = _fasta_files(bins_dir)
    if not fasta_files:
        raise ValueError(f"No FASTA bins found in {bins_dir}")

    rows: list[tuple[str, float, float]] = []
    missing: list[str] = []
    for fasta in fasta_files:
        quality = quality_rows.get(_canonical_genome(fasta.name))
        if quality is None:
            missing.append(fasta.name)
            continue
        try:
            completeness = float(quality[completeness_col])
            contamination = float(quality[contamination_col])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid completeness or contamination for {fasta.name} "
                f"in {quality_report}"
            ) from exc
        if not (math.isfinite(completeness) and math.isfinite(contamination)):
            raise ValueError(
                f"Non-finite completeness or contamination for {fasta.name} "
                f"in {quality_report}"
            )
        genome_name = _fasta_stem(fasta) if strip_extension else fasta.name
        rows.append((genome_name, completeness, contamination))

    if missing:
        preview = ", ".join(missing[:5])
        remainder = len(missing) - 5
        if remainder > 0:
            preview += f", and {remainder} more"
        raise ValueError(
            f"Quality report {quality_report} has no matching row for: {preview}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("genome", "completeness", "contamination"))
        writer.writerows(rows)


def write_metawrap_checkm_quality(
    bins_dir: Path,
    stats_files: list[Path],
    manifests: list[Path],
    output: Path,
) -> None:
    """Reuse MetaWRAP's final CheckM statistics under normalized MAG names."""
    if output.exists():
        output.unlink()
    if len(stats_files) != len(manifests):
        raise ValueError(
            "MetaWRAP quality conversion requires one stats file per manifest"
        )
    if not stats_files:
        raise ValueError("No MetaWRAP CheckM statistics were provided")

    converted: dict[str, tuple[float, float]] = {}
    for stats_path, manifest_path in zip(stats_files, manifests, strict=True):
        stats_rows, _name_col, completeness_col, contamination_col = _read_checkm2(
            stats_path
        )
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fields = reader.fieldnames or []
            if "source" not in fields or "published" not in fields:
                raise ValueError(
                    f"Unrecognized normalized MAG manifest columns in "
                    f"{manifest_path}: {fields}"
                )
            for row in reader:
                source = row.get("source", "").strip()
                published = row.get("published", "").strip()
                if not source or not published:
                    raise ValueError(f"Incomplete MAG manifest row in {manifest_path}")
                quality = stats_rows.get(_canonical_genome(source))
                if quality is None:
                    raise ValueError(
                        f"MetaWRAP stats {stats_path} have no matching row for "
                        f"{source}"
                    )
                try:
                    completeness = float(quality[completeness_col])
                    contamination = float(quality[contamination_col])
                except (KeyError, TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid MetaWRAP CheckM values for {source} in "
                        f"{stats_path}"
                    ) from exc
                if not (
                    math.isfinite(completeness)
                    and math.isfinite(contamination)
                ):
                    raise ValueError(
                        f"Non-finite MetaWRAP CheckM values for {source} in "
                        f"{stats_path}"
                    )
                name = _canonical_genome(published)
                if name in converted:
                    raise ValueError(
                        f"Duplicate normalized MAG name in MetaWRAP quality "
                        f"reports: {name}"
                    )
                converted[name] = (completeness, contamination)

    fasta_names = {
        _canonical_genome(fasta.name)
        for fasta in _fasta_files(bins_dir)
    }
    if not fasta_names:
        raise ValueError(f"No FASTA bins found in {bins_dir}")
    missing = sorted(fasta_names - converted.keys())
    extra = sorted(converted.keys() - fasta_names)
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(
                "missing quality rows for " + ", ".join(missing[:5])
            )
        if extra:
            details.append(
                "quality rows without candidate bins for " + ", ".join(extra[:5])
            )
        raise ValueError(
            "MetaWRAP CheckM results do not match candidate bins: "
            + "; ".join(details)
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("Name", "Completeness", "Contamination"))
        for name in sorted(fasta_names):
            completeness, contamination = converted[name]
            writer.writerow((name, completeness, contamination))


def filter_quality(
    bins_dir: Path,
    checkm2_report: Path,
    output_dir: Path,
    summary: Path,
    min_completeness: float,
    max_contamination: float,
    gunc_dir: Path | None,
    gunc_min_reference_representation: float = 0.30,
    fail_gunc_unscored: bool = False,
    min_quality_score: float | None = None,
    rna_summary: Path | None = None,
) -> None:
    checkm_rows, _name_col, completeness_col, contamination_col = _read_checkm2(checkm2_report)
    gunc: dict[str, bool | None] = {}
    if gunc_dir is not None:
        matches = sorted(gunc_dir.glob("GUNC.*maxCSS_level.tsv"))
        if not matches:
            raise ValueError(f"No GUNC maxCSS summary found in {gunc_dir}")
        with matches[0].open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                genome = _canonical_genome(row.get("genome", ""))
                pass_value = row.get("pass.GUNC", row.get("pass_gunc", ""))
                normalized = str(pass_value).strip().lower()
                if normalized in {"true", "t", "yes", "y", "1"}:
                    call: bool | None = True
                elif normalized in {"false", "f", "no", "n", "0"}:
                    call = False
                else:
                    call = None
                representation = row.get("reference_representation_score", "")
                if representation not in {None, ""}:
                    try:
                        if float(representation) < gunc_min_reference_representation:
                            call = None
                    except ValueError:
                        call = None
                gunc[genome] = call
    rna: dict[str, dict[str, str]] = {}
    if rna_summary is not None:
        with rna_summary.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            rna = {_canonical_genome(row.get("genome", "")): row for row in reader}
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    summary.parent.mkdir(parents=True, exist_ok=True)
    passed = 0
    evaluated = 0
    reason_counts: Counter[str] = Counter()
    with summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            ("bin", "completeness", "contamination", "pass_gunc", "selected", "reason", "quality_score", "pass_rna", "dereplication_pass")
        )
        for fasta in _fasta_files(bins_dir):
            evaluated += 1
            name = _fasta_stem(fasta)
            row = checkm_rows.get(name)
            if row is None:
                reason_counts["missing_checkm2"] += 1
                writer.writerow(
                    (name, "", "", "", "false", "missing_checkm2", "", "", "Not evaluated")
                )
                continue
            completeness = float(row[completeness_col])
            contamination = float(row[contamination_col])
            quality_score = completeness - 5 * contamination
            pass_gunc: bool | None = gunc.get(name) if gunc else True
            rna_row = rna.get(name)
            rna_not_filtered = bool(
                rna_row and rna_row.get("pass", "").strip().lower() == "not filtered"
            )
            # Report-only RNA results are informational, not failed filters.
            # Do not add this status to the generic boolean parser used by
            # other quality checks.
            pass_rna = True if not rna else bool(
                rna_row and (rna_not_filtered or _truthy(rna_row.get("pass", "")))
            )
            reasons: list[str] = []
            if completeness < min_completeness:
                reasons.append("low_completeness")
            if contamination > max_contamination:
                reasons.append("high_contamination")
            if min_quality_score is not None and quality_score < min_quality_score:
                reasons.append("low_quality_score")
            if pass_gunc is False:
                reasons.append("gunc_failed")
            elif pass_gunc is None and fail_gunc_unscored:
                reasons.append("gunc_unscored")
            if not pass_rna:
                reasons.append(
                    "rna_tool_error"
                    if rna_row and rna_row.get("error", "").strip()
                    else "rna_failed"
                )
            selected = not reasons
            reason_counts.update(reasons)
            if selected:
                _link_or_copy(fasta, output_dir / f"{name}.fa")
                passed += 1
            writer.writerow(
                (
                    name,
                    completeness,
                    contamination,
                    (
                        "unscored"
                        if pass_gunc is None
                        else str(pass_gunc).lower()
                        if gunc
                        else "not_run"
                    ),
                    str(selected).lower(),
                    ";".join(reasons) or "pass",
                    f"{quality_score:.6f}",
                    "Not filtered" if rna_not_filtered else str(pass_rna).lower() if rna else "not_run",
                    "Pending" if selected else "Not evaluated",
                )
            )
    if passed == 0:
        breakdown = ", ".join(
            f"{reason}={count}"
            for reason, count in sorted(reason_counts.items())
        )
        raise ValueError(
            "No bins passed the configured quality thresholds; "
            f"evaluated={evaluated}; reasons: {breakdown or 'unknown'}; "
            f"details={summary}"
        )


def report_dereplication_quality(
    summary: Path,
    representatives_dir: Path,
    completion_marker: Path | None = None,
) -> None:
    """Add representative status without changing quality-filter decisions.

    Also accepts older summaries without this column so completed Galah/dRep
    runs can acquire the new report without repeating genome clustering.
    """
    _clear_completion_marker(completion_marker)
    representatives = _fasta_files(representatives_dir)
    if not representatives:
        raise ValueError(f"No representative MAG FASTAs found in {representatives_dir}")
    representative_names = [_fasta_stem(path) for path in representatives]
    if len(representative_names) != len(set(representative_names)):
        raise ValueError("Duplicate representative MAG identifiers")
    with summary.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = list(reader.fieldnames or [])
        if len(fields) != len(set(fields)) or not {"bin", "selected"} <= set(fields):
            raise ValueError(f"Invalid quality summary header in {summary}; bin and selected are required")
        rows = list(reader)
    by_name: dict[str, dict[str, str]] = {}
    selected_names: set[str] = set()
    for row in rows:
        # Older filter-quality versions emitted only six fields when CheckM
        # had no record. Preserve those explicit failures during report upgrades.
        missing = {key for key, value in row.items() if value is None}
        if (
            missing
            and missing <= {"quality_score", "pass_rna", "dereplication_pass"}
            and row.get("selected", "").strip().lower() == "false"
            and row.get("reason") == "missing_checkm2"
        ):
            row.update({key: "" for key in missing})
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"Malformed quality summary row in {summary}")
        name = _canonical_genome(row["bin"])
        if not name or name in by_name:
            raise ValueError(f"Empty or duplicate MAG identifier in {summary}: {name!r}")
        selected = row["selected"].strip().lower()
        if selected not in {"true", "false", "t", "f", "yes", "no", "y", "n", "1", "0"}:
            raise ValueError(f"Invalid quality selected value for {name}: {row['selected']!r}")
        by_name[name] = row
        if _truthy(selected):
            selected_names.add(name)
    retained = set(representative_names)
    invalid = retained - selected_names
    if invalid:
        raise ValueError(
            "Representative MAGs are missing from or not selected in the quality summary: "
            + ", ".join(sorted(invalid))
        )
    if "dereplication_pass" not in fields:
        fields.append("dereplication_pass")
    for name, row in by_name.items():
        row["dereplication_pass"] = (
            "Not evaluated" if name not in selected_names else "TRUE" if name in retained else "FALSE"
        )
    # Validate all IDs before replacing the user's report, and never leave a
    # partially written table or a completion marker after an unsuccessful run.
    pending: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="", prefix=".dereplication-",
            suffix=".tsv", dir=summary.parent, delete=False,
        ) as handle:
            pending = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(pending, summary)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)
    _write_completion_marker(completion_marker)


def _marker_domains(directory: Path) -> dict[str, str]:
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    for domain, total, pattern in (
        ("bacteria", 120, "*bac120.markers_summary.tsv"),
        ("archaea", 53, "*ar53.markers_summary.tsv"),
    ):
        for path in directory.rglob(pattern):
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                for row in reader:
                    lowered = {key.lower().replace(" ", "_"): key for key in row}
                    name_col = lowered.get("genome") or lowered.get("name")
                    if not name_col:
                        continue
                    unique_col = lowered.get("number_unique_genes") or lowered.get("unique")
                    multiple_col = lowered.get("number_multiple_genes") or lowered.get("multiple")
                    try:
                        present = float(row.get(unique_col or "", 0) or 0) + float(
                            row.get(multiple_col or "", 0) or 0
                        )
                    except ValueError:
                        present = 0
                    scores[_canonical_genome(row[name_col])][domain] = present / total
    return {
        genome: max(values, key=values.get)
        for genome, values in scores.items()
        if values
    }


def _run_logged(command: list[str], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    stderr_path = output.with_name(output.name + ".stderr.log")
    with (
        output.open("w", encoding="utf-8", errors="replace") as stdout,
        stderr_path.open("w", encoding="utf-8", errors="replace") as stderr,
    ):
        process = subprocess.run(
            command,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    if process.returncode != 0:
        detail = " ".join(
            stderr_path.read_text(encoding="utf-8", errors="replace").split()
        )
        if not detail:
            detail = " ".join(
                output.read_text(encoding="utf-8", errors="replace").split()
            )
        if len(detail) > 500:
            detail = detail[:497] + "..."
        message = (
            f"Command failed ({process.returncode}): {' '.join(command)}; "
            f"stderr: {detail or 'no error text'}; see {stderr_path}"
        )
        raise RuntimeError(message)
    try:
        if stderr_path.stat().st_size == 0:
            stderr_path.unlink()
    except OSError:
        pass


def rna_qc(
    bins_dir: Path,
    marker_dir: Path,
    output_dir: Path,
    summary: Path,
    trnascan: str,
    barrnap: str,
    threads: int,
    run_trna: bool,
    trna_pass: int | None,
    run_rrna: bool,
    rrna_pass: bool,
    completion_marker: Path | None = None,
) -> None:
    run_trna = run_trna or trna_pass is not None
    run_rrna = run_rrna or rrna_pass
    filter_enabled = trna_pass is not None or rrna_pass
    if bins_dir.resolve().is_relative_to(output_dir.resolve()):
        raise ValueError(
            "RNA QC output directory would delete the input bins directory: "
            f"output={output_dir}, bins={bins_dir}"
        )
    domains = _marker_domains(marker_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    if completion_marker is not None:
        completion_marker.unlink(missing_ok=True)
    summary.parent.mkdir(parents=True, exist_ok=True)
    fastas = _fasta_files(bins_dir)
    if not fastas:
        raise ValueError(f"No candidate FASTA bins found in {bins_dir}")
    available_threads = max(1, threads)
    workers = min(len(fastas), available_threads)
    threads_per_worker = max(1, available_threads // workers)
    print(
        f"RNA QC scheduling: genomes={len(fastas)}, workers={workers}, "
        f"threads_per_worker={threads_per_worker}, "
        f"maximum_threads={workers * threads_per_worker}",
        flush=True,
    )

    def analyze(
        fasta: Path,
    ) -> tuple[str, str, set[str], set[str], str, str]:
        genome = _fasta_stem(fasta)
        domain = domains.get(genome, "bacteria")
        trna_types: set[str] = set()
        rrna_types: set[str] = set()
        trna_error = ""
        rrna_error = ""
        if run_trna:
            trna_table = output_dir / "tRNA" / f"{genome}.tsv"
            mode = "-A" if domain == "archaea" else "-B"
            try:
                _run_logged(
                    [
                        trnascan,
                        mode,
                        "--thread",
                        str(threads_per_worker),
                        "-o",
                        str(trna_table),
                        str(fasta),
                    ],
                    output_dir / "tRNA" / f"{genome}.log",
                )
                with trna_table.open(
                    "r",
                    encoding="utf-8",
                    errors="replace",
                ) as source:
                    for line in source:
                        fields = line.rstrip().split("\t")
                        if (
                            len(fields) >= 5
                            and fields[4] not in {"Type", "Undet", "Sup"}
                        ):
                            trna_types.add(fields[4])
            except (OSError, RuntimeError) as exc:
                trna_error = str(exc)
        if run_rrna:
            gff = output_dir / "rRNA" / f"{genome}.gff"
            kingdom = "arc" if domain == "archaea" else "bac"
            try:
                _run_logged(
                    [
                        barrnap,
                        "--kingdom",
                        kingdom,
                        "--threads",
                        str(threads_per_worker),
                        str(fasta),
                    ],
                    gff,
                )
                with gff.open(
                    "r",
                    encoding="utf-8",
                    errors="replace",
                ) as source:
                    for line in source:
                        if line.startswith("#"):
                            continue
                        for kind in ("5S", "16S", "23S"):
                            if (
                                f"Name={kind}_rRNA" in line
                                or f"{kind} ribosomal RNA" in line
                            ):
                                rrna_types.add(kind)
            except (OSError, RuntimeError) as exc:
                rrna_error = str(exc)
        return (
            genome,
            domain,
            trna_types,
            rrna_types,
            trna_error,
            rrna_error,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(analyze, fastas))

    with summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (
                "genome",
                "domain",
                "trna_types",
                "rrna_5S",
                "rrna_16S",
                "rrna_23S",
                "pass",
                "error",
            )
        )
        for (
            genome,
            domain,
            trna_types,
            rrna_types,
            trna_error,
            rrna_error,
        ) in results:
            errors = []
            if trna_error:
                errors.append(f"tRNA: {trna_error}")
            if rrna_error:
                errors.append(f"rRNA: {rrna_error}")
            trna_filter_passed = (
                trna_pass is None
                or (not trna_error and len(trna_types) >= trna_pass)
            )
            rrna_filter_passed = (
                not rrna_pass
                or (
                    not rrna_error
                    and {"5S", "16S", "23S"} <= rrna_types
                )
            )
            passed = trna_filter_passed and rrna_filter_passed
            if errors:
                print(
                    f"RNA QC warning for {genome}: {'; '.join(errors)}",
                    file=sys.stderr,
                    flush=True,
                )
            writer.writerow(
                (
                    genome,
                    domain,
                    len(trna_types) if run_trna else "not_run",
                    str("5S" in rrna_types).lower() if run_rrna else "not_run",
                    str("16S" in rrna_types).lower() if run_rrna else "not_run",
                    str("23S" in rrna_types).lower() if run_rrna else "not_run",
                    str(passed).lower() if filter_enabled else "Not filtered",
                    "; ".join(errors),
                )
            )
    requested_invocations = len(results) * (int(run_trna) + int(run_rrna))
    failed_invocations = sum(
        int(bool(trna_error)) + int(bool(rrna_error))
        for (
            _genome,
            _domain,
            _trna_types,
            _rrna_types,
            trna_error,
            rrna_error,
        ) in results
    )
    if requested_invocations and failed_invocations == requested_invocations:
        (
            first_genome,
            _first_domain,
            _first_trna_types,
            _first_rrna_types,
            first_trna_error,
            first_rrna_error,
        ) = results[0]
        first_errors = [
            error
            for error in (
                f"tRNA: {first_trna_error}" if first_trna_error else "",
                f"rRNA: {first_rrna_error}" if first_rrna_error else "",
            )
            if error
        ]
        raise RuntimeError(
            f"Every requested RNA tool invocation failed across "
            f"{len(results)} candidate MAGs; "
            f"first failure ({first_genome}): {'; '.join(first_errors)}; "
            f"details={summary}"
        )
    if completion_marker is not None:
        completion_marker.parent.mkdir(parents=True, exist_ok=True)
        completion_marker.write_text("complete\n", encoding="utf-8")


COVERM_METRICS = {
    "relative_abundance": ("relative abundance (%)", "relative_abundance", "rel_abd"),
    "rpkm": ("rpkm",),
    "tpm": ("tpm",),
    "mean": ("mean",),
    "count": ("read count", "count"),
    "trimmed_mean": ("trimmed mean",),
    "covered_fraction": ("covered fraction",),
    "covered_bases": ("covered bases",),
    "reads_per_base": ("reads per base",),
    "variance": ("variance",),
    "length": ("length",),
}


def _coverm_metric(field: str) -> str | None:
    normalized = field.strip().lower()
    for key, aliases in sorted(COVERM_METRICS.items(), key=lambda item: -max(map(len, item[1]))):
        if any(normalized == alias or normalized.endswith(" " + alias) for alias in aliases):
            return key
    return None


def _coverm_combined_columns(fields: list[str]) -> dict[str, dict[str, str]]:
    """Index current and legacy merged CoverM columns by sample and metric."""
    columns: dict[str, dict[str, str]] = defaultdict(dict)
    for field in fields:
        if "|" not in field:
            continue
        sample, metric_label = field.split("|", 1)
        sample = sample.strip()
        metric = _coverm_metric(metric_label)
        if not sample or metric is None:
            continue
        existing = columns[sample].get(metric)
        if existing is not None and existing != field:
            raise ValueError(
                f"Ambiguous CoverM columns for sample {sample}, metric {metric}: "
                f"{existing}, {field}"
            )
        columns[sample][metric] = field
    return dict(columns)


def _match_coverm_sample(
    requested: str,
    available: dict[str, dict[str, str]],
) -> tuple[str, dict[str, str]]:
    """Match a requested sample to exact or legacy clean-suffixed labels."""
    if requested in available:
        return requested, available[requested]
    normalized_requested = normalize_sample_name(requested)
    matches = [
        sample
        for sample in available
        if normalize_sample_name(sample) == normalized_requested
    ]
    if len(matches) == 1:
        matched = matches[0]
        return matched, available[matched]
    if len(matches) > 1:
        raise ValueError(
            f"CoverM matrix has ambiguous legacy labels for {requested}: "
            f"{sorted(matches)}"
        )
    raise ValueError(
        f"CoverM matrix has no columns for sample {requested}; "
        f"available_samples={sorted(available)}"
    )


def merge_coverm_taxonomy(
    items: list[str],
    taxonomy_dir: Path,
    output_dir: Path,
    output_suffix: str,
    niche_output: Path | None = None,
) -> None:
    """Write a full CoverM matrix plus one taxonomy-aware matrix per metric."""
    taxonomy = _read_gtdb_taxonomy(taxonomy_dir)
    samples: list[str] = []
    data: dict[str, dict[str, dict[str, str]]] = defaultdict(lambda: defaultdict(dict))
    metrics: set[str] = set()
    genomes: set[str] = set()
    for item in items:
        sample, raw_path = item.split("=", 1)
        if sample not in samples:
            samples.append(sample)
        with Path(raw_path).open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fields = reader.fieldnames or []
            if not fields:
                raise ValueError(f"Empty CoverM table: {raw_path}")
            genome_col = fields[0]
            field_metrics = {field: _coverm_metric(field) for field in fields[1:]}
            for row in reader:
                genome = _canonical_genome(row.get(genome_col, ""))
                if not genome:
                    continue
                genomes.add(genome)
                for field, metric in field_metrics.items():
                    if metric:
                        value = row.get(field, "0") or "0"
                        existing = data[metric][genome].get(sample)
                        if existing is not None and existing != value:
                            raise ValueError(
                                f"Conflicting CoverM {metric} values for "
                                f"sample {sample}, genome {genome}"
                            )
                        data[metric][genome][sample] = value
                        metrics.add(metric)
    if not metrics:
        raise ValueError("No recognized CoverM metrics were found")
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = output_suffix if output_suffix.startswith(".") else "." + output_suffix
    rank_names = ("domain", "phylum", "class", "order", "family", "genus", "species")

    def taxonomy_values(genome: str) -> list[str]:
        classification = taxonomy.get(genome, "")
        return [_rank_taxon(genome, classification, rank) for rank in rank_names]

    def write_combined(path: Path, selected_metrics: set[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            header = ["Genome", *rank_names]
            for sample in samples:
                header.extend(f"{sample}|{metric}" for metric in sorted(selected_metrics))
            writer.writerow(header)
            for genome in sorted(genomes):
                row = [genome, *taxonomy_values(genome)]
                for sample in samples:
                    row.extend(data[metric][genome].get(sample, "0") for metric in sorted(selected_metrics))
                writer.writerow(row)

    if niche_output is not None:
        if "count" not in metrics:
            raise ValueError("Internal niche support matrix requires read counts")
        write_combined(niche_output, metrics)
        metrics = metrics - {"count"}

    filenames = {
        "relative_abundance": "coverm_rel_abd",
        "rpkm": "coverm_rpkm_abd",
        "tpm": "coverm_tpm_abd",
        "mean": "coverm_mean_abd",
        "count": "coverm_counts",
    }
    manifest: dict[str, object] = {"schema_version": 2}
    for metric in sorted(metrics):
        path = output_dir / f"{filenames.get(metric, 'coverm_' + metric)}{suffix}"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(("Genome", *rank_names, *samples))
            for genome in sorted(genomes):
                writer.writerow(
                    (genome, *taxonomy_values(genome), *(data[metric][genome].get(sample, "0") for sample in samples))
                )
        manifest[metric] = path.name

    combined = output_dir / f"coverm_all_metrics{suffix}"
    write_combined(combined, metrics)
    manifest["all_metrics"] = combined.name
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "merge_schema.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "combined_column_format": "sample|metric",
                "sample_normalization": "trailing clean/cleaned tokens are ignored",
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


RANK_PREFIXES = {
    "domain": "d__",
    "phylum": "p__",
    "class": "c__",
    "order": "o__",
    "family": "f__",
    "genus": "g__",
    "species": "s__",
}


def _canonical_genome(value: str) -> str:
    name = Path(value.strip()).name
    if name.endswith(".gz"):
        name = name[:-3]
    for suffix in (".fasta", ".fna", ".fa"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


def _gtdbtk_summary_files(directory: Path) -> list[Path]:
    """Return GTDB-Tk summaries once, collapsing output-root symlink aliases."""
    summaries: list[Path] = []
    seen_targets: set[Path] = set()
    for summary in sorted(directory.rglob("gtdbtk.*.summary.tsv")):
        target = summary.resolve()
        if target in seen_targets:
            continue
        seen_targets.add(target)
        summaries.append(summary)
    return summaries


def _read_gtdb_taxonomy(directory: Path) -> dict[str, str]:
    taxonomy: dict[str, str] = {}
    summaries = _gtdbtk_summary_files(directory)
    if not summaries:
        raise ValueError(f"No GTDB-Tk summary files found in {directory}")
    for summary in summaries:
        with summary.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                genome = row.get("user_genome", "").strip()
                classification = row.get("classification", "").strip()
                if genome and classification:
                    taxonomy[_canonical_genome(genome)] = classification
    return taxonomy


def gtdbtk_result_missing(
    directory: Path,
    genomes: Iterable[Path],
) -> tuple[str, ...]:
    """Return reasons a prior GTDB-Tk result cannot annotate every input MAG."""
    if not directory.is_dir():
        return ("result directory does not exist",)
    summaries = _gtdbtk_summary_files(directory)
    if not summaries:
        return ("no gtdbtk.*.summary.tsv files",)

    reasons: list[str] = []
    taxonomy: dict[str, str] = {}
    invalid_rows: list[str] = []
    conflicting_genomes: set[str] = set()
    required_fields = {"user_genome", "classification"}
    for summary in summaries:
        try:
            with summary.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                fields = set(reader.fieldnames or ())
                missing_fields = sorted(required_fields - fields)
                if missing_fields:
                    reasons.append(
                        f"{summary.name} missing columns: {', '.join(missing_fields)}"
                    )
                    continue
                for line_number, row in enumerate(reader, start=2):
                    raw_genome = row.get("user_genome", "").strip()
                    classification = row.get("classification", "").strip()
                    if not raw_genome or not classification:
                        invalid_rows.append(f"{summary.name}:{line_number}")
                        continue
                    genome = _canonical_genome(raw_genome)
                    previous = taxonomy.get(genome)
                    if previous is not None and previous != classification:
                        conflicting_genomes.add(genome)
                        continue
                    taxonomy[genome] = classification
        except (OSError, UnicodeError, csv.Error) as exc:
            reasons.append(f"cannot read {summary.name}: {exc}")

    if invalid_rows:
        shown = ", ".join(invalid_rows[:8])
        remainder = len(invalid_rows) - 8
        suffix = f", +{remainder} more" if remainder > 0 else ""
        reasons.append(
            f"rows missing user_genome or classification: {shown}{suffix}"
        )
    if conflicting_genomes:
        shown = ", ".join(sorted(conflicting_genomes)[:8])
        remainder = len(conflicting_genomes) - 8
        suffix = f", +{remainder} more" if remainder > 0 else ""
        reasons.append(f"conflicting classifications: {shown}{suffix}")

    expected = [_canonical_genome(path.name) for path in genomes]
    duplicate_inputs = sorted(
        genome
        for genome, count in Counter(expected).items()
        if count > 1
    )
    if duplicate_inputs:
        reasons.append(
            "input MAG names are ambiguous after suffix normalization: "
            + ", ".join(duplicate_inputs)
        )
    missing_genomes = sorted(set(expected) - taxonomy.keys())
    if missing_genomes:
        shown = ", ".join(missing_genomes[:12])
        remainder = len(missing_genomes) - 12
        suffix = f", +{remainder} more" if remainder > 0 else ""
        reasons.append(f"missing MAG classifications: {shown}{suffix}")
    return tuple(reasons)


def gtdbtk_result_valid(directory: Path, genomes: Iterable[Path]) -> bool:
    return not gtdbtk_result_missing(directory, genomes)


def _rank_taxon(genome: str, classification: str, rank: str) -> str:
    if rank in {"mag", "strain"}:
        return genome
    prefix = RANK_PREFIXES[rank]
    value = next((part.strip() for part in classification.split(";") if part.strip().startswith(prefix)), prefix)
    if value == prefix:
        if rank == "species":
            return "unclassified"
        return f"{prefix}unclassified::{genome}"
    return value


def _cv_rank_quartiles(
    rows: list[dict[str, object]],
    cv_correct: bool = False,
) -> tuple[dict[str, str], dict[str, int], dict[str, object]]:
    """Rank supported taxa by the selected CV; preserve quarter quotas."""
    value_key = "corrected_si" if cv_correct else "raw_si"
    ordered = sorted(
        (row for row in rows if row["eligible"] and math.isfinite(float(row[value_key]))),
        key=lambda row: (float(row[value_key]), str(row["taxon"])),
    )
    size = len(ordered)
    tail_size = size // 4
    assignments: dict[str, str] = {}
    ranks: dict[str, int] = {}
    for position, row in enumerate(ordered, 1):
        taxon = str(row["taxon"])
        ranks[taxon] = position
        assignments[taxon] = (
            "generalist" if position <= tail_size
            else "specialist" if position > size - tail_size
            else "intermediate"
        )
    return assignments, ranks, {
        "ranking_value": "corrected_si" if cv_correct else "raw_si_cv",
        "correct_cv_classification": cv_correct,
        "eligible_taxa": size,
        "tail_size": tail_size,
        "intermediate_size": size - 2 * tail_size,
        "low_tail_max_cv": float(ordered[tail_size - 1][value_key]) if tail_size else None,
        "high_tail_min_cv": float(ordered[size - tail_size][value_key]) if tail_size else None,
        "tie_break": "taxon name ascending",
        "rounding": "floor(N/4) per tail; remaining taxa are intermediate",
        "small_cohort": "fewer than four eligible taxa: all intermediate",
    }


def classify_niche_literature(
    abundance: Path,
    taxonomy_dir: Path,
    output: Path,
    aggregated_output: Path,
    assignments_output: Path,
    provenance_output: Path,
    sample_datasets: list[str],
    rank: str,
    method: str,
    detection_percent: float,
    min_total_reads: int,
    min_prevalence: float,
    core_prevalence: float,
    min_core_datasets: int,
    abundance_method: str = "relative_abundance",
    completion_marker: Path | None = None,
    cv_correct: bool = False,
) -> None:
    """Classify niches using ranked CV quartiles or prevalence evidence.

    CV assigns the lowest/highest floor(N/4) supported taxa to generalist/
    specialist; the remaining taxa are intermediate. Ties use taxon names.
    Raw CV is the default ranking value. With cv_correct, use
    CV(density) - sqrt(K/read_count) without clipping negative values.
    Tovar-Herrera's core evidence is prevalence >=80% with support across datasets.
    """
    if rank not in {*RANK_PREFIXES, "strain"}:
        raise ValueError(f"Unsupported niche rank: {rank}")
    method = "occupancy" if method == "prevalence" else method
    if method not in {"cv", "occupancy"}:
        raise ValueError(f"Unsupported niche method: {method}")
    if cv_correct and method != "cv":
        raise ValueError("--cv-correct requires the cv niche classification method")
    if abundance_method not in {"relative_abundance", "rpkm", "tpm", "mean"}:
        raise ValueError(f"Unsupported niche abundance: {abundance_method}")
    if completion_marker is not None:
        completion_marker.unlink(missing_ok=True)
    sample_to_dataset: dict[str, str] = {}
    for item in sample_datasets:
        sample, dataset = item.split("=", 1)
        if sample in sample_to_dataset:
            raise ValueError(f"Duplicate sample metadata: {sample}")
        sample_to_dataset[sample] = dataset
    if len(sample_to_dataset) < 2:
        raise ValueError("Literature-based niche classification requires at least two samples")
    taxonomy = _read_gtdb_taxonomy(taxonomy_dir) if rank not in {"mag", "strain"} else {}

    with abundance.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = reader.fieldnames or []
        if not fields or fields[0] != "Genome":
            raise ValueError("The merged CoverM matrix must start with a Genome column")
        combined_columns = _coverm_combined_columns(fields)
        relative_columns: dict[str, str] = {}
        count_columns: dict[str, str] = {}
        for sample in sample_to_dataset:
            matched_sample, sample_columns = _match_coverm_sample(
                sample,
                combined_columns,
            )
            relative = sample_columns.get(abundance_method)
            count = sample_columns.get("count")
            if relative is None or count is None:
                raise ValueError(
                    f"CoverM matrix lacks {abundance_method} or internal count for {sample} "
                    f"(matched label {matched_sample}); "
                    f"available_metrics={sorted(sample_columns)}"
                )
            relative_columns[sample] = relative
            count_columns[sample] = count
        genome_rows = list(reader)

    taxon_relative: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    taxon_counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    taxon_members: dict[str, set[str]] = defaultdict(set)
    genome_to_taxon: dict[str, str] = {}
    for row in genome_rows:
        genome = _canonical_genome(row["Genome"])
        if not genome or genome.lower() in {"unmapped", "unbinned"}:
            continue
        classification = taxonomy.get(genome, "")
        taxon = _rank_taxon(genome, classification, rank)
        genome_to_taxon[genome] = taxon
        taxon_members[taxon].add(genome)
        for sample in sample_to_dataset:
            density = float(row.get(relative_columns[sample]) or 0)
            count = float(row.get(count_columns[sample]) or 0)
            if not math.isfinite(density) or density < 0 or not math.isfinite(count) or count < 0:
                raise ValueError(f"Non-finite/negative abundance or count: {genome}, {sample}")
            taxon_relative[taxon][sample] += density
            taxon_counts[taxon][sample] += count

    # CV uses native selected abundance values. Detection retains a percentage
    # threshold: normalize non-percentage metrics within the supplied MAG catalog.
    sample_totals = {
        sample: sum(values[sample] for values in taxon_relative.values())
        for sample in sample_to_dataset
    }
    detection_density = {
        taxon: {
            sample: (
                values[sample] if abundance_method == "relative_abundance"
                else (100 * values[sample] / sample_totals[sample] if sample_totals[sample] > 0 else 0.0)
            )
            for sample in sample_to_dataset
        }
        for taxon, values in taxon_relative.items()
    }
    detected = {
        taxon: {sample: value > 0 and value >= detection_percent for sample, value in values.items()}
        for taxon, values in detection_density.items()
    }

    datasets = sorted(set(sample_to_dataset.values()))
    dataset_samples = {
        dataset: sorted(sample for sample, value in sample_to_dataset.items() if value == dataset)
        for dataset in datasets
    }
    required_core_datasets = min(min_core_datasets, len(datasets))
    rows: list[dict[str, object]] = []
    k_habitat_classes = len(sample_to_dataset)
    for taxon in sorted(taxon_members):
        densities = [taxon_relative[taxon][sample] for sample in sample_to_dataset]
        counts = [taxon_counts[taxon][sample] for sample in sample_to_dataset]
        overall_prevalence = sum(detected[taxon].values()) / len(densities)
        total_reads = sum(counts)
        mean_density = statistics.fmean(densities)
        raw_si = statistics.stdev(densities) / mean_density if mean_density > 0 else math.inf
        si_bias = math.sqrt(k_habitat_classes / total_reads) if total_reads > 0 else math.inf
        corrected_si = raw_si - si_bias if math.isfinite(raw_si) and math.isfinite(si_bias) else math.inf
        prevalences = {
            dataset: sum(detected[taxon][sample] for sample in samples) / len(samples)
            for dataset, samples in dataset_samples.items()
        }
        core_dataset_count = sum(value >= core_prevalence for value in prevalences.values())
        eligible = mean_density > 0 and total_reads >= min_total_reads and (
            method == "cv" or overall_prevalence >= min_prevalence
        )
        rows.append(
            {
                "taxon": taxon,
                "members": len(taxon_members[taxon]),
                "overall_prevalence": overall_prevalence,
                "total_reads": total_reads,
                "raw_si": raw_si,
                "si_bias": si_bias,
                "corrected_si": corrected_si,
                "core_dataset_count": core_dataset_count,
                "eligible": eligible,
                "prevalences": prevalences,
            }
        )
    eligible_si = [float(row["corrected_si"]) for row in rows if row["eligible"] and math.isfinite(float(row["corrected_si"]))]
    community_mean_si = statistics.fmean(eligible_si) if eligible_si else None
    cv_assignments, cv_ranks, cv_quartiles = _cv_rank_quartiles(rows, cv_correct) if method == "cv" else ({}, {}, None)
    ranking_value = "corrected_si" if cv_correct else "raw_si_cv"
    ranking_label = "corrected_cv" if cv_correct else "raw_cv"
    taxon_niche: dict[str, str] = {}
    for row in rows:
        if not row["eligible"]:
            niche = "unclassified_low_support"
            reason = (
                "zero_abundance_or_below_min_total_reads"
                if method == "cv"
                else "zero_abundance_or_below_min_total_reads_or_occupancy"
            )
        elif method == "cv":
            taxon = str(row["taxon"])
            if taxon not in cv_assignments:
                row["eligible"] = False
                niche = "unclassified_low_support"
                reason = "non_finite_corrected_cv" if cv_correct else "non_finite_cv"
            else:
                niche = cv_assignments[taxon]
                reason = {
                    "generalist": f"{ranking_label}_lowest_rank_quarter",
                    "specialist": f"{ranking_label}_highest_rank_quarter",
                    "intermediate": f"{ranking_label}_middle_rank_half_or_remainder",
                }[niche]
        else:
            core_supported = int(row["core_dataset_count"]) >= required_core_datasets
            if core_supported:
                niche = "generalist"
                reason = "core_prevalence_supported"
            else:
                niche = "specialist"
                reason = "core_prevalence_not_supported"
        row["niche"] = niche
        row["reason"] = reason
        taxon_niche[str(row["taxon"])] = niche

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(
            (
                "taxon",
                "rank",
                "method",
                "member_mags",
                "abundance_metric",
                "overall_occupancy",
                "total_reads",
                "raw_si_cv",
                "expected_si_bias_sqrt_K_over_N",
                "corrected_si",
                "community_mean_corrected_si",
                "correct_cv_classification",
                "eligible",
                "niche",
                "reason",
                "cv_quartile_rank",
                "cv_low_tail_max",
                "cv_high_tail_min",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    row["taxon"],
                    rank,
                    method,
                    row["members"],
                    abundance_method,
                    f"{float(row['overall_prevalence']):.6f}",
                    f"{float(row['total_reads']):.3f}",
                    f"{float(row['raw_si']):.8f}",
                    f"{float(row['si_bias']):.8f}",
                    f"{float(row['corrected_si']):.8f}",
                    f"{community_mean_si:.8f}" if community_mean_si is not None else "NA",
                    "TRUE" if cv_correct else "FALSE",
                    str(row["eligible"]).lower(),
                    row["niche"],
                    row["reason"],
                    cv_ranks.get(str(row["taxon"]), "NA"),
                    f"{cv_quartiles['low_tail_max_cv']:.8f}" if cv_quartiles and cv_quartiles["low_tail_max_cv"] is not None else "NA",
                    f"{cv_quartiles['high_tail_min_cv']:.8f}" if cv_quartiles and cv_quartiles["high_tail_min_cv"] is not None else "NA",
                )
            )

    aggregated_output.parent.mkdir(parents=True, exist_ok=True)
    with aggregated_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        header = ["Taxon"]
        for sample in sample_to_dataset:
            header.extend((f"{sample}|{abundance_method}", f"{sample}|internal_read_count"))
        writer.writerow(header)
        for taxon in sorted(taxon_members):
            values: list[object] = [taxon]
            for sample in sample_to_dataset:
                values.extend((taxon_relative[taxon][sample], taxon_counts[taxon][sample]))
            writer.writerow(values)

    assignments_output.parent.mkdir(parents=True, exist_ok=True)
    with assignments_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("Genome", "Taxon", "Rank", "Niche", "correct_cv_classification"))
        for genome, taxon in sorted(genome_to_taxon.items()):
            writer.writerow((genome, taxon, rank, taxon_niche[taxon], "TRUE" if cv_correct else "FALSE"))

    provenance_output.parent.mkdir(parents=True, exist_ok=True)
    provenance_output.write_text(
        json.dumps(
            {
                "method": (
                    f"metabaw_{ranking_label}_rank_quartiles_v3"
                    if method == "cv"
                    else "tovar_core_prevalence_v1"
                ),
                "selected_criterion": method,
                "correct_cv_classification": cv_correct,
                "abundance_metric": abundance_method,
                "abundance_extension": (
                    "original relative-abundance input" if abundance_method == "relative_abundance"
                    else "MetaBAW extension using native selected abundance for CV; not an exact reproduction of the cited abundance scale"
                ),
                "rank": rank,
                "strain_definition": "dereplicated representative MAG proxy" if rank in {"strain", "mag"} else None,
                "references": (
                    [
                        {
                        "citation": "Chen et al. 2021, The ISME Journal",
                        "doi": "10.1038/s41396-021-00988-w",
                        "implemented_evidence": (
                            "corrected_SI = CV(taxon density across samples) - sqrt(K/N); "
                            + ("used for ranking with --cv-correct; " if cv_correct else "retained as a diagnostic only; ")
                            + "rank-quartile classification is the MetaBAW rule"
                        ),
                        }
                    ]
                    if method == "cv"
                    else [
                        {
                        "citation": "Tovar-Herrera et al. 2025, Nature Ecology & Evolution",
                        "doi": "10.1038/s41559-025-02904-3",
                        "implemented_evidence": "core prevalence threshold with cross-dataset support",
                        }
                    ]
                ),
                "parameters": {
                    "detection_percent": detection_percent,
                    "detection_scale": (
                        "CoverM relative abundance percent, without renormalization"
                        if abundance_method == "relative_abundance"
                        else "100 * selected taxon abundance / sum selected MAG abundance in each sample"
                    ),
                    "cv_input_scale": "native " + abundance_method,
                    "cv_correct": cv_correct,
                    "cv_ranking_value": ranking_value if method == "cv" else None,
                    "read_count_role": "internal support filter and sqrt(K/N) CV correction; not an abundance method",
                    "min_total_reads": min_total_reads,
                    "min_prevalence": min_prevalence,
                    "core_prevalence": core_prevalence,
                    "configured_min_core_datasets": min_core_datasets,
                    "effective_min_core_datasets": required_core_datasets,
                    "datasets": datasets,
                    "K_sample_profiles": k_habitat_classes,
                    "community_mean_corrected_si": community_mean_si,
                    "cv_rank_quartiles": cv_quartiles,
                },
                "classification_rules": (
                    {
                        "generalist": f"lowest floor(N/4) taxa ranked by {ranking_value}, then taxon name",
                        "specialist": f"highest floor(N/4) taxa ranked by {ranking_value}, then taxon name",
                        "intermediate": "remaining ranked taxa, including the rounding remainder",
                        "unclassified_low_support": "zero abundance, below minimum total reads, or non-finite CV",
                    }
                    if method == "cv"
                    else {
                        "generalist": "core prevalence supported across the required datasets",
                        "specialist": "core prevalence not supported",
                        "unclassified_low_support": (
                            "below minimum total reads or prevalence"
                        ),
                    }
                ),
                "interpretation_note": (
                    "CV-mode labels use raw CV by default or corrected SI with --cv-correct. "
                    "The community mean is not a classification threshold. The selected method is "
                    "applied independently; CV and prevalence evidence are not combined "
                    "into one decision rule."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if completion_marker is not None:
        completion_marker.parent.mkdir(parents=True, exist_ok=True)
        completion_marker.write_text("complete\n", encoding="utf-8")


def _labeled_paths(items: list[str]) -> list[tuple[str, Path]]:
    labeled: list[tuple[str, Path]] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected NAME=PATH, received: {item}")
        label, raw_path = item.split("=", 1)
        if not label.strip() or not raw_path.strip():
            raise ValueError(f"Expected non-empty NAME=PATH, received: {item}")
        labeled.append((label.strip(), Path(raw_path).expanduser().resolve()))
    return labeled


def merge_kofam_annotations(items: list[str], output: Path) -> None:
    """Merge KofamScan mapper tables while retaining the source MAG."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(("Genome", "Gene", "KO"))
        for genome, path in _labeled_paths(items):
            if not path.is_file():
                raise FileNotFoundError(f"KofamScan mapper output is missing: {path}")
            with path.open("r", encoding="utf-8-sig") as source:
                for line_number, raw in enumerate(source, start=1):
                    line = raw.strip()
                    if not line or line.startswith("#"):
                        continue
                    fields = line.split("\t")
                    if len(fields) < 2:
                        fields = line.split()
                    if len(fields) < 2:
                        raise ValueError(
                            f"Unexpected KofamScan mapper row in {path} "
                            f"at line {line_number}: {line}"
                        )
                    if fields[0] == "*" and len(fields) >= 3:
                        gene, ko = fields[1], fields[2]
                    else:
                        gene, ko = fields[0].lstrip("*"), fields[1]
                    writer.writerow((genome, gene, ko))


def merge_dbcan_annotations(items: list[str], output: Path) -> None:
    """Merge run_dbCAN overview tables while retaining every upstream column."""
    tables: list[tuple[str, list[str], list[dict[str, str]]]] = []
    all_fields: list[str] = []
    for genome, directory in _labeled_paths(items):
        candidates = (directory / "overview.tsv", directory / "overview.txt")
        overview = next((path for path in candidates if path.is_file()), None)
        if overview is None:
            raise FileNotFoundError(
                f"run_dbCAN overview.tsv or overview.txt is missing under {directory}"
            )
        with overview.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            fields = [str(field) for field in (reader.fieldnames or ()) if field]
            if not fields:
                raise ValueError(f"run_dbCAN overview has no header: {overview}")
            rows = [
                {field: str(row.get(field) or "") for field in fields}
                for row in reader
            ]
        for field in fields:
            if field not in all_fields:
                all_fields.append(field)
        tables.append((genome, fields, rows))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Genome", *all_fields],
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        for genome, _fields, rows in tables:
            for row in rows:
                writer.writerow({"Genome": genome, **row})


FUNCTIONAL_PREFIX_COLUMNS = (
    "MAG",
    "MAG_contig_raw_id",
    "MAG_contig_id",
    "MAG_contig_gene_id",
    "Gene",
)


def build_functional_gene_map(
    mag: Path,
    proteins: Path,
    genome: str,
    output: Path,
) -> None:
    """Map Prodigal protein IDs to raw and MAG-normalized contig/gene IDs."""
    contig_ids = [identifier for identifier, _header, _sequence in iter_fasta(mag)]
    if not contig_ids:
        raise ValueError(f"MAG FASTA contains no contigs: {mag}")
    if len(contig_ids) != len(set(contig_ids)):
        raise ValueError(f"MAG FASTA contains duplicate contig IDs: {mag}")
    already_named = all(
        identifier == genome or identifier.startswith(f"{genome}_")
        for identifier in contig_ids
    )
    normalized_contigs = {
        identifier: identifier if already_named else f"{genome}_{index}"
        for index, identifier in enumerate(contig_ids, start=1)
    }
    longest_first = sorted(contig_ids, key=len, reverse=True)
    rows: list[dict[str, str]] = []
    seen_proteins: set[str] = set()
    for protein_id, _header, _sequence in iter_fasta(proteins):
        if protein_id in seen_proteins:
            raise ValueError(f"Duplicate Prodigal protein ID {protein_id!r}: {proteins}")
        seen_proteins.add(protein_id)
        raw_contig = next(
            (
                contig
                for contig in longest_first
                if protein_id.startswith(f"{contig}_")
                and protein_id[len(contig) + 1 :]
            ),
            None,
        )
        if raw_contig is None:
            raise ValueError(
                f"Cannot associate Prodigal protein {protein_id!r} with a contig in {mag}"
            )
        gene_suffix = protein_id[len(raw_contig) + 1 :]
        normalized_contig = normalized_contigs[raw_contig]
        rows.append(
            {
                "MAG": genome,
                "MAG_contig_raw_id": raw_contig,
                "MAG_contig_id": normalized_contig,
                "MAG_contig_gene_id": f"{normalized_contig}_{gene_suffix}",
                "Protein_id": protein_id,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(*FUNCTIONAL_PREFIX_COLUMNS[:-1], "Protein_id"),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def combine_functional_proteins(
    items: list[str],
    gene_maps: list[str],
    output: Path,
    output_gene_map: Path,
) -> None:
    """Combine per-MAG proteins with collision-free IDs and a reversible map."""
    protein_paths = dict(_labeled_paths(items))
    gene_map_paths = dict(_labeled_paths(gene_maps))
    if protein_paths.keys() != gene_map_paths.keys():
        missing_maps = sorted(protein_paths.keys() - gene_map_paths.keys())
        missing_proteins = sorted(gene_map_paths.keys() - protein_paths.keys())
        raise ValueError(
            "Protein and gene-map labels differ; "
            f"missing gene maps={missing_maps}, missing proteins={missing_proteins}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    output_gene_map.parent.mkdir(parents=True, exist_ok=True)
    combined_rows: list[dict[str, str]] = []
    sequence_count = 0
    with output.open("w", encoding="utf-8") as fasta_handle:
        for genome, proteins in protein_paths.items():
            gene_map_path = gene_map_paths[genome]
            with gene_map_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle, delimiter="\t")
                mapped = {
                    str(row.get("Protein_id") or ""): dict(row)
                    for row in reader
                    if str(row.get("Protein_id") or "")
                }
            seen: set[str] = set()
            for original_id, _header, sequence in iter_fasta(proteins):
                if original_id in seen:
                    raise ValueError(
                        f"Duplicate protein identifier {original_id!r} in {proteins}"
                    )
                seen.add(original_id)
                source = mapped.get(original_id)
                if source is None:
                    raise ValueError(
                        f"Protein {genome}/{original_id} is absent from {gene_map_path}"
                    )
                if str(source.get("MAG") or "") != genome:
                    raise ValueError(
                        f"Gene map MAG does not match label {genome!r}: {gene_map_path}"
                    )
                sequence_count += 1
                combined_id = f"MBWPROT{sequence_count:012d}"
                write_record(fasta_handle, combined_id, sequence)
                combined_rows.append(
                    {
                        **source,
                        "Protein_id": combined_id,
                        "Original_protein_id": original_id,
                    }
                )
            unexpected = sorted(set(mapped) - seen)
            if unexpected:
                raise ValueError(
                    f"Gene map contains proteins absent from {proteins}: "
                    + ", ".join(unexpected[:10])
                )
    if sequence_count == 0:
        output.unlink(missing_ok=True)
        raise ValueError("No predicted proteins were available for functional annotation")
    with output_gene_map.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                *FUNCTIONAL_PREFIX_COLUMNS[:-1],
                "Protein_id",
                "Original_protein_id",
            ),
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(combined_rows)


def _functional_annotation_value(kind: str, row: dict[str, str]) -> tuple[str, str]:
    if kind == "kegg":
        return str(row.get("KO") or "Unclassified"), "KO"
    if kind == "hydrogenase":
        return str(row.get("Hydrogenase_type") or "Unclassified"), "Hydrogenase_type"
    if kind == "terminal":
        return str(row.get("Terminal_enzyme") or "Unclassified"), "Terminal_enzyme"
    candidates = (
        "Recommend Results",
        "Recommend Results ",
        "dbCAN_hmm",
        "HMMER",
        "DIAMOND",
        "EC#",
    )
    for field in candidates:
        value = str(row.get(field) or "").strip()
        if value and value not in {"-", "N"}:
            return value, field
    return "Unclassified", ""


def _functional_query_field(kind: str, fields: Iterable[str]) -> str:
    if kind in {"kegg", "hydrogenase", "terminal"}:
        return "Gene"
    candidates = ("Gene ID", "Gene_ID", "GeneID", "gene_id", "Gene")
    available = set(fields)
    for field in candidates:
        if field in available:
            return field
    raise ValueError("run_dbCAN annotation table has no recognizable gene-ID column")


def read_kegg_descriptions(path: Path) -> dict[str, str]:
    """Read ``Kid descript`` or equivalent two-column KO descriptions."""
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"KEGG KO description table is missing or empty: {path}")
    descriptions: dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.rstrip("\r\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            fields = line.split("\t", 1)
            if len(fields) < 2:
                fields = line.split(None, 1)
            if len(fields) < 2:
                raise ValueError(
                    f"Unexpected KEGG description row in {path} at line "
                    f"{line_number}: {line}"
                )
            ko, description = (field.strip() for field in fields)
            if ko.lower() in {"kid", "knum", "ko"}:
                continue
            if not re.fullmatch(r"K\d{5}", ko):
                raise ValueError(
                    f"Invalid KO identifier in {path} at line {line_number}: {ko}"
                )
            if not description:
                raise ValueError(
                    f"Empty KO description in {path} at line {line_number}: {ko}"
                )
            previous = descriptions.get(ko)
            if previous is not None and previous != description:
                raise ValueError(
                    f"Conflicting duplicate KO description in {path}: {ko}"
                )
            descriptions[ko] = description
    if not descriptions:
        raise ValueError(f"KEGG description table contains no KO records: {path}")
    return descriptions


def format_functional_annotations(
    input_path: Path,
    gene_maps: list[str],
    output: Path,
    kind: str,
    ko_descriptions: Path | None = None,
) -> None:
    """Apply the common five-column MAG/contig/gene schema to an annotation table."""
    mapping: dict[tuple[str, str], dict[str, str]] = {}
    global_mapping: dict[str, dict[str, str]] = {}
    ambiguous_protein_ids: set[str] = set()
    for genome, path in _labeled_paths(gene_maps):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                protein_id = str(row.get("Protein_id") or "")
                if not protein_id:
                    raise ValueError(f"Functional gene map lacks Protein_id: {path}")
                key = (genome, protein_id)
                if key in mapping:
                    raise ValueError(f"Duplicate functional gene mapping: {genome}/{protein_id}")
                mapping[key] = dict(row)
                if protein_id in ambiguous_protein_ids:
                    continue
                if protein_id in global_mapping:
                    global_mapping.pop(protein_id)
                    ambiguous_protein_ids.add(protein_id)
                else:
                    global_mapping[protein_id] = dict(row)
    with input_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        input_fields = [str(field) for field in (reader.fieldnames or ()) if field]
        query_field = _functional_query_field(kind, input_fields)
        source_rows = [dict(row) for row in reader]
    descriptions: dict[str, str] | None = None
    if kind == "kegg" and ko_descriptions is not None:
        descriptions = read_kegg_descriptions(ko_descriptions)
        observed_kos = {
            str(row.get("KO") or "").strip()
            for row in source_rows
            if str(row.get("KO") or "").strip()
        }
        missing_kos = sorted(observed_kos - descriptions.keys())
        if missing_kos:
            preview = ", ".join(missing_kos[:10])
            remaining = len(missing_kos) - 10
            if remaining > 0:
                preview += f", +{remaining} more"
            print(
                f"[WARNING] K.descript.txt lacks descriptions for "
                f"{len(missing_kos)} annotated KO(s); writing NA to "
                f"KO_gene_name and KO_description; missing={preview}; "
                f"file={ko_descriptions}",
                file=sys.stderr,
            )
    formatted: list[dict[str, str]] = []
    additional_fields: list[str] = (
        ["KO_gene_name", "KO_description"]
        if descriptions is not None
        else []
    )
    for row in source_rows:
        genome = str(row.get("Genome") or row.get("MAG") or "")
        protein_id = str(row.get(query_field) or "")
        mapped = mapping.get((genome, protein_id)) or global_mapping.get(protein_id)
        if mapped is None:
            raise ValueError(
                f"No MAG/contig mapping for functional gene {genome}/{protein_id}"
            )
        annotation, annotation_field = _functional_annotation_value(kind, row)
        excluded = {"Genome", "MAG", query_field, annotation_field}
        extras = {field: str(row.get(field) or "") for field in input_fields if field not in excluded}
        for field in extras:
            if field not in additional_fields and field not in FUNCTIONAL_PREFIX_COLUMNS:
                additional_fields.append(field)
        formatted_row = {
            "MAG": str(mapped["MAG"]),
            "MAG_contig_raw_id": str(mapped["MAG_contig_raw_id"]),
            "MAG_contig_id": str(mapped["MAG_contig_id"]),
            "MAG_contig_gene_id": str(mapped["MAG_contig_gene_id"]),
            "Gene": annotation,
            **extras,
        }
        if descriptions is not None:
            description = descriptions.get(annotation, "NA")
            formatted_row["KO_gene_name"] = (
                description.split(";", 1)[0].strip() if description != "NA" else "NA"
            )
            formatted_row["KO_description"] = description
        formatted.append(formatted_row)
    formatted.sort(
        key=lambda row: (row["MAG"], row["MAG_contig_id"], row["MAG_contig_gene_id"])
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(*FUNCTIONAL_PREFIX_COLUMNS, *additional_fields),
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(formatted)


HYDROGENASE_COLUMNS = (
    "Genome",
    "Gene",
    "Hydrogenase_type",
    "HydDB_subject",
    "HydDB_group",
    "Identity",
    "Query_coverage",
    "Evalue",
    "Bitscore",
)
HYDROGENASE_FINAL_COLUMNS = (
    *HYDROGENASE_COLUMNS,
    "FeFe_subject",
    "FeFe_identity",
    "FeFe_evalue",
    "FeFe_bitscore",
)


def _hydrogenase_type(value: str) -> str | None:
    normalized = re.sub(r"[^a-z]", "", value.lower())
    if "nife" in normalized:
        return "NiFe"
    if "fefe" in normalized:
        return "FeFe"
    if normalized == "fe" or normalized.startswith("fehydrogenase"):
        return "Fe"
    return None


def _hydrogenase_mapping(path: Path) -> dict[str, tuple[str, str]]:
    mapping: dict[str, tuple[str, str]] = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split("\t") if "\t" in line else line.split()
            if len(fields) < 2:
                raise ValueError(
                    f"Hydrogenase mapping line {line_number} has fewer than two columns: {path}"
                )
            reference, group = fields[0].strip(), fields[1].strip()
            if reference.lower() == "id" and group.lower() in {"gene", "type", "class"}:
                continue
            kind = _hydrogenase_type(group)
            if kind is None:
                continue
            mapping[reference] = (kind, group)
    if not mapping:
        raise ValueError(
            f"Hydrogenase mapping contains no Fe, NiFe, or FeFe entries: {path}"
        )
    return mapping


def _mapped_hydrogenase_subject(
    subject: str,
    mapping: dict[str, tuple[str, str]],
) -> tuple[str, str, str] | None:
    candidates = [subject, *subject.split("|")]
    for candidate in candidates:
        candidate = candidate.strip()
        if candidate in mapping:
            kind, group = mapping[candidate]
            return candidate, kind, group
    return None


def filter_hydrogenase_hits(
    hits: Path,
    proteins: Path,
    mapping_path: Path,
    genome: str,
    filtered_output: Path,
    fefe_fasta: Path,
    min_query_coverage: float,
    min_identity: float,
) -> None:
    """Select the best HydDB hit per gene and extract high-confidence FeFe hits."""
    mapping = _hydrogenase_mapping(mapping_path)
    best: dict[str, tuple[tuple[float, float, float, float], dict[str, str]]] = {}
    with hits.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip() or raw.startswith("#"):
                continue
            fields = raw.rstrip("\r\n").split("\t")
            if len(fields) < 13:
                raise ValueError(
                    f"HydDB BLASTP line {line_number} has {len(fields)} columns; expected 13"
                )
            query, subject = fields[0], fields[1]
            try:
                identity = float(fields[2]) / 100.0
                query_start = int(fields[6])
                query_end = int(fields[7])
                query_length = int(fields[8])
                evalue = float(fields[11])
                bitscore = float(fields[12])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid numeric value in HydDB BLASTP line {line_number}: {hits}"
                ) from exc
            if query_length <= 0:
                raise ValueError(
                    f"Invalid query length in HydDB BLASTP line {line_number}: {query_length}"
                )
            coverage = (abs(query_end - query_start) + 1) / query_length
            mapped = _mapped_hydrogenase_subject(subject, mapping)
            if mapped is None:
                raise ValueError(
                    f"HydDB subject {subject!r} is absent from {mapping_path}"
                )
            reference, kind, group = mapped
            row = {
                "Genome": genome,
                "Gene": query,
                "Hydrogenase_type": kind,
                "HydDB_subject": reference,
                "HydDB_group": group,
                "Identity": f"{identity:.6f}",
                "Query_coverage": f"{coverage:.6f}",
                "Evalue": fields[11],
                "Bitscore": fields[12],
            }
            score = (bitscore, -evalue, identity, coverage)
            if query not in best or score > best[query][0]:
                best[query] = (score, row)

    retained = [
        row
        for _score, row in best.values()
        if float(row["Query_coverage"]) >= min_query_coverage
        and float(row["Identity"]) >= min_identity
    ]
    retained.sort(key=lambda row: row["Gene"])
    filtered_output.parent.mkdir(parents=True, exist_ok=True)
    with filtered_output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=HYDROGENASE_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(retained)

    fefe_genes = {
        row["Gene"] for row in retained if row["Hydrogenase_type"] == "FeFe"
    }
    sequences = {
        identifier: (header, sequence)
        for identifier, header, sequence in iter_fasta(proteins)
    }
    missing_sequences = sorted(fefe_genes - sequences.keys())
    if missing_sequences:
        raise ValueError(
            "FeFe candidate proteins are missing from the predicted protein FASTA: "
            + ", ".join(missing_sequences[:10])
        )
    fefe_fasta.parent.mkdir(parents=True, exist_ok=True)
    with fefe_fasta.open("w", encoding="utf-8") as handle:
        for gene in sorted(fefe_genes):
            header, sequence = sequences[gene]
            write_record(handle, header, sequence)


def finalize_hydrogenase_hits(
    filtered: Path,
    fefe_hits: Path,
    nife_output: Path,
    fe_output: Path,
    fefe_output: Path,
    combined_output: Path,
) -> None:
    """Confirm FeFe candidates and write the three class tables plus a union."""
    with filtered.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = [dict(row) for row in reader]
    fefe_best: dict[str, tuple[float, dict[str, str]]] = {}
    with fefe_hits.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip() or raw.startswith("#"):
                continue
            fields = raw.rstrip("\r\n").split("\t")
            if len(fields) < 12:
                raise ValueError(
                    f"FeFe DIAMOND line {line_number} has {len(fields)} columns; expected 12"
                )
            try:
                bitscore = float(fields[11])
                identity = float(fields[2]) / 100.0
            except ValueError as exc:
                raise ValueError(
                    f"Invalid numeric value in FeFe DIAMOND line {line_number}: {fefe_hits}"
                ) from exc
            annotation = {
                "FeFe_subject": fields[1],
                "FeFe_identity": f"{identity:.6f}",
                "FeFe_evalue": fields[10],
                "FeFe_bitscore": fields[11],
            }
            if fields[0] not in fefe_best or bitscore > fefe_best[fields[0]][0]:
                fefe_best[fields[0]] = (bitscore, annotation)

    finalized: list[dict[str, str]] = []
    for row in rows:
        if row.get("Hydrogenase_type") == "FeFe":
            match = fefe_best.get(str(row.get("Gene", "")))
            if match is None:
                continue
            extra = match[1]
        else:
            extra = {
                "FeFe_subject": "",
                "FeFe_identity": "",
                "FeFe_evalue": "",
                "FeFe_bitscore": "",
            }
        finalized.append({**row, **extra})
    finalized.sort(key=lambda row: (row["Hydrogenase_type"], row["Gene"]))

    outputs = (
        (nife_output, [row for row in finalized if row["Hydrogenase_type"] == "NiFe"]),
        (fe_output, [row for row in finalized if row["Hydrogenase_type"] == "Fe"]),
        (fefe_output, [row for row in finalized if row["Hydrogenase_type"] == "FeFe"]),
        (combined_output, finalized),
    )
    for path, selected in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=HYDROGENASE_FINAL_COLUMNS,
                delimiter="\t",
                lineterminator="\n",
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(selected)


def merge_hydrogenase_annotations(items: list[str], output: Path) -> None:
    """Merge per-MAG hydrogenase tables while preserving genome identity."""
    rows: list[dict[str, str]] = []
    for genome, path in _labeled_paths(items):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                record = dict(row)
                record["Genome"] = record.get("Genome") or genome
                rows.append(record)
    rows.sort(key=lambda row: (row.get("Genome", ""), row.get("Gene", "")))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=HYDROGENASE_FINAL_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


TERMINAL_ENZYME_COLUMNS = (
    "Genome",
    "Gene",
    "Terminal_enzyme",
    "Reference",
    "Reference_title",
    "Identity",
    "Query_coverage",
    "Required_identity",
    "Evalue",
    "Bitscore",
)
TERMINAL_ENZYME_NAMES = (
    "AclB", "AcsB", "AmoA", "ARO", "AsrA", "AtpA", "CcoN", "CooS",
    "CoxA", "CoxL", "Cyc2", "CydA", "CyoA", "DsrA", "FCC", "FdhA",
    "FrdA", "HbsC", "HbsT", "HzsA", "IsoA", "McrA", "Mcr", "MmoX",
    "MtrB", "NapA", "NarG", "NifH", "NirK", "NirS", "NorB", "NosZ",
    "NrfA", "NuoF", "NxrA", "OmcB", "PmoA", "PsaA", "PsbA", "RbcL",
    "RdhA", "RHO", "SdhA", "Sor", "SoxB", "Sqr", "YgfK",
)
TERMINAL_IDENTITY_THRESHOLDS = {
    "PsaA": 0.80,
    "HbsT": 0.75,
    "PsbA": 0.70,
    "IsoA": 0.70,
    "AtpA": 0.70,
    "YgfK": 0.70,
    "ARO": 0.70,
    "CoxL": 0.60,
    "MmoX": 0.60,
    "AmoA": 0.60,
    "NxrA": 0.60,
    "NuoF": 0.60,
    "RbcL": 0.60,
    "NiFe": 0.60,
    "FeFe": 0.60,
    "Fe": 0.60,
    "RHO": 0.40,
}


def _terminal_enzyme_name(subject: str, title: str) -> str:
    text = f"{subject} {title}"
    lowered = text.lower()
    if re.search(r"(?<![a-z0-9])nife(?:[-_ ]?hydrogenase)?(?![a-z0-9])", lowered):
        return "NiFe"
    if re.search(r"(?<![a-z0-9])fefe(?:[-_ ]?hydrogenase)?(?![a-z0-9])", lowered):
        return "FeFe"
    if re.search(r"(?<![a-z0-9])fe[-_ ]?hydrogenase(?![a-z0-9])", lowered):
        return "Fe"
    if re.search(r"(?<![a-z0-9])sdha(?![a-z0-9])", lowered) and re.search(
        r"(?<![a-z0-9])frda(?![a-z0-9])", lowered
    ):
        return "SdhA/FrdA"
    for name in sorted(TERMINAL_ENZYME_NAMES, key=len, reverse=True):
        if re.search(
            rf"(?<![A-Za-z0-9]){re.escape(name)}(?![A-Za-z0-9])",
            text,
            flags=re.IGNORECASE,
        ):
            return name
    return "Other"


def filter_terminal_enzyme_hits(
    hits: Path,
    genome: str,
    output: Path,
    min_query_coverage: float = 0.80,
) -> None:
    """Apply marker-specific identity thresholds to terminal-enzyme hits."""
    best: dict[str, tuple[tuple[float, float, float, float], dict[str, str]]] = {}
    with hits.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, start=1):
            if not raw.strip() or raw.startswith("#"):
                continue
            fields = raw.rstrip("\r\n").split("\t")
            if len(fields) < 10:
                raise ValueError(
                    f"Terminal DIAMOND line {line_number} has {len(fields)} columns; expected 10"
                )
            query, subject, title = fields[0], fields[1], fields[2]
            try:
                identity = float(fields[3]) / 100.0
                query_start = int(fields[5])
                query_end = int(fields[6])
                query_length = int(fields[7])
                evalue = float(fields[8])
                bitscore = float(fields[9])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid numeric value in terminal DIAMOND line {line_number}: {hits}"
                ) from exc
            if query_length <= 0:
                raise ValueError(
                    f"Invalid query length in terminal DIAMOND line {line_number}: {query_length}"
                )
            coverage = (abs(query_end - query_start) + 1) / query_length
            enzyme = _terminal_enzyme_name(subject, title)
            required_identity = TERMINAL_IDENTITY_THRESHOLDS.get(enzyme, 0.50)
            if coverage < min_query_coverage or identity < required_identity:
                continue
            row = {
                "Genome": genome,
                "Gene": query,
                "Terminal_enzyme": enzyme,
                "Reference": subject,
                "Reference_title": title,
                "Identity": f"{identity:.6f}",
                "Query_coverage": f"{coverage:.6f}",
                "Required_identity": f"{required_identity:.6f}",
                "Evalue": fields[8],
                "Bitscore": fields[9],
            }
            score = (bitscore, -evalue, identity, coverage)
            if query not in best or score > best[query][0]:
                best[query] = (score, row)
    rows = [row for _score, row in best.values()]
    rows.sort(key=lambda row: row["Gene"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=TERMINAL_ENZYME_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def merge_terminal_enzyme_annotations(items: list[str], output: Path) -> None:
    """Merge per-MAG terminal-enzyme tables into one deterministic table."""
    rows: list[dict[str, str]] = []
    for genome, path in _labeled_paths(items):
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                record = dict(row)
                record["Genome"] = record.get("Genome") or genome
                rows.append(record)
    rows.sort(key=lambda row: (row.get("Genome", ""), row.get("Gene", "")))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=TERMINAL_ENZYME_COLUMNS,
            delimiter="\t",
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def _link_dbcan_asset(source: Path, destination: Path) -> None:
    """Create a zero-copy database alias, including across filesystems."""
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        destination.symlink_to(source.resolve())
        return
    except OSError as symlink_error:
        try:
            os.link(source, destination)
            return
        except OSError as hardlink_error:
            raise OSError(
                f"Cannot link dbCAN asset {source} to {destination}; "
                "the temporary directory must permit symbolic links, or reside "
                "on the same filesystem as the database"
            ) from hardlink_error


def prepare_dbcan_database(source: Path, output: Path, marker: Path) -> None:
    """Build a non-destructive run_dbCAN view supporting old and new names."""
    source = source.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"dbCAN database directory does not exist: {source}")

    def choose(*names: str) -> Path:
        for name in names:
            candidate = source / name
            if candidate.is_file() and candidate.stat().st_size > 0:
                return candidate
        raise FileNotFoundError(
            f"dbCAN database is missing a non-empty {' or '.join(names)} under {source}"
        )

    cazy = choose("CAZy.dmnd")
    family = choose("dbCAN.hmm", "dbCAN.txt")
    subfamily = choose("dbCAN-sub.hmm", "dbCAN_sub.hmm")
    mapping = choose("fam-substrate-mapping.tsv")

    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    aliases = {
        "CAZy.dmnd": cazy,
        "dbCAN.hmm": family,
        "dbCAN.txt": family,
        "dbCAN-sub.hmm": subfamily,
        "dbCAN_sub.hmm": subfamily,
        "fam-substrate-mapping.tsv": mapping,
    }
    for name, asset in aliases.items():
        _link_dbcan_asset(asset, output / name)

    # Older run_dbCAN releases call HMMER binaries and require hmmpress files.
    # Mirror any available indexes under both historical naming conventions.
    for asset, names in (
        (family, ("dbCAN.hmm", "dbCAN.txt")),
        (subfamily, ("dbCAN-sub.hmm", "dbCAN_sub.hmm")),
    ):
        for suffix in (".h3f", ".h3i", ".h3m", ".h3p"):
            sidecar = Path(str(asset) + suffix)
            if sidecar.is_file() and sidecar.stat().st_size > 0:
                for name in names:
                    _link_dbcan_asset(sidecar, output / f"{name}{suffix}")

    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps(
            {
                "source": str(source),
                "family_hmm": str(family),
                "subfamily_hmm": str(subfamily),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _comebin_training_progress(training_log: Path) -> str:
    """Summarize completed epochs from old and current COMEBin log formats."""
    if not training_log.is_file():
        return "training_not_started (data augmentation or preprocessing is active)"
    try:
        with training_log.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 1024 * 1024), os.SEEK_SET)
            content = handle.read().decode("utf-8", errors="replace")
    except OSError as exc:
        return f"training_progress_unavailable ({exc})"

    total_matches = re.findall(
        r"Start SimCLR training for\s+(\d+)\s+epochs", content
    )
    total = int(total_matches[-1]) if total_matches else 200
    completed: list[int] = []
    # COMEBin 1.0.x numbers the first completed epoch as Epoch: 0.
    completed.extend(int(value) + 1 for value in re.findall(r"Epoch:\s*(\d+)", content))
    # COMEBin 1.1.x emits one-based epoch=N/T progress records.
    completed.extend(
        int(value)
        for value in re.findall(r"NN training:\s*epoch=(\d+)/\d+", content)
    )
    latest = min(total, max(completed, default=0))
    if latest == 0:
        return f"completed_epochs=0/{total} (first epoch in progress)"
    return f"completed_epochs={latest}/{total}"


def run_comebin_with_heartbeat(
    command: str,
    output_dir: Path,
    device: str,
    heartbeat_seconds: int,
) -> None:
    """Run COMEBin while making old launchers' long first epoch observable."""
    if heartbeat_seconds < 1:
        raise ValueError("COMEBin heartbeat interval must be positive")
    output_dir.mkdir(parents=True, exist_ok=True)
    training_log = output_dir / "comebin_res" / "training.log"
    started = time.monotonic()
    print(
        f"[COMEBIN] Starting {device} training; liveness heartbeat interval="
        f"{heartbeat_seconds}s. COMEBin 1.0.x writes epoch progress only after "
        "an epoch completes.",
        flush=True,
    )
    print(f"[COMEBIN] Command: {command}", flush=True)
    shell_executable = shutil.which("bash") if os.name == "posix" else None
    process = subprocess.Popen(
        command,
        env=os.environ.copy(),
        shell=True,
        executable=shell_executable,
    )
    while True:
        try:
            return_code = process.wait(timeout=heartbeat_seconds)
            break
        except subprocess.TimeoutExpired:
            elapsed_seconds = int(time.monotonic() - started)
            elapsed = f"{elapsed_seconds // 3600:02d}:{elapsed_seconds % 3600 // 60:02d}:{elapsed_seconds % 60:02d}"
            progress = _comebin_training_progress(training_log)
            message = (
                f"[COMEBIN HEARTBEAT] process={process.pid} is still running; "
                f"device={device}; elapsed={elapsed}; {progress}. This confirms "
                "process liveness; training progress is based only on completed "
                "epochs reported by COMEBin."
            )
            print(message, flush=True)
            if training_log.parent.is_dir():
                try:
                    with training_log.open("a", encoding="utf-8") as handle:
                        handle.write(f"INFO:MetaBAW:{message}\n")
                except OSError as exc:
                    print(
                        f"[COMEBIN HEARTBEAT] Could not append to "
                        f"{training_log}: {exc}",
                        file=sys.stderr,
                        flush=True,
                    )
    if return_code != 0:
        raise RuntimeError(f"COMEBin exited with code {return_code}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m metabaw.internal")
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("prepare-samtools-compat")
    command.add_argument("--output", type=Path, required=True)
    command = sub.add_parser("run-checkm")
    command.add_argument("--command", dest="external_command", required=True)
    command.add_argument("--tmpdir", type=Path, required=True)
    command = sub.add_parser("run-comebin")
    command.add_argument("--command", dest="external_command", required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--device", choices=("cpu", "cuda"), required=True)
    command.add_argument("--heartbeat-seconds", type=int, default=1200)
    command = sub.add_parser("run-metadecoder-coverage")
    command.add_argument("--bam", type=Path, action="append", required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--completion-marker", type=Path, required=True)
    command.add_argument("--threads", type=int, required=True)
    command = sub.add_parser("run-gtdbtk-classify")
    command.add_argument("--genome-dir", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--extension", required=True)
    command.add_argument("--threads", type=int, required=True)
    command.add_argument("--place-species", action="store_true")
    command.add_argument("--pplacer-threads", type=int, default=1)
    command.add_argument("--scratch-dir", type=Path)
    command.add_argument("--ipc-tmpdir", type=Path, default=Path("/tmp"))
    command = sub.add_parser("fasta-filter")
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--min-length", type=int, required=True)
    command = sub.add_parser("rename-contigs")
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--prefix", required=True)
    command = sub.add_parser("concat-fasta")
    command.add_argument("--input", action="append", required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--min-length", type=int, required=True)
    command.add_argument("--separator", default="__")
    command = sub.add_parser("bins-to-map")
    command.add_argument("--bins-dir", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--label", required=True)
    command.add_argument("--min-bin-bp", type=int, default=0)
    command = sub.add_parser("normalize-map")
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--label", required=True)
    command.add_argument("--order", choices=("bin-contig", "contig-bin"), default="bin-contig")
    command = sub.add_parser("combine-maps")
    command.add_argument("--input", type=Path, action="append", required=True)
    command.add_argument("--output", type=Path, required=True)
    command = sub.add_parser("map-for-dastool")
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command = sub.add_parser("resolve-semibin-multisample")
    command.add_argument("--tool-root", type=Path, required=True)
    command.add_argument("--sample", action="append", required=True)
    command.add_argument("--output-root", type=Path, required=True)
    command.add_argument("--completion-marker", type=Path, required=True)
    command = sub.add_parser("run-dastool-refinement")
    command.add_argument("--binner-spec", action="append", required=True)
    command.add_argument("--assembly", type=Path, required=True)
    command.add_argument("--output-prefix", type=Path, required=True)
    command.add_argument("--output-bins", type=Path, required=True)
    command.add_argument("--completion-marker", type=Path, required=True)
    command.add_argument("--threads", type=int, required=True)
    command.add_argument("--extra-args", default="")
    command = sub.add_parser("run-metawrap-refinement")
    command.add_argument("--input", action="append", required=True)
    command.add_argument("--runner-json", required=True)
    command.add_argument("--output-root", type=Path, required=True)
    command.add_argument("--output-bins", type=Path, required=True)
    command.add_argument("--stats", type=Path, required=True)
    command.add_argument("--completion-marker", type=Path, required=True)
    command.add_argument("--threads", type=int, required=True)
    command.add_argument("--min-completeness", type=float, required=True)
    command.add_argument("--max-contamination", type=float, required=True)
    command.add_argument("--extra-args", default="")
    command = sub.add_parser("merge-aemb")
    command.add_argument("--input", action="append", required=True)
    command.add_argument("--output", type=Path, required=True)
    command = sub.add_parser("run-strobealign-aemb")
    command.add_argument("--contigs", type=Path, required=True)
    command.add_argument("--read", type=Path, action="append", required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--threads", type=int, required=True)
    command = sub.add_parser("parse-magscot-hmm")
    command.add_argument("--pfam", type=Path, required=True)
    command.add_argument("--tigr", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command = sub.add_parser("finalize-magscot")
    command.add_argument("--combined", type=Path, required=True)
    command.add_argument("--mapping", type=Path, required=True)
    command.add_argument("--run-log", type=Path)
    command = sub.add_parser("run-magscot")
    command.add_argument("--combined", type=Path, required=True)
    command.add_argument("--mapping", type=Path, required=True)
    command.add_argument("--run-log", type=Path, required=True)
    command.add_argument("--native-command-json", required=True)
    command = sub.add_parser("materialize-bins")
    command.add_argument("--assembly", type=Path, required=True)
    command.add_argument("--mapping", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--prefix", required=True)
    command.add_argument("--suffix", default="fa", help="MAG file extension (default: fa)")
    command.add_argument("--completion-marker", type=Path)
    command = sub.add_parser("materialize-refined")
    command.add_argument("--assembly", type=Path, required=True)
    command.add_argument("--mapping", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--prefix", required=True)
    command.add_argument("--source", type=Path, action="append", default=[])
    command.add_argument("--suffix", default="fa", help="MAG file extension (default: fa)")
    command.add_argument("--completion-marker", type=Path)
    command = sub.add_parser("stage-bams")
    command.add_argument("--bam", action="append", required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command = sub.add_parser("collect-fasta")
    command.add_argument("--source", type=Path, action="append", required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command = sub.add_parser("stage-annotation-genomes")
    command.add_argument("--genome", action="append", required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command = sub.add_parser("publish-fasta")
    command.add_argument("--source-dir", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--prefix", required=True)
    command.add_argument("--suffix", default="fa", help="MAG file extension (default: fa)")
    command = sub.add_parser("normalize-refined-fasta")
    command.add_argument("--source-dir", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--prefix", required=True)
    command.add_argument("--source", type=Path, action="append", default=[])
    command.add_argument("--suffix", default="fa", help="MAG file extension (default: fa)")
    command.add_argument("--completion-marker", type=Path)
    command = sub.add_parser("tag-contigs")
    command.add_argument("--input-dir", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--suffix", default="fa", help="MAG file extension (default: fa)")
    command = sub.add_parser("report-coassembly-provenance")
    command.add_argument("--bins-dir", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--suffix", default="fa")
    command.add_argument("--coassembly", action="append", default=[])
    command.add_argument("--individual", action="append", default=[])
    command = sub.add_parser("filter-quality")
    command.add_argument("--bins-dir", type=Path, required=True)
    command.add_argument("--checkm2", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--summary", type=Path, required=True)
    command.add_argument("--min-completeness", type=float, required=True)
    command.add_argument("--max-contamination", type=float, required=True)
    command.add_argument("--gunc-dir", type=Path)
    command.add_argument("--gunc-min-reference-representation", type=float, default=0.30)
    command.add_argument("--fail-gunc-unscored", action="store_true")
    command.add_argument("--min-quality-score", type=float)
    command.add_argument("--rna-summary", type=Path)
    command = sub.add_parser("report-dereplication-quality")
    command.add_argument("--summary", type=Path, required=True)
    command.add_argument("--representatives-dir", type=Path, required=True)
    command.add_argument("--completion-marker", type=Path)
    command = sub.add_parser("drep-genome-info")
    command.add_argument("--bins-dir", type=Path, required=True)
    command.add_argument("--quality-report", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--strip-extension", action="store_true")
    command = sub.add_parser("metawrap-checkm-quality")
    command.add_argument("--bins-dir", type=Path, required=True)
    command.add_argument("--stats", type=Path, action="append", required=True)
    command.add_argument("--manifest", type=Path, action="append", required=True)
    command.add_argument("--output", type=Path, required=True)
    command = sub.add_parser("rna-qc")
    command.add_argument("--bins-dir", type=Path, required=True)
    command.add_argument("--marker-dir", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--summary", type=Path, required=True)
    command.add_argument("--trnascan", default="tRNAscan-SE")
    command.add_argument("--barrnap", default="barrnap")
    command.add_argument("--threads", type=int, default=1)
    command.add_argument("--trna", action="store_true")
    command.add_argument("--trna-pass", type=int)
    command.add_argument("--rrna", action="store_true")
    command.add_argument("--rrna-pass", action="store_true")
    command.add_argument("--completion-marker", type=Path)
    command = sub.add_parser("merge-coverm-taxonomy")
    command.add_argument("--input", action="append", required=True)
    command.add_argument("--taxonomy-dir", type=Path, required=True)
    command.add_argument("--output-dir", type=Path, required=True)
    command.add_argument("--output-suffix", default=".tsv")
    command.add_argument("--niche-output", type=Path)
    command = sub.add_parser("classify-niche-literature")
    command.add_argument("--abundance", type=Path, required=True)
    command.add_argument("--taxonomy-dir", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--aggregated-output", type=Path, required=True)
    command.add_argument("--assignments-output", type=Path, required=True)
    command.add_argument("--provenance-output", type=Path, required=True)
    command.add_argument("--completion-marker", type=Path)
    command.add_argument("--sample-dataset", action="append", required=True)
    command.add_argument(
        "--rank",
        choices=("domain", "phylum", "class", "order", "family", "genus", "species", "strain"),
        required=True,
    )
    command.add_argument("--method", choices=("cv", "occupancy", "prevalence"), required=True)
    command.add_argument("--cv-correct", action="store_true")
    command.add_argument("--abundance-method", choices=("relative_abundance", "rpkm", "tpm", "mean"), default="relative_abundance")
    command.add_argument("--detection-percent", type=float, required=True)
    command.add_argument("--min-total-reads", type=int, required=True)
    command.add_argument("--min-prevalence", type=float, required=True)
    command.add_argument("--core-prevalence", type=float, required=True)
    command.add_argument("--min-core-datasets", type=int, required=True)
    command = sub.add_parser("merge-kofam")
    command.add_argument("--input", action="append", required=True)
    command.add_argument("--output", type=Path, required=True)
    command = sub.add_parser("merge-dbcan")
    command.add_argument("--input", action="append", required=True)
    command.add_argument("--output", type=Path, required=True)
    command = sub.add_parser("prepare-dbcan-database")
    command.add_argument("--source", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--marker", type=Path, required=True)
    command = sub.add_parser("build-functional-gene-map")
    command.add_argument("--mag", type=Path, required=True)
    command.add_argument("--proteins", type=Path, required=True)
    command.add_argument("--genome", required=True)
    command.add_argument("--output", type=Path, required=True)
    command = sub.add_parser("combine-functional-proteins")
    command.add_argument("--input", action="append", required=True)
    command.add_argument("--gene-map", action="append", required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--output-gene-map", type=Path, required=True)
    command = sub.add_parser("format-functional-annotations")
    command.add_argument("--input", type=Path, required=True)
    command.add_argument("--gene-map", action="append", required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--ko-descriptions", type=Path)
    command.add_argument(
        "--kind",
        choices=("kegg", "cazy", "hydrogenase", "terminal"),
        required=True,
    )
    command = sub.add_parser("filter-hydrogenase")
    command.add_argument("--hits", type=Path, required=True)
    command.add_argument("--proteins", type=Path, required=True)
    command.add_argument("--mapping", type=Path, required=True)
    command.add_argument("--genome", required=True)
    command.add_argument("--filtered-output", type=Path, required=True)
    command.add_argument("--fefe-fasta", type=Path, required=True)
    command.add_argument("--min-query-coverage", type=float, default=0.9)
    command.add_argument("--min-identity", type=float, default=0.5)
    command = sub.add_parser("finalize-hydrogenase")
    command.add_argument("--filtered", type=Path, required=True)
    command.add_argument("--fefe-hits", type=Path, required=True)
    command.add_argument("--nife-output", type=Path, required=True)
    command.add_argument("--fe-output", type=Path, required=True)
    command.add_argument("--fefe-output", type=Path, required=True)
    command.add_argument("--combined-output", type=Path, required=True)
    command = sub.add_parser("merge-hydrogenase")
    command.add_argument("--input", action="append", required=True)
    command.add_argument("--output", type=Path, required=True)
    command = sub.add_parser("filter-terminal-enzymes")
    command.add_argument("--hits", type=Path, required=True)
    command.add_argument("--genome", required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--min-query-coverage", type=float, default=0.80)
    command = sub.add_parser("merge-terminal-enzymes")
    command.add_argument("--input", action="append", required=True)
    command.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    apply_memory_limit_from_environment()
    args = build_parser().parse_args(argv)
    dispatch = {
        "prepare-samtools-compat": lambda: prepare_samtools_compat(args.output),
        "run-checkm": lambda: run_checkm(args.external_command, args.tmpdir),
        "run-metadecoder-coverage": lambda: run_metadecoder_coverage(
            tuple(args.bam), args.output, args.completion_marker, args.threads
        ),
        "run-comebin": lambda: run_comebin_with_heartbeat(
            args.external_command,
            args.output_dir,
            args.device,
            args.heartbeat_seconds,
        ),
        "run-gtdbtk-classify": lambda: run_gtdbtk_classify(
            args.genome_dir,
            args.output_dir,
            args.extension,
            args.threads,
            args.place_species,
            args.pplacer_threads,
            args.scratch_dir,
            args.ipc_tmpdir,
        ),
        "fasta-filter": lambda: fasta_filter(args.input, args.output, args.min_length),
        "rename-contigs": lambda: rename_fasta_records(
            args.input, args.output, args.prefix
        ),
        "concat-fasta": lambda: concatenate_fastas(
            args.input,
            args.output,
            args.min_length,
            args.separator,
        ),
        "bins-to-map": lambda: bins_to_map(
            args.bins_dir,
            args.output,
            args.label,
            args.min_bin_bp,
        ),
        "normalize-map": lambda: normalize_map(args.input, args.output, args.label, args.order),
        "combine-maps": lambda: combine_maps(args.input, args.output),
        "map-for-dastool": lambda: map_for_dastool(args.input, args.output),
        "resolve-semibin-multisample": lambda: resolve_semibin_multisample(
            args.tool_root,
            args.sample,
            args.output_root,
            args.completion_marker,
        ),
        "run-dastool-refinement": lambda: run_dastool_refinement(
            args.binner_spec,
            args.assembly,
            args.output_prefix,
            args.output_bins,
            args.completion_marker,
            args.threads,
            args.extra_args,
        ),
        "run-metawrap-refinement": lambda: run_metawrap_refinement(
            args.input,
            args.runner_json,
            args.output_root,
            args.output_bins,
            args.stats,
            args.completion_marker,
            args.threads,
            args.min_completeness,
            args.max_contamination,
            args.extra_args,
        ),
        "merge-aemb": lambda: merge_aemb(args.input, args.output),
        "run-strobealign-aemb": lambda: run_strobealign_aemb(
            args.contigs,
            args.read,
            args.output,
            args.threads,
        ),
        "parse-magscot-hmm": lambda: parse_magscot_hmm(args.pfam, args.tigr, args.output),
        "finalize-magscot": lambda: finalize_magscot_mapping(
            args.combined, args.mapping, args.run_log
        ),
        "run-magscot": lambda: run_magscot_refinement(
            args.combined, args.mapping, args.run_log, args.native_command_json
        ),
        "materialize-bins": lambda: materialize_bins(
            args.assembly,
            args.mapping,
            args.output_dir,
            args.prefix,
            args.suffix,
            args.completion_marker,
        ),
        "materialize-refined": lambda: materialize_refined(
            args.assembly,
            args.mapping,
            args.output_dir,
            args.prefix,
            args.source,
            args.suffix,
            args.completion_marker,
        ),
        "stage-bams": lambda: stage_bams(args.bam, args.output_dir),
        "stage-annotation-genomes": lambda: stage_annotation_genomes(args.genome, args.output_dir),
        "collect-fasta": lambda: collect_fasta(args.source, args.output_dir),
        "publish-fasta": lambda: publish_fasta(
            args.source_dir, args.output_dir, args.prefix, args.suffix
        ),
        "normalize-refined-fasta": lambda: normalize_refined_fasta(
            args.source_dir,
            args.output_dir,
            args.prefix,
            args.source,
            args.suffix,
            args.completion_marker,
        ),
        "tag-contigs": lambda: tag_contigs(args.input_dir, args.output_dir, args.suffix),
        "report-coassembly-provenance": lambda: report_coassembly_provenance(
            args.bins_dir,
            args.output,
            args.suffix,
            args.coassembly,
            args.individual,
        ),
        "filter-quality": lambda: filter_quality(
            args.bins_dir,
            args.checkm2,
            args.output_dir,
            args.summary,
            args.min_completeness,
            args.max_contamination,
            args.gunc_dir,
            args.gunc_min_reference_representation,
            args.fail_gunc_unscored,
            args.min_quality_score,
            args.rna_summary,
        ),
        "report-dereplication-quality": lambda: report_dereplication_quality(
            args.summary,
            args.representatives_dir,
            args.completion_marker,
        ),
        "drep-genome-info": lambda: write_drep_genome_info(
            args.bins_dir,
            args.quality_report,
            args.output,
            args.strip_extension,
        ),
        "metawrap-checkm-quality": lambda: write_metawrap_checkm_quality(
            args.bins_dir,
            args.stats,
            args.manifest,
            args.output,
        ),
        "rna-qc": lambda: rna_qc(
            args.bins_dir,
            args.marker_dir,
            args.output_dir,
            args.summary,
            args.trnascan,
            args.barrnap,
            args.threads,
            args.trna,
            args.trna_pass,
            args.rrna,
            args.rrna_pass,
            args.completion_marker,
        ),
        "merge-coverm-taxonomy": lambda: merge_coverm_taxonomy(
            args.input, args.taxonomy_dir, args.output_dir, args.output_suffix, args.niche_output
        ),
        "classify-niche-literature": lambda: classify_niche_literature(
            args.abundance,
            args.taxonomy_dir,
            args.output,
            args.aggregated_output,
            args.assignments_output,
            args.provenance_output,
            args.sample_dataset,
            args.rank,
            args.method,
            args.detection_percent,
            args.min_total_reads,
            args.min_prevalence,
            args.core_prevalence,
            args.min_core_datasets,
            args.abundance_method,
            args.completion_marker,
            args.cv_correct,
        ),
        "merge-kofam": lambda: merge_kofam_annotations(args.input, args.output),
        "merge-dbcan": lambda: merge_dbcan_annotations(args.input, args.output),
        "prepare-dbcan-database": lambda: prepare_dbcan_database(
            args.source, args.output, args.marker
        ),
        "build-functional-gene-map": lambda: build_functional_gene_map(
            args.mag, args.proteins, args.genome, args.output
        ),
        "combine-functional-proteins": lambda: combine_functional_proteins(
            args.input, args.gene_map, args.output, args.output_gene_map
        ),
        "format-functional-annotations": lambda: format_functional_annotations(
            args.input,
            args.gene_map,
            args.output,
            args.kind,
            args.ko_descriptions,
        ),
        "filter-hydrogenase": lambda: filter_hydrogenase_hits(
            args.hits,
            args.proteins,
            args.mapping,
            args.genome,
            args.filtered_output,
            args.fefe_fasta,
            args.min_query_coverage,
            args.min_identity,
        ),
        "finalize-hydrogenase": lambda: finalize_hydrogenase_hits(
            args.filtered,
            args.fefe_hits,
            args.nife_output,
            args.fe_output,
            args.fefe_output,
            args.combined_output,
        ),
        "merge-hydrogenase": lambda: merge_hydrogenase_annotations(
            args.input, args.output
        ),
        "filter-terminal-enzymes": lambda: filter_terminal_enzyme_hits(
            args.hits,
            args.genome,
            args.output,
            args.min_query_coverage,
        ),
        "merge-terminal-enzymes": lambda: merge_terminal_enzyme_annotations(
            args.input, args.output
        ),
    }
    try:
        dispatch[args.command]()
    except Exception as exc:
        print(f"metaBAW internal error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
