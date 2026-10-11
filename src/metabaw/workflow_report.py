"""Human-readable, pre-execution methods and reproducibility record."""
from __future__ import annotations

import argparse
from hashlib import sha256
from importlib import metadata, resources
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import sys

from . import __version__
from .metadecoder_coverage import metadecoder_executable
from .dependencies import ISOLATED_TOOLS, SOFTWARE
from .model import Task, topological_order


LABELS = {
    "metabat2": "MetaBAT2", "metadecoder": "MetaDecoder", "vamb": "VAMB",
    "run_comebin.sh": "COMEBin", "SemiBin2": "SemiBin2", "LorBin": "LorBin",
    "MAGScoT.R": "MAGScoT", "DAS_Tool": "DAS Tool", "metawrap": "MetaWRAP",
    "checkm": "CheckM", "checkm2": "CheckM2", "gunc": "GUNC",
    "gtdbtk": "GTDB-Tk", "Rscript": "R", "coverm": "CoverM",
    "exec_annotation": "KofamScan (KEGG)", "run_dbcan": "dbCAN (CAZy)",
    "blastp": "BLASTP (NCBI BLAST+)", "makeblastdb": "makeblastdb (NCBI BLAST+)",
    "hmmsearch": "hmmsearch (HMMER)", "megahit": "MEGAHIT", "flye": "Flye",
}
PACKAGES = {item.executable: item.package.split("=")[0] for item in SOFTWARE.values()}
PACKAGES.update({item.executable: item.package.split("=")[0] for item in ISOLATED_TOOLS.values()})
PACKAGES.update({
    "LorBin": "lorbin", "minibwa": "minibwa", "MAGScoT.R": "magscot", "bowtie2-build": "bowtie2",
    "jgi_summarize_bam_contig_depths": "metabat2", "hmmscan": "hmmer",
    "hmmpress": "hmmer",
})
WRAPPED_TOOLS = {
    "run-checkm": ("checkm",),
    "run-comebin": ("run_comebin.sh",),
    "run-metadecoder-coverage": ("metadecoder", "samtools"),
    "run-strobealign-aemb": ("strobealign",),
    "run-dastool-refinement": ("DAS_Tool", "diamond"),
    "run-metawrap-refinement": ("metawrap",),
    "run-gtdbtk-classify": ("gtdbtk",),
    "prepare-samtools-compat": ("samtools",),
}
WRAPPER_NOTES = {
    "run-checkm": "Native CheckM lineage_wf in the resolved CheckM console-launcher Python. A process-local, literal-only compatibility reader accepts finite NumPy scalar representations in statistics; native calculations, thread budget and final QC thresholds are unchanged.",
    "run-magscot": "MAGScoT native scoring and refinement; managed thresholds use 0.8 x the requested completeness and 1.2 x contamination (capped at 100%). Final CheckM/CheckM2 filtering retains the original thresholds. Confirmed no-bin results contribute zero refined MAGs without substituting rejected source bins.",
    "run-metadecoder-coverage": "MetaDecoder native coverage with --bin_size 2000; fresh private BAM indexes, supervised workers and complete coverage validation. See the task log for the resolved invocation.",
    "run-dastool-refinement": "DAS Tool uses --search_engine diamond --write_bins --write_bin_evals; available binner inputs are selected at runtime. Verified no-bin score outcomes contribute zero MAGs.",
    "run-gtdbtk-classify": "GTDB-Tk classify_wf; --place_species is passed only if the installed version supports it. See the task log for the resolved invocation.",
    "run-metawrap-refinement": "MetaWRAP bin_refinement uses native CheckM. With downstream CheckM2, managed refinement thresholds use 0.8 x the requested completeness and 1.2 x contamination (capped at 100%); with downstream CheckM, thresholds remain unchanged. Final quality filtering retains the original thresholds. Available binner directories are selected at runtime.",
    "metawrap-checkm-quality": "Reuses CheckM estimates produced during MetaWRAP refinement; does not launch a new CheckM run.",
    "run-comebin": "COMEBin launcher settings and heartbeat interval are recorded below; upstream training defaults depend on the installed COMEBin version.",
}


def _tokens(command: str | tuple[str, ...]) -> list[str]:
    if not isinstance(command, str):
        return list(command)
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        lexer.commenters = ""
        return list(lexer)
    except ValueError:
        return []  # The full, unmodified command is always retained in the report.


def _command_tokens(command):
    """Include wrapped native child commands without executing them."""
    tokens = _tokens(command)
    yield tokens
    for index, token in enumerate(tokens[:-1]):
        if token in {"--command", "--pipe"}:
            yield from _command_tokens(tokens[index + 1])
        elif token == "--native-command-json":
            try:
                native = json.loads(tokens[index + 1])
            except (TypeError, ValueError):
                continue
            if isinstance(native, list) and all(isinstance(value, str) for value in native):
                yield from _command_tokens(tuple(native))


def _tool_position(tokens, index):
    """Recognize executable positions, not directories, labels or conda env names."""
    if tokens[index] == "metabaw.internal":
        return index > 0 and tokens[index - 1] == "-m"
    if index == 0 or tokens[index - 1] in {"&&", "||", "|", ";", "then", "do", "exec"}:
        return True
    if tokens[index - 1].replace("\\", "/").rsplit("/", 1)[-1] == "Rscript":
        return True
    start = index - 1
    while start > 0 and tokens[start - 1] not in {"&&", "||", "|", ";", "then", "do"}:
        start -= 1
    while start < index and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tokens[start]):
        start += 1
    if start == index:
        return True
    if tokens[start].replace("\\", "/").rsplit("/", 1)[-1] in {"conda", "mamba", "micromamba"}:
        start += 1
        if start < index and tokens[start] == "run":
            start += 1
            while start < index and tokens[start].startswith("-"):
                start += 2 if tokens[start] in {"-n", "--name", "-p", "--prefix", "--cwd"} else 1
            return start == index
    return False


def task_software(task: Task) -> tuple[list[str], list[str]]:
    software: list[str] = []
    notes: list[str] = []
    for tokens in _command_tokens(task.command):
        for index, token in enumerate(tokens):
            name = token.replace("\\", "/").rsplit("/", 1)[-1]
            if name in PACKAGES and _tool_position(tokens, index):
                software.append(name)
            if token in ("minimap2-sr", "minimap2-ont", "minimap2-pb", "minimap2-hifi"):
                software.append("minimap2")
            if token == "metabaw.internal" and index + 1 < len(tokens) and _tool_position(tokens, index):
                helper = tokens[index + 1]
                software.append("MetaBAW")
                software.extend(WRAPPED_TOOLS.get(helper, ()))
                if helper == "rna-qc":
                    if "--trna" in tokens:
                        software.append("tRNAscan-SE")
                    if "--rrna" in tokens:
                        software.append("barrnap")
                if helper in WRAPPER_NOTES:
                    notes.append(WRAPPER_NOTES[helper])
    return list(dict.fromkeys(software or ["shell/runtime utilities"])), notes


def _package_version(executable: str | None, package: str) -> tuple[str, str]:
    """Read metadata belonging to this executable, without launching tools."""
    if not executable:
        return "not detected", "executable unavailable during report generation"
    path = Path(executable).expanduser().resolve()
    prefix = path.parent.parent
    normalized = re.sub(r"[-_.]+", "-", package.lower())
    for record in sorted((prefix / "conda-meta").glob(f"{package}-*.json")):
        try:
            data = json.loads(record.read_text(encoding="utf-8"))
            if re.sub(r"[-_.]+", "-", data.get("name", "").lower()) == normalized and data.get("version"):
                return str(data["version"]), str(record)
        except (OSError, ValueError, AttributeError):
            continue
    sites = [prefix / "Lib" / "site-packages", *sorted((prefix / "lib").glob("python*/site-packages"))]
    for site in sites:
        if not site.is_dir():
            continue
        for distribution in metadata.distributions(path=[str(site)]):
            try:
                name = distribution.metadata.get("Name", "")
                if re.sub(r"[-_.]+", "-", name.lower()) == normalized:
                    return distribution.version, f"package metadata under {site}"
            except (OSError, ValueError):
                continue
    return "not detected", "no matching package metadata for the resolved executable"


def _inventory(software: list[str], args: argparse.Namespace) -> list[dict[str, str]]:
    detected = getattr(args, "_software_executables", {})
    isolated = {spec.executable: key for key, spec in ISOLATED_TOOLS.items()}
    rows = []
    for tool in software:
        if tool == "MetaBAW":
            rows.append({"software": tool, "version": __version__, "executable": sys.executable,
                         "version_source": "metabaw.__version__", "environment": sys.prefix})
            continue
        if tool == "shell/runtime utilities":
            rows.append({"software": tool, "version": "not detected", "executable": "see task command",
                         "version_source": "runtime-dependent", "environment": "task working directory / PATH"})
            continue
        key = isolated.get(tool)
        executable = detected.get(tool)
        if tool == "metadecoder":
            executable = metadecoder_executable()
        if tool == "MAGScoT.R" and getattr(args, "magscot_dir", None):
            candidate = Path(args.magscot_dir) / tool
            executable = str(candidate) if candidate.is_file() else None
        if not executable and key is None and tool != "metadecoder":
            executable = shutil.which(tool)
        version, source = _package_version(executable, PACKAGES[tool])
        if tool == "run_comebin.sh" and getattr(args, "comebin_version", None):
            version, source = args.comebin_version, "COMEBin launcher metadata read during preflight"
        row = {"software": LABELS.get(tool, tool), "version": version,
               "executable": executable or "not detected", "version_source": source,
               "environment": str(getattr(args, f"{key}_env", "unknown")) if key else "main environment / PATH"}
        if tool == "MAGScoT.R" and executable:
            row["script_sha256"] = sha256(Path(executable).read_bytes()).hexdigest()
        rows.append(row)
    return rows


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


PHASES = {
    "00_prepare": "Genome preparation", "01_prepare": "Contig preparation",
    "01_taxonomy": "Taxonomic classification", "02_mapping": "Alignment",
    "02_abundance": "Abundance estimation", "03_binning": "Binning",
    "03_merge": "Taxonomy and abundance integration", "04_refinement": "Refinement",
    "04_niche": "Niche classification", "05_quality": "Quality control",
    "05_gene_prediction": "Gene prediction", "06_dereplication": "Dereplication",
    "06_kegg": "KEGG annotation", "07_cazy": "CAZy annotation",
    "07_catalog": "MAG dataset and provenance", "08_hydrogenase": "Hydrogenase annotation",
    "09_terminal_enzymes": "Hydrogen-metabolism terminal enzymes",
}
INTERNAL_METHODS = {
    "fasta-filter": "contig filtering", "concat-fasta": "contig filtering",
    "filter-quality": "MAG quality filtering", "classify-niche-literature": "niche classification",
    "filter-hydrogenase": "HydDB hit filtering", "finalize-hydrogenase": "hydrogenase classification",
    "filter-terminal-enzymes": "terminal-enzyme hit filtering",
}
OMIT_OPTIONS = {
    "-i", "-o", "-1", "-2", "-U", "--input", "--output", "--prefix", "--sample",
    "--sample-dataset", "--genome", "--label", "--labels", "--kind", "--binner-spec",
    "--runner-json", "--native-command-json", "--run-log", "--completion-marker", "--heartbeat-seconds",
    "--tmpdir", "--tmp-dir", "--out-dir", "--outdir", "--output-dir", "--pipe",
}
SHELL_BOUNDARIES = {"&&", "||", "|", ";", ">", ">>", "<", "2>", "&", "(", ")"}
MODES = {"index", "view", "sort", "genome", "predict", "lineage_wf", "cluster", "recluster",
         "dereplicate", "classify_wf", "CAZyme_annotation", "single_easy_bin", "multi_easy_bin",
         "bin_refinement", "bin", "default", "coverage", "seed", "kmer", "aamb", "aemb", "hmm",
         "run", "identify", "makedb", "blastp"}
BOOLEAN_OPTIONS = {"--cut_nc", "--cut_ga", "--cut_tc", "--noali", "--notextw", "--force",
                   "--no-report-unannotated", "--very-sensitive", "--self-supervised", "--aemb",
                   "--cv-correct",
                   "--pacbio-raw", "--pacbio-corr", "--pacbio-hifi",
                   "--nano-raw", "--nano-corr", "--nano-hq"}


def _file_operand(value):
    return ("/" in value or "\\" in value or value.startswith(("{", "[", "$"))
            or re.search(r"\.(?:fa|fna|fasta|faa|fq|fastq|bam|sam|tsv|txt|csv|hmm|dmnd|gz)$", value))


def _is_flag(value: str) -> bool:
    return value.startswith("-") and value != "-" and not re.fullmatch(r"-\d+(?:\.\d+)?", value)


def _parameters(tokens: list[str]) -> str:
    """Keep effective numeric/method options, omitting per-sample file operands."""
    parameters = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not _is_flag(token):
            if token in MODES:
                parameters.append(token)
            index += 1
            continue
        option, separator, inline = token.partition("=")
        values = [inline] if separator else []
        index += 1
        if not separator:
            while index < len(tokens) and not _is_flag(tokens[index]):
                values.append(tokens[index])
                index += 1
        if option == "--extra-args":
            extra = _parameters(_tokens(" ".join(values)))
            if extra:
                parameters.append(extra)
            continue
        if option in OMIT_OPTIONS and not (option == "-o" and not values):
            continue
        # Input/output/database paths and sample labels remain in run_manifest.json.
        first_file = next((i for i, value in enumerate(values) if _file_operand(value)), None)
        if first_file is not None:
            if first_file == 0 and option not in BOOLEAN_OPTIONS:
                continue
            values = values[:first_file]
        values = [value for value in values if value != "-"]
        parameters.append(" ".join([option, *values]))
    return "; ".join(parameters)


def _method_calls(task: Task) -> list[tuple[str, str]]:
    """Summarize actual tool calls, including tools invoked through internal wrappers."""
    calls = []
    for tokens in _command_tokens(task.command):
        for index, token in enumerate(tokens):
            name = token.replace("\\", "/").rsplit("/", 1)[-1]
            if name not in PACKAGES and token != "metabaw.internal":
                continue
            if not _tool_position(tokens, index) or name == "parallel":
                continue
            stop = next((j for j in range(index + 1, len(tokens))
                         if tokens[j] in SHELL_BOUNDARIES), len(tokens))
            tail = tokens[index + 1:stop]
            if token == "metabaw.internal":
                if not tail:
                    continue
                helper, arguments = tail[0], tail[1:]
                parameters = _parameters(arguments)
                if helper in INTERNAL_METHODS:
                    calls.append(("MetaBAW", "; ".join(filter(None, (INTERNAL_METHODS[helper], parameters)))))
                elif helper == "run-gtdbtk-classify":
                    parameters = parameters.replace("--threads", "--cpus").replace("--pplacer-threads", "--pplacer_cpus")
                    parameters = parameters.replace("--place-species", "species placement=requested (version-dependent)")
                    calls.append(("gtdbtk", "classify_wf; " + parameters))
                elif helper == "run-metadecoder-coverage":
                    calls.append(("metadecoder", "coverage; --bin_size 2000; " + parameters))
                    calls.append(("samtools", "quickcheck -v; view -H"))
                elif helper == "run-dastool-refinement":
                    calls.append(("DAS_Tool", "--search_engine diamond; --write_bins; --write_bin_evals; " + parameters.replace("--threads", "-t")))
                elif helper == "run-metawrap-refinement":
                    calls.append(("metawrap", "bin_refinement; " + parameters.replace("--threads", "-t")
                                  .replace("--min-completeness", "-c").replace("--max-contamination", "-x")))
                elif helper == "run-strobealign-aemb":
                    calls.append(("strobealign", "--aemb; " + parameters.replace("--threads", "-t")))
                elif helper == "rna-qc":
                    if "--trna" in arguments:
                        calls.append(("tRNAscan-SE", "-A for archaea / -B for bacteria; threads allocated per genome"))
                    if "--rrna" in arguments:
                        calls.append(("barrnap", "--kingdom arc / bac; threads allocated per genome"))
                continue
            # The script itself supplies the relevant refinement parameters.
            if name == "Rscript" and any(value.endswith("MAGScoT.R") for value in tail):
                continue
            parameters = _parameters(tail)
            if name in {"vamb", "run_comebin.sh", "SemiBin2", "LorBin"}:
                parameters = "; ".join(filter(None, (parameters, "device=" + ("GPU" if task.gpus else "CPU"))))
            calls.append((name, parameters or "installed defaults"))
    return calls


def _variant_details(variants: list[str]) -> str:
    if len(variants) == 1:
        return variants[0]
    parts = [variant.split("; ") for variant in variants]
    common = [item for item in parts[0] if all(item in other for other in parts[1:])]
    if not common:
        return " | ".join(variants)
    differences = ["; ".join(item for item in variant if item not in common) or "no additional options"
                   for variant in parts]
    return "; ".join(common) + "; variants: [" + " | ".join(differences) + "]"


def _compact_steps(tasks: list[Task], versions: dict[str, str], payload: dict) -> str:
    phases: dict[str, dict[str, list[str]]] = {}
    for task in tasks:
        phase = (
            "Coassembly" if task.id.startswith("01.coassemble.")
            else "Assembly" if task.id.startswith("01.assemble.")
            else PHASES.get(task.stage, task.stage)
        )
        tools = phases.setdefault(phase, {})
        for tool, parameters in _method_calls(task):
            variants = tools.setdefault(tool, [])
            if parameters not in variants:
                variants.append(parameters)
    lines = []
    for number, (phase, tools) in enumerate(phases.items(), 1):
        lines.extend([f"# Step{number}", "", phase, ""])
        for tool, variants in (tools or {"MetaBAW": [phase.lower()]}).items():
            label = LABELS.get(tool, tool)
            version = versions.get(label)
            details = _variant_details(variants)
            if version and version != "not detected":
                details = f"version={version}; {details}"
            lines.extend([f"Software: {label} ({details})", ""])
    if payload.get("gtdbtk_classification_skipped"):
        lines.extend(["Taxonomy: reused validated GTDB-Tk results.", ""])
    niche = payload.get("niche")
    if isinstance(niche, dict) and not niche.get("enabled"):
        lines.extend([f"Niche classification: skipped; {niche.get('reason', 'not selected')}.", ""])
    lines.append("Planned methods. Full commands and provenance: run_manifest.json; startup diagnostics: start_info.txt.")
    return "\n".join(lines) + "\n"


def _code_fingerprint() -> str:
    digest = sha256()
    for source in sorted(resources.files("metabaw").iterdir(), key=lambda item: item.name):
        if source.name.endswith(".py"):
            digest.update(source.name.encode("utf-8") + b"\0")
            digest.update(source.read_bytes())
    return digest.hexdigest()


def write_workflow_details(
    path: Path, args: argparse.Namespace, tasks: list[Task], payload: dict[str, object]
) -> dict[str, object]:
    """Persist the resolved plan before any workflow task is launched."""
    ordered = topological_order(tasks)
    detected = {task.id: task_software(task) for task in ordered}
    software = list(dict.fromkeys(["MetaBAW", *(tool for tools, _notes in detected.values() for tool in tools)]))
    inventory = _inventory(software, args)
    code_hash = _code_fingerprint()
    invocation = getattr(args, "_invocation", None)
    options = {name: value for name, value in vars(args).items()
               if not name.startswith("_") and name != "func" and not callable(value)}
    manifest_records = []
    for kind, value in payload.get("input_sources", {}).items():
        source = value.get("path") if isinstance(value, dict) else value
        if not source:
            continue
        record = {"kind": kind, "path": str(source)}
        try:
            content = Path(source).read_bytes()
            record.update({"sha256": sha256(content).hexdigest(), "content": content.decode("utf-8-sig")})
        except (OSError, UnicodeError) as exc:
            record["snapshot_error"] = str(exc)
        manifest_records.append(record)
    runtime = {"created_utc": payload.get("created_utc"), "metabaw_version": __version__,
               "metabaw_source_sha256": code_hash, "python": sys.version,
               "python_executable": sys.executable, "platform": platform.platform(),
               "invocation_cwd": getattr(args, "_invocation_cwd", str(Path.cwd())),
               "selected_database_environment": {key: os.environ[key] for key in
                   ("GTDBTK_DATA_PATH", "CHECKM_DATA_PATH", "KOFAM_DB", "DBCAN_DB", "HYDROGENASE_DB")
                   if key in os.environ}}
    versions = {row["software"]: row["version"] for row in inventory}
    path.write_text(_compact_steps(ordered, versions, payload), encoding="utf-8")
    return {
        "path": str(path), "metabaw_source_sha256": code_hash,
        "software_inventory": inventory, "input_manifests": manifest_records,
        "runtime": runtime, "invocation": invocation,
        "resolved_options": json.loads(_json(options)),
    }
