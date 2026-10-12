"""CUDA libraries scoped to MetaDecoder, using the main environment's Python."""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import sys
import tempfile


@dataclass(frozen=True)
class CupyInstallPlan:
    cuda_version: str
    runtime_prefix: Path
    commands: tuple[tuple[str, ...], ...]


def runtime_root() -> Path:
    return Path(sys.prefix).resolve() / "share" / "metabaw" / "cuda"


def _runtime_environment(prefix: Path) -> dict[str, str]:
    """Put real CUDA libraries first; never expose the driver stub libraries."""
    targets = sorted((prefix / "targets").glob("*"))
    candidates = [*targets, prefix, prefix / "Library"]
    cuda_path = next((path for path in candidates
                      if (path / "include" / "cuda_runtime.h").is_file()), None)
    if cuda_path is None:
        raise RuntimeError(f"MetaDecoder CUDA runtime headers are missing: {prefix}")
    libraries = [path for base in candidates for name in ("lib", "lib64", "bin")
                 if (path := base / name).is_dir()]
    if not libraries:
        raise RuntimeError(f"MetaDecoder CUDA runtime libraries are missing: {prefix}")
    environment = {"CUDA_PATH": str(cuda_path)}
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        [*(str(path) for path in libraries), *filter(None, [os.environ.get("LD_LIBRARY_PATH")])]
    )
    bins = [path for base in (prefix, prefix / "Library") if (path := base / "bin").is_dir()]
    if bins:
        environment["PATH"] = os.pathsep.join(
            [*(str(path) for path in bins), *filter(None, [os.environ.get("PATH")])]
        )
    return environment


def save_cupy_runtime(plan: CupyInstallPlan) -> None:
    """Publish routing only after installation and runtime layout validation."""
    _runtime_environment(plan.runtime_prefix)
    root = runtime_root()
    root.mkdir(parents=True, exist_ok=True)
    configuration = root / "runtime.json"
    payload = json.dumps({"cuda_version": plan.cuda_version}, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=root,
                                     prefix=".runtime-", suffix=".tmp", delete=False) as handle:
        pending = Path(handle.name)
        try:
            handle.write(payload)
        except BaseException:
            handle.close()
            pending.unlink(missing_ok=True)
            raise
    try:
        pending.replace(configuration)
    finally:
        pending.unlink(missing_ok=True)


def cupy_runtime_environment() -> dict[str, str]:
    """Read this Python prefix's configuration without altering parent or PyTorch."""
    root = runtime_root()
    configuration = root / "runtime.json"
    if not configuration.exists():
        return {}
    try:
        payload = json.loads(configuration.read_text(encoding="utf-8"))
        version = payload["cuda_version"]
        if not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+", version):
            raise ValueError("invalid CUDA version")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"Invalid MetaDecoder CUDA runtime configuration: {configuration}") from exc
    environment = _runtime_environment(root / version)
    environment["METABAW_CUPY_CUDA_VERSION"] = version
    return environment
