"""Test and build a clean MetaBAW wheel without leaving build/test artifacts."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib

from verify_wheel import verify_wheel


ROOT_FILES = frozenset({
    ".gitignore", "AGENTS.md", "LICENSE", "pyproject.toml", "README.md",
    "RELEASE_NOTES.md", "SHA256SUMS", "release.py", "verify_wheel.py",
    "metabaw_title_logo.png",
})
EXAMPLE_FILES = frozenset({
    "reads.tsv.example", "contigs.tsv.example", "genomes.txt.example",
    "multi.txt.example", "run.sh", "run.coassembly.sh", "run.annotation.sh",
})


def unwanted_paths(root: Path) -> list[Path]:
    """Return unapproved paths; do not delete user data automatically."""
    unwanted: list[Path] = []
    wheels: list[Path] = []
    for path in sorted(root.iterdir()):
        if path.name == ".git":
            continue
        if path.is_symlink():
            unwanted.append(path)
        elif path.is_file():
            if path.name.startswith("metabaw-") and path.name.endswith("-py3-none-any.whl"):
                wheels.append(path)
            elif path.name not in ROOT_FILES:
                unwanted.append(path)
        elif path.is_dir() and path.name in {"src", "tests", "examples"}:
            for child in sorted(path.rglob("*")):
                relative = child.relative_to(root).parts
                if child.is_symlink():
                    unwanted.append(child)
                elif child.is_dir():
                    if relative != ("src", "metabaw"):
                        unwanted.append(child)
                elif path.name == "src":
                    if len(relative) != 3 or relative[1] != "metabaw" or child.suffix != ".py":
                        unwanted.append(child)
                elif path.name == "tests":
                    if len(relative) != 2 or child.suffix != ".py":
                        unwanted.append(child)
                elif len(relative) != 2 or child.name not in EXAMPLE_FILES:
                    unwanted.append(child)
        else:
            unwanted.append(path)
    if len(wheels) > 1:
        unwanted.extend(wheels)
    # Report a forbidden directory once, not every file below it.
    return [path for path in unwanted if not any(parent in unwanted for parent in path.parents)]


def check_layout(root: Path) -> None:
    unwanted = unwanted_paths(root)
    if unwanted:
        raise RuntimeError("Unapproved release files/directories:\n" + "\n".join(
            str(path.relative_to(root)) for path in unwanted
        ))


def development_environment(root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join((str(root / "src"), str(root)))
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return environment


def run_tests(root: Path, temporary: Path) -> None:
    work = temporary / "test-work"
    work.mkdir()
    subprocess.run(
        [sys.executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
         "--basetemp", str(temporary / "pytest"), str(root / "tests")],
        cwd=work, env=development_environment(root), check=True,
    )
    check_layout(root)


def build_wheel(root: Path, temporary: Path) -> Path:
    """Build from a minimal source snapshot located outside the release tree."""
    snapshot = temporary / "source"
    snapshot.mkdir()
    for name in ("LICENSE", "README.md", "pyproject.toml"):
        shutil.copy2(root / name, snapshot / name)
    logo = root / "metabaw_title_logo.png"
    if logo.is_file():
        shutil.copy2(logo, snapshot / logo.name)
    package = snapshot / "src" / "metabaw"
    package.mkdir(parents=True)
    for path in sorted((root / "src" / "metabaw").glob("*.py")):
        shutil.copy2(path, package / path.name)
    destination = temporary / "wheel"
    environment = development_environment(snapshot)
    subprocess.run(
        [sys.executable, "-B", "-m", "build", "--wheel", "--no-isolation",
         "--outdir", str(destination), str(snapshot)],
        cwd=temporary, env=environment, check=True,
    )
    wheels = list(destination.glob("*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"Expected one built wheel, found {len(wheels)}")
    verify_wheel(wheels[0], root)
    return wheels[0]


def atomic_write(path: Path, data: bytes) -> None:
    """Publish on the destination filesystem and remove temporary files."""
    with tempfile.NamedTemporaryFile(prefix=".metabaw-release-", suffix=".tmp", dir=path.parent, delete=False) as handle:
        pending = Path(handle.name)
        try:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            handle.close()
            pending.unlink(missing_ok=True)
            raise
    try:
        os.replace(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def publish_wheel(root: Path, wheel: Path) -> Path:
    check_layout(root)
    verify_wheel(wheel, root)
    previous = list(root.glob("metabaw-*-py3-none-any.whl"))
    destination = root / wheel.name
    data = wheel.read_bytes()
    atomic_write(destination, data)
    digest = hashlib.sha256(data).hexdigest()
    atomic_write(root / "SHA256SUMS", f"{digest}  {wheel.name}\n".encode("ascii"))
    for old in previous:
        if old != destination:
            # Exact top-level release wheels only; never recurse into user paths.
            old.unlink()
    check_layout(root)
    verify_wheel(destination, root)
    print(f"Published: {destination}\nSHA256: {digest}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="check directory, wheel contents and checksum only")
    mode.add_argument("--test", action="store_true", help="run tests outside the package directory without publishing")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    check_layout(root)
    if args.check:
        project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]
        wheel = root / f"metabaw-{project['version']}-py3-none-any.whl"
        verify_wheel(wheel, root)
        expected = f"{hashlib.sha256(wheel.read_bytes()).hexdigest()}  {wheel.name}"
        if (root / "SHA256SUMS").read_text(encoding="ascii").strip() != expected:
            raise RuntimeError("SHA256SUMS does not match the release wheel")
        print("OK: clean release directory, wheel content allowlist and SHA256SUMS")
        return
    with tempfile.TemporaryDirectory(prefix="metabaw-release-") as directory:
        temporary = Path(directory)
        run_tests(root, temporary)
        if not args.test:
            publish_wheel(root, build_wheel(root, temporary))
    check_layout(root)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"[RELEASE ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
