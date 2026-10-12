from __future__ import annotations

from contextlib import redirect_stdout
import base64
import csv
import hashlib
import io
from pathlib import Path
import tempfile
import tomllib
import unittest
from unittest.mock import patch
import zipfile

from release import (
    atomic_write, build_wheel, check_layout, development_environment,
    publish_wheel, unwanted_paths,
)
from verify_wheel import verify_wheel


class ReleaseHygieneTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.parent = Path(temporary.name)
        self.root = self.parent / "package"
        (self.root / "src" / "metabaw").mkdir(parents=True)
        (self.root / "tests").mkdir()
        (self.root / "examples").mkdir()
        (self.root / "src" / "metabaw" / "__init__.py").write_text("# source\n")
        (self.root / "tests" / "test_example.py").write_text("# test\n")
        (self.root / "examples" / "reads.tsv.example").write_text("S1\treads.fq\n")
        (self.root / "README.md").write_text("# MetaBAW\n")
        (self.root / "LICENSE").write_text("MIT\n")
        (self.root / "pyproject.toml").write_text(
            '[project]\nversion = "0.1.0"\n', encoding="utf-8",
        )

    def wheel(self, extra=None):
        source = self.root / "src" / "metabaw" / "__init__.py"
        prefix = "metabaw-0.1.0.dist-info"
        members = {
            "metabaw/__init__.py": source.read_bytes(),
            f"{prefix}/METADATA": (
                b"Version: 0.1.0\n"
                b"Summary: MetaBAW: Metagenome Binning Automated Workflow\n"
                b"Requires-Python: <3.12,>=3.11\n"
            ),
            f"{prefix}/WHEEL": b"Wheel-Version: 1.0\n",
            f"{prefix}/entry_points.txt": b"[console_scripts]\n",
            f"{prefix}/top_level.txt": b"metabaw\n",
            f"{prefix}/licenses/LICENSE": b"MIT\n",
        }
        if extra:
            members.update(extra)
        record = io.StringIO()
        writer = csv.writer(record, lineterminator="\n")
        for name, data in members.items():
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
            writer.writerow((name, f"sha256={digest}", len(data)))
        writer.writerow((f"{prefix}/RECORD", "", ""))
        members[f"{prefix}/RECORD"] = record.getvalue().encode("utf-8")
        staging = self.parent / "staged"
        staging.mkdir(exist_ok=True)
        wheel = staging / "metabaw-0.1.0-py3-none-any.whl"
        with zipfile.ZipFile(wheel, "w") as archive:
            for name, data in members.items():
                archive.writestr(name, data)
        return wheel

    def test_maintained_sources_tests_and_examples_are_allowed(self):
        self.assertEqual(unwanted_paths(self.root), [])
        check_layout(self.root)

    def test_generated_artifacts_and_results_are_rejected_not_deleted(self):
        for name in ("build", ".package_build", "metabaw_annotation_result", "src/metabaw.egg-info", "tests/__pycache__"):
            directory = self.root / name
            directory.mkdir(parents=True)
            (directory / "retained.txt").write_text("must not be deleted by audit")
        (self.root / "unused.log").write_text("log")
        bad = {path.relative_to(self.root).as_posix() for path in unwanted_paths(self.root)}
        self.assertEqual(bad, {
            "build", ".package_build", "metabaw_annotation_result", "src/metabaw.egg-info",
            "tests/__pycache__", "unused.log",
        })
        with self.assertRaisesRegex(RuntimeError, "Unapproved release"):
            check_layout(self.root)
        self.assertTrue((self.root / "metabaw_annotation_result" / "retained.txt").is_file())

    def test_multiple_release_wheels_are_rejected(self):
        for version in ("0.1.0", "0.2.0"):
            (self.root / f"metabaw-{version}-py3-none-any.whl").touch()
        self.assertEqual(len(unwanted_paths(self.root)), 2)

    def test_test_environment_disables_bytecode_and_plugin_side_effects(self):
        environment = development_environment(self.root)
        self.assertEqual(environment["PYTHONDONTWRITEBYTECODE"], "1")
        self.assertEqual(environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"], "1")

    def test_build_runs_from_minimal_external_snapshot(self):
        temporary = self.parent / "build-temp"
        temporary.mkdir()
        logo = self.root / "metabaw_title_logo.png"
        logo.write_bytes(b"logo fixture")
        def build(command, **options):
            self.assertNotEqual(options["cwd"], self.root)
            snapshot = Path(command[-1])
            self.assertFalse(snapshot.is_relative_to(self.root))
            self.assertTrue((snapshot / "src" / "metabaw" / "__init__.py").is_file())
            self.assertEqual((snapshot / logo.name).read_bytes(), logo.read_bytes())
            self.assertFalse((snapshot / "tests").exists())
            self.assertFalse((snapshot / "examples").exists())
            destination = Path(command[command.index("--outdir") + 1])
            destination.mkdir()
            (destination / "metabaw-0.1.0-py3-none-any.whl").write_bytes(b"wheel")
        with patch("release.subprocess.run", side_effect=build), patch("release.verify_wheel") as verify:
            wheel = build_wheel(self.root, temporary)
        verify.assert_called_once_with(wheel, self.root)
        self.assertEqual(unwanted_paths(self.root), [])

    def test_atomic_write_failure_keeps_original_and_removes_staging_file(self):
        target = self.root / "SHA256SUMS"
        target.write_bytes(b"old")
        with patch("release.os.replace", side_effect=PermissionError("blocked")):
            with self.assertRaises(PermissionError):
                atomic_write(target, b"new")
        self.assertEqual(target.read_bytes(), b"old")
        self.assertEqual(list(self.root.glob(".metabaw-release-*")), [])

    def test_only_runtime_sources_and_required_metadata_enter_wheel(self):
        wheel = self.wheel()
        self.assertEqual(verify_wheel(wheel, self.root), (7, 1))
        for extra in (
            "metabaw_annotation_result/start_info.txt", "tests/test_example.py",
            "metabaw/__pycache__/__init__.pyc", "README.md", "metabaw_title_logo.png", "unexpected.dist-info/file.txt",
        ):
            with self.subTest(extra=extra):
                wheel = self.wheel({extra: b"unwanted"})
                with self.assertRaisesRegex(ValueError, "content allowlist failed"):
                    verify_wheel(wheel, self.root)

    def test_publishing_retains_only_current_wheel_and_matching_checksum(self):
        old = self.root / "metabaw-0.0.9-py3-none-any.whl"
        old.write_bytes(b"obsolete wheel")
        staged = self.wheel()
        with redirect_stdout(io.StringIO()):
            current = publish_wheel(self.root, staged)
        self.assertFalse(old.exists())
        self.assertEqual(list(self.root.glob("*.whl")), [current])
        digest = hashlib.sha256(current.read_bytes()).hexdigest()
        self.assertEqual((self.root / "SHA256SUMS").read_text().strip(), f"{digest}  {current.name}")
        self.assertEqual(unwanted_paths(self.root), [])

    def test_setuptools_disables_automatic_extra_package_data(self):
        config = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertFalse(config["tool"]["setuptools"]["include-package-data"])
        self.assertEqual(config["tool"]["setuptools"]["packages"]["find"]["include"], ["metabaw", "metabaw.*"])

    def test_spdx_license_requires_supported_build_backend(self):
        root = Path(__file__).resolve().parents[1]
        config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
        self.assertEqual(config["build-system"]["requires"], ["setuptools>=77"])
        self.assertEqual(config["build-system"]["build-backend"], "setuptools.build_meta")
        self.assertEqual(config["project"]["license"], "MIT")
        self.assertEqual(config["project"]["dependencies"], ["setuptools<82"])
        self.assertIn(
            'python -m pip install "setuptools>=77,<82" build pytest',
            (root / "README.md").read_text(encoding="utf-8"),
        )

    def test_project_files_omit_deprecated_edition_suffix(self):
        root = Path(__file__).resolve().parents[1]
        deprecated_suffix = (" (" + "named-input" + " edition)").encode("utf-8")
        for path in root.rglob("*"):
            if path.is_file() and path.suffix != ".whl":
                with self.subTest(path=path.relative_to(root).as_posix()):
                    self.assertNotIn(deprecated_suffix, path.read_bytes())

    def test_wheel_rejects_deprecated_edition_suffix(self):
        deprecated_suffix = " (" + "named-input" + " edition)"
        metadata = (
            "Version: 0.1.0\n"
            "Summary: MetaBAW: Metagenome Binning Automated Workflow"
            f"{deprecated_suffix}\n"
            "Requires-Python: <3.12,>=3.11\n"
        ).encode("utf-8")
        wheel = self.wheel({"metabaw-0.1.0.dist-info/METADATA": metadata})
        with self.assertRaisesRegex(ValueError, "Deprecated edition suffix"):
            verify_wheel(wheel, self.root)


if __name__ == "__main__":
    unittest.main()
