from __future__ import annotations

import csv
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from metabaw.direct import DirectBinBuilder
from metabaw.executor import Executor
from metabaw.internal import filter_quality, report_dereplication_quality
from metabaw.model import Task
from metabaw.state import StateStore


HEADER = 'bin\tcompleteness\tcontamination\tpass_gunc\tselected\treason\tquality_score\tpass_rna'


def read_rows(path: Path) -> tuple[list[str], dict[str, dict[str, str]]]:
    with path.open(encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle, delimiter='\t')
        fields = list(reader.fieldnames or [])
        return fields, {row['bin']: row for row in reader}


class DereplicationReportingTests(unittest.TestCase):
    def setup_files(self, root: Path) -> tuple[Path, Path, Path]:
        summary = root / 'quality_summary.tsv'
        summary.write_text(
            HEADER + '\textra\n'
            'retained\t90\t1\ttrue\ttrue\tpass\t85\tNot filtered\tone\n'
            'redundant\t80\t2\tunscored\ttrue\tpass\t70\ttrue\ttwo\n'
            'excluded\t40\t1\ttrue\tfalse\tlow_completeness\t35\tnot_run\tthree\n',
            encoding='utf-8',
        )
        representatives = root / 'representatives'
        representatives.mkdir()
        (representatives / 'retained.fa').write_text('>contig\nACGT\n', encoding='utf-8')
        return summary, representatives, root / 'report.complete'

    def test_upgrade_preserves_quality_and_marks_all_three_states(self):
        with TemporaryDirectory() as directory:
            summary, representatives, marker = self.setup_files(Path(directory))
            old_fields, old_rows = read_rows(summary)
            report_dereplication_quality(summary, representatives, marker)
            fields, rows = read_rows(summary)
            self.assertEqual(fields, old_fields + ['dereplication_pass'])
            self.assertEqual({name: row['dereplication_pass'] for name, row in rows.items()},
                             {'retained': 'TRUE', 'redundant': 'FALSE', 'excluded': 'Not evaluated'})
            for name in rows:
                self.assertEqual({field: rows[name][field] for field in old_fields}, old_rows[name])
            self.assertTrue(marker.is_file())
            first = summary.read_bytes()
            report_dereplication_quality(summary, representatives, marker)
            self.assertEqual(summary.read_bytes(), first)
            self.assertEqual(read_rows(summary)[0].count('dereplication_pass'), 1)
            self.assertEqual(list(summary.parent.glob('.dereplication-*.tsv')), [])

    def test_statuses_are_recalculated_when_representatives_change(self):
        with TemporaryDirectory() as directory:
            summary, representatives, marker = self.setup_files(Path(directory))
            report_dereplication_quality(summary, representatives, marker)
            (representatives / 'retained.fa').unlink()
            (representatives / 'redundant.fna').write_text('>contig\nACGT\n', encoding='utf-8')
            report_dereplication_quality(summary, representatives, marker)
            _, rows = read_rows(summary)
            self.assertEqual(rows['retained']['dereplication_pass'], 'FALSE')
            self.assertEqual(rows['redundant']['dereplication_pass'], 'TRUE')

    def test_empty_unknown_or_quality_excluded_representatives_do_not_publish(self):
        for invalid in ('empty', 'unknown', 'excluded', 'duplicate'):
            with self.subTest(invalid=invalid), TemporaryDirectory() as directory:
                summary, representatives, marker = self.setup_files(Path(directory))
                original = summary.read_bytes()
                marker.write_text('stale\n', encoding='utf-8')
                if invalid == 'empty':
                    (representatives / 'retained.fa').unlink()
                else:
                    name = 'retained.fna' if invalid == 'duplicate' else invalid + '.fa'
                    (representatives / name).write_text('>contig\nACGT\n', encoding='utf-8')
                with self.assertRaises(ValueError):
                    report_dereplication_quality(summary, representatives, marker)
                self.assertEqual(summary.read_bytes(), original)
                self.assertFalse(marker.exists())

    def test_invalid_summary_rows_do_not_publish(self):
        for invalid in ('duplicate', 'invalid_selected', 'missing_bin', 'truncated', 'extra_field'):
            with self.subTest(invalid=invalid), TemporaryDirectory() as directory:
                summary, representatives, marker = self.setup_files(Path(directory))
                text = summary.read_text(encoding='utf-8')
                if invalid == 'duplicate':
                    text += text.splitlines()[1] + '\n'
                elif invalid == 'invalid_selected':
                    text = text.replace('true\tpass', 'unknown\tpass', 1)
                elif invalid == 'missing_bin':
                    text = text.replace('bin\tcompleteness', 'name\tcompleteness', 1)
                elif invalid == 'truncated':
                    text += 'truncated\t90\t1\n'
                else:
                    text += text.splitlines()[1] + '\textra\n'
                summary.write_text(text, encoding='utf-8')
                original = summary.read_bytes()
                with self.assertRaises(ValueError):
                    report_dereplication_quality(summary, representatives, marker)
                self.assertEqual(summary.read_bytes(), original)
                self.assertFalse(marker.exists())

    def test_new_quality_report_has_pending_and_not_evaluated_states(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            bins = root / 'bins'
            bins.mkdir()
            for name in ('retained', 'excluded', 'missing'):
                (bins / (name + '.fa')).write_text('>contig\nACGT\n', encoding='utf-8')
            quality = root / 'checkm.tsv'
            quality.write_text('Name\tCompleteness\tContamination\nretained\t50\t10\nexcluded\t49\t0\n',
                               encoding='utf-8')
            summary = root / 'quality.tsv'
            filter_quality(bins, quality, root / 'filtered', summary, 50, 10, None)
            _, rows = read_rows(summary)
            self.assertEqual(rows['retained']['dereplication_pass'], 'Pending')
            self.assertEqual(rows['excluded']['dereplication_pass'], 'Not evaluated')
            self.assertEqual(rows['missing']['dereplication_pass'], 'Not evaluated')
            self.assertFalse(any(value is None for row in rows.values() for value in row.values()))

    def test_legacy_short_missing_checkm_row_is_preserved(self):
        with TemporaryDirectory() as directory:
            summary, representatives, marker = self.setup_files(Path(directory))
            summary.write_text(HEADER + '\nretained\t90\t1\ttrue\ttrue\tpass\t85\tnot_run\n'
                               'missing\t\t\t\tfalse\tmissing_checkm2\n', encoding='utf-8')
            report_dereplication_quality(summary, representatives, marker)
            _, rows = read_rows(summary)
            self.assertEqual(rows['missing']['dereplication_pass'], 'Not evaluated')
            self.assertEqual(rows['missing']['reason'], 'missing_checkm2')

    def test_resume_adds_report_without_rerunning_galah_or_drep(self):
        for tool in ('galah', 'drep'):
            for tag in (False, True):
                with self.subTest(tool=tool, tag=tag), TemporaryDirectory() as directory:
                    root = Path(directory)
                    options = SimpleNamespace(
                        outdir=root / 'results', workdir=root / 'work', tag_contigs=tag,
                        dereplicator=tool, mag_suffix='fa', min_completeness=50,
                        max_contamination=10, ani=99, min_aligned_fraction=30,
                        threads=2, extra_args={},
                    )
                    quality_dir = options.outdir / 'quality_control_files'
                    quality_dir.mkdir(parents=True)
                    summary, _unused, _marker = self.setup_files(quality_dir)
                    filtered = quality_dir / 'filtered_bins'
                    filtered.mkdir()
                    for name in ('retained', 'redundant'):
                        (filtered / (name + '.fa')).write_text('>contig\nACGT\n', encoding='utf-8')
                    quality = root / 'checkm.tsv'
                    quality.write_text('Name\tCompleteness\tContamination\n', encoding='utf-8')
                    filtered_task = Task('05.qc.filter', '05_quality', ('must-not-rerun',), root,
                                         inputs=(quality,), outputs=(filtered, summary))
                    builder = DirectBinBuilder([], options, root)
                    final_task, final_dir = builder._dereplicate(filtered_task, filtered, quality)
                    final_dir.mkdir(parents=True)
                    (final_dir / 'retained.fa').write_text('>contig\nACGT\n', encoding='utf-8')
                    if tool == 'galah':
                        (options.outdir / 'galah_clusters.tsv').write_text('cluster\n', encoding='utf-8')
                    report_task = builder._dereplication_report(final_task, final_dir)
                    self.assertEqual(report_task.deps, (final_task.id,))
                    self.assertIn('report-dereplication-quality', report_task.command)
                    fingerprints = (filtered_task.fingerprint([]), final_task.fingerprint(['quality']),
                                    report_task.fingerprint(['dereplication']))
                    state = StateStore(root / 'state.sqlite')
                    try:
                        executor = Executor(state, root / 'logs', max_cpus=2, max_parallel=1)
                        tasks = [filtered_task, final_task, report_task]
                        statuses = executor.run(tasks)
                        self.assertEqual(statuses[filtered_task.id], 'skipped')
                        self.assertEqual(statuses[final_task.id], 'skipped')
                        self.assertEqual(statuses[report_task.id], 'success')
                        self.assertEqual(executor.run(tasks), {task.id: 'skipped' for task in tasks})
                    finally:
                        state.close()
                    self.assertEqual(fingerprints, (filtered_task.fingerprint([]),
                                                   final_task.fingerprint(['quality']),
                                                   report_task.fingerprint(['dereplication'])))
                    self.assertEqual(read_rows(summary)[1]['retained']['dereplication_pass'], 'TRUE')

    def test_report_precedes_coassembly_provenance_and_contig_tagging(self):
        for strategy in ('default', 'coassembly'):
            for tag in (False, True):
                with self.subTest(strategy=strategy, tag=tag), TemporaryDirectory() as directory:
                    root = Path(directory)
                    options = SimpleNamespace(outdir=root / 'results', workdir=root / 'work',
                                              tag_contigs=tag, assembly_strategy=strategy, mag_suffix='fa')
                    builder = DirectBinBuilder([], options, root)
                    catalog = Task('catalog', '05_quality', ('true',), root)
                    quality = Task('quality', '05_quality', ('true',), root, deps=(catalog.id,))
                    filtered = Task('filter', '05_quality', ('true',), root, deps=(quality.id,))
                    final = Task('dereplicate', '06_dereplication', ('true',), root, deps=(filtered.id,))
                    for task in (catalog, quality, filtered, final):
                        builder.add(task)
                    final_dir = root / 'representatives'

                    def provenance(dependency, _final_dir):
                        return builder.add(Task('provenance', '07_catalog', ('true',), root,
                                                deps=(dependency.id,)))

                    with patch.object(builder, '_prepare_semibin_multisample'), \
                         patch.object(builder, '_add_binning_barrier'), \
                         patch.object(builder, '_catalog', return_value=(catalog, root / 'candidates')), \
                         patch.object(builder, '_quality', return_value=(quality, root / 'quality.tsv')), \
                         patch.object(builder, '_gunc', return_value=None), \
                         patch.object(builder, '_rna', return_value=(None, None)), \
                         patch.object(builder, '_filter', return_value=(filtered, root / 'filtered')), \
                         patch.object(builder, '_dereplicate', return_value=(final, final_dir)), \
                         patch.object(builder, '_coassembly_report', side_effect=provenance):
                        tasks = {task.id: task for task in builder.build()}
                    self.assertEqual(tasks['06.dereplicate.report'].deps, (final.id,))
                    if strategy == 'coassembly':
                        self.assertEqual(tasks['provenance'].deps, ('06.dereplicate.report',))
                    if tag:
                        expected = 'provenance' if strategy == 'coassembly' else '06.dereplicate.report'
                        self.assertEqual(tasks['07.tag_contigs'].deps, (expected,))


if __name__ == '__main__':
    unittest.main()
