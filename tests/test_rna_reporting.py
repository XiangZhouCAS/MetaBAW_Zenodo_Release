from __future__ import annotations

import csv
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from metabaw.direct import DirectBinBuilder
from metabaw.internal import filter_quality, rna_qc
from metabaw.model import Task


AMINO_ACIDS = (
    'Ala', 'Arg', 'Asn', 'Asp', 'Cys', 'Gln', 'Glu', 'Gly', 'His', 'Ile',
    'Leu', 'Lys', 'Met', 'Phe', 'Pro', 'Ser', 'Thr', 'Trp', 'Tyr', 'Val',
)


def read_table(path):
    with path.open(encoding='utf-8', newline='') as handle:
        return {row.get('genome', row.get('bin')): row
                for row in csv.DictReader(handle, delimiter='\t')}


class RnaReportingTests(TestCase):
    def run_report(self, *, run_trna=True, trna_pass=None, run_rrna=True,
                   rrna_pass=False, failures=()):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            bins = root / 'bins'
            bins.mkdir()
            genomes = {'complete': 20, 'missing_rrna': 20, 'low_trna': 16}
            for genome in genomes:
                (bins / f'{genome}.fa').write_text('>contig_1\nACGT\n', encoding='utf-8')
            output = root / 'rna'
            summary = root / 'rna_quality.tsv'
            marker = output / 'rna_qc.v3.complete'

            def fake_tool(command, destination):
                genome = Path(command[-1]).stem
                if (command[0], genome) in failures:
                    raise RuntimeError('synthetic tool failure')
                if command[0] == 'fake_trnascan':
                    destination = Path(command[command.index('-o') + 1])
                    content = ''.join(f'contig_1\t1\t1\t70\t{kind}\n'
                                      for kind in AMINO_ACIDS[:genomes[genome]])
                else:
                    kinds = () if genome == 'missing_rrna' else ('5S', '16S', '23S')
                    content = ''.join(f'contig_1\tbarrnap\trRNA\t1\t100\t.\t+\t.\tName={kind}_rRNA\n'
                                      for kind in kinds)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(content, encoding='utf-8')

            with patch('metabaw.internal._run_logged', side_effect=fake_tool):
                rna_qc(bins, root / 'markers', output, summary, 'fake_trnascan',
                       'fake_barrnap', 2, run_trna, trna_pass, run_rrna, rrna_pass, marker)
            self.assertEqual(marker.read_text(encoding='utf-8'), 'complete\n')
            return read_table(summary)

    def test_report_only_status_does_not_imply_complete_rna(self):
        rows = self.run_report()
        self.assertEqual({row['pass'] for row in rows.values()}, {'Not filtered'})
        self.assertEqual(rows['missing_rrna']['rrna_5S'], 'false')
        self.assertEqual(rows['missing_rrna']['rrna_16S'], 'false')
        self.assertEqual(rows['missing_rrna']['rrna_23S'], 'false')
        self.assertEqual(rows['low_trna']['trna_types'], '16')
        self.assertTrue(all(not row['error'] for row in rows.values()))

    def test_single_report_only_module_keeps_unrun_fields(self):
        for trna, rrna in ((True, False), (False, True)):
            with self.subTest(trna=trna, rrna=rrna):
                rows = self.run_report(run_trna=trna, run_rrna=rrna)
                self.assertTrue(all(row['pass'] == 'Not filtered' for row in rows.values()))
                for row in rows.values():
                    self.assertEqual(row['trna_types'] == 'not_run', not trna)
                    self.assertEqual(row['rrna_5S'] == 'not_run', not rrna)

    def test_report_only_tool_errors_are_still_reported(self):
        rows = self.run_report(failures={('fake_barrnap', 'missing_rrna')})
        self.assertEqual(rows['missing_rrna']['pass'], 'Not filtered')
        self.assertIn('rRNA: synthetic tool failure', rows['missing_rrna']['error'])

    def test_requested_filters_keep_boolean_statuses(self):
        cases = (
            (18, False, {'complete': 'true', 'missing_rrna': 'true', 'low_trna': 'false'}),
            (None, True, {'complete': 'true', 'missing_rrna': 'false', 'low_trna': 'true'}),
            (18, True, {'complete': 'true', 'missing_rrna': 'false', 'low_trna': 'false'}),
        )
        for trna_pass, rrna_pass, expected in cases:
            with self.subTest(trna_pass=trna_pass, rrna_pass=rrna_pass):
                rows = self.run_report(trna_pass=trna_pass, rrna_pass=rrna_pass)
                self.assertEqual({name: row['pass'] for name, row in rows.items()}, expected)

    def test_hard_filters_implicitly_enable_their_prediction_tools(self):
        rows = self.run_report(run_trna=False, run_rrna=False, trna_pass=18, rrna_pass=True)
        self.assertEqual(rows['complete']['trna_types'], '20')
        self.assertEqual(rows['complete']['rrna_5S'], 'true')
        self.assertEqual(rows['complete']['pass'], 'true')
        self.assertEqual(rows['missing_rrna']['pass'], 'false')
        self.assertEqual(rows['low_trna']['pass'], 'false')

    def test_requested_filter_still_fails_on_its_tool_error(self):
        rows = self.run_report(rrna_pass=True, failures={('fake_barrnap', 'complete')})
        self.assertEqual(rows['complete']['pass'], 'false')
        self.assertIn('synthetic tool failure', rows['complete']['error'])

    def test_report_only_status_does_not_hide_total_tool_failure(self):
        failures = {(tool, genome) for tool in ('fake_trnascan', 'fake_barrnap')
                    for genome in ('complete', 'missing_rrna', 'low_trna')}
        with self.assertRaisesRegex(RuntimeError, 'Every requested RNA tool invocation failed'):
            self.run_report(failures=failures)

    def test_downstream_accepts_report_only_status_without_weakening_other_filters(self):
        statuses = {
            'not_filtered': ('Not filtered', '', 90, 1),
            'not_filtered_error': (' not FILTERED ', 'rRNA: failure', 90, 1),
            'legacy_pass': ('TRUE', '', 90, 1),
            'failed': ('false', '', 90, 1),
            'tool_failed': ('false', 'rRNA: failure', 90, 1),
            'unknown': ('unknown', '', 90, 1),
            'low_quality': ('Not filtered', '', 49, 1),
            'contaminated': ('Not filtered', '', 90, 11),
            'missing_row': (None, '', 90, 1),
        }
        with TemporaryDirectory() as directory:
            root = Path(directory)
            bins = root / 'bins'
            bins.mkdir()
            for genome in statuses:
                (bins / f'{genome}.fa').write_text('>contig_1\nACGT\n', encoding='utf-8')
            quality = root / 'checkm2.tsv'
            quality.write_text('Name\tCompleteness\tContamination\n' + ''.join(
                f'{genome}\t{com}\t{con}\n' for genome, (_status, _error, com, con) in statuses.items()
            ), encoding='utf-8')
            report = root / 'rna_quality.tsv'
            report.write_text('genome\tpass\terror\n' + ''.join(
                f'{genome}\t{status}\t{error}\n' for genome, (status, error, _com, _con) in statuses.items()
                if status is not None
            ), encoding='utf-8')
            output = root / 'filtered_bins'
            summary = root / 'quality_summary.tsv'
            filter_quality(bins, quality, output, summary, 50, 10, None, rna_summary=report)
            rows = read_table(summary)
            self.assertEqual({path.stem for path in output.iterdir()},
                             {'not_filtered', 'not_filtered_error', 'legacy_pass'})
            self.assertEqual(rows['not_filtered']['pass_rna'], 'Not filtered')
            self.assertEqual(rows['not_filtered_error']['pass_rna'], 'Not filtered')
            self.assertEqual(rows['legacy_pass']['pass_rna'], 'true')
            self.assertEqual(rows['failed']['reason'], 'rna_failed')
            self.assertEqual(rows['tool_failed']['reason'], 'rna_tool_error')
            self.assertEqual(rows['unknown']['reason'], 'rna_failed')
            self.assertEqual(rows['missing_row']['reason'], 'rna_failed')
            self.assertEqual(rows['low_quality']['reason'], 'low_completeness')
            self.assertEqual(rows['contaminated']['reason'], 'high_contamination')

    def test_resume_marker_versions_report_and_report_only_is_not_a_filter(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            options = SimpleNamespace(
                run_trna=True, run_rrna=True, trna_pass=None, rrna_pass=False,
                threads=2, gtdbtk_data=None, mag_suffix='fa', workdir=root / 'tmp',
                outdir=root / 'results', min_completeness=50, max_contamination=10,
                min_quality_score=None,
            )
            builder = DirectBinBuilder([], options, root)
            candidates = root / 'candidates'
            catalog = Task('catalog', '05_quality', ('true',), root)
            task, summary = builder._rna((catalog, candidates))
            marker = options.outdir / 'quality_control_files' / 'rna' / 'rna_qc.v3.complete'
            self.assertIn(marker, task.outputs)
            self.assertIn(str(marker), task.display_command())
            legacy = replace(task, command=task.display_command().replace('rna_qc.v3.complete', 'rna_qc.v2.complete'),
                             outputs=tuple(Path(str(path).replace('rna_qc.v3.complete', 'rna_qc.v2.complete'))
                                           for path in task.outputs))
            self.assertNotEqual(task.fingerprint(['domain']), legacy.fingerprint(['domain']))
            filter_task, _ = builder._filter((catalog, candidates),
                                            Task('checkm2', '05_quality', ('true',), root),
                                            root / 'checkm2.tsv', None, task, summary)
            self.assertNotIn('--rna-summary', filter_task.display_command())
            options.trna_pass = 18
            filter_task, _ = builder._filter((catalog, candidates),
                                            Task('checkm2', '05_quality', ('true',), root),
                                            root / 'checkm2.tsv', None, task, summary)
            self.assertIn('--rna-summary', filter_task.display_command())
