from __future__ import annotations

from collections import Counter
import csv
from dataclasses import replace
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

from metabaw.internal import _cv_rank_quartiles, classify_niche_literature
from metabaw.model import Task


class CvRankQuartileTests(TestCase):
    def test_extreme_quarters_and_integer_remainder(self):
        for size in (0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 101, 400):
            with self.subTest(size=size):
                rows = [{'taxon': f'T{i:03d}', 'raw_si': i / 10, 'eligible': True}
                        for i in reversed(range(size))]
                labels, ranks, metadata = _cv_rank_quartiles(rows)
                tail = size // 4
                counts = Counter(labels.values())
                self.assertEqual(counts['generalist'], tail)
                self.assertEqual(counts['specialist'], tail)
                self.assertEqual(counts['intermediate'], size - 2 * tail)
                self.assertEqual(list(ranks.values()), list(range(1, size + 1)))
                for i in range(size):
                    expected = ('generalist' if i < tail else 'specialist'
                                if i >= size - tail else 'intermediate')
                    self.assertEqual(labels[f'T{i:03d}'], expected)
                self.assertEqual(metadata['eligible_taxa'], size)
                self.assertEqual(metadata['tail_size'], tail)
                self.assertEqual(metadata['intermediate_size'], size - 2 * tail)
                self.assertEqual(metadata['low_tail_max_cv'], (tail - 1) / 10 if tail else None)
                self.assertEqual(metadata['high_tail_min_cv'], (size - tail) / 10 if tail else None)

    def test_exact_ties_use_taxon_names_without_inflating_tail_sizes(self):
        rows = [{'taxon': name, 'raw_si': 1.0, 'eligible': True} for name in 'hgfedcba']
        first = _cv_rank_quartiles(rows)
        second = _cv_rank_quartiles(list(reversed(rows)))
        self.assertEqual(first, second)
        labels, ranks, metadata = first
        self.assertEqual(labels, {
            'a': 'generalist', 'b': 'generalist', 'c': 'intermediate',
            'd': 'intermediate', 'e': 'intermediate', 'f': 'intermediate',
            'g': 'specialist', 'h': 'specialist',
        })
        self.assertEqual(ranks['a'], 1)
        self.assertEqual(ranks['h'], 8)
        self.assertEqual(metadata['low_tail_max_cv'], metadata['high_tail_min_cv'])

    def test_support_filter_is_applied_before_computing_quotas(self):
        rows = [{'taxon': f'T{i}', 'raw_si': i, 'eligible': True} for i in range(4)]
        rows += [{'taxon': f'unsupported{i}', 'raw_si': -100, 'eligible': False}
                 for i in range(20)]
        rows += [{'taxon': 'invalid', 'raw_si': math.inf, 'eligible': True}]
        labels, ranks, metadata = _cv_rank_quartiles(rows)
        self.assertEqual(metadata['eligible_taxa'], 4)
        self.assertEqual(metadata['tail_size'], 1)
        self.assertEqual(set(labels), {'T0', 'T1', 'T2', 'T3'})
        self.assertEqual(set(ranks), set(labels))

    def run_analysis(self, root, *, method='cv', reversed_input=False, marker=None, cv_correct=False):
        # Two represented families have two MAGs each. Ranking is at the
        # selected family level, not weighted by the number of member MAGs.
        profiles = {
            'A1': [0.5, 0.5, 0.5, 0.5], 'A2': [0.5, 0.5, 0.5, 0.5],
            'B1': [0.99, 1.01, 0.99, 1.01],
            'C1': [0.4, 0.6, 0.4, 0.6], 'C2': [0.4, 0.6, 0.4, 0.6],
            'D1': [4, 0, 0, 0], 'Z1': [0, 0, 0, 0], 'U1': [0.1, 0.1, 0.1, 0.1],
        }
        matrix = root / 'matrix.tsv'
        with matrix.open('w', encoding='utf-8', newline='') as handle:
            writer = csv.writer(handle, delimiter='\t')
            writer.writerow(['Genome'] + [f'S{s}|{metric}' for s in range(4)
                                           for metric in ('relative_abundance', 'count')])
            names = list(profiles)
            if reversed_input:
                names.reverse()
            for genome in names:
                # B's diagnostic correction is large enough to reverse the
                # A/B corrected-SI ordering, distinguishing the requested raw
                # CV ranking from an accidental corrected-SI implementation.
                count = 5 if genome.startswith('B') else 1 if genome.startswith('U') else 100
                writer.writerow([genome] + [value for s in range(4)
                                             for value in (profiles[genome][s], count)])
        taxonomy = root / 'gtdbtk.bac120.summary.tsv'
        taxonomy.write_text('user_genome\tclassification\n' + ''.join(
            f'{genome}\td__Bacteria;p__P;c__C;o__O;f__{genome[0]};g__G;s__S\n'
            for genome in profiles
        ), encoding='utf-8')
        output = root / 'niche.tsv'
        provenance = root / 'provenance.json'
        assignments = root / 'assignments.tsv'
        classify_niche_literature(matrix, root, output, root / 'aggregated.tsv', assignments,
                                  provenance, [f'S{i}=default' for i in range(4)],
                                  'family', method, .01, 20, .2, .8, 1,
                                  completion_marker=marker, cv_correct=cv_correct)
        with output.open(encoding='utf-8', newline='') as handle:
            rows = {row['taxon']: row for row in csv.DictReader(handle, delimiter='\t')}
        with assignments.open(encoding='utf-8', newline='') as handle:
            genome_labels = {row['Genome']: row for row in csv.DictReader(handle, delimiter='\t')}
        return rows, json.loads(provenance.read_text(encoding='utf-8')), genome_labels

    def test_full_report_uses_raw_cv_and_propagates_middle_labels_to_mags(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / 'niche' / 'niche_classification.v3.raw_cv.complete'
            rows, provenance, genomes = self.run_analysis(root, marker=marker)
            self.assertEqual({name: row['niche'] for name, row in rows.items()}, {
                'f__A': 'generalist', 'f__B': 'intermediate', 'f__C': 'intermediate',
                'f__D': 'specialist', 'f__U': 'unclassified_low_support',
                'f__Z': 'unclassified_low_support',
            })
            self.assertEqual(rows['f__A']['cv_quartile_rank'], '1')
            self.assertEqual(rows['f__D']['cv_quartile_rank'], '4')
            self.assertLess(float(rows['f__B']['corrected_si']),
                            float(rows['f__A']['corrected_si']))
            self.assertEqual(rows['f__U']['cv_quartile_rank'], 'NA')
            self.assertEqual(rows['f__Z']['eligible'], 'false')
            self.assertEqual(genomes['B1']['Niche'], 'intermediate')
            self.assertEqual(genomes['C1']['Niche'], genomes['C2']['Niche'])
            self.assertEqual(genomes['D1']['Niche'], 'specialist')
            self.assertEqual(provenance['method'], 'metabaw_raw_cv_rank_quartiles_v3')
            self.assertTrue(all(row['correct_cv_classification'] == 'FALSE' for row in rows.values()))
            self.assertTrue(all(row['correct_cv_classification'] == 'FALSE' for row in genomes.values()))
            self.assertFalse(provenance['correct_cv_classification'])
            for row in rows.values():
                self.assertNotIn('core_datasets', row)
                self.assertNotIn('required_core_datasets', row)
                self.assertFalse(any(field.startswith('prevalence::') for field in row))
            quota = provenance['parameters']['cv_rank_quartiles']
            self.assertEqual(quota['eligible_taxa'], 4)
            self.assertEqual(quota['tail_size'], 1)
            self.assertIn('intermediate', provenance['classification_rules'])
            self.assertEqual(marker.read_text(encoding='utf-8'), 'complete\n')
            repeated, _, _ = self.run_analysis(root, reversed_input=True)
            self.assertEqual(repeated, rows)

    def test_raw_and_corrected_rankings_differ_but_correction_does_not_drive_labels(self):
        # Same raw CV values, deliberately reversed diagnostic values.
        rows = [{'taxon': f'T{i}', 'raw_si': i / 10, 'corrected_si': 10 - i,
                 'eligible': True} for i in range(4)]
        labels, _, _ = _cv_rank_quartiles(rows)
        self.assertEqual(labels['T0'], 'generalist')
        self.assertEqual(labels['T3'], 'specialist')

    def test_corrected_mode_drives_labels_and_propagates_flag_to_all_rows(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / 'niche' / 'niche_classification.v3.corrected_cv.complete'
            rows, provenance, genomes = self.run_analysis(root, marker=marker, cv_correct=True)
            self.assertEqual(rows['f__B']['niche'], 'generalist')
            self.assertEqual(rows['f__A']['niche'], 'intermediate')
            self.assertEqual(rows['f__D']['niche'], 'specialist')
            self.assertEqual(rows['f__B']['cv_quartile_rank'], '1')
            self.assertLess(float(rows['f__B']['corrected_si']), 0)
            self.assertAlmostEqual(float(rows['f__B']['cv_low_tail_max']),
                                   float(rows['f__B']['corrected_si']))
            self.assertAlmostEqual(float(rows['f__B']['cv_high_tail_min']),
                                   float(rows['f__D']['corrected_si']))
            self.assertTrue(all(row['correct_cv_classification'] == 'TRUE' for row in rows.values()))
            self.assertTrue(all(row['correct_cv_classification'] == 'TRUE' for row in genomes.values()))
            self.assertEqual(genomes['B1']['Niche'], 'generalist')
            self.assertTrue(provenance['correct_cv_classification'])
            self.assertTrue(provenance['parameters']['cv_correct'])
            self.assertEqual(provenance['method'], 'metabaw_corrected_cv_rank_quartiles_v3')
            self.assertEqual(provenance['parameters']['cv_rank_quartiles']['ranking_value'], 'corrected_si')
            self.assertIn('corrected_si', provenance['classification_rules']['generalist'])
            self.assertTrue(rows['f__B']['reason'].startswith('corrected_cv_'))
            self.assertEqual(marker.read_text(encoding='utf-8'), 'complete\n')
            repeated, _, _ = self.run_analysis(root, reversed_input=True, cv_correct=True)
            self.assertEqual(repeated, rows)

    def test_corrected_quartiles_preserve_negative_values_ties_and_remainders(self):
        for size in (0, 1, 3, 4, 5, 10, 101):
            with self.subTest(size=size):
                rows = [{'taxon': f'T{i:03d}', 'raw_si': size - i,
                         'corrected_si': -1.0, 'eligible': True}
                        for i in reversed(range(size))]
                labels, ranks, metadata = _cv_rank_quartiles(rows, cv_correct=True)
                counts = Counter(labels.values())
                self.assertEqual(counts['generalist'], size // 4)
                self.assertEqual(counts['specialist'], size // 4)
                self.assertEqual(counts['intermediate'], size - 2 * (size // 4))
                self.assertEqual(list(ranks), [f'T{i:03d}' for i in range(size)])
                self.assertEqual(metadata['ranking_value'], 'corrected_si')
                self.assertEqual(metadata['low_tail_max_cv'], -1.0 if size >= 4 else None)

    def test_corrected_mode_excludes_nonfinite_selected_values_before_ranking(self):
        rows = [{'taxon': f'T{i}', 'raw_si': i, 'corrected_si': i, 'eligible': True}
                for i in range(4)]
        rows += [{'taxon': 'invalid', 'raw_si': 0, 'corrected_si': math.nan, 'eligible': True},
                 {'taxon': 'infinite', 'raw_si': 0, 'corrected_si': math.inf, 'eligible': True},
                 {'taxon': 'unsupported', 'raw_si': 0, 'corrected_si': -100, 'eligible': False}]
        labels, _, metadata = _cv_rank_quartiles(rows, cv_correct=True)
        self.assertEqual(set(labels), {'T0', 'T1', 'T2', 'T3'})
        self.assertEqual(metadata['eligible_taxa'], 4)

    def test_cv_correction_rejects_occupancy_mode(self):
        with TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, '--cv-correct.*cv'):
                self.run_analysis(Path(directory), method='occupancy', cv_correct=True)

    def test_occupancy_mode_keeps_its_prevalence_classification(self):
        with TemporaryDirectory() as directory:
            rows, provenance, _ = self.run_analysis(Path(directory), method='occupancy')
            self.assertEqual(rows['f__A']['niche'], 'generalist')
            self.assertEqual(rows['f__B']['niche'], 'generalist')
            self.assertEqual(rows['f__C']['niche'], 'generalist')
            self.assertEqual(rows['f__D']['niche'], 'specialist')
            self.assertNotIn('intermediate', provenance['classification_rules'])
            self.assertIsNone(provenance['parameters']['cv_rank_quartiles'])
            self.assertTrue(all(row['cv_quartile_rank'] == 'NA' for row in rows.values()))
            self.assertTrue(all(row['correct_cv_classification'] == 'FALSE' for row in rows.values()))
            for row in rows.values():
                self.assertNotIn('core_datasets', row)
                self.assertNotIn('required_core_datasets', row)
                self.assertFalse(any(field.startswith('prevalence::') for field in row))

    def test_failed_classification_does_not_leave_a_stale_completion_marker(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / 'niche' / 'niche_classification.v3.raw_cv.complete'
            self.run_analysis(root, marker=marker)
            self.assertTrue(marker.exists())
            with patch('metabaw.internal._cv_rank_quartiles', side_effect=ValueError('test failure')):
                with self.assertRaisesRegex(ValueError, 'test failure'):
                    self.run_analysis(root, marker=marker)
            self.assertFalse(marker.exists())

    def test_new_marker_prevents_old_binary_output_fingerprints_from_matching(self):
        root = Path('/example')
        outputs = tuple(root / name for name in ('niche.tsv', 'taxon.tsv', 'assignments.tsv', 'provenance.json'))
        previous = Task('niche', '04_niche', ('python', '-m', 'metabaw.internal',
                        'classify-niche-literature', '--method', 'cv'), root, outputs=outputs)
        marker = root / 'niche_classification.v3.raw_cv.complete'
        current = replace(previous, command=(*previous.command, '--completion-marker', str(marker)),
                          outputs=(*outputs, marker))
        self.assertNotEqual(current.fingerprint(['merge']), previous.fingerprint(['merge']))
