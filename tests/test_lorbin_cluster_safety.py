from __future__ import annotations

import math
import sys
import types
import unittest
from contextlib import contextmanager, nullcontext
from unittest.mock import Mock, patch

import numpy as np

from metabaw.lorbin_compat import (
    _birch_thresholds, _compile_bin_cluster, _dbscan_thresholds,
    _clustering_runtime, _install_cluster_compat, _marker_mapping,
    _stable_kneighbors_graph, _validate_clustering_inputs, main as lorbin_main,
)


# A reduced offline fixture exercises the actual upstream boundary statement
# shapes. Neural bin selection is represented by supplied primary assignments;
# tests do not claim to validate LorBin's trained models or biological outputs.
CLUSTER_SOURCE = """
def bin_cluster(logger, latent, contig2marker, contig_dict, contig_list, contig_all,
                minfasta, feature='no_markers', a=0.6):
    min_k_1 = min(200, latent.shape[0] - 1)
    dist_matrix = kneighbors_graph(latent, min_k_1, mode='distance', p=2, n_jobs=10)
    dist_matrix_cos = kneighbors_graph(latent, min_k_1, mode='distance', metric='cosine', p=2, n_jobs=10)
    p2_distance_data = dist_matrix.data
    eps_p2 = []
    for k in range(1, min(80, min_k_1)):
        value = np.partition(p2_distance_data, len(latent) * k - 1)[len(latent) * k - 1]
        eps_p2.extend([math.floor(0.5 * value), math.floor(value), math.floor(1.2 * value)])
    eps_p2 = sorted(set(eps_p2))
    eps_p2 = eps_p2 + [0.3 * eps_p2[0], 0.5 * eps_p2[0], eps_p2[0] + 0.5, eps_p2[1] + 0.5]
    eps_p2.sort()
    observed['dbscan'] = eps_p2
    observed['marker_count'] = len(contig2marker[contig_all[0]])
    contig_labels = list(primary_labels)
    keep_label = list(primary_keep)
    recluster_index = [i for i, label in enumerate(contig_labels) if label == -1]
    recluster_latent = latent[recluster_index]
    min_k_2 = min(200, len(recluster_latent) - 1)
    dist_matrix = kneighbors_graph(recluster_latent, min_k_2, mode='distance', p=2, n_jobs=10)
    p2_distance = dist_matrix.data
    eps_p2_2 = []
    for k in range(5, min(80, min_k_2), 10):
        value = np.partition(p2_distance, len(recluster_latent) * k - 1)[len(recluster_latent) * k - 1]
        eps_p2_2.extend([math.floor(0.8 * value), math.floor(value), math.floor(1.2 * value)])
    eps_p2_2 = sorted(set(eps_p2_2))
    threds = eps_p2_2 + [0.1 * eps_p2_2[0], 0.3 * eps_p2_2[0], 0.5 * eps_p2_2[0], 10, 15, 30]
    threds.sort()
    observed['birch'] = threds
    return contig_labels, keep_label
"""


def numeric_neighbor_modules():
    """Small deterministic NumPy backend; no optional sklearn/SciPy required."""
    calls = []

    class Neighbors:
        def __init__(self, *, metric, p, metric_params, **kwargs):
            self.metric, self.p = metric, p

        def fit(self, data):
            self.data = data
            return self

        def kneighbors(self, query, *, n_neighbors, return_distance):
            if query is None:
                raise AssertionError("unstable self-removal path used")
            calls.append((len(query), n_neighbors, self.metric))
            if self.metric == 'cosine':
                normed = self.data / np.linalg.norm(self.data, axis=1, keepdims=True)
                distances = np.maximum(0.0, 1.0 - normed @ normed.T)
            else:
                distances = np.linalg.norm(query[:, None, :] - self.data[None, :, :], axis=2)
            indices = np.argsort(distances, axis=1, kind='stable')[:, :n_neighbors]
            return np.take_along_axis(distances, indices, axis=1), indices

    sklearn = types.ModuleType('sklearn')
    sklearn.__path__ = []
    sklearn.config_context = Mock(side_effect=lambda **kwargs: nullcontext())
    neighbors = types.ModuleType('sklearn.neighbors')
    neighbors.NearestNeighbors = Neighbors
    neighbors.kneighbors_graph = Mock(side_effect=AssertionError('unstable graph used'))
    metrics = types.ModuleType('sklearn.metrics')
    metrics.pairwise_distances = Mock(side_effect=AssertionError('unexpected row fallback'))
    scipy = types.ModuleType('scipy')
    scipy.__path__ = []
    sparse = types.ModuleType('scipy.sparse')
    sparse.csr_matrix = lambda values, *, shape: types.SimpleNamespace(
        data=values[0], indices=values[1], indptr=values[2], shape=shape,
    )
    threadpoolctl = types.ModuleType('threadpoolctl')
    threadpoolctl.threadpool_limits = Mock(side_effect=lambda **kwargs: nullcontext())
    return {
        'sklearn': sklearn, 'sklearn.neighbors': neighbors, 'sklearn.metrics': metrics,
        'scipy': scipy, 'scipy.sparse': sparse, 'threadpoolctl': threadpoolctl,
    }, calls


def controlled_runtime_modules(modules=None, *, fast=True):
    """Model sklearn's scoped config and a pool reset by prior Torch training."""
    modules = dict(modules or {})
    sklearn = modules.setdefault('sklearn', types.ModuleType('sklearn'))
    sklearn.__path__ = []
    config = {'enable_cython_pairwise_dist': fast, 'working_memory': 1024}
    pools = {'threads': 384}

    @contextmanager
    def configuration(**kwargs):
        previous = dict(config)
        config.update(kwargs)
        try:
            yield
        finally:
            config.clear()
            config.update(previous)

    @contextmanager
    def limits(*, limits):
        previous = pools['threads']
        pools['threads'] = limits
        try:
            yield
        finally:
            pools['threads'] = previous

    sklearn.config_context = configuration
    threadpoolctl = types.ModuleType('threadpoolctl')
    threadpoolctl.threadpool_limits = limits
    modules['threadpoolctl'] = threadpoolctl
    return modules, config, pools


class LorBinClusterSafetyTests(unittest.TestCase):
    def test_normal_threshold_lists_are_exactly_unchanged(self):
        logger = Mock()
        for base in ([0, 1, 2], [2, 3, 5], [10, 20]):
            with self.subTest(base=base):
                self.assertEqual(_dbscan_thresholds(base, None, logger), base + [
                    0.3 * base[0], 0.5 * base[0], base[0] + 0.5, base[1] + 0.5,
                ])
                self.assertEqual(_birch_thresholds(base, None, logger), base + [
                    0.1 * base[0], 0.3 * base[0], 0.5 * base[0], 10, 15, 30,
                ])
        logger.warning.assert_not_called()

    def test_one_threshold_reuses_original_offsets_without_retuning_distances(self):
        for value in (0, 2, 10):
            with self.subTest(value=value):
                logger = Mock()
                actual = _dbscan_thresholds([value], np.array([0.1]), logger)
                self.assertEqual(actual, [value, 0.3 * value, 0.5 * value, value + 0.5, value + 0.5])
                logger.warning.assert_called_once()

    def test_empty_sampling_range_uses_observed_distance_and_native_scale(self):
        for distance, base in ((0.2, 0), (4.8, 4), (0.0, 0)):
            with self.subTest(distance=distance):
                logger = Mock()
                self.assertEqual(_dbscan_thresholds([], np.array([distance, distance + 1]), logger), [
                    base, 0.3 * base, 0.5 * base, base + 0.5, base + 0.5,
                ])
                self.assertEqual(_birch_thresholds([], np.array([distance, distance + 1]), logger), [
                    base, 0.1 * base, 0.3 * base, 0.5 * base, 10, 15, 30,
                ])

    def test_invalid_thresholds_and_distances_are_not_silently_replaced(self):
        for base, distances in (
            ([float('nan')], [0.1]), ([-1], [0.1]), ([], []),
            ([], [float('inf')]), ([], [float('nan')]), ([], [-0.1]),
        ):
            with self.subTest(base=base, distances=distances):
                with self.assertRaisesRegex(ValueError, 'invalid|no valid'):
                    _dbscan_thresholds(base, np.array(distances), Mock())

    def run_fixture(self, count, primary_labels, scale=0.0001, markers=None):
        data = np.arange(1, count + 1)[:, None] * np.arange(1, 33)[None, :] * scale
        names = [f'contig_{i}' for i in range(count)]
        marker_map = {name: ['marker'] for name in names} if markers is None else markers
        namespace = {
            'np': np, 'math': math, 'primary_labels': primary_labels,
            'primary_keep': [True], 'observed': {}, '__file__': '/isolated/lorbin/cluster.py',
        }
        function = _compile_bin_cluster(CLUSTER_SOURCE, namespace)
        modules, calls = numeric_neighbor_modules()
        logger = Mock()
        with patch.dict(sys.modules, modules):
            result = function(logger, data, marker_map, {name: 'A' * 2000 for name in names},
                              names, np.array(names), 1500)
        return result, namespace['observed'], logger, calls, function

    def test_collapsed_thresholds_and_no_recluster_contigs_preserve_primary_bins(self):
        result, observed, logger, calls, _ = self.run_fixture(10, [0] * 10)
        self.assertEqual(result, ([0] * 10, [True]))
        self.assertEqual(observed['dbscan'], [0, 0, 0, 0.5, 0.5])
        self.assertNotIn('birch', observed)
        self.assertEqual(len(calls), 2)
        logger.warning.assert_called_once()

    def test_unmodified_boundary_fixture_reproduces_the_reported_index_error(self):
        data = np.arange(1, 11)[:, None] * np.arange(1, 33)[None, :] * 0.0001
        names = [f'contig_{i}' for i in range(10)]
        namespace = {
            'np': np, 'math': math, 'kneighbors_graph': _stable_kneighbors_graph,
            'observed': {}, 'primary_labels': [0] * 10, 'primary_keep': [True],
        }
        exec(CLUSTER_SOURCE, namespace)
        modules, _ = numeric_neighbor_modules()
        with patch.dict(sys.modules, modules), self.assertRaisesRegex(IndexError, 'list index'):
            namespace['bin_cluster'](
                Mock(), data, {name: ['marker'] for name in names},
                {name: 'A' * 2000 for name in names}, names, np.array(names), 1500,
            )

    def test_normal_clustering_fixture_matches_the_unmodified_threshold_strategy(self):
        count = 20
        labels = [-1] * count
        result, observed, logger, _, _ = self.run_fixture(count, labels, scale=10)
        self.assertEqual(result, (labels, [True]))
        logger.warning.assert_not_called()
        namespace = {
            'np': np, 'math': math, 'kneighbors_graph': _stable_kneighbors_graph,
            'observed': {}, 'primary_labels': labels, 'primary_keep': [True],
        }
        exec(CLUSTER_SOURCE, namespace)
        data = np.arange(1, count + 1)[:, None] * np.arange(1, 33)[None, :] * 10
        names = [f'contig_{i}' for i in range(count)]
        modules, _ = numeric_neighbor_modules()
        with patch.dict(sys.modules, modules):
            reference = namespace['bin_cluster'](
                Mock(), data, {name: ['marker'] for name in names},
                {name: 'A' * 2000 for name in names}, names, np.array(names), 1500,
            )
        self.assertEqual(reference, result)
        self.assertEqual(namespace['observed'], observed)

    def test_one_unassigned_contig_is_not_assigned_to_an_invented_bin(self):
        labels = [0] * 9 + [-1]
        result, observed, logger, calls, _ = self.run_fixture(10, labels)
        self.assertEqual(result, (labels, [True]))
        self.assertNotIn('birch', observed)
        self.assertEqual(len(calls), 2)
        logger.info.assert_any_call(
            '[LORBIN COMPAT] Fewer than two unassigned contigs; '
            'skip reclustering and preserve primary assignments'
        )

    def test_small_recluster_sets_have_safe_thresholds(self):
        for count in range(2, 8):
            with self.subTest(contigs=count):
                result, observed, logger, calls, _ = self.run_fixture(count, [-1] * count)
                self.assertEqual(result, ([-1] * count, [True]))
                self.assertTrue(any(value > 0 for value in observed['dbscan']))
                self.assertIn(10, observed['birch'])
                self.assertIn(15, observed['birch'])
                self.assertIn(30, observed['birch'])
                self.assertEqual(len(calls), 3)
                if count <= 6:
                    self.assertTrue(logger.warning.called)

    def test_preserves_model_directory_and_does_not_edit_upstream_namespace(self):
        _, _, _, _, function = self.run_fixture(10, [0] * 10)
        self.assertEqual(function.__globals__['__file__'], '/isolated/lorbin/cluster.py')
        self.assertTrue(function._metabaw_cluster_compat)

    def test_invalid_embedding_inputs_fail_clearly(self):
        valid = np.array([[1.0, 2.0], [2.0, 3.0]])
        for data, names, fasta, message in (
            (valid[:1], ['a'], {'a': 'A'}, 'at least two'),
            (valid, ['a'], {'a': 'A'}, 'shape/order'),
            (valid[:, :0], ['a', 'b'], {'a': 'A', 'b': 'C'}, 'shape/order'),
            (np.array([[np.nan, 1.0], [1.0, 2.0]]), ['a', 'b'], {'a': 'A', 'b': 'C'}, 'NaN'),
            (np.array([[np.inf, 1.0], [1.0, 2.0]]), ['a', 'b'], {'a': 'A', 'b': 'C'}, 'infinite'),
            (valid, ['a', 'a'], {'a': 'A'}, 'duplicate'),
            (valid, ['a', 'b'], {'a': 'A'}, 'absent'),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    _validate_clustering_inputs(data, names, names, fasta)

    def test_marker_free_contigs_do_not_gain_fake_support(self):
        logger = Mock()
        markers = _marker_mapping([], logger)
        self.assertEqual(markers['contig_without_markers'], [])
        result, observed, _, _, _ = self.run_fixture(10, [0] * 10, markers=[])
        self.assertEqual(observed['marker_count'], 0)
        logger.warning.assert_called_once()

    def test_unknown_upstream_implementation_is_rejected_before_running(self):
        for source in (
            'def different(): return None',
            CLUSTER_SOURCE.replace('eps_p2[1] + 0.5', '0.5'),
            CLUSTER_SOURCE.replace('recluster_latent = latent[recluster_index]', 'recluster_latent = latent'),
        ):
            with self.subTest(source=source[:50]):
                with self.assertRaisesRegex(RuntimeError, 'Unsupported LorBin'):
                    _compile_bin_cluster(source, {})

    def test_installation_is_idempotent_and_leaves_original_globals_untouched(self):
        namespace = {'__file__': '/isolated/lorbin/cluster.py'}
        exec(CLUSTER_SOURCE, namespace)
        original = namespace['bin_cluster']
        cluster = types.SimpleNamespace(bin_cluster=original)
        with patch('inspect.getsource', return_value=CLUSTER_SOURCE) as getsource:
            _install_cluster_compat(cluster)
            installed = cluster.bin_cluster
            _install_cluster_compat(cluster)
        self.assertIs(cluster.bin_cluster, installed)
        self.assertIsNot(cluster.bin_cluster, original)
        self.assertIs(namespace['bin_cluster'], original)
        self.assertNotIn('_metabaw_dbscan_thresholds', namespace)
        getsource.assert_called_once_with(original)

    def test_runtime_settings_are_scoped_and_restored_on_return_or_error(self):
        for fast in (False, True):
            for error in (False, True):
                with self.subTest(fast=fast, error=error):
                    modules, config, pools = controlled_runtime_modules(fast=fast)
                    previous = dict(config)
                    logger = Mock()
                    with patch.dict(sys.modules, modules):
                        try:
                            with _clustering_runtime(logger):
                                self.assertEqual(config, {
                                    'enable_cython_pairwise_dist': False, 'working_memory': 64,
                                })
                                self.assertEqual(pools['threads'], 1)
                                if error:
                                    raise ValueError('unrelated model failure')
                        except ValueError as exc:
                            self.assertEqual(str(exc), 'unrelated model failure')
                    self.assertEqual(config, previous)
                    self.assertEqual(pools['threads'], 384)
                    logger.info.assert_called_once()

    def test_full_launcher_scopes_all_graphs_and_birch_but_not_training(self):
        modules, graph_calls = numeric_neighbor_modules()
        modules, config, pools = controlled_runtime_modules(modules)
        data = np.arange(1, 11)[:, None] * np.arange(1, 33)[None, :] * 0.0001
        unchanged_data = data.copy()
        names = [f'contig_{i}' for i in range(len(data))]
        observed = {}
        checked_graphs = []
        original_neighbors = modules['sklearn.neighbors'].NearestNeighbors

        class CheckedNeighbors(original_neighbors):
            def fit(self, values):
                self.assert_stable_settings()
                checked_graphs.append(len(values))
                return super().fit(values)

            @staticmethod
            def assert_stable_settings():
                if config['enable_cython_pairwise_dist'] or pools['threads'] != 1:
                    raise AssertionError('neighbor graph entered unstable distance runtime')

        modules['sklearn.neighbors'].NearestNeighbors = CheckedNeighbors

        class SimulatedBirch:
            # This is fault injection, not a reproduction of the Linux kernel.
            def __init__(self, **kwargs):
                observed['birch_parameters'] = kwargs

            def fit_predict(self, values):
                centers = values[::2]
                labels = np.arange(len(centers))
                if config['enable_cython_pairwise_dist']:
                    indices = np.full(len(values), 94100130012672, dtype=np.int64)
                else:
                    self.assert_stable_settings()
                    indices = np.linalg.norm(values[:, None] - centers[None, :], axis=2).argmin(axis=1)
                return labels[indices]

            assert_stable_settings = staticmethod(CheckedNeighbors.assert_stable_settings)

        # First show that the injected failure has the same kind of absurd
        # closest-centroid index as the report; no clipping can make it valid.
        with self.assertRaisesRegex(IndexError, '94100130012672'):
            SimulatedBirch(threshold=0.5, n_clusters=None).fit_predict(data)
        observed.clear()
        source = CLUSTER_SOURCE.replace(
            '    return contig_labels, keep_label',
            "    observed['predictions'] = Birch(threshold=0.5, n_clusters=None).fit_predict(recluster_latent)\n"
            '    return contig_labels, keep_label',
        )
        namespace = {
            'np': np, 'math': math, 'Birch': SimulatedBirch, 'observed': observed,
            'primary_labels': [-1] * len(data), 'primary_keep': [True],
        }
        exec(source, namespace)
        original = namespace['bin_cluster']
        package = types.ModuleType('lorbin')
        package.__path__ = []
        cluster = types.ModuleType('lorbin.cluster')
        cluster.bin_cluster = original
        package.cluster = cluster
        entry = types.ModuleType('lorbin.lorbin')
        entry.bin_cluster = original
        training_settings = []

        def run():
            # Training sees its original settings, including CUDA selection;
            # only the subsequent bin_cluster call enters the compatibility scope.
            training_settings.append((dict(config), pools['threads']))
            return entry.bin_cluster(Mock(), data, {name: ['marker'] for name in names},
                                     {name: 'A' * 2000 for name in names}, names, np.array(names), 1500)

        entry.main = Mock(side_effect=run)
        modules.update({'lorbin': package, 'lorbin.cluster': cluster, 'lorbin.lorbin': entry})
        with patch.dict(sys.modules, modules), patch('inspect.getsource', return_value=source):
            lorbin_main()
        self.assertEqual(training_settings, [({
            'enable_cython_pairwise_dist': True, 'working_memory': 1024,
        }, 384)])
        self.assertEqual(checked_graphs, [10, 10, 10])
        self.assertEqual(len(graph_calls), 3)
        self.assertEqual(observed['birch_parameters'], {'threshold': 0.5, 'n_clusters': None})
        centers = data[::2]
        expected = np.linalg.norm(data[:, None] - centers[None, :], axis=2).argmin(axis=1)
        np.testing.assert_array_equal(observed['predictions'], expected)
        np.testing.assert_array_equal(data, unchanged_data)
        self.assertEqual(config, training_settings[0][0])
        self.assertEqual(pools['threads'], 384)
        self.assertIs(namespace['bin_cluster'], original)


if __name__ == '__main__':
    unittest.main()
