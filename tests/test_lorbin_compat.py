from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from metabaw.direct import BinOptions, DirectBinBuilder
from metabaw.discovery import Analysis, ReadSample
from metabaw.lorbin_compat import _exclude_self, _stable_kneighbors_graph, main as lorbin_main
from metabaw.model import Task


def _graph_backend(distances, indices):
    model = Mock()
    model.fit.return_value = model
    model.kneighbors.return_value = (distances, indices)
    nearest = Mock(return_value=model)
    legacy = Mock(side_effect=AssertionError("unstable path used"))
    pairwise = Mock(side_effect=AssertionError("unexpected fallback"))
    csr = Mock(side_effect=lambda values, *, shape: types.SimpleNamespace(values=values, shape=shape))

    sklearn = types.ModuleType("sklearn")
    sklearn.__path__ = []
    neighbors = types.ModuleType("sklearn.neighbors")
    neighbors.NearestNeighbors = nearest
    neighbors.kneighbors_graph = legacy
    metrics = types.ModuleType("sklearn.metrics")
    metrics.pairwise_distances = pairwise
    scipy = types.ModuleType("scipy")
    scipy.__path__ = []
    sparse = types.ModuleType("scipy.sparse")
    sparse.csr_matrix = csr
    return types.SimpleNamespace(
        model=model, nearest=nearest, legacy=legacy, pairwise=pairwise, csr=csr,
        modules={
            "sklearn": sklearn,
            "sklearn.neighbors": neighbors,
            "sklearn.metrics": metrics,
            "scipy": scipy,
            "scipy.sparse": sparse,
        },
    )


class LorBinCompatTests(unittest.TestCase):
    def test_self_is_removed_row_by_row_even_when_not_first(self) -> None:
        distances = np.array([
            [0.0, 0.2, 0.3],
            [0.1, 0.0, 0.4],
            [0.1, 0.2, 0.0],
        ])
        indices = np.array([
            [0, 1, 2],
            [0, 1, 2],
            [0, 1, 2],
        ])
        selected_distances, selected_indices = _exclude_self(distances, indices, 2)
        np.testing.assert_array_equal(selected_indices, [[1, 2], [0, 2], [0, 1]])
        np.testing.assert_allclose(selected_distances, [[0.2, 0.3], [0.1, 0.4], [0.1, 0.2]])

    def test_missing_self_and_repeated_self_are_handled(self) -> None:
        distances = np.array([[0.1, 0.2, 0.3], [0.0, 0.0, 0.5]])
        indices = np.array([[1, 2, 3], [1, 1, 0]])
        calls = []

        def fallback(row):
            calls.append(row)
            return np.array([0.4, 0.6]), np.array([0, 2])

        selected_distances, selected_indices = _exclude_self(
            distances, indices, 2, fallback=fallback
        )
        self.assertEqual(calls, [1])
        np.testing.assert_array_equal(selected_indices, [[1, 2], [0, 2]])
        np.testing.assert_allclose(selected_distances, [[0.1, 0.2], [0.4, 0.6]])

    def test_all_lorbin_graphs_explicitly_query_data_and_have_no_self_edges(self) -> None:
        data = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        distances = np.array([[0.0, 0.2, 0.3], [0.2, 0.0, 0.4], [0.3, 0.4, 0.0]])
        indices = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
        # Match the actual LorBin calls: no metric argument for both Euclidean
        # graphs, explicit cosine for the other graph. Also check the L2 alias.
        for name, options in (
            ("initial Euclidean", {}),
            ("cosine", {"metric": "cosine"}),
            ("reclustering Euclidean", {}),
            ("explicit Euclidean", {"metric": "euclidean"}),
        ):
            with self.subTest(graph=name):
                backend = _graph_backend(distances, indices)
                with patch.dict(sys.modules, backend.modules):
                    graph = _stable_kneighbors_graph(
                        data, 2, mode="distance", p=2, n_jobs=10, **options
                    )
                backend.legacy.assert_not_called()
                backend.pairwise.assert_not_called()
                backend.nearest.assert_called_once_with(
                    n_neighbors=3, metric=options.get("metric", "minkowski"),
                    p=2, metric_params=None, n_jobs=10,
                )
                self.assertIs(backend.model.fit.call_args.args[0], data)
                self.assertIs(backend.model.kneighbors.call_args.args[0], data)
                self.assertEqual(backend.model.kneighbors.call_args.kwargs, {
                    "n_neighbors": 3, "return_distance": True,
                })
                self.assertEqual(graph.shape, (3, 3))
                np.testing.assert_allclose(graph.values[0], [0.2, 0.3, 0.2, 0.4, 0.3, 0.4])
                np.testing.assert_array_equal(graph.values[1], [1, 2, 0, 2, 0, 1])
                np.testing.assert_array_equal(graph.values[2], [0, 2, 4, 6])

    def test_repeated_self_fallback_preserves_the_requested_distance_parameters(self) -> None:
        data = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        distances = np.array([[0.0, 0.0, 1.0], [0.0, 2.0, 3.0], [0.0, 4.0, 5.0]])
        indices = np.array([[0, 0, 1], [1, 0, 2], [2, 0, 1]])
        weights = np.array([1.0, 2.0])
        for name, options, expected_params in (
            ("default Euclidean", {}, {"p": 2}),
            ("explicit Euclidean", {"metric": "euclidean"}, {}),
            ("cosine", {"metric": "cosine"}, {}),
            ("Minkowski p=1", {"p": 1}, {"p": 1}),
            ("metric parameter override", {"p": 1, "metric_params": {"p": 3, "w": weights}},
             {"p": 3, "w": weights}),
        ):
            with self.subTest(metric=name):
                backend = _graph_backend(distances, indices)
                backend.pairwise.side_effect = None
                backend.pairwise.return_value = np.array([[0.0, 4.0, 2.0]])
                with patch.dict(sys.modules, backend.modules):
                    graph = _stable_kneighbors_graph(data, 2, mode="distance", **options)
                backend.pairwise.assert_called_once()
                np.testing.assert_array_equal(backend.pairwise.call_args.args[0], data[:1])
                self.assertIs(backend.pairwise.call_args.args[1], data)
                self.assertEqual(backend.pairwise.call_args.kwargs, {
                    "metric": options.get("metric", "minkowski"), **expected_params,
                })
                if "metric_params" in options:
                    self.assertEqual(options["metric_params"]["p"], 3)
                    self.assertIs(options["metric_params"]["w"], weights)
                np.testing.assert_array_equal(graph.values[0], [2.0, 4.0, 2.0, 3.0, 4.0, 5.0])
                np.testing.assert_array_equal(graph.values[1], [2, 1, 0, 2, 0, 1])
                np.testing.assert_array_equal(graph.values[2], [0, 2, 4, 6])
                backend.legacy.assert_not_called()

    def test_other_modes_and_metrics_keep_the_original_delegation(self) -> None:
        data = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        for options in (
            {"mode": "connectivity", "metric": "cosine"},
            {"mode": "distance", "metric": "cosine", "include_self": True},
            {"mode": "distance", "metric": "manhattan"},
        ):
            with self.subTest(options=options):
                backend = _graph_backend(None, None)
                backend.legacy.side_effect = None
                backend.legacy.return_value = sentinel = object()
                with patch.dict(sys.modules, backend.modules):
                    graph = _stable_kneighbors_graph(data, 2, **options)
                self.assertIs(graph, sentinel)
                backend.legacy.assert_called_once()
                self.assertIs(backend.legacy.call_args.args[0], data)
                self.assertEqual(backend.legacy.call_args.args[1], 2)
                self.assertEqual(backend.legacy.call_args.kwargs, {
                    "mode": "connectivity", "metric": "minkowski", "p": 2,
                    "metric_params": None, "include_self": False, "n_jobs": None,
                    **options,
                })
                backend.nearest.assert_not_called()

    def test_launcher_patches_initial_cosine_and_reclustering_graphs(self) -> None:
        data = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        distances = np.array([[0.0, 0.2, 0.3], [0.2, 0.0, 0.4], [0.3, 0.4, 0.0]])
        indices = np.array([[0, 1, 2], [0, 1, 2], [0, 1, 2]])
        backend = _graph_backend(distances, indices)
        package = types.ModuleType("lorbin")
        package.__path__ = []
        cluster = types.ModuleType("lorbin.cluster")
        cluster.kneighbors_graph = backend.legacy
        cluster.bin_cluster = Mock()
        package.cluster = cluster
        entry = types.ModuleType("lorbin.lorbin")

        def run_clustering():
            for options in ({}, {"metric": "cosine"}, {}):
                cluster.kneighbors_graph(data, 2, mode="distance", p=2, n_jobs=10, **options)

        entry.main = Mock(side_effect=run_clustering)
        with patch.dict(sys.modules, {
            **backend.modules, "lorbin": package, "lorbin.cluster": cluster, "lorbin.lorbin": entry,
        }), patch("metabaw.lorbin_compat._install_cluster_compat") as install:
            lorbin_main()
        install.assert_called_once_with(cluster)
        self.assertIs(cluster.kneighbors_graph, _stable_kneighbors_graph)
        self.assertIs(entry.bin_cluster, cluster.bin_cluster)
        entry.main.assert_called_once_with()
        backend.legacy.assert_not_called()
        self.assertEqual(backend.model.kneighbors.call_count, 3)
        self.assertEqual([call.kwargs["metric"] for call in backend.nearest.call_args_list], [
            "minkowski", "cosine", "minkowski",
        ])

    def test_lorbin_task_uses_compat_launcher_without_automatic_retry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            contigs = root / "HMI_1.fa"
            sample = ReadSample("HMI_1", root / "HMI_1.fastq.gz", contigs=contigs)
            analysis = Analysis("HMI_1", (sample,), (contigs,), False)
            options = BinOptions(
                outdir=root / "results",
                workdir=root / "results" / "tmp" / "work",
                threads=4,
                read_type="long",
                align_tool="minimap2",
                flye_read_type="--nano-raw",
                binners=("lorbin",),
                min_contig_length=1500,
                min_fasta_kbs=200,
                batch_size=1024,
                refiner="magscot",
                quality_control="checkm2",
                min_completeness=50.0,
                max_contamination=10.0,
                min_quality_score=None,
                run_gunc=False,
                run_trna=False,
                trna_pass=None,
                run_rrna=False,
                rrna_pass=False,
                dereplicator="galah",
                environment=None,
                tag_contigs=False,
                gpu=True,
                max_gpu_memory="100%",
                assembly_strategy="default",
                magscot_dir=root / "MAGScoT",
                checkm2_db=None,
                gunc_db=None,
                gtdbtk_data=None,
                extra_args={},
                gpu_binners=("lorbin",),
            )
            builder = DirectBinBuilder([], options, root)
            builder._binners(
                analysis,
                Task("prepare", "01", ("true",), root),
                contigs,
                Task("bamset", "02", ("true",), root),
                [root / "HMI_1.bam"],
                root / "bamset",
            )
            run = next(task for task in builder.tasks if task.id == "03.bin.lorbin.HMI_1")
            self.assertIn("lorbin_compat.py", run.display_command())
            self.assertIn(" bin -o ", run.display_command())
            self.assertIn("--cuda", run.display_command())
            self.assertEqual(run.automatic_retries, 0)
            self.assertNotIn("OMP_THREAD_LIMIT", run.env)
            self.assertEqual(run.env["OMP_NUM_THREADS"], "1")
            self.assertEqual(run.env["CUDA_VISIBLE_DEVICES"], "0")


if __name__ == "__main__":
    unittest.main()
