"""Run LorBin with stable distance kernels and guarded clustering boundaries.

This file is executed by the isolated LorBin Python environment.  It must not
import MetaBAW's main-environment dependencies.
"""

from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def _clustering_runtime(logger):
    """Scope distance-backend and native-pool settings to clustering only."""
    from sklearn import config_context
    from threadpoolctl import threadpool_limits

    # In sklearn 1.1.x, BIRCH's closest-centroid assignment and brute-force
    # neighbors can use the same Cython/OpenMP distance-reduction backend.
    # An invalid centroid index cannot be repaired by clipping or dropping
    # contigs. Select sklearn's exact, chunked fallback before any distances
    # are computed, and cap the temporary distance chunks at 64 MiB.
    # Runtime pool control also covers libraries that reset OpenMP's thread
    # count after process startup. Neither setting touches CUDA training.
    with config_context(enable_cython_pairwise_dist=False, working_memory=64):
        with threadpool_limits(limits=1):
            logger.info(
                "[LORBIN COMPAT] Clustering uses chunked pairwise distances "
                "(working_memory=64 MiB) and single-thread native pools; "
                "CUDA training and --num_process are unchanged"
            )
            yield


def _threshold_base(values, distances, logger, stage):
    """Keep upstream candidates; handle an empty small-sample candidate list."""
    import math
    import numpy as np

    base = list(values)
    if any(not math.isfinite(value) or value < 0 for value in base):
        raise ValueError(f"LorBin {stage} has invalid distance thresholds")
    if not base:
        distances = np.asarray(distances)
        if not distances.size or not np.isfinite(distances).all() or (distances < 0).any():
            raise ValueError(f"LorBin {stage} has no valid neighbor distances")
        # With few contigs the upstream sampling range is empty. Use the
        # smallest observed distance at the same integer scale, not a new
        # distance metric or an arbitrary synthetic second candidate.
        base = [math.floor(float(distances.min()))]
        logger.warning(
            "[LORBIN COMPAT] %s: no sampled thresholds; using the smallest "
            "observed distance at the original integer scale (%s)", stage, base[0],
        )
    return base


def _dbscan_thresholds(values, distances, logger):
    base = _threshold_base(values, distances, logger, "DBSCAN")
    first = base[0]
    if len(base) > 1:
        second = base[1]
    else:
        second = first
        logger.warning(
            "[LORBIN COMPAT] DBSCAN: only one distinct threshold (%s); "
            "reuse it for the second offset, preserving the original +0.5 rule", first,
        )
    return base + [0.3 * first, 0.5 * first, first + 0.5, second + 0.5]


def _birch_thresholds(values, distances, logger):
    base = _threshold_base(values, distances, logger, "BIRCH")
    first = base[0]
    return base + [0.1 * first, 0.3 * first, 0.5 * first, 10, 15, 30]


def _validate_clustering_inputs(latent, contig_all, contig_list, contig_dict):
    import numpy as np

    if (
        np.ndim(latent) != 2
        or latent.shape[1] == 0
        or len(latent) != len(contig_all)
        or len(latent) != len(contig_list)
        or list(contig_all) != list(contig_list)
    ):
        raise ValueError("LorBin embedding shape/order does not match its contig IDs")
    if len(latent) < 2:
        raise ValueError("LorBin requires at least two contigs for clustering")
    if len(set(contig_all)) != len(contig_all):
        raise ValueError("LorBin embedding contains duplicate contig IDs")
    if not np.isfinite(latent).all():
        raise ValueError("LorBin embedding contains NaN or infinite values; retained for diagnosis")
    if any(name not in contig_dict for name in contig_all):
        raise ValueError("LorBin embedding contains contig IDs absent from the input FASTA")


def _marker_mapping(markers, logger):
    # LorBin returns [] rather than a mapping when no markers were found.
    # Keep those contigs marker-free rather than inventing marker support.
    from collections import defaultdict

    if not markers:
        logger.warning("[LORBIN COMPAT] No marker genes found; no marker-supported bins will be invented")
        return defaultdict(list)
    return markers


def _compile_bin_cluster(source, namespace, filename="<metabaw-lorbin-cluster>", *, first_line=1):
    """Patch only known upstream boundary expressions, without editing LorBin."""
    import ast
    import textwrap

    tree = ast.parse(textwrap.dedent(source))
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise RuntimeError("Unsupported LorBin clustering source; expected one bin_cluster function")
    function = tree.body[0]
    expected_args = {
        "logger", "latent", "contig2marker", "contig_dict", "contig_list", "contig_all", "minfasta",
    }
    if function.name != "bin_cluster" or not expected_args.issubset(
        {argument.arg for argument in function.args.args}
    ):
        raise RuntimeError("Unsupported LorBin bin_cluster interface; use MetaBAW's pinned LorBin release")

    replacements = {"DBSCAN": 0, "BIRCH": 0, "recluster": 0}
    body = []
    for statement in function.body:
        target = (
            statement.targets[0].id
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name) else None
        )
        value = getattr(statement, "value", None)
        if target in ("eps_p2", "threds") and isinstance(value, ast.BinOp) and isinstance(value.op, ast.Add):
            base_name, helper, distance_name, stage = (
                ("eps_p2", "_metabaw_dbscan_thresholds", "p2_distance_data", "DBSCAN")
                if target == "eps_p2" else
                ("eps_p2_2", "_metabaw_birch_thresholds", "p2_distance", "BIRCH")
            )
            # Require the known first/second-index expansion, not merely a
            # coincidentally matching variable name in another LorBin version.
            indices = [
                item.slice.value for item in ast.walk(value)
                if isinstance(item, ast.Subscript) and isinstance(item.value, ast.Name)
                and item.value.id == base_name and isinstance(item.slice, ast.Constant)
            ]
            required_index = 1 if stage == "DBSCAN" else 0
            if (
                isinstance(value.left, ast.Name) and value.left.id == base_name
                and required_index in indices
            ):
                statement.value = ast.Call(
                    func=ast.Name(id=helper, ctx=ast.Load()),
                    args=[ast.Name(id=name, ctx=ast.Load()) for name in (base_name, distance_name, "logger")],
                    keywords=[],
                )
                replacements[stage] += 1
        if (
            target == "recluster_latent" and isinstance(value, ast.Subscript)
            and isinstance(value.value, ast.Name) and value.value.id == "latent"
            and isinstance(value.slice, ast.Name) and value.slice.id == "recluster_index"
        ):
            guard = ast.parse(
                "if len(recluster_index) < 2:\n"
                "    logger.info('[LORBIN COMPAT] Fewer than two unassigned contigs; "
                "skip reclustering and preserve primary assignments')\n"
                "    return contig_labels, keep_label\n"
            ).body[0]
            body.append(ast.copy_location(guard, statement))
            replacements["recluster"] += 1
        body.append(statement)
    if replacements != {"DBSCAN": 1, "BIRCH": 1, "recluster": 1}:
        raise RuntimeError(
            "Unsupported LorBin clustering implementation; cannot safely install boundary guards. "
            "Use MetaBAW's pinned LorBin release."
        )
    preamble = ast.parse(
        "_metabaw_validate_inputs(latent, contig_all, contig_list, contig_dict)\n"
        "contig2marker = _metabaw_marker_mapping(contig2marker, logger)\n"
    ).body
    runtime_guard = ast.parse("with _metabaw_clustering_runtime(logger):\n    pass\n").body[0]
    runtime_guard.body = preamble + body
    function.body = [ast.copy_location(runtime_guard, body[0])]
    ast.fix_missing_locations(tree)
    ast.increment_lineno(tree, first_line - 1)
    patched_namespace = dict(namespace)
    patched_namespace.update({
        "kneighbors_graph": _stable_kneighbors_graph,
        "_metabaw_dbscan_thresholds": _dbscan_thresholds,
        "_metabaw_birch_thresholds": _birch_thresholds,
        "_metabaw_validate_inputs": _validate_clustering_inputs,
        "_metabaw_marker_mapping": _marker_mapping,
        "_metabaw_clustering_runtime": _clustering_runtime,
    })
    exec(compile(tree, filename, "exec"), patched_namespace)
    patched = patched_namespace["bin_cluster"]
    patched._metabaw_cluster_compat = True
    return patched


def _install_cluster_compat(cluster):
    import inspect

    original = cluster.bin_cluster
    if getattr(original, "_metabaw_cluster_compat", False):
        return
    try:
        source = inspect.getsource(original)
    except (OSError, TypeError) as exc:
        raise RuntimeError("Cannot inspect LorBin clustering source; use MetaBAW's pinned LorBin release") from exc
    cluster.bin_cluster = _compile_bin_cluster(
        source, original.__globals__, original.__code__.co_filename,
        first_line=original.__code__.co_firstlineno,
    )


def _exclude_self(distances, indices, n_neighbors, *, fallback=None):
    """Select exactly ``n_neighbors`` non-self edges from every query row."""
    import numpy as np

    if distances.shape != indices.shape or distances.ndim != 2:
        raise ValueError("LorBin neighbor distances and indices have different shapes")
    n_rows = indices.shape[0]
    if indices.shape[1] != n_neighbors + 1:
        raise ValueError("LorBin neighbor query returned an unexpected width")

    selected_distances = np.empty((n_rows, n_neighbors), dtype=distances.dtype)
    selected_indices = np.empty((n_rows, n_neighbors), dtype=indices.dtype)
    for row in range(n_rows):
        keep = np.flatnonzero(indices[row] != row)
        if len(keep) < n_neighbors:
            if fallback is None:
                raise ValueError(f"LorBin row {row} has too few non-self neighbors")
            row_distances, row_indices = fallback(row)
            if len(row_distances) != n_neighbors or len(row_indices) != n_neighbors:
                raise ValueError(f"LorBin fallback for row {row} returned the wrong width")
            selected_distances[row] = row_distances
            selected_indices[row] = row_indices
            continue
        # Tied distances can exclude the query point from the first k+1 hits.
        # In that case discard the last hit, retaining k non-self neighbors.
        keep = keep[:n_neighbors]
        selected_distances[row] = distances[row, keep]
        selected_indices[row] = indices[row, keep]
    return selected_distances, selected_indices


def _stable_kneighbors_graph(
    X,
    n_neighbors,
    *,
    mode="connectivity",
    metric="minkowski",
    p=2,
    metric_params=None,
    include_self=False,
    n_jobs=None,
):
    from sklearn.neighbors import NearestNeighbors, kneighbors_graph

    # LorBin's first graph and reclustering graph use the default Minkowski
    # metric with p=2 (Euclidean); only its second graph uses cosine distance.
    # Cover both without changing the requested metric or neighbor count.
    if (
        metric not in ("cosine", "minkowski", "euclidean")
        or mode != "distance"
        or include_self
    ):
        return kneighbors_graph(
            X,
            n_neighbors,
            mode=mode,
            metric=metric,
            p=p,
            metric_params=metric_params,
            include_self=include_self,
            n_jobs=n_jobs,
        )

    import numpy as np
    from scipy.sparse import csr_matrix
    from sklearn.metrics import pairwise_distances

    n_rows = len(X)
    if not 0 < n_neighbors < n_rows:
        raise ValueError("LorBin neighbor graph requires 1 <= neighbors < contigs")

    # Passing X explicitly bypasses sklearn 1.1.x's X=None self-removal path.
    # That path can raise a reshape ValueError even with finite, unique rows.
    model = NearestNeighbors(
        n_neighbors=n_neighbors + 1,
        metric=metric,
        p=p,
        metric_params=metric_params,
        n_jobs=n_jobs,
    ).fit(X)
    distances, indices = model.kneighbors(
        X, n_neighbors=n_neighbors + 1, return_distance=True
    )

    def recompute_row(row):
        # Only needed if a backend returns the query index more than once.
        distance_params = dict(metric_params or {})
        if metric == "minkowski":
            # As in NearestNeighbors, metric_params['p'] overrides the p option.
            distance_params.setdefault("p", p)
        values = pairwise_distances(
            X[row : row + 1], X, metric=metric, **distance_params
        )[0]
        values[row] = np.inf
        if np.isfinite(values).sum() < n_neighbors:
            raise ValueError(f"LorBin row {row} has too few finite neighbor distances")
        chosen = np.argpartition(values, n_neighbors - 1)[:n_neighbors]
        chosen = chosen[np.argsort(values[chosen], kind="stable")]
        return values[chosen], chosen

    selected_distances, selected_indices = _exclude_self(
        distances, indices, n_neighbors, fallback=recompute_row
    )
    indptr = np.arange(n_rows + 1, dtype=np.intp) * n_neighbors
    return csr_matrix(
        (selected_distances.ravel(), selected_indices.ravel(), indptr),
        shape=(n_rows, n_rows),
    )


def main() -> None:
    import lorbin.cluster as cluster

    cluster.kneighbors_graph = _stable_kneighbors_graph
    _install_cluster_compat(cluster)
    import lorbin.lorbin as entry

    # Also cover a previously imported entry module's from-import binding.
    entry.bin_cluster = cluster.bin_cluster

    entry.main()


if __name__ == "__main__":
    main()
