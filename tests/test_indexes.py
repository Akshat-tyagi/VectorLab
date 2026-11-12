import numpy as np
import pytest

from vectorlab.indexes import BruteForce, HNSW, KDTree


@pytest.mark.parametrize("metric", ["cosine", "euclidean", "manhattan"])
@pytest.mark.parametrize("dim", [1, 3, 16])
def test_kdtree_matches_oracle(metric, dim):
    rng = np.random.default_rng(13)
    matrix = rng.normal(size=(80, dim))
    matrix[0] = 0
    matrix[1] = matrix[2]
    exact, tree = BruteForce(dim, metric), KDTree(dim, metric)
    for id, row in enumerate(matrix):
        exact.insert(id, row)
        tree.insert(id, row)
    for q in [*rng.normal(size=(8, dim)), matrix[1], np.zeros(dim)]:
        for k in [1, 7, 100]:
            expected, actual = exact.search(q, k), tree.search(q, k)
            # Cosine ties can differ by a floating point rounding unit after normalization.
            np.testing.assert_allclose([h[1] for h in actual], [h[1] for h in expected], atol=1e-12)
            if metric != "cosine":
                assert [h[0] for h in actual] == [h[0] for h in expected]
            else:
                for id, d in actual:
                    assert abs(dict(exact.search(q, 100))[id] - d) < 1e-12
    for id in [0, 20, 79]:
        assert tree.delete(id) and exact.delete(id)
    q = rng.normal(size=dim)
    np.testing.assert_allclose([d for _, d in tree.search(q, 100)],
                               [d for _, d in exact.search(q, 100)], atol=1e-12)


@pytest.mark.parametrize("kind", [BruteForce, KDTree, HNSW])
@pytest.mark.parametrize("metric", ["cosine", "euclidean", "manhattan"])
def test_edge_cases_and_persistence(kind, metric, tmp_path):
    index = kind(3, metric)
    assert index.search([1, 0, 0]) == []
    assert not index.delete(5)
    for id in range(5):
        index.insert(id, [1, 0, 0])
    assert len(index.search([1, 0, 0], 20)) == 5
    assert index.delete(0)
    assert not index.delete(0)
    assert 0 not in dict(index.search([1, 0, 0], 20))
    path = tmp_path / "index.npz"
    index.save(path)
    restored = kind.load(path)
    assert restored.search([1, 0, 0], 20) == index.search([1, 0, 0], 20)
    restored.insert(99, [0, 1, 0])
    assert 99 in dict(restored.search([0, 1, 0], 20))
    for id in [1, 2, 3, 4, 99]:
        restored.delete(id)
    assert restored.search([1, 0, 0]) == []
    restored.insert(100, [0, 0, 1])
    assert restored.search([0, 0, 1])[0][0] == 100


@pytest.mark.parametrize("kind", [BruteForce, KDTree, HNSW])
def test_validation(kind):
    index = kind(2)
    for vector in [[1], [float("nan"), 0], [float("inf"), 0], [[1, 2]]]:
        with pytest.raises(ValueError):
            index.insert(1, vector)
        with pytest.raises(ValueError):
            index.search(vector)
    index.insert(1, [1, 2])
    with pytest.raises(ValueError):
        index.insert(1, [2, 3])
    for k in [0, -1, True]:
        with pytest.raises(ValueError):
            index.search([1, 2], k)


def test_hnsw_recall_and_graph_roundtrip(tmp_path):
    rng = np.random.default_rng(42)
    index, oracle = HNSW(12, M=8, ef_construction=80), BruteForce(12)
    for id, row in enumerate(rng.normal(size=(250, 12))):
        index.insert(id, row)
        oracle.insert(id, row)
    recalls = []
    for query in rng.normal(size=(20, 12)):
        recalls.append(len(set(dict(index.search(query, 10, ef=100))) &
                           set(dict(oracle.search(query, 10)))) / 10)
    assert np.mean(recalls) >= 0.95
    path = tmp_path / "graph.npz"
    index.delete(index.entry)
    index.save(path)
    restored = HNSW.load(path)
    assert restored.graph == index.graph
    assert restored.deleted == index.deleted
    extra = rng.normal(size=12)
    index.insert(900, extra)
    restored.insert(900, extra)
    assert restored.graph == index.graph
    query = rng.normal(size=12)
    assert restored.search(query, 10) == index.search(query, 10)
    restored.compact()
    assert not restored.deleted
    assert len(restored.vectors) == 250


def test_empty_snapshot(tmp_path):
    for kind in [BruteForce, KDTree, HNSW]:
        path = tmp_path / f"{kind.__name__}.npz"
        kind(2).save(path)
        assert kind.load(path).search([0, 0]) == []


def test_independent_metric_examples():
    from vectorlab.indexes import distances
    matrix = np.array([[0., 0.], [3., 4.], [-3., -4.]])
    query = np.array([3., 4.])
    np.testing.assert_allclose(distances(matrix, query, "euclidean"), [5, 0, 10])
    np.testing.assert_allclose(distances(matrix, query, "manhattan"), [7, 0, 14])
    np.testing.assert_allclose(distances(matrix, query, "cosine"), [1, 0, 2], atol=1e-15)


@pytest.mark.parametrize("metric", ["cosine", "euclidean", "manhattan"])
def test_kdtree_identical_vector_tie_order(metric):
    tree = KDTree(2, metric)
    for id in [9, 3, 7, 1]:
        tree.insert(id, [1, 0])
    assert tree.search([1, 0], 2) == [(1, 0.0), (3, 0.0)]


def test_cosine_pruning_regression():
    # Raw axis differences are invalid lower bounds for cosine distance.
    tree, oracle = KDTree(2), BruteForce(2)
    for id, row in enumerate([[100, 1], [1, 1], [0.1, 0.001], [-1, 5], [0, 0]]):
        tree.insert(id, row)
        oracle.insert(id, row)
    np.testing.assert_allclose([d for _, d in tree.search([10, 0.1], 3)],
                               [d for _, d in oracle.search([10, 0.1], 3)], atol=1e-12)
