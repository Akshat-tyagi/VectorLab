"""Exact and approximate indexes; no external ANN implementation.

Cosine KD-tree search uses normalized vectors and Euclidean bounds:
||a-b||² = 2 * cosine_distance(a,b) for nonzero unit vectors.
Zero vectors are searched separately, since their cosine distance is 1.
"""
from __future__ import annotations

import heapq
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Literal, Self, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

Metric = Literal["euclidean", "manhattan", "cosine"]
Hit = tuple[int, float]
METRICS = {"euclidean", "manhattan", "cosine"}


def distances(matrix: NDArray, query: NDArray, metric: Metric) -> NDArray:
    """Vectorized float64 distances; zero-vector cosine distance is defined as 1."""
    if metric == "euclidean":
        return np.linalg.norm(matrix - query, axis=1)
    if metric == "manhattan":
        return np.abs(matrix - query).sum(axis=1)
    norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query)
    similarity = np.divide(matrix @ query, norms, out=np.zeros(len(matrix)), where=norms > 0)
    return np.clip(1 - similarity, 0, 2)


class BruteForce:
    """Mutable exact baseline; stable ordering by (distance, external ID)."""

    def __init__(self, dim: int, metric: Metric = "cosine", **_: object):
        if not isinstance(dim, int) or isinstance(dim, bool) or dim < 1 or metric not in METRICS:
            raise ValueError("positive dimension and a supported metric required")
        self.dim, self.metric = dim, metric
        self.vectors: dict[int, NDArray] = {}
        self._search_dirty = True

    def _vector(self, value: ArrayLike) -> NDArray:
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (self.dim,) or not np.isfinite(array).all():
            raise ValueError(f"expected {self.dim} finite numbers")
        return array.copy()

    def insert(self, id: int, vector: ArrayLike) -> None:
        if (not isinstance(id, int) or isinstance(id, bool) or not -(2**63) <= id < 2**63
                or id in self.vectors):
            raise ValueError("ID must be a unique signed 64-bit integer")
        self.vectors[id] = self._vector(vector)
        self._search_dirty = True

    def delete(self, id: int) -> bool:
        removed = self.vectors.pop(id, None) is not None
        self._search_dirty |= removed
        return removed

    def build(self) -> None:
        """Cache the contiguous reference matrix and norms outside query timing."""
        self._search_ids = np.array(list(self.vectors), dtype=np.int64)
        self._search_matrix = (np.stack(list(self.vectors.values())) if self.vectors
                               else np.empty((0, self.dim)))
        self._search_norms = np.linalg.norm(self._search_matrix, axis=1)
        self._search_dirty = False

    def search(self, query: ArrayLike, k: int = 5, **_: object) -> list[Hit]:
        q = self._vector(query)
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError("k must be a positive integer")
        if not self.vectors:
            return []
        if self._search_dirty:
            self.build()
        ids = self._search_ids
        if self.metric == "cosine":
            norms = self._search_norms * np.linalg.norm(q)
            similarity = np.divide(self._search_matrix @ q, norms, out=np.zeros(len(ids)), where=norms > 0)
            ds = np.clip(1 - similarity, 0, 2)
        else:
            ds = distances(self._search_matrix, q, self.metric)
        order = np.lexsort((ids, ds))[:k]
        return [(int(ids[i]), float(ds[i])) for i in order]

    def _state(self) -> dict:
        return {"version": 1, "kind": type(self).__name__, "dim": self.dim,
                "metric": self.metric, "ids": list(self.vectors)}

    def save(self, path: str | Path, extra_metadata: dict | None = None) -> None:
        """Atomic single-file, non-pickle snapshot, including graph topology."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        matrix = np.stack(list(self.vectors.values())) if self.vectors else np.empty((0, self.dim))
        fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".npz")
        try:
            with os.fdopen(fd, "wb") as stream:
                state = {**(extra_metadata or {}), **self._state()}
                np.savez_compressed(stream, metadata=json.dumps(state), vectors=matrix)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        with np.load(path, allow_pickle=False) as archive:
            state = json.loads(str(archive["metadata"]))
            matrix = archive["vectors"].copy()
        kinds = {"BruteForce": BruteForce, "KDTree": KDTree, "HNSW": HNSW}
        if state.get("version") != 1 or state.get("kind") not in kinds:
            raise ValueError("unsupported snapshot")
        kind = kinds[state["kind"]]
        if cls is not BruteForce and cls is not kind:
            raise ValueError("snapshot index type mismatch")
        index = kind(state["dim"], state["metric"], **state.get("config", {}))
        if matrix.shape != (len(state["ids"]), index.dim):
            raise ValueError("invalid snapshot dimensions")
        for id, row in zip(state["ids"], matrix):
            BruteForce.insert(index, id, row)
        if isinstance(index, HNSW):
            index.graph = {int(id): layers for id, layers in state["graph"].items()}
            index.entry = state["entry"]
            index.deleted = set(state["deleted"])
            index.rng.bit_generator.state = state["rng"]
            for id, row in index.vectors.items():
                index._cache(id, row)
        return cast(Self, index)


class KDTree(BruteForce):
    """Balanced median tree, lazily rebuilt after mutation; exact for all metrics."""

    def __init__(self, dim: int, metric: Metric = "cosine", **kwargs: object):
        super().__init__(dim, metric)
        self._dirty = True
        self._root = None

    def insert(self, id: int, vector: ArrayLike) -> None:
        super().insert(id, vector)
        self._dirty = True

    def delete(self, id: int) -> bool:
        deleted = super().delete(id)
        self._dirty |= deleted
        return deleted

    def build(self) -> None:
        ids = list(self.vectors)
        self._zeros = []
        self._points = {}
        for id in ids:
            point = self.vectors[id]
            norm = np.linalg.norm(point)
            if self.metric == "cosine" and norm == 0:
                self._zeros.append(id)
            else:
                self._points[id] = point / norm if self.metric == "cosine" else point

        def build_node(keys: list[int], depth: int):
            if not keys:
                return None
            axis = depth % self.dim
            keys.sort(key=lambda id: (self._points[id][axis], id))
            middle = len(keys) // 2
            return (keys[middle], axis, build_node(keys[:middle], depth + 1),
                    build_node(keys[middle + 1:], depth + 1))

        self._root = build_node(list(self._points), 0)
        self._dirty = False

    def search(self, query: ArrayLike, k: int = 5, **_: object) -> list[Hit]:
        q = self._vector(query)
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise ValueError("k must be positive")
        if self._dirty:
            self.build()
        if self.metric == "cosine":
            norm = np.linalg.norm(q)
            if norm == 0:
                return [(id, 1.0) for id in sorted(self.vectors)[:k]]
            q = q / norm
        heap: list[tuple[float, int]] = []

        def offer(id: int, distance: float):
            candidate = (-distance, -id)
            if len(heap) < k:
                heapq.heappush(heap, candidate)
            elif candidate > heap[0]:
                heapq.heapreplace(heap, candidate)

        for id in self._zeros:
            offer(id, 1.0)

        def visit(node):
            if node is None:
                return
            id, axis, left, right = node
            point = self._points[id]
            distance = float(distances(point[None, :], q, self.metric)[0])
            offer(id, distance)
            diff = float(q[axis] - point[axis])
            near, far = (left, right) if diff < 0 else (right, left)
            visit(near)
            bound = diff * diff / 2 if self.metric == "cosine" else abs(diff)
            if len(heap) < k or bound <= -heap[0][0] + 1e-12:
                visit(far)

        visit(self._root)
        return sorted([(-id, -distance) for distance, id in heap], key=lambda h: (h[1], h[0]))


class HNSW(BruteForce):
    """Layered proximity graph with beam search and diversity neighbor selection.

    Deletions are tombstones: deleted nodes remain traversable until compact().
    This preserves graph routes; snapshots retain tombstones and random state.
    """

    def __init__(self, dim: int, metric: Metric = "cosine", M: int = 16,
                 ef_construction: int = 100, seed: int = 42, **kwargs: object):
        super().__init__(dim, metric)
        if (not isinstance(M, int) or isinstance(M, bool) or not isinstance(ef_construction, int)
                or isinstance(ef_construction, bool) or M < 2 or ef_construction < M):
            raise ValueError("M >= 2 and ef_construction >= M required")
        self.M, self.ef_construction, self.seed = M, ef_construction, seed
        self.rng = np.random.default_rng(seed)
        self.graph: dict[int, list[list[int]]] = {}
        self.entry: int | None = None
        self.deleted: set[int] = set()
        self._positions: dict[int, int] = {}
        self._matrix = np.empty((64, dim), dtype=np.float64)

    def _cache(self, id: int, vector: NDArray) -> None:
        position = len(self._positions)
        if position == len(self._matrix):
            expanded = np.empty((2 * len(self._matrix), self.dim), dtype=np.float64)
            expanded[:position] = self._matrix
            self._matrix = expanded
        self._positions[id] = position
        norm = np.linalg.norm(vector)
        self._matrix[position] = vector / norm if self.metric == "cosine" and norm else vector

    def _dist(self, q: NDArray, ids: list[int]) -> NDArray:
        rows = self._matrix[[self._positions[id] for id in ids]]
        if self.metric == "cosine":
            norm = np.linalg.norm(q)
            return np.clip(1 - rows @ (q / norm), 0, 2) if norm else np.ones(len(ids))
        return distances(rows, q, self.metric)

    def _layer(self, q: NDArray, entries: list[int], ef: int, level: int) -> list[tuple[float, int]]:
        initial = [(float(d), id) for d, id in zip(self._dist(q, entries), entries)]
        candidates = initial.copy()
        heapq.heapify(candidates)
        best = [(-d, -id) for d, id in initial]
        heapq.heapify(best)
        visited = set(entries)
        while candidates:
            distance, id = heapq.heappop(candidates)
            if len(best) >= ef and distance > -best[0][0]:
                break
            neighbors = [n for n in self.graph[id][level] if n not in visited]
            visited.update(neighbors)
            if not neighbors:
                continue
            for d, neighbor in zip(self._dist(q, neighbors), neighbors):
                pair = (float(d), neighbor)
                if len(best) < ef or pair < (-best[0][0], -best[0][1]):
                    heapq.heappush(candidates, pair)
                    heapq.heappush(best, (-pair[0], -neighbor))
                    if len(best) > ef:
                        heapq.heappop(best)
        return sorted((-d, -id) for d, id in best)

    def _select(self, q: NDArray, candidates: list[tuple[float, int]], limit: int) -> list[int]:
        if self.metric == "cosine" and candidates:
            # One BLAS call replaces a Python/NumPy call for each candidate.
            rows = self._matrix[[self._positions[id] for _, id in candidates]]
            pairwise = np.clip(1 - rows @ rows.T, 0, 2)
            query_distances = np.array([d for d, _ in candidates])
            eligible = np.ones(len(candidates), dtype=bool)
            chosen: list[int] = []
            for position, (_, id) in enumerate(candidates):
                if eligible[position]:
                    chosen.append(id)
                    if len(chosen) == limit:
                        break
                    eligible &= pairwise[:, position] >= query_distances
            return chosen
        selected: list[int] = []
        for distance, id in candidates:
            if not selected or np.all(self._dist(self.vectors[id], selected) >= distance):
                selected.append(id)
                if len(selected) == limit:
                    break
        return selected

    def insert(self, id: int, vector: ArrayLike) -> None:
        super().insert(id, vector)
        q = self.vectors[id]
        self._cache(id, q)
        level = int(-math.log(max(float(self.rng.random()), 1e-15)) / math.log(self.M))
        self.graph[id] = [[] for _ in range(level + 1)]
        if self.entry is None:
            self.entry = id
            return
        entry = self.entry
        top = len(self.graph[entry]) - 1
        for layer in range(top, level, -1):
            entry = self._layer(q, [entry], 1, layer)[0][1]
        for layer in range(min(top, level), -1, -1):
            candidates = self._layer(q, [entry], self.ef_construction, layer)
            limit = self.M * 2 if layer == 0 else self.M
            selected = self._select(q, candidates, limit)
            self.graph[id][layer] = selected
            for neighbor in selected:
                links = self.graph[neighbor][layer]
                links.append(id)
                if len(links) > limit:
                    ranked = sorted((float(d), n) for d, n in zip(self._dist(self.vectors[neighbor], links), links))
                    self.graph[neighbor][layer] = self._select(self.vectors[neighbor], ranked, limit)
            entry = candidates[0][1]
        if level > top:
            self.entry = id

    def delete(self, id: int) -> bool:
        if id not in self.vectors or id in self.deleted:
            return False
        self.deleted.add(id)
        return True

    def search(self, query: ArrayLike, k: int = 5, ef: int = 50, **_: object) -> list[Hit]:
        q = self._vector(query)
        if not isinstance(k, int) or isinstance(k, bool) or k < 1 or ef < 1:
            raise ValueError("positive k and ef required")
        if self.entry is None or len(self.deleted) == len(self.vectors):
            return []
        entry = self.entry
        for layer in range(len(self.graph[entry]) - 1, 0, -1):
            entry = self._layer(q, [entry], 1, layer)[0][1]
        found = self._layer(q, [entry], max(ef, k + len(self.deleted)), 0)
        return [(id, d) for d, id in found if id not in self.deleted][:k]

    def compact(self) -> None:
        active = [(id, v.copy()) for id, v in self.vectors.items() if id not in self.deleted]
        self.__init__(self.dim, self.metric, self.M, self.ef_construction, self.seed)
        for id, vector in active:
            self.insert(id, vector)

    def _state(self) -> dict:
        return {**super()._state(), "config": {"M": self.M, "ef_construction": self.ef_construction,
                "seed": self.seed}, "graph": self.graph, "entry": self.entry,
                "deleted": sorted(self.deleted), "rng": self.rng.bit_generator.state}

