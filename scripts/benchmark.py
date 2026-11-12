"""Reproducible real-embedding benchmark with incremental CSV output."""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from vectorlab import BruteForce, HNSW, KDTree


def measure(index, queries, truth, k, ef=50):
    for query in queries[:3]:
        index.search(query, k, ef=ef)
    times, recalls = [], []
    for query, target in zip(queries, truth):
        start = time.perf_counter_ns()
        hits = index.search(query, k, ef=ef)
        times.append((time.perf_counter_ns() - start) / 1e6)
        recalls.append(len(set(dict(hits)) & target) / len(target))
    return {"recall_at_k": float(np.mean(recalls)), "p50_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)), "mean_ms": float(np.mean(times))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/news.npz"))
    parser.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 100000])
    parser.add_argument("--m", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--ef", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--ef-construction", type=int, default=100)
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--skip-kdtree", action="store_true", help="Explicitly omit expensive high-dimensional exact KD-tree")
    args = parser.parse_args()
    if args.queries < 1 or args.k < 1 or any(n < args.k for n in args.sizes):
        parser.error("positive queries and k <= every size required")
    with np.load(args.data, allow_pickle=False) as archive:
        matrix, queries = archive["vectors"], archive["queries"][:args.queries]
    if max(args.sizes) > len(matrix) or not len(queries):
        parser.error("dataset lacks requested vectors or queries; run prepare_data first")
    args.output.mkdir(parents=True, exist_ok=True)
    metadata = {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
                "processor": platform.processor(), "cpu_count": os.cpu_count(), "dim": matrix.shape[1],
                "queries": len(queries), "k": args.k, "metric": "cosine", "seed": 42,
                "threads": {key: os.getenv(key) for key in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"]},
                "data_path": str(args.data), "arguments": vars(args) | {"data": str(args.data), "output": str(args.output)}}
    provenance_path = args.data.with_suffix(".json")
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        metadata["dataset"] = {key: provenance[key] for key in
                               ["dataset", "split", "query_split", "model", "backend", "corpus_sha256"]}
    (args.output / "benchmark_environment.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    fields = ["n", "algorithm", "M", "ef", "ef_construction", "build_seconds", "snapshot_bytes",
              "recall_at_k", "p50_ms", "p95_ms", "mean_ms"]
    with (args.output / "benchmark.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for n in args.sizes:
            oracle = BruteForce(matrix.shape[1])
            start = time.perf_counter()
            for id, row in enumerate(matrix[:n]):
                oracle.insert(id, row)
            bf_build = time.perf_counter() - start
            truth = [set(dict(oracle.search(q, args.k))) for q in queries]
            def record(index, algorithm, build_seconds, M="", ef=""):
                path = args.output / "benchmark_index.npz"
                index.save(path)
                row = {"n": n, "algorithm": algorithm, "M": M, "ef": ef,
                       "ef_construction": args.ef_construction if M else "", "build_seconds": build_seconds,
                       "snapshot_bytes": path.stat().st_size,
                       **measure(index, queries, truth, args.k, ef or 50)}
                writer.writerow(row)
                stream.flush()
                path.unlink()
                print(json.dumps(row), flush=True)
            record(oracle, "bruteforce", bf_build)
            if not args.skip_kdtree:
                tree = KDTree(matrix.shape[1])
                start = time.perf_counter()
                for id, row in enumerate(matrix[:n]):
                    tree.insert(id, row)
                tree.build()  # Never charge lazy construction to first-query latency.
                record(tree, "kdtree", time.perf_counter() - start)
            for m in args.m:
                index = HNSW(matrix.shape[1], M=m, ef_construction=args.ef_construction)
                checkpoint = args.output / f"hnsw-n{n}-m{m}-efc{args.ef_construction}.npz"
                start = time.perf_counter()
                for id, row in enumerate(matrix[:n]):
                    index.insert(id, row)
                    if (id + 1) % 1000 == 0:
                        print(f"HNSW n={n} M={m}: {id + 1} inserted", flush=True)
                build_seconds = time.perf_counter() - start
                index.save(checkpoint)
                for ef in args.ef:
                    record(index, "hnsw", build_seconds, m, ef)


if __name__ == "__main__":
    main()
