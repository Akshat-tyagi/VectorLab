"""Build HNSW from ordered encoding checkpoints while data preparation continues.

This optional pipeline overlaps graph construction and embedding computation.
Recorded construction time excludes waiting for batches and snapshot writes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from vectorlab import HNSW


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, default=Path("data/embedding_batches"))
    parser.add_argument("--prefix", required=True, help="Exact corpus hash prefix used in batch filenames")
    parser.add_argument("--count", type=int, default=100000)
    parser.add_argument("--m", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--ef-construction", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("data/prebuilt"))
    parser.add_argument("--wait-timeout", type=float, default=3600)
    args = parser.parse_args()
    if args.count < 1 or args.wait_timeout <= 0 or not args.prefix.isalnum():
        parser.error("positive count, positive timeout, and alphanumeric prefix required")
    args.output.mkdir(parents=True, exist_ok=True)
    indexes: dict[int, HNSW] = {}
    times = {m: 0.0 for m in args.m}
    digest = hashlib.sha256()
    position = 0
    while position < args.count:
        batch_path = args.batch_dir / f"{args.prefix}-{position}.npy"
        deadline = time.monotonic() + args.wait_timeout
        while True:
            try:
                batch = np.load(batch_path, allow_pickle=False)
                break
            except (FileNotFoundError, ValueError, EOFError):
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"encoding batch not ready: {batch_path}")
                time.sleep(2)
        if batch.ndim != 2 or not len(batch) or not np.isfinite(batch).all():
            raise ValueError(f"invalid batch: {batch_path}")
        batch = np.ascontiguousarray(batch[:args.count - position], dtype=np.float64)
        digest.update(memoryview(batch))
        for m in args.m:
            if m not in indexes:
                indexes[m] = HNSW(batch.shape[1], M=m, ef_construction=args.ef_construction)
            start = time.perf_counter()
            for offset, row in enumerate(batch):
                indexes[m].insert(position + offset, row)
            times[m] += time.perf_counter() - start
        position += len(batch)
        print(f"Built both graphs through {position}/{args.count} real vectors", flush=True)
    for m, index in indexes.items():
        path = args.output / f"hnsw-n{args.count}-m{m}-efc{args.ef_construction}.npz"
        index.save(path)
        path.with_suffix(".json").write_text(json.dumps({"build_seconds": times[m],
            "vector_sha256": digest.hexdigest(), "count": position, "M": m,
            "ef_construction": args.ef_construction,
            "timing": "sum of batch insertion wall times; excludes waiting, encoding, and save"}, indent=2), encoding="utf-8")
        print(f"Saved {path}: construction {times[m]:.2f}s", flush=True)


if __name__ == "__main__":
    main()
