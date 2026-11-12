"""Import prepared AG News embeddings into the durable API document store."""
import argparse
import json
from pathlib import Path

import numpy as np

from vectorlab import HNSW
from vectorlab.rag import DocumentStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--limit", type=int, default=3000)
    parser.add_argument("--snapshot", type=Path, default=Path("data/documents.npz"))
    args = parser.parse_args()
    if args.snapshot.exists():
        parser.error("snapshot exists; choose a new path to avoid overwriting documents")
    metadata = json.loads((args.data / "news.json").read_text(encoding="utf-8"))
    with np.load(args.data / "news.npz", allow_pickle=False) as archive:
        vectors = archive["vectors"]
    if not 1 <= args.limit <= len(vectors):
        parser.error("--limit must lie between 1 and corpus size")
    class Identity:
        model = metadata["model"]
        def encode(self, texts):
            raise RuntimeError("import uses precomputed embeddings")
    store = DocumentStore(args.snapshot, Identity())
    store.index = HNSW(vectors.shape[1])
    for position, (text, vector) in enumerate(zip(metadata["texts"][:args.limit], vectors[:args.limit])):
        id = position + 1
        store.index.insert(id, vector)
        store.documents[id] = {"id": id, "title": f"AG News row {metadata['source_rows'][position]}",
                                "text": text, "chunk": 0, "chunks": 1}
    store.next_id = args.limit + 1
    store.save()
    print(f"Saved {args.limit} real documents to {args.snapshot}")


if __name__ == "__main__":
    main()
