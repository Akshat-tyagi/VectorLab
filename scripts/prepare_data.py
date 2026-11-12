"""Download AG News and SQuAD, then encode real text. Run from repository root."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from vectorlab.embeddings import create_embedder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100_000)
    parser.add_argument("--backend", choices=["sentence-transformers", "ollama"], default="sentence-transformers")
    parser.add_argument("--model")
    parser.add_argument("--output", type=Path, default=Path("data"))
    args = parser.parse_args()
    if not 1000 <= args.count <= 120_000:
        parser.error("--count must be between 1000 and 120000")
    from datasets import load_dataset
    args.output.mkdir(parents=True, exist_ok=True)
    encoder = create_embedder(args.backend, args.model)
    start = time.perf_counter()
    news = load_dataset("fancyzhx/ag_news", split="train")
    held_out = load_dataset("fancyzhx/ag_news", split="test")
    permutation = np.random.default_rng(42).permutation(len(news))[:args.count]
    texts = [news[int(i)]["text"] for i in permutation]
    queries = [held_out[i]["text"] for i in range(100)]
    corpus_texts = set(texts)
    query_overlap = sum(query in corpus_texts for query in queries)
    print(f"Encoding {len(texts)} AG News documents with {encoder.model}", flush=True)
    # Checkpoint each batch to permit resuming expensive encoding.
    batch_dir = args.output / "embedding_batches"
    batch_dir.mkdir(exist_ok=True)
    identity = hashlib.sha256(json.dumps([encoder.model, texts], ensure_ascii=False).encode()).hexdigest()
    parts = []
    for start_row in range(0, len(texts), 512):
        batch_path = batch_dir / f"{identity[:16]}-{start_row}.npy"
        if batch_path.exists():
            batch = np.load(batch_path, allow_pickle=False)
        else:
            batch = encoder.encode(texts[start_row:start_row + 512])
            np.save(batch_path, batch)
        parts.append(batch)
        print(f"Encoded {min(start_row + 512, len(texts))}/{len(texts)}", flush=True)
    vectors = np.concatenate(parts)
    np.savez_compressed(args.output / "news.npz", vectors=vectors, queries=encoder.encode(queries))
    (args.output / "news.json").write_text(json.dumps({"dataset": "fancyzhx/ag_news", "split": "train",
        "query_split": "test", "seed": 42, "model": encoder.model, "backend": args.backend,
        "exact_text_query_overlap_count": query_overlap,
        "count": len(texts), "dim": int(vectors.shape[1]), "corpus_sha256": identity,
        "encoding_seconds": time.perf_counter() - start,
        "source_rows": permutation.tolist(), "texts": texts, "queries": queries}, ensure_ascii=False), encoding="utf-8")

    # Ground truth comes from SQuAD's human annotations, not generated answers.
    squad = load_dataset("rajpurkar/squad", split="validation")
    contexts = list(dict.fromkeys(row["context"] for row in squad))
    questions, selected_contexts = [], set()
    for row in squad:
        if row["context"] in selected_contexts:
            continue
        selected_contexts.add(row["context"])
        questions.append({"id": row["id"], "question": row["question"],
                          "answers": row["answers"]["text"],
                          "relevant_ids": [contexts.index(row["context"])],
                          "answer_starts": row["answers"]["answer_start"]})
        if len(questions) == 25:
            break
    negatives = ["What is the secret password to my personal bank account?",
                 "What will my unpublished diary entry say tomorrow?",
                 "What is the current temperature inside my refrigerator?"]
    np.savez_compressed(args.output / "rag.npz", vectors=encoder.encode(contexts),
                        queries=encoder.encode([q["question"] for q in questions]),
                        negatives=encoder.encode(negatives))
    (args.output / "rag.json").write_text(json.dumps({"dataset": "rajpurkar/squad", "split": "validation",
        "model": encoder.model, "contexts": contexts, "questions": questions,
        "negative_questions": negatives, "note": "25 distinct source paragraphs; not used to tune threshold"},
        ensure_ascii=False, indent=2), encoding="utf-8")
    print("Wrote real embeddings, source provenance, and 25 answer-labeled questions.", flush=True)


if __name__ == "__main__":
    main()
