"""Score 25 SQuAD questions against labeled source paragraphs, plus abstentions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from vectorlab import BruteForce, HNSW


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("reports/rag.json"))
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--max-distance", type=float, default=0.4)
    args = parser.parse_args()
    if args.k < 1 or not 0 <= args.max_distance <= 2:
        parser.error("require k > 0 and distance in [0,2]")
    labels = json.loads((args.data / "rag.json").read_text(encoding="utf-8"))
    with np.load(args.data / "rag.npz", allow_pickle=False) as archive:
        vectors, queries, negatives = archive["vectors"], archive["queries"], archive["negatives"]
    exact, index = BruteForce(vectors.shape[1]), HNSW(vectors.shape[1])
    for id, vector in enumerate(vectors):
        exact.insert(id, vector)
        index.insert(id, vector)
    results = []
    for question, query in zip(labels["questions"], queries):
        hits = [(id, d) for id, d in index.search(query, args.k) if d <= args.max_distance]
        truth = set(question["relevant_ids"])
        results.append({**question, "retrieved": hits, "hit": bool(set(dict(hits)) & truth),
                        "exact_hit": bool(set(dict(exact.search(query, args.k))) & truth),
                        "abstained": not hits})
    negative_results = []
    for question, query in zip(labels["negative_questions"], negatives):
        hits = [(id, d) for id, d in index.search(query, args.k) if d <= args.max_distance]
        negative_results.append({"question": question, "abstained": not hits, "retrieved": hits})
    report = {"dataset": labels["dataset"], "model": labels["model"], "corpus_size": len(vectors),
              "k": args.k, "max_distance": args.max_distance, "question_count": len(results),
              "retrieval_hit_rate": float(np.mean([row["hit"] for row in results])),
              "exact_unfiltered_hit_rate": float(np.mean([row["exact_hit"] for row in results])),
              "negative_abstention_rate": float(np.mean([row["abstained"] for row in negative_results])),
              "questions": results, "negative_controls": negative_results,
              "limitations": "Retrieval evaluation only; does not score generation faithfulness. Threshold fixed before evaluation. Small non-random sample; no population-level claim."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"questions", "negative_controls"}}, indent=2))


if __name__ == "__main__":
    main()
