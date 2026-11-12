"""Create a readable report exclusively from completed measured artifacts."""
import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    args = parser.parse_args()
    with (args.reports / "benchmark.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    rag = json.loads((args.reports / "rag.json").read_text(encoding="utf-8"))
    environment = json.loads((args.reports / "benchmark_environment.json").read_text(encoding="utf-8"))
    k, query_count, dim = environment["k"], environment["queries"], environment["dim"]
    model = environment.get("dataset", {}).get("model", rag["model"])
    lines = ["# Measured results", "", "These are measurements from this checkout, not target numbers or simulated timings.", "",
             "## Search", "", f"AG News documents encoded with {model} ({dim} dimensions). "
             f"{query_count} held-out AG News test documents serve as queries; k={k}, cosine distance, seed=42, ef_construction=100. "
             "Exact brute-force top-k IDs define recall. Index construction and embedding inference are outside query timing. "
             "Three warmups precede each configuration. The brute-force baseline caches its matrix and norms.", "",
             f"| Vectors | Index | M | ef | Recall@{k} | Median ms | p95 ms | Build s |", "|---:|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(f"| {int(row['n']):,} | {row['algorithm']} | {row['M'] or '—'} | {row['ef'] or '—'} | "
                     f"{float(row['recall_at_k']):.3f} | {float(row['p50_ms']):.3f} | {float(row['p95_ms']):.3f} | "
                     f"{float(row['build_seconds']):.2f} |")
    lines += ["", "![Recall and latency by M and ef](recall_latency.png)", "",
              "Raw configuration results, mean latency, and compressed snapshot sizes: [benchmark.csv](benchmark.csv). "
              "Machine, package, source provenance, and exact run arguments: [benchmark_environment.json](benchmark_environment.json) "
              "and [requirements-measured.txt](requirements-measured.txt).", "",
              "This is a shared host, not a controlled hardware lab. Treat timings as observations; small differences "
              "can be noise. KD-tree pruning degrades at 384 dimensions. Python heap, dictionary, and allocation costs "
              "can make HNSW slower than vectorized brute force, especially at small N. Larger ef increases work and "
              "usually recall; larger M increases graph storage and neighbor work. No universal speedup is claimed.", "",
              "Graphs were constructed once per size/M. Small graphs used direct insertion; the 100k graphs used "
              "the streaming batch builder while encoding continued. Their reported build times sum timed insertions, "
              "excluding waits, model inference, and snapshot writes. Final query measurements reuse hash-verified graphs. "
              "Snapshot size is disk usage, not peak RAM. The 100k graph needs both original vectors and a normalized cache.", "",
              "## RAG retrieval", "",
              f"Corpus: {rag['corpus_size']:,} distinct SQuAD validation paragraphs. "
              f"Evaluation: {rag['question_count']} questions with human-annotated answers and relevant source IDs. "
              f"k={rag['k']}; fixed maximum cosine distance={rag['max_distance']}.", "",
              f"* Filtered HNSW retrieval hit rate: **{rag['retrieval_hit_rate']:.0%}** "
              f"({sum(q['hit'] for q in rag['questions'])}/{rag['question_count']}).",
              f"* Unfiltered exact retrieval hit rate: **{rag['exact_unfiltered_hit_rate']:.0%}**.",
              f"* Negative-control abstention: **{rag['negative_abstention_rate']:.0%}** "
              f"on {len(rag['negative_controls'])} unrelated questions.", "",
              "Per-question answers, ground truth, returned IDs/distances, and failures: [rag.json](rag.json). "
              "The conservative threshold rejects some valid evidence. It is not a calibrated confidence score. "
              "The 25 deterministically selected questions are a small diagnostic set, not a representative quality estimate. "
              "SQuAD paragraphs are retrieval units here; the API's overlapping word chunker is separately unit-tested. "
              "Retrieval success does not establish answer correctness. These retrieval metrics "
              "do not score generated-answer faithfulness.", "",
              "## Reproduce", "", "```sh", "python -m scripts.prepare_data --count 100000",
              "python -m scripts.benchmark --sizes 1000 10000 100000 --m 8 16 --ef 20 50 100 --queries 100",
              "python -m scripts.evaluate_rag", "python -m scripts.plot_results",
              "python -m scripts.summarize_results", "```", "",
              f"Measured Python {environment['python']}, NumPy {environment['numpy']}; "
              "BLAS/OMP thread settings are recorded in the environment artifact.", ""]
    (args.reports / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
