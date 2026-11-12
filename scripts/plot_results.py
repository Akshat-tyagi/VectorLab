"""Plot measured CSV results, never illustrative or invented numbers."""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("reports/benchmark.csv"))
    parser.add_argument("--output", type=Path, default=Path("reports/recall_latency.png"))
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with args.input.open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    sizes = sorted({int(row["n"]) for row in rows})
    figure, axes = plt.subplots(1, len(sizes), figsize=(6 * len(sizes), 4), squeeze=False)
    for axis, n in zip(axes[0], sizes):
        selected = [row for row in rows if int(row["n"]) == n]
        for m in sorted({row["M"] for row in selected if row["algorithm"] == "hnsw"}):
            points = sorted([row for row in selected if row["M"] == m], key=lambda row: int(row["ef"]))
            x, y = [float(r["p50_ms"]) for r in points], [float(r["recall_at_k"]) for r in points]
            axis.plot(x, y, "o-", label=f"HNSW M={m}")
            for row, px, py in zip(points, x, y):
                axis.annotate(f"ef={row['ef']}", (px, py), xytext=(3, 5), textcoords="offset points", fontsize=8)
        for row in selected:
            if row["algorithm"] != "hnsw":
                axis.scatter(float(row["p50_ms"]), float(row["recall_at_k"]), marker="x", label=row["algorithm"])
        axis.set(title=f"{n:,} real document vectors", xlabel="Median query latency (ms; log scale)", ylabel="Recall@k")
        axis.set_xscale("log")
        axis.set_ylim(0, 1.06)
        axis.legend()
        axis.grid(alpha=0.2)
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
