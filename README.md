# VectorLab NumPy

An inspectable Python vector database: hand-built HNSW, an exact KD-tree, a NumPy brute-force baseline, durable document retrieval, a browser workspace, and reproducible evaluation on real text embeddings.

The indexes are implemented directly in Python and NumPy without wrapping FAISS, hnswlib, or Chroma. This is an educational implementation, not a production database.

## Setup

Use Python 3.11 or newer; Python 3.11 or 3.12 is recommended for the broadest ML wheel availability. Run every command from this repository directory. Windows, macOS, and Linux use the same Python commands after activating a virtual environment:

```sh
python -m venv .venv
```

Activate on macOS/Linux with `source .venv/bin/activate`, on Windows PowerShell with `.venv\Scripts\Activate.ps1`, or in Windows Command Prompt with `.venv\Scripts\activate.bat`. Alternatively, invoke the virtual environment's Python executable directly if activation is restricted.

```sh
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m pytest -q
```

For just the indexes, API, and tests, use `python -m pip install -e ".[test]"`. The real-data experiments additionally need `python -m pip install -e ".[evaluation]"`. On CPU-only Linux, installing PyTorch from its CPU wheel index before the requirements can avoid large CUDA downloads:

```sh
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Dependencies use bounded version ranges, not a fully locked environment. CI checks pass for the configured Python 3.11/3.12 matrix on Linux, macOS, and Windows.

## Indexes

```python
from vectorlab import HNSW, KDTree, BruteForce

index = HNSW(dim=3, metric="cosine", M=16, ef_construction=100)
index.insert(10, [1, 0, 0])
index.insert(20, [0, 1, 0])
print(index.search([1, 0, 0], k=2, ef=50))  # (external_id, distance)
index.delete(20)
index.save("data/example.npz")
restored = HNSW.load("data/example.npz")
```

All indexes support Euclidean, Manhattan, and cosine **distance** (smaller is better), finite vectors, unique signed 64-bit integer IDs, and explicit dimension validation. Cosine distance to a zero vector is defined as 1, including zero against zero. Repeated vectors are allowed; repeated IDs are rejected. Deleted HNSW IDs are reserved until compaction. `k` larger than the live population returns at most that population; empty indexes return an empty list. Nonpositive `k` is rejected.

* **BruteForce** caches a contiguous matrix and vector norms, vectorizes distance calculations, and returns exact neighbors sorted by distance then ID. Cache construction is outside benchmark query timing. Sorting is O(N log N); selection is not optimized with partitioning.
* **KDTree** builds a balanced median tree and performs exact branch-and-bound search. Euclidean and Manhattan use axis-distance bounds. Cosine uses unit-normalized vectors and the bound `axis_difference² / 2`; zero vectors are handled separately. Mutations trigger a lazy rebuild. At high dimensions, pruning often fails and Python recursion/NumPy call overhead can make it slower than brute force. No logarithmic query-time guarantee is claimed.
* **HNSW** assigns exponentially distributed random levels, performs greedy upper-layer descent and beam search at the base, and selects diverse neighbors. `M` limits upper-layer neighbors; the base limit is `2*M`. `ef_construction` controls insertion search breadth; query `ef` controls recall/latency. It is approximate, with no universal recall or complexity guarantee. Deletions remain traversable tombstones to preserve routes. `compact()` rebuilds live nodes; many tombstones increase search work.

Snapshots are versioned compressed NumPy archives with JSON metadata, not pickle. They preserve vectors, graph links, entry point, tombstones, parameters, and RNG state. Writes use a temporary file, fsync, and atomic replacement on the same filesystem. Only load snapshots you trust; malformed or extremely large archives are not hardened against hostile inputs.

## Real datasets and embeddings

```sh
python -m scripts.prepare_data --count 100000
```

This downloads **AG News** (`fancyzhx/ag_news`) and **SQuAD** (`rajpurkar/squad`) using Hugging Face Datasets. AG News training documents are shuffled with seed 42; 100 test documents become held-out benchmark queries, avoiding self-neighbor inflation. Text is encoded by `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions). No manually written vectors or synthetic padding are used. Encoding is checkpointed in batches, so rerunning can reuse completed batches for the same model and corpus.

The command also encodes all distinct SQuAD validation paragraphs and saves 25 questions with human-annotated answers, offsets, and relevant paragraph IDs. The questions use distinct paragraphs and are selected deterministically. These are paragraph retrieval units rather than the API's 250-word overlapping chunks. This evaluates the embedding/index retrieval path; it does not measure how the API chunking strategy changes retrieval.

Generated files under `data/` include embeddings, source row IDs, source texts, model identity, corpus hash, and evaluation labels. They are ignored by Git. Downloads require internet access and disk space; model inference and pure Python graph construction can take substantial CPU time. Dataset/model revisions currently follow the upstream defaults; archive generated metadata and embeddings when reproducing a particular run. Check upstream dataset terms before redistributing source text.

## Benchmark and charts

```sh
python -m scripts.benchmark --sizes 1000 10000 100000 --m 8 16 --ef 20 50 100 --queries 100
python -m scripts.plot_results
```

The benchmark compares both custom indexes with exact brute force. Each corpus size is a prefix of the same shuffled real corpus. Recall@k is `|approximate IDs intersect exact top-k IDs| / k`, with k=10 by default. Three warmups precede each timed configuration. It reports per-query mean, median and p95 milliseconds, recall, build seconds, and compressed snapshot bytes. Index construction is outside query timing, including the KD-tree's lazy build. Embedding time, network time, and LLM generation are excluded from search latency. Full metadata is written beside results. Snapshot bytes measure disk space, not process RSS.

`reports/benchmark.csv` is flushed after each configuration. `reports/recall_latency.png` plots recall against median latency, labels ef, and uses separate curves for M. The script records environment details and can run with `--skip-kdtree` only when explicitly requested; omitted configurations must not be represented as measured. For less noisy results, set `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` in your shell before running and record the hardware. Do not compare results from different machines as if they were controlled.

An optional `scripts.build_stream` command can consume the ordered embedding batch files while preparation is running. `scripts.benchmark --prebuilt-dir PATH` reuses its saved graphs after checking the corpus hash. In this mode, reported build seconds are the sum of timed insertion batches, excluding waits for embeddings and snapshot writes. The default benchmark builds indexes directly.

Python spends time on heap operations, sets, dictionaries, recursion, neighbor selection, and many small NumPy allocations. NumPy accelerates distance arithmetic but does not remove those graph-traversal costs. Increasing M can improve connectivity but costs build time, memory, and neighbor evaluation. Increasing ef typically raises recall and query time; the actual tradeoff depends on the data. A 100k-vector graph in this implementation can be slow to build. The vectorized exact baseline may win at small N or when approximate recall requirements are high.

Measured results are documented in [reports/RESULTS.md](reports/RESULTS.md). No performance numbers should be inferred from the example commands.

The committed report contains all 24 completed configurations, including 100k vectors. At 100k, M=16 and ef=100 reached 99.2% recall@10 with 3.90 ms median latency; the exact baseline measured 36.52 ms. HNSW construction took about nine minutes per 100k graph on the measured machine. Brute force was faster at 1k. These observations apply to this dataset and environment.

## RAG evaluation

```sh
python -m scripts.evaluate_rag --k 3 --max-distance 0.4
```

`reports/rag.json` contains every question, accepted answers, ground-truth IDs, retrieved IDs/distances, per-question success, overall retrieval hit rate, and an exact unfiltered retrieval baseline. The 25 positives measure whether the right source paragraph appears in the returned top-k. Three separate out-of-corpus negative controls measure abstention. The fixed cosine threshold is a simple heuristic, not a calibrated confidence estimate. It may reject valid evidence or accept irrelevant material. Tune on a separate development set before evaluating new held-out questions. This small deterministic evaluation is not an estimate of all-domain quality. Generation faithfulness and answer correctness are not scored by the retrieval metric.

## API and persistence

Create `.env` from `.env.example` if it is missing (copy it with your file manager, or run `python -c "from pathlib import Path; p=Path('.env'); p.exists() or p.write_text(Path('.env.example').read_text())"`). Paste your key from the [Groq console](https://console.groq.com/keys):

```dotenv
GROQ_API_KEY=your_key_here
GROQ_MODEL=openai/gpt-oss-20b
```

The server automatically loads this repository's `.env`; existing shell environment variables take precedence. Restart after changing settings. `.env` is ignored by Git; never put the key in frontend JavaScript. Search and document management work without a Groq key; AI answers with evidence require one. Missing credentials, rate limits, and connection failures appear as actionable errors.

Only answer generation uses Groq Cloud. Questions and retrieved source passages are sent to Groq; embeddings, the custom vector index, and document snapshots remain local. Sentence-transformers is a local embedding encoder, not the answer-generating LLM. Keeping its model preserves compatibility with existing snapshots. The default generation model is `openai/gpt-oss-20b` ([provider documentation](https://console.groq.com/docs/model/openai/gpt-oss-20b)); availability and limits depend on your Groq account. Live Groq answer generation and GitHub CI have been verified working by the project maintainer.

```sh
python -m uvicorn vectorlab.api:create_app --factory --host 127.0.0.1 --port 8080
```

Open **<http://127.0.0.1:8080/>** for the frontend. Use **Semantic search** to find relevant passages with a choice of HNSW (approximate), KD-tree (exact), or brute force (exact), **Documents** to add, filter, browse, and delete stored chunks, and **Ask AI** to get an answer with its retrieved sources. The interface includes loading states, empty-result messages, and error feedback. It uses the same server as the API; no frontend installation or build step is needed.

Interactive request/response documentation remains available at <http://127.0.0.1:8080/docs>. The default sentence-transformer embedding model downloads on first startup. No Ollama or local answer-generation model is needed. `VECTORLAB_SNAPSHOT` selects the snapshot path (default `data/documents.npz`); `VECTORLAB_MAX_DISTANCE` sets the relevance threshold (default 0.4).

| Method | Endpoint | Input / behavior |
|---|---|---|
| GET | `/status` | Counts, embedding identity, dimensions, threshold |
| POST | `/documents` | `{"title":"Graph notes","text":"A tree is a connected graph without cycles."}` |
| GET | `/documents` | Stored document chunks |
| DELETE | `/documents/{id}` | Delete a chunk and persist |
| POST | `/search` | `{"question":"What is a tree?","k":3,"algorithm":"hnsw"}`; also `kd-tree` or `brute-force` |
| POST | `/ask` | Same input; retrieve evidence, then generate with Groq |

For generation, configure `GROQ_API_KEY` in `.env`. `/ask` returns `abstained: true` and a clear no-evidence response when retrieval finds nothing within the threshold; it does not invoke the LLM in that case. With evidence, the prompt requests evidence-only answers and citations, but prompts cannot guarantee model faithfulness.

To start the API with a few thousand prepared real documents without embedding them twice, run `python -m scripts.ingest_data --limit 3000` before starting the server. The import refuses to overwrite an existing snapshot. Use the same embedding model for data preparation and the server.

Semantic search defaults to HNSW. Alternate indexes reuse the same stored vectors and cosine threshold. They are built on first use, cached, and invalidated after insertion or deletion; the first query after a change includes rebuild time. KD-tree can be slower than brute force for high-dimensional embeddings. Ask AI continues to use HNSW.

Document insertion uses 250-word chunks with 30-word overlap. All chunk embeddings are generated and validated before mutation, preventing partial inserts after an embedding failure. Graph, text, IDs, and model identity share one atomic snapshot, and startup reloads them. Persistence failure raises an error; mutations are not rolled back in memory on disk failure. The document store serializes index access with a lock. Use **one API worker**, because multiple worker processes would have independent memory and could overwrite the snapshot. The core index classes themselves are not thread-safe.

This is a local developer application: no authentication, replication, transactions, multi-process coordination, or streaming LLM output is included. Keep the default localhost binding.

## Project map

```text
vectorlab/indexes.py       NumPy metrics, brute force, KD-tree, HNSW, snapshots
vectorlab/embeddings.py    Local sentence-transformer embeddings
vectorlab/generation.py    Groq Cloud answer generation and safe errors
vectorlab/rag.py           Chunking, persistent document store, retrieval and abstention
vectorlab/api.py           FastAPI app factory and Pydantic inputs
vectorlab/static/          Browser interface (HTML, CSS, JavaScript)
scripts/prepare_data.py    Download and encode real datasets
scripts/ingest_data.py     Seed the API with prepared real documents
scripts/build_stream.py    Optional overlapping graph construction
scripts/benchmark.py       Recall, latency, build time and snapshot size
scripts/evaluate_rag.py    Answer-labeled retrieval evaluation
scripts/plot_results.py    Chart from measured CSV
scripts/summarize_results.py  Readable report from measured artifacts
tests/                    Correctness, persistence and API tests
reports/                   Measured artifacts and limitations
```

The tests deliberately use small deterministic arrays and fake embedders to run without downloads; those fixtures never feed the reported real-data experiments. Git history records implementation milestones. The project directory and package are named `vectorlab-numpy`; the project title is **VectorLab NumPy**.
