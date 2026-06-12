"""
IVF (Inverted File Index) demo — vector database approximate nearest neighbor.

Run:
    python src/main.py

What this script does:
  1. Generates (or loads) dummy product-embedding vectors in data/vectors.csv
  2. Builds an IVF index over those vectors
  3. Runs several example searches
  4. Benchmarks recall@k for different nprobe values
"""

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from ivf import IVFIndex

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_PATH = os.path.join(ROOT, "data", "vectors.csv")

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------
DIM = 8          # vector dimensionality
N_PER_CLUSTER = 60

# Three semantic clusters mimicking product-category embeddings
CLUSTERS = {
    "electronics": np.array([8.0, 2.0, 7.0, 3.0, 8.0, 2.0, 7.0, 3.0]),
    "furniture":   np.array([2.0, 8.0, 3.0, 7.0, 2.0, 8.0, 3.0, 7.0]),
    "clothing":    np.array([5.0, 5.0, 1.0, 9.0, 5.0, 5.0, 1.0, 9.0]),
}


def generate_and_save(path: str) -> None:
    """Create synthetic clustered vectors and write them to a CSV file."""
    rng = np.random.default_rng(seed=42)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "label"] + [f"v{i}" for i in range(DIM)])
        vid = 0
        for label, center in CLUSTERS.items():
            for _ in range(N_PER_CLUSTER):
                vec = center + rng.standard_normal(DIM) * 1.2
                writer.writerow([vid, label] + [f"{x:.4f}" for x in vec])
                vid += 1

    print(f"Generated {vid} vectors → {os.path.relpath(path)}\n")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_vectors(path: str) -> tuple[np.ndarray, list[int], list[str]]:
    """Return (vectors, ids, labels) from a CSV produced by generate_and_save."""
    ids, labels, rows = [], [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        dim = sum(1 for k in (reader.fieldnames or []) if k.startswith("v"))
        for row in reader:
            ids.append(int(row["id"]))
            labels.append(row["label"])
            rows.append([float(row[f"v{i}"]) for i in range(dim)])
    return np.array(rows), ids, labels


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def recall_at_k(
    index: IVFIndex,
    vectors: np.ndarray,
    query: np.ndarray,
    k: int,
    nprobe: int,
) -> float:
    """Fraction of true top-k neighbors found by IVF search."""
    exact_dists = np.linalg.norm(vectors - query, axis=1)
    true_top_k = set(np.argsort(exact_dists)[:k])
    ivf_results = {vid for _, vid in index.search(query, k=k, nprobe=nprobe)}
    return len(true_top_k & ivf_results) / k


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ── Step 0: Data ──────────────────────────────────────────────────────
    if not os.path.exists(DATA_PATH):
        print("No data found — generating dummy vectors...")
        generate_and_save(DATA_PATH)

    vectors, ids, labels = load_vectors(DATA_PATH)
    dim = vectors.shape[1]
    n = len(vectors)
    print(f"Loaded {n} vectors  |  dim={dim}  |  labels={set(labels)}")

    # ── Step 1: Build IVF index ───────────────────────────────────────────
    #
    # nlist: how many Voronoi cells to create.
    # Rule of thumb: sqrt(n) ≤ nlist ≤ 4*sqrt(n).
    # With 180 vectors and 3 natural clusters, nlist=9 gives ~3 sub-cells per
    # original cluster — a good balance.
    nlist = 9
    print(f"\nBuilding IVF index  |  nlist={nlist}")
    index = IVFIndex(dim=dim, nlist=nlist)
    index.train(vectors, n_iter=40)
    index.add(vectors, ids)
    index.stats()

    # ── Step 2: Example searches ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Example searches (k=5)")
    print("=" * 60)

    query_examples = {
        "electronics": CLUSTERS["electronics"],
        "furniture":   CLUSTERS["furniture"],
        "clothing":    CLUSTERS["clothing"],
    }

    for qname, query in query_examples.items():
        print(f"\nQuery category: {qname}")
        print(f"  Query vector : [{', '.join(f'{x:.1f}' for x in query)}]")
        for nprobe in [1, 3, nlist]:
            results = index.search(query, k=5, nprobe=nprobe)
            result_labels = [labels[vid] for _, vid in results]
            top_dists = [f"{d:.3f}" for d, _ in results]
            print(f"  nprobe={nprobe:2d}  distances={top_dists}  labels={result_labels}")

    # ── Step 3: Recall benchmark ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Recall@5 benchmark — IVF vs exact search")
    print("=" * 60)
    print("(Higher nprobe = more cells scanned = better recall, but slower)\n")

    rng = np.random.default_rng(seed=7)
    test_queries = [
        CLUSTERS[label] + rng.standard_normal(dim) * 0.8
        for label in list(CLUSTERS.keys()) * 5   # 5 queries per cluster
    ]

    for nprobe in [1, 2, 3, 4, 6, nlist]:
        recalls = [
            recall_at_k(index, vectors, q, k=5, nprobe=nprobe)
            for q in test_queries
        ]
        bar_len = int(np.mean(recalls) * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"  nprobe={nprobe:2d}  [{bar}]  recall={np.mean(recalls):.0%}")

    print("\nDone. Increase nprobe to trade speed for accuracy.")


if __name__ == "__main__":
    main()
