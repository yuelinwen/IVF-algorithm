import numpy as np


class IVFIndex:
    """
    Inverted File Index (IVF) for approximate nearest neighbor search.

    Three-phase workflow:
      1. train()  — K-means clustering to partition the vector space into
                    `nlist` Voronoi cells (cluster centroids).
      2. add()    — Assign each database vector to its nearest centroid and
                    store it in the corresponding inverted list.
      3. search() — For a query, probe only `nprobe` nearest cells instead of
                    the entire dataset, trading recall for speed.

    Accuracy vs. speed knob: increasing `nprobe` toward `nlist` converges to
    exact brute-force search; setting nprobe=1 is fastest but misses some
    neighbors that lie near cell boundaries.
    """

    def __init__(self, dim: int, nlist: int):
        """
        dim   — dimensionality of vectors
        nlist — number of Voronoi cells (cluster centroids)
        """
        self.dim = dim
        self.nlist = nlist
        self.centroids: np.ndarray | None = None   # shape (nlist, dim)
        self.inverted_lists: dict[int, list] = {}  # cell_id -> [(vector, id), ...]
        self.is_trained = False

    # ------------------------------------------------------------------
    # Phase 1: Train
    # ------------------------------------------------------------------

    def train(self, vectors: np.ndarray, n_iter: int = 25) -> None:
        """
        Run K-means on `vectors` to find `nlist` centroids.
        The centroids define the Voronoi partition of the space.
        """
        n = len(vectors)
        if n < self.nlist:
            raise ValueError(f"Need at least {self.nlist} training vectors, got {n}")

        vectors = vectors.astype(float)

        # Initialize: pick nlist random vectors as starting centroids
        rng = np.random.default_rng(seed=0)
        centroids = vectors[rng.choice(n, self.nlist, replace=False)].copy()

        for i in range(n_iter):
            assignments = self._assign(vectors, centroids)

            new_centroids = np.zeros_like(centroids)
            for c in range(self.nlist):
                members = vectors[assignments == c]
                if len(members):
                    new_centroids[c] = members.mean(axis=0)
                else:
                    # Empty cluster — reinitialize to a random data point
                    new_centroids[c] = vectors[rng.integers(n)]

            if np.allclose(centroids, new_centroids, atol=1e-6):
                print(f"  K-means converged after {i + 1} iterations.")
                break
            centroids = new_centroids

        self.centroids = centroids
        self.inverted_lists = {c: [] for c in range(self.nlist)}
        self.is_trained = True

    # ------------------------------------------------------------------
    # Phase 2: Add
    # ------------------------------------------------------------------

    def add(self, vectors: np.ndarray, ids: list) -> None:
        """
        Assign each vector to its nearest centroid and store it in the
        corresponding inverted list.
        """
        if not self.is_trained:
            raise RuntimeError("Call train() before add()")

        vectors = vectors.astype(float)
        assignments = self._assign(vectors, self.centroids)

        for vec, vid, cell in zip(vectors, ids, assignments):
            self.inverted_lists[cell].append((vec, vid))

    # ------------------------------------------------------------------
    # Phase 3: Search
    # ------------------------------------------------------------------

    def search(
        self, query: np.ndarray, k: int, nprobe: int = 1
    ) -> list[tuple[float, int]]:
        """
        Return the k approximate nearest neighbors of `query`.

        nprobe — number of Voronoi cells to scan.
                 nprobe=1  → fastest, lowest recall
                 nprobe=nlist → exact brute-force, highest recall
        """
        if not self.is_trained:
            raise RuntimeError("Call train() before search()")

        query = query.astype(float)

        # Step A: find the nprobe closest centroids
        dist_to_centroids = np.linalg.norm(self.centroids - query, axis=1)
        probe_cells = np.argsort(dist_to_centroids)[:nprobe]

        # Step B: collect candidate (distance, id) pairs from those cells
        candidates: list[tuple[float, int]] = []
        for cell in probe_cells:
            for vec, vid in self.inverted_lists[cell]:
                dist = float(np.linalg.norm(vec - query))
                candidates.append((dist, vid))

        # Step C: return top-k by ascending distance
        candidates.sort(key=lambda x: x[0])
        return candidates[:k]

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def stats(self) -> None:
        sizes = [len(v) for v in self.inverted_lists.values()]
        total = sum(sizes)
        print(f"\nIVF Index statistics:")
        print(f"  nlist  = {self.nlist}")
        print(f"  total  = {total} vectors indexed")
        print(f"  list sizes: min={min(sizes)}, max={max(sizes)}, "
              f"avg={np.mean(sizes):.1f}")

    def _assign(self, vectors: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        """
        Assign each row of `vectors` to its nearest centroid.

        Uses the identity  ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b
        for an efficient batched computation instead of a Python loop.
        """
        a2 = (vectors ** 2).sum(axis=1, keepdims=True)   # (n, 1)
        b2 = (centroids ** 2).sum(axis=1)                 # (nlist,)
        ab = vectors @ centroids.T                         # (n, nlist)
        dist_sq = a2 + b2 - 2 * ab                        # (n, nlist)  — broadcast
        return np.argmin(dist_sq, axis=1)
