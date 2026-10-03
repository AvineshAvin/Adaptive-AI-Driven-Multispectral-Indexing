import time
import numpy as np
from typing import List, Dict, Any, Tuple

def kmeans_clustering_numpy(X: np.ndarray, k: int, max_iter: int = 40, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pure NumPy implementation of spherical K-Means clustering.
    Avoids external Cython DLL dependencies and Windows AppControl blocking.
    """
    np.random.seed(seed)
    n_samples, n_features = X.shape
    k = min(k, n_samples)

    # Initialize centroids randomly from data
    init_indices = np.random.choice(n_samples, size=k, replace=False)
    centroids = X[init_indices].copy()

    # Normalize centroids
    c_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    c_norms[c_norms == 0] = 1e-7
    centroids = centroids / c_norms

    labels = np.zeros(n_samples, dtype=np.int32)

    for _ in range(max_iter):
        # Compute cosine similarity: (N, K)
        sims = np.dot(X, centroids.T)
        new_labels = np.argmax(sims, axis=1)

        if np.array_equal(new_labels, labels) and _ > 0:
            break
        labels = new_labels

        new_centroids = np.zeros_like(centroids)
        for c_id in range(k):
            mask = (labels == c_id)
            if np.any(mask):
                new_centroids[c_id] = np.mean(X[mask], axis=0)
            else:
                new_centroids[c_id] = X[np.random.choice(n_samples)]

        c_norms = np.linalg.norm(new_centroids, axis=1, keepdims=True)
        c_norms[c_norms == 0] = 1e-7
        centroids = new_centroids / c_norms

    return centroids, labels

class StaticInvertedIndex:
    """
    Baseline 2: Static Inverted File Index (IVF).
    Clusters feature representations into K visual centroids.
    Constructs inverted posting lists mapping clusters to item indices.
    Retrieval only probes the top-W closest clusters.
    """
    def __init__(self, ids: List[str], features: np.ndarray, metadata: List[Dict[str, Any]], 
                 n_clusters: int = 16, n_probe: int = 3):
        self.ids = ids
        self.features = features
        self.metadata = metadata
        self.total_count = len(ids)
        self.n_clusters = min(n_clusters, max(2, self.total_count // 20))
        self.n_probe = min(n_probe, self.n_clusters)

        # Train cluster centroids with pure NumPy spherical K-means
        self.centroids, cluster_assignments = kmeans_clustering_numpy(
            self.features, k=self.n_clusters, max_iter=30
        )

        # Build inverted posting lists
        self.posting_lists: Dict[int, List[int]] = {c: [] for c in range(self.n_clusters)}
        for idx, c in enumerate(cluster_assignments):
            self.posting_lists[int(c)].append(idx)

    def search(self, query_feat: np.ndarray, top_k: int = 10, n_probe: int = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        start_time = time.perf_counter()
        
        q = query_feat.flatten()
        q_norm = np.linalg.norm(q)
        if q_norm > 1e-7:
            q = q / q_norm

        probe_count = n_probe if n_probe is not None else self.n_probe

        # 1. Identify closest cluster centroids
        centroid_sims = np.dot(self.centroids, q)
        probed_clusters = np.argsort(-centroid_sims)[:probe_count]

        # 2. Collect candidate items from inverted posting lists
        candidate_indices = []
        for c in probed_clusters:
            candidate_indices.extend(self.posting_lists[int(c)])

        candidates_evaluated = len(candidate_indices)
        if candidates_evaluated == 0:
            candidate_indices = list(range(self.total_count))
            candidates_evaluated = self.total_count

        cand_indices_arr = np.array(candidate_indices, dtype=np.int32)
        cand_features = self.features[cand_indices_arr]

        # 3. Compute cosine similarities among candidates
        sims = np.dot(cand_features, q)
        
        if top_k >= len(cand_indices_arr):
            top_sub_idx = np.argsort(-sims)
        else:
            top_sub_idx = np.argpartition(-sims, top_k)[:top_k]
            top_sub_idx = top_sub_idx[np.argsort(-sims[top_sub_idx])]

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        results = []
        for s_idx in top_sub_idx:
            orig_idx = cand_indices_arr[s_idx]
            results.append({
                "id": self.ids[orig_idx],
                "score": float(sims[s_idx]),
                "class_name": self.metadata[orig_idx]["class_name"],
                "path": self.metadata[orig_idx]["path"],
                "tier": "STATIC_IVF"
            })

        stats = {
            "search_time_ms": elapsed_ms,
            "candidates_evaluated": candidates_evaluated,
            "search_space_ratio": candidates_evaluated / max(1, self.total_count),
            "probed_clusters": int(probe_count),
            "mode": "Static Inverted Index"
        }
        return results, stats
