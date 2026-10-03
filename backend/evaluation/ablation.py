import numpy as np
from typing import List, Dict, Any
from .metrics import compute_precision_at_k, compute_average_precision
from ..retrieval.engine import RetrievalEngine

class AblationStudyRunner:
    """
    Executes systematic ablation studies to evaluate individual contributions
    of the framework's components.
    """
    def __init__(self, engine: RetrievalEngine):
        self.engine = engine

    def run_ablation(self, query_indices: List[int], top_k: int = 10) -> Dict[str, Any]:
        experiments = [
            {"id": "full_proposed", "name": "Full Adaptive Framework (Proposed)", "alpha": None, "use_gating": True},
            {"id": "spectral_only", "name": "Spectral Branch Only (alpha=1.0)", "alpha": 1.0, "use_gating": True},
            {"id": "spatial_only", "name": "Spatial Branch Only (alpha=0.0)", "alpha": 0.0, "use_gating": True},
            {"id": "no_confidence_gating", "name": "Without Confidence Early-Stopping", "alpha": None, "use_gating": False},
        ]

        ablation_results = []

        for exp in experiments:
            p1_list, p10_list, map_list, lat_list, red_list = [], [], [], [], []

            for q_idx in query_indices:
                q_meta = self.engine.tiered_index.metadata[q_idx]
                q_class = q_meta["class_name"]
                q_feat_dict = {
                    "spectral_features": self.engine.tiered_index.spectral_features[q_idx],
                    "spatial_features": self.engine.tiered_index.spatial_features[q_idx],
                    "fused_features": self.engine.tiered_index.features[q_idx],
                    "alpha": 0.55 if exp["alpha"] is None else exp["alpha"]
                }

                if exp["use_gating"]:
                    res, stats = self.engine.retrieve_adaptive(
                        q_feat_dict["spectral_features"],
                        q_feat_dict["spatial_features"],
                        q_feat_dict["fused_features"],
                        alpha=q_feat_dict["alpha"],
                        top_k=top_k + 1
                    )
                else:
                    # Scan without early stopping
                    res, stats = self.engine.retrieve(q_feat_dict, mode="flat", top_k=top_k + 1)

                filtered = [it for it in res if it.get("id") != q_meta["id"]][:top_k]
                p1_list.append(compute_precision_at_k(filtered, q_class, 1))
                p10_list.append(compute_precision_at_k(filtered, q_class, 10))
                map_list.append(compute_average_precision(filtered, q_class, 10))
                lat_list.append(stats["search_time_ms"])
                red_list.append(stats["search_space_reduction_pct"])

            ablation_results.append({
                "experiment_id": exp["id"],
                "name": exp["name"],
                "precision_at_1": round(float(np.mean(p1_list)), 4),
                "precision_at_10": round(float(np.mean(p10_list)), 4),
                "mAP": round(float(np.mean(map_list)), 4),
                "avg_search_time_ms": round(float(np.mean(lat_list)), 3),
                "search_space_reduction_pct": round(float(np.mean(red_list)), 2)
            })

        return {
            "num_queries": len(query_indices),
            "ablations": ablation_results
        }
