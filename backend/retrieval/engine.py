import time
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from ..multispectral.feature_extractor import DualBranchFeatureExtractor
from ..indexing.adaptive_tiers import AdaptiveTieredIndex
from ..indexing.tier_manager import DynamicTierManager
from ..indexing.inverted_index import StaticInvertedIndex
from ..indexing.exhaustive_index import ExhaustiveIndex
from .confidence import calculate_retrieval_confidence
from ..config import HOT_CONFIDENCE_THRESHOLD, WARM_CONFIDENCE_THRESHOLD

class RetrievalEngine:
    """
    Unified Remote Sensing Image Retrieval Engine.
    Coordinates between:
      1. Proposed Adaptive Multi-Tier Retrieval (HOT-WARM-COLD)
      2. Baseline Static Inverted Index (IVF)
      3. Baseline Exhaustive Flat Search
    """
    def __init__(self,
                 tiered_index: AdaptiveTieredIndex,
                 tier_manager: DynamicTierManager,
                 inverted_index: StaticInvertedIndex,
                 exhaustive_index: ExhaustiveIndex,
                 extractor: DualBranchFeatureExtractor = None):
        self.tiered_index = tiered_index
        self.tier_manager = tier_manager
        self.inverted_index = inverted_index
        self.exhaustive_index = exhaustive_index
        self.extractor = extractor if extractor is not None else DualBranchFeatureExtractor()
        self.query_history: List[Dict[str, Any]] = []

    def retrieve_adaptive(self,
                          query_spec: np.ndarray,
                          query_spat: np.ndarray,
                          query_fused: np.ndarray,
                          alpha: float,
                          top_k: int = 10,
                          early_stop_threshold: Optional[float] = None,
                          early_stop_policy: str = "recall_aware") -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Adaptive AI-Driven Multi-Tier Search:
          Tier 1: HOT Tier (In-Memory BLAS-vectorized search)
          Tier 2: WARM Tier (Selective partition expansion)
          Tier 3: COLD Tier (Compressed storage fallback)
        
        Confidence-Aware Early Stopping:
          - 'recall_aware' (Recommended): Evaluates top-1 precision & candidate density across top-k.
            Escalates to WARM if candidate support is sparse, preventing Top-K accuracy drop.
          - 'aggressive': Checks top-1 score alone; delivers maximum 88% search pruning.
        """
        start_time = time.perf_counter()
        candidates_searched = 0
        tier_path = []
        tau = early_stop_threshold if early_stop_threshold is not None else HOT_CONFIDENCE_THRESHOLD

        hot_arr = getattr(self.tiered_index, 'hot_arr', None)
        if hot_arr is None or len(hot_arr) == 0:
            hot_arr = np.array(list(self.tiered_index.hot_indices), dtype=np.int32)

        candidates_searched += len(hot_arr)
        tier_path.append(f"HOT ({len(hot_arr)} cands)")

        early_stopped = False
        tier_status = "COLD_TIER_FALLBACK"

        # 1. Search HOT Tier using high-speed BLAS dot product
        comb_hot, spec_hot, spat_hot = self.tiered_index.search_candidates(
            hot_arr, query_spec, query_spat, alpha
        )

        hot_order = np.argsort(-comb_hot)
        top_hot_scores = [float(comb_hot[idx]) for idx in hot_order[:top_k]]
        agg_conf, conf_list = calculate_retrieval_confidence(
            top_hot_scores, policy=early_stop_policy, top_k=top_k
        )

        # Early-Stopping Condition on HOT Tier
        # In recall_aware mode, require at least top_k candidates and agg_conf >= tau
        if len(hot_arr) >= top_k and agg_conf >= tau:
            early_stopped = True
            tier_status = "HOT_CACHE_HIT (Early Termination)"
            selected_indices = hot_arr[hot_order[:top_k * 2]]
            selected_scores = comb_hot[hot_order[:top_k * 2]]
            selected_spec = spec_hot[hot_order[:top_k * 2]]
            selected_spat = spat_hot[hot_order[:top_k * 2]]
            selected_tiers = ["HOT"] * len(selected_indices)
        else:
            # 2. Escalate to WARM Tier (recall recovery)
            warm_arr = getattr(self.tiered_index, 'warm_arr', None)
            if warm_arr is None or len(warm_arr) == 0:
                warm_arr = np.array(list(self.tiered_index.warm_indices), dtype=np.int32)

            candidates_searched += len(warm_arr)
            tier_path.append(f"WARM ({len(warm_arr)} cands)")

            comb_warm, spec_warm, spat_warm = self.tiered_index.search_candidates(
                warm_arr, query_spec, query_spat, alpha
            )

            # Fuse HOT + WARM candidates
            all_sims = np.concatenate([comb_hot, comb_warm])
            all_spec = np.concatenate([spec_hot, spec_warm])
            all_spat = np.concatenate([spat_hot, spat_warm])
            all_indices = np.concatenate([hot_arr, warm_arr])
            all_tiers = ["HOT"] * len(hot_arr) + ["WARM"] * len(warm_arr)

            warm_order = np.argsort(-all_sims)
            top_hw_scores = [float(all_sims[idx]) for idx in warm_order[:top_k]]
            agg_conf, conf_list = calculate_retrieval_confidence(
                top_hw_scores, policy=early_stop_policy, top_k=top_k
            )

            warm_tau = max(0.60, tau * 0.85)
            if len(all_sims) >= top_k and agg_conf >= warm_tau:
                early_stopped = True
                tier_status = "WARM_ESCALATION_RESOLVED"
                selected_indices = all_indices[warm_order[:top_k * 2]]
                selected_scores = all_sims[warm_order[:top_k * 2]]
                selected_spec = all_spec[warm_order[:top_k * 2]]
                selected_spat = all_spat[warm_order[:top_k * 2]]
                selected_tiers = [all_tiers[idx] for idx in warm_order[:top_k * 2]]
            else:
                # 3. Escalate to COLD Tier as ultimate fallback
                cold_arr = getattr(self.tiered_index, 'cold_arr', None)
                if cold_arr is None or len(cold_arr) == 0:
                    cold_arr = np.array(list(self.tiered_index.cold_indices), dtype=np.int32)

                candidates_searched += len(cold_arr)
                tier_path.append(f"COLD ({len(cold_arr)} cands)")

                comb_cold, spec_cold, spat_cold = self.tiered_index.search_candidates(
                    cold_arr, query_spec, query_spat, alpha
                )

                all_sims = np.concatenate([all_sims, comb_cold])
                all_spec = np.concatenate([all_spec, spec_cold])
                all_spat = np.concatenate([all_spat, spat_cold])
                all_indices = np.concatenate([all_indices, cold_arr])
                all_tiers = all_tiers + ["COLD"] * len(cold_arr)

                cold_order = np.argsort(-all_sims)
                tier_status = "COLD_TIER_FALLBACK"
                selected_indices = all_indices[cold_order[:top_k * 2]]
                selected_scores = all_sims[cold_order[:top_k * 2]]
                selected_spec = all_spec[cold_order[:top_k * 2]]
                selected_spat = all_spat[cold_order[:top_k * 2]]
                selected_tiers = [all_tiers[idx] for idx in cold_order[:top_k * 2]]

        # Construct final top-k result objects
        unique_results = []
        seen_ids = set()
        for i in range(len(selected_indices)):
            orig_idx = int(selected_indices[i])
            img_id = self.tiered_index.ids[orig_idx]
            if img_id not in seen_ids:
                seen_ids.add(img_id)
                score_val = float(selected_scores[i])
                meta_item = self.tiered_index.metadata[orig_idx]
                unique_results.append({
                    "idx": orig_idx,
                    "id": img_id,
                    "score": score_val,
                    "spectral_score": float(selected_spec[i]),
                    "spatial_score": float(selected_spat[i]),
                    "confidence": float(score_val),
                    "class_name": meta_item["class_name"],
                    "path": meta_item["path"],
                    "tier": selected_tiers[i]
                })
            if len(unique_results) == top_k:
                break

        # Re-compute per-candidate confidence on final list
        final_scores = [r["score"] for r in unique_results]
        final_agg_conf, final_conf_list = calculate_retrieval_confidence(
            final_scores, policy=early_stop_policy, top_k=top_k
        )
        for i, c in enumerate(final_conf_list):
            if i < len(unique_results):
                unique_results[i]["confidence"] = c

        elapsed_ms = (time.perf_counter() - start_time) * 1000.0

        # Self-learning feedback update
        for rank, item in enumerate(unique_results):
            self.tiered_index.record_access(item["idx"], relevance=0.1 if rank < 3 else 0.05)

        total_repo = self.tiered_index.total_count
        reduction_pct = (1.0 - (candidates_searched / max(1, total_repo))) * 100.0

        stats = {
            "search_time_ms": round(elapsed_ms, 3),
            "candidates_evaluated": candidates_searched,
            "total_repository_size": total_repo,
            "search_space_reduction_pct": round(max(0.0, reduction_pct), 2),
            "tier_status": tier_status,
            "tier_path": " -> ".join(tier_path),
            "aggregate_confidence": round(final_agg_conf, 4),
            "alpha_used": round(alpha, 3),
            "early_stopped": early_stopped,
            "early_stop_threshold": round(tau, 3),
            "early_stop_policy": early_stop_policy,
            "mode": "Adaptive AI Multi-Tier (Proposed)"
        }

        return unique_results, stats

    def retrieve(self,
                 query_geotiff_or_features: Any,
                 mode: str = "adaptive",
                 top_k: int = 10,
                 custom_alpha: Optional[float] = None,
                 early_stop_threshold: Optional[float] = None,
                 early_stop_policy: str = "recall_aware",
                 n_probe: Optional[int] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Dispatches search to selected indexing mode:
          - 'adaptive': Proposed multi-tier adaptive framework
          - 'inverted': Baseline static inverted index
          - 'flat': Baseline exhaustive flat search
        """
        if isinstance(query_geotiff_or_features, np.ndarray) and query_geotiff_or_features.ndim == 3:
            extracted = self.extractor.extract_all(query_geotiff_or_features)
            q_spec = extracted["spectral_features"]
            q_spat = extracted["spatial_features"]
            q_fused = extracted["fused_features"]
            alpha = custom_alpha if custom_alpha is not None else extracted["alpha"]
            query_band_sig = extracted["band_signature"]
        else:
            if isinstance(query_geotiff_or_features, dict):
                q_spec = query_geotiff_or_features["spectral_features"]
                q_spat = query_geotiff_or_features["spatial_features"]
                q_fused = query_geotiff_or_features["fused_features"]
                alpha = custom_alpha if custom_alpha is not None else query_geotiff_or_features.get("alpha", 0.55)
                query_band_sig = query_geotiff_or_features.get("band_signature", [])
            else:
                q_fused = query_geotiff_or_features
                q_spec = q_fused
                q_spat = q_fused
                alpha = 0.55
                query_band_sig = []

        if mode == "inverted":
            results, stats = self.inverted_index.search(q_fused, top_k=top_k, n_probe=n_probe)
            scores = [r["score"] for r in results]
            agg_c, c_list = calculate_retrieval_confidence(scores, policy=early_stop_policy, top_k=top_k)
            for i, c in enumerate(c_list):
                results[i]["confidence"] = c
            stats["aggregate_confidence"] = round(agg_c, 4)
            stats["search_space_reduction_pct"] = round((1.0 - stats["search_space_ratio"]) * 100.0, 2)
            stats["n_probe_used"] = n_probe if n_probe is not None else self.inverted_index.n_probe
        elif mode == "flat":
            results, stats = self.exhaustive_index.search(q_fused, top_k=top_k)
            scores = [r["score"] for r in results]
            agg_c, c_list = calculate_retrieval_confidence(scores, policy=early_stop_policy, top_k=top_k)
            for i, c in enumerate(c_list):
                results[i]["confidence"] = c
            stats["aggregate_confidence"] = round(agg_c, 4)
            stats["search_space_reduction_pct"] = 0.0
        else:
            results, stats = self.retrieve_adaptive(
                q_spec, q_spat, q_fused,
                alpha=alpha,
                top_k=top_k,
                early_stop_threshold=early_stop_threshold,
                early_stop_policy=early_stop_policy
            )

        stats["query_band_signature"] = query_band_sig
        return results, stats
