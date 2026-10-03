import numpy as np
from typing import List, Dict, Any

def compute_precision_at_k(retrieved_items: List[Dict[str, Any]], query_class: str, k: int) -> float:
    """
    Computes Precision@K:
    Fraction of retrieved items in top-K that share the same class as the query.
    """
    if not retrieved_items or k <= 0:
        return 0.0
    slice_items = retrieved_items[:k]
    matches = sum(1 for item in slice_items if item.get("class_name") == query_class)
    return matches / float(len(slice_items))

def compute_average_precision(retrieved_items: List[Dict[str, Any]], query_class: str, k: int = 10) -> float:
    """
    Computes Average Precision (AP@K).
    """
    if not retrieved_items or k <= 0:
        return 0.0
    
    slice_items = retrieved_items[:k]
    running_correct = 0
    precisions = []

    for rank, item in enumerate(slice_items, start=1):
        if item.get("class_name") == query_class:
            running_correct += 1
            precisions.append(running_correct / float(rank))

    if not precisions:
        return 0.0
    return sum(precisions) / float(len(precisions))
