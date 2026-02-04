"""
Ordering and label consistency metrics.
"""

from __future__ import annotations

from typing import Iterable, Sequence


def neighbor_vertebra_consistency(
    gt_order: Sequence[int],
    pred_order: Sequence[int],
    *,
    consider_direction: bool = True,
) -> float:
    """
    Fraction of GT adjacent pairs that are adjacent in prediction.

    If consider_direction is True, adjacency must preserve order (a,b).
    If False, adjacency is undirected (a,b) or (b,a).
    """
    gt_pairs = []
    for i in range(len(gt_order) - 1):
        gt_pairs.append((gt_order[i], gt_order[i + 1]))

    if not gt_pairs:
        return 0.0

    pred_index = {label: idx for idx, label in enumerate(pred_order)}
    correct = 0
    for a, b in gt_pairs:
        if a not in pred_index or b not in pred_index:
            continue
        ia = pred_index[a]
        ib = pred_index[b]
        if abs(ia - ib) == 1:
            if consider_direction and ia > ib:
                continue
            correct += 1

    return correct / len(gt_pairs)


def vertebra_level_accuracy(gt_labels: Sequence[int], pred_labels: Sequence[int]) -> float:
    """
    Per-vertebra level accuracy for aligned label sequences.
    """
    if len(gt_labels) == 0:
        return 0.0
    if len(gt_labels) != len(pred_labels):
        raise ValueError("gt_labels and pred_labels must have the same length.")
    correct = sum(int(g == p) for g, p in zip(gt_labels, pred_labels))
    return correct / len(gt_labels)


def spine_ordering_edit_distance(gt_order: Sequence[int], pred_order: Sequence[int]) -> float:
    """
    Normalized edit distance between GT and predicted ordering.
    """
    if len(gt_order) == 0:
        return 0.0
    dist = _levenshtein_distance(gt_order, pred_order)
    return dist / max(1, len(gt_order))


def _levenshtein_distance(a: Sequence[int], b: Sequence[int]) -> int:
    """
    Classic Levenshtein distance for sequences.
    """
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


