"""
SOED: Spine Ordering Edit Distance (weighted edit distance).

We keep this separate from model code to make the metric definition explicit and reusable.
"""

from __future__ import annotations

from typing import Sequence


def weighted_edit_distance(
    a: Sequence[int],
    b: Sequence[int],
    *,
    w_ins: float = 1.0,
    w_del: float = 1.0,
    w_sub: float = 1.0,
) -> float:
    """
    Weighted Levenshtein distance for sequences.

    Args:
      a: GT sequence
      b: predicted sequence
      w_ins: insertion cost
      w_del: deletion cost
      w_sub: substitution cost (0 if equal)
    """
    n = len(a)
    m = len(b)
    if n == 0:
        return float(m) * float(w_ins)
    if m == 0:
        return float(n) * float(w_del)

    # dp[i][j] = min cost to transform a[:i] -> b[:j]
    prev = [0.0] * (m + 1)
    for j in range(1, m + 1):
        prev[j] = prev[j - 1] + float(w_ins)

    for i in range(1, n + 1):
        cur = [0.0] * (m + 1)
        cur[0] = prev[0] + float(w_del)
        ai = int(a[i - 1])
        for j in range(1, m + 1):
            bj = int(b[j - 1])
            sub_cost = 0.0 if ai == bj else float(w_sub)
            cur[j] = min(
                prev[j] + float(w_del),        # delete ai
                cur[j - 1] + float(w_ins),     # insert bj
                prev[j - 1] + sub_cost,        # substitute
            )
        prev = cur
    return float(prev[m])


def soed(
    gt_order: Sequence[int],
    pred_order: Sequence[int],
    *,
    w_ins: float = 1.0,
    w_del: float = 1.0,
    w_sub: float = 1.0,
    normalize_by: str = "gt",
) -> float:
    """
    Normalized weighted edit distance.

    normalize_by:
      - "gt": divide by len(gt_order) (paper default)
      - "max": divide by max(len(gt), len(pred))
      - "none": no normalization
    """
    dist = weighted_edit_distance(gt_order, pred_order, w_ins=w_ins, w_del=w_del, w_sub=w_sub)
    if normalize_by == "none":
        return dist
    if normalize_by == "max":
        denom = max(1, max(len(gt_order), len(pred_order)))
        return dist / denom
    denom = max(1, len(gt_order))
    return dist / denom


