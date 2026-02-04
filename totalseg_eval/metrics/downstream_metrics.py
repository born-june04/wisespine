"""
Downstream-aware metrics.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence


def cobb_angle_error(gt_angle: float, pred_angle: float) -> float:
    """
    Absolute Cobb angle error in degrees.
    """
    return abs(float(pred_angle) - float(gt_angle))


def level_conditioned_failure_rate(
    level_errors: Mapping[int, Iterable[bool]],
) -> dict:
    """
    Compute failure rate per vertebra level.

    Args:
        level_errors: dict[level] -> iterable of booleans (True = error).
    """
    rates = {}
    for level, errs in level_errors.items():
        errs = list(errs)
        if not errs:
            rates[int(level)] = 0.0
            continue
        rates[int(level)] = sum(bool(e) for e in errs) / len(errs)
    return rates


