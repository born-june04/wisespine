"""
Anatomy transition rules (paper-ready, code-ready).

This module is the "single source of truth" for:
- Vertebra label indexing (C1..S1)
- validTransition rules used by NVC/strictNVC-style structural metrics

We intentionally keep the rules minimal and unambiguous:
- Adjacent vertebrae in a valid spine chain must be consecutive levels.

Notes on direction:
- Some pipelines order vertebrae superior→inferior, others inferior→superior.
- For structure validity we often want to ignore direction and only require adjacency.
  Use `consider_direction=True` if you want to enforce a specific direction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


def vertebra_names() -> list[str]:
    names: list[str] = []
    names += [f"C{i}" for i in range(1, 8)]
    names += [f"T{i}" for i in range(1, 13)]
    names += [f"L{i}" for i in range(1, 6)]
    names += ["S1"]
    return names


def name_to_level_id_1based() -> dict[str, int]:
    """
    Returns mapping {C1:1, ..., S1:25}
    """
    return {name: i + 1 for i, name in enumerate(vertebra_names())}


def level_id_1based_to_name() -> dict[int, str]:
    m = name_to_level_id_1based()
    return {v: k for k, v in m.items()}


def is_valid_transition(a: int, b: int, *, consider_direction: bool = False) -> bool:
    """
    validTransition(ŷ_k, ŷ_{k+1})

    If consider_direction=False (default):
      - undirected adjacency: abs(b-a) == 1

    If consider_direction=True:
      - directed adjacency: b == a+1 (i.e., increasing level id)
    """
    if consider_direction:
        return b == a + 1
    return abs(b - a) == 1


def nvc_transition(seq: Sequence[int], *, consider_direction: bool = False) -> float:
    """
    NVC as "fraction of adjacent pairs satisfying validTransition".
    This matches the formal definition in Spine_Field_Assembly.md (Section 13).
    """
    if len(seq) < 2:
        return 0.0
    ok = 0
    for a, b in zip(seq[:-1], seq[1:]):
        ok += int(is_valid_transition(int(a), int(b), consider_direction=consider_direction))
    return ok / (len(seq) - 1)


def strict_nvc_transition(seq: Sequence[int], *, consider_direction: bool = False) -> float:
    """
    strict NVC = (length of longest contiguous subsequence with no violated transitions) / (N-1)

    Implementation note:
    - We compute the longest run of consecutive valid transitions.
    - Returns 0 for sequences shorter than 2.
    """
    if len(seq) < 2:
        return 0.0
    best = 0
    cur = 0
    for a, b in zip(seq[:-1], seq[1:]):
        if is_valid_transition(int(a), int(b), consider_direction=consider_direction):
            cur += 1
        else:
            best = max(best, cur)
            cur = 0
    best = max(best, cur)
    return best / (len(seq) - 1)


@dataclass(frozen=True)
class TransitionViolation:
    index: int  # violation happens between seq[index] -> seq[index+1]
    a: int
    b: int


def transition_violations(seq: Sequence[int], *, consider_direction: bool = False) -> list[TransitionViolation]:
    out: list[TransitionViolation] = []
    for i, (a, b) in enumerate(zip(seq[:-1], seq[1:])):
        if not is_valid_transition(int(a), int(b), consider_direction=consider_direction):
            out.append(TransitionViolation(index=i, a=int(a), b=int(b)))
    return out


def duplicates(seq: Iterable[int]) -> dict[int, int]:
    """
    Returns {label: count} for labels that appear more than once.
    """
    counts: dict[int, int] = {}
    for x in seq:
        xi = int(x)
        counts[xi] = counts.get(xi, 0) + 1
    return {k: v for k, v in counts.items() if v > 1}


