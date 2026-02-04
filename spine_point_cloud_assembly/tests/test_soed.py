import sys
from pathlib import Path

# Ensure project import works when running tests from repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from spine_point_cloud_assembly.utils.soed import soed, weighted_edit_distance  # noqa: E402


def test_weighted_edit_distance_identity():
    assert weighted_edit_distance([1, 2, 3], [1, 2, 3]) == 0.0


def test_weighted_edit_distance_insertion_deletion():
    # insert one
    assert weighted_edit_distance([1, 2, 3], [1, 2, 3, 4]) == 1.0
    # delete one
    assert weighted_edit_distance([1, 2, 3, 4], [1, 2, 3]) == 1.0


def test_weighted_edit_distance_substitution():
    assert weighted_edit_distance([1, 2, 3], [1, 9, 3]) == 1.0
    assert weighted_edit_distance([1, 2, 3], [1, 9, 3], w_sub=2.0) == 2.0


def test_soed_normalization_gt():
    # dist=1, denom=len(gt)=4
    val = soed([1, 2, 3, 4], [1, 2, 3], normalize_by="gt")
    assert abs(val - 0.25) < 1e-8


