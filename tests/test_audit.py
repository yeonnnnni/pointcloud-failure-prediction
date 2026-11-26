"""독립 감사의 수학적 기준과 입력 검증."""

import numpy as np
import pytest

from pfp.audit import CORRUPTIONS, condition_scope, geometry_reference, optimum_f1


def test_bruteforce_geometry_keeps_duplicate_points():
    points = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [1, 0, 0]], dtype=float)
    np.testing.assert_allclose(geometry_reference(points), [1, 0, 0, 1, 0, 0, 0.5, 0.5])


def test_bruteforce_geometry_handles_zero_variance():
    np.testing.assert_array_equal(geometry_reference(np.zeros((3, 3))), np.zeros(8))


def test_bruteforce_geometry_rejects_nonfinite():
    with pytest.raises(AssertionError, match="비유한"):
        geometry_reference(np.array([[0, 0, 0], [1, np.nan, 1]]))


def test_independent_validation_threshold_handles_ties():
    threshold, f1 = optimum_f1(np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.7, 0.1]))
    assert threshold == 0.7
    assert f1 == 0.8


def test_default_audit_still_requires_all_75_conditions():
    conditions = {(c, s) for c in CORRUPTIONS for s in range(1, 6)}
    assert not condition_scope(conditions, pilot=False)["partial"]
    conditions.remove(("gaussian", 3))
    with pytest.raises(AssertionError, match="조건 수 불일치"):
        condition_scope(conditions, pilot=False)


def test_partial_audit_requires_explicit_exact_count_and_warns():
    conditions = {(c, s) for c in CORRUPTIONS for s in range(1, 6)} - {("gaussian", 3)}
    scope = condition_scope(conditions, pilot=False, expected_conditions=74)
    assert scope["partial"]
    assert scope["expected_conditions"] == 74
    assert scope["scope_warning"]
    assert scope["missing_conditions"] == [{"corruption": "gaussian", "severity": 3}]
    with pytest.raises(AssertionError, match="조건 수 불일치"):
        condition_scope(conditions, pilot=False, expected_conditions=73)


def test_pilot_audit_keeps_exact_four_conditions():
    conditions = {(c, s) for c in ["gaussian", "cutout"] for s in [1, 5]}
    assert condition_scope(conditions, pilot=True)["expected_conditions"] == 4
    assert condition_scope(conditions, pilot=True, expected_conditions=4)["scope_warning"]
    with pytest.raises(AssertionError, match="파일럿"):
        condition_scope(conditions - {("cutout", 5)}, pilot=True)
