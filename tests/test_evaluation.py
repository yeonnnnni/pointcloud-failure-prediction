"""검토 예산·동점·임계값·ID 누출의 계산 회귀 검사."""

import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from pfp.evaluation import (
    FEATURE_SETS, curves, paired_object_bootstrap, ranking_order, review_stats,
    validate_table, validation_f1_threshold,
)


class EvaluationTests(unittest.TestCase):
    def test_budget_ceil_and_ties(self):
        y = np.array([0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0])
        scores = np.array([0.9, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])
        stats = review_stats(y, scores, 0.1)
        self.assertEqual(stats["n_reviewed"], 2)
        self.assertEqual(stats["n_captured_errors"], 1)
        self.assertEqual(stats["n_total_errors"], 3)
        self.assertAlmostEqual(stats["capture_rate"], 1 / 3)
        self.assertEqual(ranking_order(scores).tolist()[:2], [0, 1])

    def test_threshold_reproduces_maximum_validation_f1(self):
        y = np.array([1, 0, 1, 0, 0, 1])
        scores = np.array([0.8, 0.8, 0.5, 0.2, 0.1, 0.1])
        threshold, maximum = validation_f1_threshold(y, scores)
        direct = [f1_score(y, scores >= t) for t in np.unique(scores)]
        self.assertAlmostEqual(maximum, max(direct))
        self.assertAlmostEqual(f1_score(y, scores >= threshold), max(direct))

    def test_risk_curve_excludes_high_scores(self):
        review, risk = curves(np.array([1, 1, 0, 0]), np.array([0.9, 0.8, 0.2, 0.1]), "x")
        point = next(row for row in risk if row["n_excluded"] == 2)
        self.assertEqual(point["coverage"], 0.5)
        self.assertEqual(point["risk"], 0)
        self.assertEqual(review[-1]["capture_rate"], 1)

    def test_validation_rejects_object_leakage(self):
        table = pd.DataFrame({"object_id": [0, 1, 0, 3], "corruption": ["a", "a", "b", "b"],
                              "severity": [1] * 4, "split": ["train", "train", "validation", "validation"],
                              "true_label": [0] * 4, "pred_label": [0, 1, 0, 1], "error": [0, 1, 0, 1]})
        for column in FEATURE_SETS["combined"]:
            table[column] = 0.5
        with self.assertRaisesRegex(ValueError, "여러 split"):
            validate_table(table, pilot=True)

    def test_validation_rejects_inconsistent_target(self):
        table = pd.DataFrame({"object_id": [0, 1, 2, 3], "corruption": ["a"] * 4,
                              "severity": [1] * 4, "split": ["train", "train", "validation", "validation"],
                              "true_label": [0] * 4, "pred_label": [0, 1, 0, 1], "error": [1, 1, 0, 1]})
        for column in FEATURE_SETS["combined"]:
            table[column] = 0.5
        with self.assertRaisesRegex(ValueError, "정답/예측"):
            validate_table(table, pilot=True)

    def test_paired_bootstrap_identical_models_have_zero_difference(self):
        # 각 원본 ID는 조건 2개의 묶음이며 두 모델에 같은 재표집을 적용한다.
        table = pd.DataFrame({"object_id": [0, 0, 1, 1, 2, 2], "error": [0, 1, 0, 1, 0, 1]})
        scores = np.array([0.2, 0.8, 0.3, 0.6, 0.4, 0.7])
        result = paired_object_bootstrap(table, scores, scores, n_bootstrap=20)
        self.assertEqual(result["n_valid"], 20)
        self.assertEqual(result["n_objects"], 3)
        for metric in result["metrics"].values():
            self.assertEqual(metric["mean_delta"], 0)
            self.assertEqual(metric["ci95_low"], 0)
            self.assertEqual(metric["ci95_high"], 0)

    def test_bootstrap_ties_use_canonical_keys(self):
        # 재표집된 행 순서가 바뀌어도 동일 점수의 선택은 원본 키 순서를 따른다.
        y = np.array([1, 0, 1, 0])
        scores = np.repeat(0.5, 4)
        canonical_indices = np.array([2, 0, 3, 1])
        stats = review_stats(y, scores, 0.5, canonical_indices)
        self.assertEqual(stats["n_reviewed"], 2)
        self.assertEqual(stats["n_captured_errors"], 0)


if __name__ == "__main__":
    unittest.main()
