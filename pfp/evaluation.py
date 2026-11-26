"""ID 단위 분할을 보존하는 오분류 탐지 평가.

파일럿은 validation에서 동작만 점검하며 최종 test 성능을 의미하지 않는다.
학습형 모델은 train에서 fit하고 validation AP로 설정을 선택한다.
모든 모델은 동일한 (object_id, corruption, severity) 동점 규칙을 사용한다.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SEED = 42
KEYS = ["object_id", "corruption", "severity"]
CONFIDENCE_FEATURES = ["msp", "entropy", "margin"]
GEOMETRY_FEATURES = [
    "pca_ratio_1", "pca_ratio_2", "pca_ratio_3",
    "bbox_x", "bbox_y", "bbox_z", "knn_dist_mean", "knn_dist_std",
]
FEATURE_SETS = {
    "confidence": CONFIDENCE_FEATURES,
    "geometry": GEOMETRY_FEATURES,
    "combined": CONFIDENCE_FEATURES + GEOMETRY_FEATURES,
}


def write_json(path: Path, value: dict | list) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def validate_table(table: pd.DataFrame, pilot: bool = False) -> dict:
    required = KEYS + ["split", "true_label", "pred_label", "error"] + FEATURE_SETS["combined"]
    missing = sorted(set(required) - set(table.columns))
    if missing:
        raise ValueError(f"필수 열 누락: {missing}")
    if table.empty or table[required].isna().any().any():
        raise ValueError("비어 있는 표 또는 누락값이 있습니다.")
    if table.duplicated(KEYS).any():
        raise ValueError("조인 키 중복이 있습니다.")
    if not np.isfinite(table[FEATURE_SETS["combined"]].to_numpy(dtype=float)).all():
        raise ValueError("비유한 feature가 있습니다.")
    if not table["error"].isin([0, 1]).all():
        raise ValueError("error는 0 또는 1이어야 합니다.")
    target = (table["pred_label"] != table["true_label"]).astype(int)
    if not np.array_equal(target, table["error"]):
        raise ValueError("error와 정답/예측 클래스가 일치하지 않습니다.")
    if table.groupby("object_id")["split"].nunique().max() != 1:
        raise ValueError("동일 원본 ID가 여러 split에 존재합니다.")
    if table.groupby("object_id")["true_label"].nunique().max() != 1:
        raise ValueError("동일 원본 ID의 정답 클래스가 조건마다 다릅니다.")
    expected_splits = {"train", "validation"} if pilot else {"train", "validation", "test"}
    actual_splits = set(table["split"])
    if actual_splits != expected_splits:
        raise ValueError(f"split 불일치: {actual_splits}; 기대값 {expected_splits}")
    for split in sorted(expected_splits):
        if table.loc[table["split"] == split, "error"].nunique() != 2:
            raise ValueError(f"{split}에 양성·음성이 모두 필요합니다. 파일럿 ID를 늘리세요.")
    if not table["msp"].between(0, 1).all():
        raise ValueError("MSP가 확률 범위를 벗어났습니다.")
    stats = []
    for split, part in table.groupby("split", sort=True):
        positives = int(part["error"].sum())
        stats.append({"split": split, "n_objects": int(part["object_id"].nunique()),
                      "n_rows": len(part), "n_errors": positives,
                      "n_correct": len(part) - positives, "error_rate": positives / len(part)})
    return {"n_rows": len(table), "n_objects": int(table["object_id"].nunique()),
            "n_conditions": len(table[["corruption", "severity"]].drop_duplicates()),
            "duplicate_keys": 0, "missing_values": 0, "split_id_intersections": 0,
            "feature_columns": FEATURE_SETS, "splits": stats,
            "conditions": table[["corruption", "severity"]].drop_duplicates().sort_values(
                ["corruption", "severity"]).to_dict("records")}


def ranking_order(scores: np.ndarray, tie_keys: np.ndarray | None = None) -> np.ndarray:
    """동점은 이미 정렬된 원본 행 인덱스(공통 조인 키 순서)로 결정한다."""
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("비유한 탐지 점수가 있습니다.")
    if tie_keys is None:
        tie_keys = np.arange(len(scores))
    return np.lexsort((tie_keys, -scores))


def review_stats(y: np.ndarray, scores: np.ndarray, budget: float,
                 tie_keys: np.ndarray | None = None) -> dict:
    if not 0 <= budget <= 1:
        raise ValueError("검토 예산은 [0, 1]이어야 합니다.")
    n = len(y)
    k = int(math.ceil(budget * n))
    detected = int(np.asarray(y)[ranking_order(scores, tie_keys)[:k]].sum())
    total = int(np.asarray(y).sum())
    return {"requested_budget": budget, "n_reviewed": k,
            "actual_review_fraction": k / n, "n_captured_errors": detected,
            "n_total_errors": total, "capture_rate": detected / total if total else None,
            "random_expected_capture": k / n}


def validation_f1_threshold(y: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y, scores)
    denom = precision[:-1] + recall[:-1]
    values = np.divide(2 * precision[:-1] * recall[:-1], denom,
                       out=np.zeros_like(denom), where=denom > 0)
    # F1 동점이면 높은 임계값을 사용한다. test는 이 선택에 사용하지 않는다.
    index = int(np.flatnonzero(values == values.max())[-1])
    return float(thresholds[index]), float(values[index])


def metric_row(name: str, y: np.ndarray, scores: np.ndarray,
               threshold: float, val_f1: float, val_ap: float,
               probability: bool = True) -> dict:
    row = {"model": name, "n_rows": len(y), "n_errors": int(y.sum()),
           "error_prevalence": float(y.mean()), "ap": float(average_precision_score(y, scores)),
           "auroc": float(roc_auc_score(y, scores)),
           "f1": float(f1_score(y, scores >= threshold, zero_division=0)),
           "threshold_from_validation": threshold, "validation_f1": val_f1,
           "validation_ap": val_ap,
           "brier": float(brier_score_loss(y, scores)) if probability else None,
           "calibration": "uncalibrated" if probability else "not_probability_of_error"}
    for q in [0.1, 0.2]:
        result = review_stats(y, scores, q)
        suffix = str(int(q * 100))
        for key in ["capture_rate", "n_reviewed", "n_captured_errors", "n_total_errors",
                    "actual_review_fraction", "random_expected_capture"]:
            row[f"{key}_{suffix}"] = result[key]
    return row


def curves(y: np.ndarray, scores: np.ndarray, name: str) -> tuple[list, list]:
    ordered_y = y[ranking_order(scores)]
    cumulative = np.concatenate([[0], np.cumsum(ordered_y)])
    review, risk = [], []
    n = len(y)
    total = int(y.sum())
    for q in np.linspace(0, 1, 101):
        k = min(n, int(math.ceil(float(q) * n)))
        selected_errors = int(cumulative[k])
        review.append({"model": name, "requested_budget": float(q), "n_reviewed": k,
                       "actual_review_fraction": k / n, "n_captured_errors": selected_errors,
                       "n_total_errors": total, "capture_rate": selected_errors / total,
                       "random_expected_capture": k / n})
        if k < n:
            risk.append({"model": name, "n_excluded": k, "n_retained": n - k,
                         "coverage": (n - k) / n, "n_remaining_errors": total - selected_errors,
                         "risk": (total - selected_errors) / (n - k)})
    return review, risk


def robustness_tables(table: pd.DataFrame, output_dir: Path) -> dict:
    """행 모집단 전체에 대해 PointNet의 손상별 robustness를 별도 집계한다."""
    condition = table.groupby(["corruption", "severity"])["error"].agg(
        n_rows="size", n_errors="sum", error_rate="mean").reset_index()
    condition["accuracy"] = 1 - condition["error_rate"]
    condition.to_csv(output_dir / "robustness_by_condition.csv", index=False)
    per_type = []
    for name, part in table.groupby("corruption", sort=True):
        per_type.append({"corruption": name, "n_rows": len(part),
                         "n_objects": int(part.object_id.nunique()),
                         "n_levels": int(part.severity.nunique()),
                         "accuracy": float(accuracy_score(part.true_label, part.pred_label)),
                         "macro_f1": float(f1_score(part.true_label, part.pred_label,
                                                    labels=np.arange(40), average="macro", zero_division=0)),
                         "error_rate": float(part.error.mean())})
    pd.DataFrame(per_type).to_csv(output_dir / "robustness_by_corruption.csv", index=False)
    table.groupby("true_label")["error"].agg(n_rows="size", n_errors="sum", error_rate="mean").reset_index().to_csv(
        output_dir / "robustness_by_class.csv", index=False)
    table.groupby("error")[CONFIDENCE_FEATURES].agg(["count", "mean", "std", "median"]).to_csv(
        output_dir / "confidence_distribution.csv")
    return {"n_rows": len(table), "n_errors": int(table.error.sum()),
            "accuracy": float(accuracy_score(table.true_label, table.pred_label)),
            "error_rate": float(table.error.mean()),
            "macro_f1": float(f1_score(table.true_label, table.pred_label,
                                       labels=np.arange(40), average="macro", zero_division=0))}


def paired_object_bootstrap(table: pd.DataFrame, scores_a: np.ndarray,
                            scores_b: np.ndarray, n_bootstrap: int = 300) -> dict:
    """각 원본 ID의 모든 조건을 함께 재표집한 B-A 차이의 percentile CI."""
    rng = np.random.default_rng(SEED)
    ids = table.object_id.to_numpy()
    groups = [np.flatnonzero(ids == value) for value in np.unique(ids)]
    y_all = table.error.to_numpy(dtype=int)
    samples = {"ap": [], "capture_rate_10": [], "capture_rate_20": []}
    for _ in range(n_bootstrap):
        indices = np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))])
        y = y_all[indices]
        if np.unique(y).size != 2:
            continue
        a, b = scores_a[indices], scores_b[indices]
        samples["ap"].append(float(average_precision_score(y, b) - average_precision_score(y, a)))
        for q in [0.1, 0.2]:
            left, right = review_stats(y, a, q, indices), review_stats(y, b, q, indices)
            samples[f"capture_rate_{int(q * 100)}"].append(right["capture_rate"] - left["capture_rate"])
    return {"method": "paired object_id bootstrap; percentile 95% CI; fixed fitted detectors",
            "seed": SEED, "n_requested": n_bootstrap, "n_valid": len(samples["ap"]),
            "n_objects": len(groups), "metrics": {
                key: {"mean_delta": float(np.mean(values)),
                      "ci95_low": float(np.quantile(values, 0.025)),
                      "ci95_high": float(np.quantile(values, 0.975))}
                for key, values in samples.items()}}


def evaluate(table: pd.DataFrame, output_dir: str | Path, pilot: bool = False,
             *, n_bootstrap: int = 300, importance_samples: int = 6000,
             n_jobs: int = 8) -> dict:
    started = time.perf_counter()
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    table = table.sort_values(KEYS, kind="mergesort").reset_index(drop=True).copy()
    validation = validate_table(table, pilot)
    write_json(out / "table_validation.json", validation)
    train = table.loc[table.split == "train"].reset_index(drop=True)
    val = table.loc[table.split == "validation"].reset_index(drop=True)
    target_split = "validation" if pilot else "test"
    test = table.loc[table.split == target_split].reset_index(drop=True)
    y_train, y_val, y_test = (part.error.to_numpy(dtype=int) for part in [train, val, test])
    model_dir = out / "models"
    model_dir.mkdir(exist_ok=True)
    metrics, model_choices, review_curves, risk_curves = [], {}, [], []
    predictions = test.copy()
    fitted = {}

    def record(name, val_scores, test_scores, model=None, columns=None, config=None):
        val_scores, test_scores = np.asarray(val_scores), np.asarray(test_scores)
        for scores in [val_scores, test_scores]:
            if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
                raise ValueError(f"{name}: 탐지 확률 유효성 검사 실패")
        threshold, val_f1 = validation_f1_threshold(y_val, val_scores)
        val_ap = float(average_precision_score(y_val, val_scores))
        row = metric_row(name, y_test, test_scores, threshold, val_f1, val_ap, model is not None)
        row["evaluation_split"] = target_split
        row["pilot"] = pilot
        metrics.append(row)
        predictions[f"score_{name}"] = test_scores
        ranks = np.empty(len(test_scores), dtype=int)
        ranks[ranking_order(test_scores)] = np.arange(1, len(test_scores) + 1)
        predictions[f"rank_{name}"] = ranks
        review, risk = curves(y_test, test_scores, name)
        review_curves.extend(review)
        risk_curves.extend(risk)
        if model is not None:
            fitted[name] = model
            joblib.dump({"model": model, "features": columns, "threshold": threshold,
                         "seed": SEED, "configuration": config}, model_dir / f"{name}.joblib")
        model_choices[name] = {"features": columns or ["msp"], "configuration": config,
                               "validation_ap": val_ap, "threshold": threshold,
                               "calibration": "none", "fit_split": "train" if model is not None else None}
        print(f"{name}: {target_split} AP={row['ap']:.6f}, AUROC={row['auroc']:.6f}, "
              f"capture@10={row['capture_rate_10']:.6f}, capture@20={row['capture_rate_20']:.6f}", flush=True)

    record("baseline_msp", 1 - val.msp.to_numpy(), 1 - test.msp.to_numpy())

    for features_name, columns in FEATURE_SETS.items():
        best, best_ap, best_config, candidates = None, -np.inf, None, []
        for c in [0.1, 1.0, 10.0]:
            model = make_pipeline(StandardScaler(), LogisticRegression(
                C=c, max_iter=2000, random_state=SEED, solver="lbfgs"))
            model.fit(train[columns], y_train)
            score = float(average_precision_score(y_val, model.predict_proba(val[columns])[:, 1]))
            config = {"C": c, "max_iter": 2000, "solver": "lbfgs", "scaler_fit": "train"}
            candidates.append({**config, "validation_ap": score})
            if score > best_ap:
                best, best_ap, best_config = model, score, config
        record(f"lr_{features_name}", best.predict_proba(val[columns])[:, 1],
               best.predict_proba(test[columns])[:, 1], best, columns,
               {"selected": best_config, "candidates": candidates, "selection_metric": "validation AP"})

    if not pilot:
        from lightgbm import LGBMClassifier, early_stopping, log_evaluation

        for features_name, columns in FEATURE_SETS.items():
            best, best_ap, best_config, candidates = None, -np.inf, None, []
            for leaves in [15, 31]:
                params = {"n_estimators": 500, "learning_rate": 0.05, "num_leaves": leaves,
                          "min_child_samples": 100, "reg_lambda": 1.0, "random_state": SEED,
                          "n_jobs": n_jobs, "verbosity": -1, "deterministic": True,
                          "force_col_wise": True, "metric": "average_precision"}
                model = LGBMClassifier(**params)
                model.fit(train[columns], y_train, eval_set=[(val[columns], y_val)],
                          eval_metric="average_precision",
                          callbacks=[early_stopping(40, first_metric_only=True, verbose=False), log_evaluation(0)])
                score = float(average_precision_score(y_val, model.predict_proba(val[columns])[:, 1]))
                config = {**params, "best_iteration": int(model.best_iteration_)}
                candidates.append({**config, "validation_ap": score})
                if score > best_ap:
                    best, best_ap, best_config = model, score, config
            record(f"lgbm_{features_name}", best.predict_proba(val[columns])[:, 1],
                   best.predict_proba(test[columns])[:, 1], best, columns,
                   {"selected": best_config, "candidates": candidates, "selection_metric": "validation AP",
                    "early_stopping_rounds": 40, "early_stopping_split": "validation"})

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(out / "detection_metrics.csv", index=False)
    write_json(out / "detection_metrics.json", metrics)
    write_json(out / "model_selection.json", model_choices)
    predictions.to_parquet(out / ("validation_predictions.parquet" if pilot else "test_predictions.parquet"), index=False)
    pd.DataFrame(review_curves).to_csv(out / "review_budget_curve.csv", index=False)
    pd.DataFrame(risk_curves).to_csv(out / "risk_coverage_curve.csv", index=False)
    robustness = robustness_tables(table, out)

    summary = {"pilot": pilot, "evaluation_split": target_split,
               "scope_warning": "파일럿 validation 동작 점검이며 최종 test 성능이 아님" if pilot else None,
               "seed": SEED, "data": validation, "pointnet_robustness": robustness,
               "n_evaluation_rows": len(test), "n_evaluation_objects": int(test.object_id.nunique()),
               "evaluation_error_prevalence": float(y_test.mean()),
               "tie_break": "descending score, ascending object_id, corruption, severity",
               "f1_threshold_rule": "maximize validation F1; ties choose highest threshold",
               "calibration": "none; learned models output uncalibrated P(error=1)",
               "model_selection_uses_test": False,
               "features_exclude": ["object_id", "corruption", "severity", "split", "true_label", "pred_label", "error"]}

    if not pilot:
        left = metrics_df.set_index("model").loc["lgbm_confidence"]
        right = metrics_df.set_index("model").loc["lgbm_combined"]
        summary["geometry_addition_delta"] = {key: float(right[key] - left[key])
            for key in ["ap", "auroc", "capture_rate_10", "capture_rate_20", "f1", "brier"]}
        per_condition = []
        for (corruption, severity), part in predictions.groupby(["corruption", "severity"]):
            y = part.error.to_numpy(dtype=int)
            if np.unique(y).size < 2:
                continue
            a = part.score_lgbm_confidence.to_numpy()
            b = part.score_lgbm_combined.to_numpy()
            per_condition.append({"corruption": corruption, "severity": int(severity), "n_rows": len(part),
                                  "n_errors": int(y.sum()), "ap_confidence": float(average_precision_score(y, a)),
                                  "ap_combined": float(average_precision_score(y, b)),
                                  "delta_ap": float(average_precision_score(y, b) - average_precision_score(y, a)),
                                  **{f"delta_capture_{int(q*100)}": review_stats(y, b, q)["capture_rate"] -
                                     review_stats(y, a, q)["capture_rate"] for q in [0.1, 0.2]}})
        pd.DataFrame(per_condition).to_csv(out / "geometry_delta_by_condition.csv", index=False)
        if n_bootstrap:
            bootstrap = paired_object_bootstrap(test, predictions.score_lgbm_confidence.to_numpy(),
                                               predictions.score_lgbm_combined.to_numpy(), n_bootstrap)
            write_json(out / "paired_bootstrap.json", bootstrap)
            summary["paired_bootstrap"] = bootstrap
        if importance_samples:
            selected = np.sort(np.random.default_rng(SEED).choice(len(test), min(importance_samples, len(test)), replace=False))
            part = test.iloc[selected]
            columns = FEATURE_SETS["combined"]
            result = permutation_importance(fitted["lgbm_combined"], part[columns], part.error,
                                            scoring="average_precision", n_repeats=5,
                                            random_state=SEED, n_jobs=1)
            importance = pd.DataFrame({"feature": columns, "ap_drop_mean": result.importances_mean,
                                       "ap_drop_std": result.importances_std,
                                       "n_rows": len(part), "n_repeats": 5, "seed": SEED})
            importance.sort_values("ap_drop_mean", ascending=False).to_csv(out / "permutation_importance.csv", index=False)
            pd.DataFrame(part[KEYS]).to_csv(out / "permutation_sample_keys.csv", index=False)
            summary["permutation_importance"] = {"model": "lgbm_combined", "split": "test", "n_rows": len(part),
                                                  "n_repeats": 5, "seed": SEED, "metric": "AP decrease",
                                                  "interpretation": "관찰된 연관성이며 인과 효과가 아님; feature 상관에 영향받음"}
    summary["elapsed_seconds"] = time.perf_counter() - started
    write_json(out / "evaluation_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description="PointNet 오분류 탐지기 비교 평가")
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--importance-samples", type=int, default=6000)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    table = pd.read_parquet(args.table) if args.table.suffix == ".parquet" else pd.read_csv(args.table)
    evaluate(table, args.output, args.pilot, n_bootstrap=args.bootstrap,
             importance_samples=args.importance_samples, n_jobs=args.jobs)


if __name__ == "__main__":
    main()
