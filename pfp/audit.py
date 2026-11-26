"""저장된 실측 결과를 독립 계산으로 대조하는 사후 감사.

추론·geometry·평가 모듈의 계산 함수를 재사용하지 않는다. 전체 logits와
confidence 표, 원본 ID 분할, 유형별 raw geometry 1개, 검토 예산과 지표를
다시 계산한다. 공개 JSON에는 서버 절대 경로를 저장하지 않는다.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, roc_auc_score


KEYS = ["object_id", "corruption", "severity"]
CONFIDENCE = ["msp", "entropy", "margin"]
GEOMETRY = ["pca_ratio_1", "pca_ratio_2", "pca_ratio_3", "bbox_x", "bbox_y", "bbox_z",
            "knn_dist_mean", "knn_dist_std"]
CORRUPTIONS = {"uniform", "gaussian", "background", "impulse", "upsampling", "distortion",
               "distortion_rbf", "distortion_rbf_inv", "density", "density_inc", "shear",
               "rotation", "cutout", "occlusion", "lidar"}
FEATURE_SETS = {"confidence": CONFIDENCE, "geometry": GEOMETRY, "combined": CONFIDENCE + GEOMETRY}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def close(actual, expected, message, rtol=1e-10, atol=1e-12) -> float:
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    require(a.shape == b.shape, f"{message}: shape 불일치 {a.shape} / {b.shape}")
    require(np.isfinite(a).all() and np.isfinite(b).all(), f"{message}: 비유한 값")
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=message)
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def geometry_reference(points: np.ndarray) -> np.ndarray:
    """np.cov와 완전 거리행렬을 사용한 독립 geometry 계산."""
    xyz = np.asarray(points, dtype=np.float64)
    require(xyz.ndim == 2 and xyz.shape[1] == 3 and len(xyz) >= 2, "raw XYZ shape 오류")
    require(np.isfinite(xyz).all(), "raw XYZ 비유한 좌표")
    values = np.linalg.eigvalsh(np.cov(xyz, rowvar=False, ddof=1))
    values = np.sort(np.maximum(values, 0))[::-1]
    ratios = values / values.sum() if values.sum() > np.finfo(np.float64).eps else np.zeros(3)
    box = np.max(xyz, axis=0) - np.min(xyz, axis=0)
    distances = cdist(xyz, xyz, metric="euclidean")
    np.fill_diagonal(distances, np.inf)
    nearest = np.sort(distances, axis=1)[:, :min(8, len(xyz) - 1)]
    return np.concatenate([ratios, box, [np.mean(nearest), np.std(nearest, ddof=0)]])


def optimum_f1(y: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """score별 누적 TP/FP로 validation F1을 독립 계산한다."""
    grouped = pd.DataFrame({"score": scores, "positive": y}).groupby("score", sort=True).agg(
        count=("positive", "size"), positive=("positive", "sum")).sort_index(ascending=False)
    selected, true_positive = grouped["count"].cumsum(), grouped["positive"].cumsum()
    # 2 TP / (선택 수 + 전체 양성 수). 내림차순이므로 첫 최대가 높은 임계값.
    f1 = 2 * true_positive / (selected + int(np.sum(y)))
    threshold = float(f1.idxmax())
    return threshold, float(f1.max())


def condition_scope(conditions: set[tuple[str, int]], *, pilot: bool,
                    expected_conditions: int = 75) -> dict:
    """부분 범위는 명시한 조건 수와 정확히 같을 때만 허용한다."""
    official = {(c, s) for c in CORRUPTIONS for s in range(1, 6)}
    require(1 <= expected_conditions <= 75, "기대 조건 수는 1~75여야 함")
    require(conditions.issubset(official), "공식 범위 외 조건 포함")
    if pilot:
        require(expected_conditions in {4, 75}, "파일럿은 4조건만 허용")
        expected_count = 4
        required = {(c, s) for c in {"gaussian", "cutout"} for s in [1, 5]}
        require(conditions == required, "파일럿 Gaussian/Cutout level1·5 조건 불일치")
    else:
        expected_count = expected_conditions
        require(len(conditions) == expected_count, f"조건 수 불일치: {len(conditions)} != {expected_count}")
        if expected_count == 75:
            require(conditions == official, "전체 75조건 누락 또는 추가")
    partial = not pilot and expected_count < 75
    warning = (f"부분 결과: 공식 75조건 중 {expected_count}조건만 사용했으며 전체 75조건 최종 성능이 아님"
               if partial else "파일럿 validation 동작 점검이며 최종 test 성능이 아님" if pilot else None)
    return {"partial": partial, "scope_warning": warning, "expected_conditions": expected_count,
            "official_total_conditions": 75,
            "missing_conditions": [{"corruption": c, "severity": int(s)} for c, s in sorted(official - conditions)]}


def audit(table_path: Path, data_dir: Path, cache_dir: Path, evaluation_dir: Path,
          output: Path, *, pilot: bool = False, expected_conditions: int = 75) -> dict:
    start = time.perf_counter()
    table = pd.read_parquet(table_path).sort_values(KEYS).reset_index(drop=True)
    required = KEYS + ["split", "true_label", "pred_label", "error"] + CONFIDENCE + GEOMETRY
    require(set(required).issubset(table.columns), "표 필수 열 누락")
    require(not table[required].isna().any().any(), "표 누락값")
    require(not table.duplicated(KEYS).any(), "표 조인 키 중복")
    require(np.isfinite(table[CONFIDENCE + GEOMETRY].to_numpy()).all(), "표 비유한 feature")
    require(table.true_label.between(0, 39).all() and table.pred_label.between(0, 39).all(), "클래스 범위 오류")
    labels = np.load(data_dir / "label.npy").reshape(-1)
    require(len(labels) == 2468 and set(labels) == set(range(40)), "공식 label 크기/클래스 오류")
    object_ids = np.sort(table.object_id.unique())
    require(np.equal(object_ids, object_ids.astype(int)).all(), "정수가 아닌 원본 ID")
    require(object_ids.min() >= 0 and object_ids.max() < len(labels), "원본 ID 범위 오류")
    require(np.array_equal(labels[table.object_id.to_numpy(dtype=int)], table.true_label), "원본 라벨 매핑 오류")
    require(np.array_equal((table.pred_label != table.true_label).astype(int), table.error), "오분류 타깃 오류")
    conditions = set(map(tuple, table[["corruption", "severity"]].drop_duplicates().to_numpy()))
    scope = condition_scope(conditions, pilot=pilot, expected_conditions=expected_conditions)
    if not pilot:
        require(np.array_equal(object_ids, np.arange(2468)), "전체 원본 ID 2468개 불일치")
        require(len(table) == 2468 * scope["expected_conditions"],
                f"표 {2468 * scope['expected_conditions']}행 불일치")
    require(len(table) == len(object_ids) * len(conditions), "ID×조건 행 수 불일치")
    require(table.groupby("object_id").size().eq(len(conditions)).all(), "물체별 조건 수 불일치")
    require(table.groupby("object_id").split.nunique().eq(1).all(), "split 사이 원본 ID 누출")
    expected_splits = {"train", "validation"} if pilot else {"train", "validation", "test"}
    require(set(table.split) == expected_splits, "split 집합 오류")
    split_counts = []
    for split, part in table.groupby("split", sort=True):
        split_counts.append({"split": split, "objects": int(part.object_id.nunique()), "rows": len(part),
                             "positive": int(part.error.sum()), "negative": int(len(part) - part.error.sum())})

    configuration_path = cache_dir.parent / "configuration.json"
    configuration = json.loads(configuration_path.read_text())
    environment = json.loads((cache_dir.parent / "environment.local.json").read_text())
    reverse_levels = {int(display): int(stored) for stored, display in environment["level_mapping"].items()}
    require(configuration["label_sha256"] == digest(data_dir / "label.npy"), "label hash 불일치")
    require(np.array_equal(configuration["ids"], object_ids), "configuration ID 불일치")

    confidence_max_difference, cache_table_max_difference = 0.0, 0.0
    for corruption, severity in sorted(conditions):
        name = f"{corruption}_{severity}"
        directory = cache_dir / name
        metadata = json.loads((directory / "metadata.json").read_text())
        cache_table_path, logits_path = directory / "table.parquet", directory / "logits.npz"
        require(metadata["table_sha256"] == digest(cache_table_path), f"{name} 표 hash 불일치")
        require(metadata["logits_sha256"] == digest(logits_path), f"{name} logits hash 불일치")
        require(metadata["passed"], f"{name} 조건 완료 검증 기록 없음")
        signature = {**configuration, "data_sha256": metadata["data_sha256"], "condition": name,
                     "shape": metadata["shape"]}
        expected_fingerprint = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        require(metadata["fingerprint"] == expected_fingerprint, f"{name} 설정 fingerprint 불일치")
        part = table.loc[(table.corruption == corruption) & (table.severity == severity)].sort_values("object_id")
        cached_table = pd.read_parquet(cache_table_path).sort_values("object_id")
        require(np.array_equal(cached_table[KEYS + ["split"]], part[KEYS + ["split"]]), f"{name} 표 키/split 불일치")
        numeric = ["true_label", "pred_label", "error"] + CONFIDENCE + GEOMETRY
        cache_table_max_difference = max(cache_table_max_difference, close(
            cached_table[numeric], part[numeric], f"{name} 캐시 표 대조", rtol=0, atol=0))
        with np.load(logits_path) as saved:
            ids = saved["object_id"].astype(int)
            logits = saved["logits"].astype(np.float64)
        require(np.array_equal(ids, object_ids), f"{name} logits 원본 ID 불일치")
        require(logits.shape == (len(object_ids), 40) and np.isfinite(logits).all(), f"{name} logits shape/유한값 오류")
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        sorted_probs = np.sort(probs, axis=1)
        entropy = -np.sum(np.where(probs > 0, probs * np.log(np.clip(probs, 1e-300, 1)), 0), axis=1)
        recomputed = np.column_stack([sorted_probs[:, -1], entropy, sorted_probs[:, -1] - sorted_probs[:, -2]])
        confidence_max_difference = max(confidence_max_difference, close(
            part[CONFIDENCE], recomputed, f"{name} confidence 재계산"))
        predicted = np.argmax(logits, axis=1)
        require(np.array_equal(predicted, part.pred_label), f"{name} logits 예측 불일치")
        require(np.array_equal((predicted != labels[ids]).astype(int), part.error), f"{name} logits 오분류 타깃 불일치")
        raw_path = data_dir / f"data_{corruption}_{reverse_levels[int(severity)]}.npy"
        raw = np.load(raw_path, mmap_mode="r")
        require(list(raw.shape) == metadata["shape"] and raw.shape[0] == 2468 and raw.shape[-1] == 3,
                f"{name} raw 데이터 shape 불일치")
        del raw

    geometry_checks = []
    for corruption in sorted(table.corruption.unique()):
        # 표본 선택 규칙은 실측값이나 오류 여부와 무관한 SHA-256(name) 정수다.
        seed = int(hashlib.sha256(corruption.encode()).hexdigest()[:16], 16)
        available_levels = sorted(table.loc[table.corruption == corruption, "severity"].unique())
        severity = int(available_levels[seed % len(available_levels)])
        object_id = int(object_ids[(seed // len(available_levels)) % len(object_ids)])
        raw_path = data_dir / f"data_{corruption}_{reverse_levels[severity]}.npy"
        raw = np.load(raw_path, mmap_mode="r")
        points = np.asarray(raw[object_id], dtype=np.float64)
        reference = geometry_reference(points)
        row = table.loc[(table.corruption == corruption) & (table.severity == severity) & (table.object_id == object_id)]
        require(len(row) == 1, "geometry 대조 표본 키 오류")
        difference = close(row[GEOMETRY].to_numpy().reshape(-1), reference,
                           f"{corruption}_{severity}/{object_id} raw geometry", rtol=1e-9, atol=1e-12)
        geometry_checks.append({"corruption": corruption, "severity": severity, "object_id": object_id,
                                "n_points": len(points), "max_abs_difference": difference})
        del raw

    target_split = "validation" if pilot else "test"
    target = table.loc[table.split == target_split].sort_values(KEYS).reset_index(drop=True)
    validation = table.loc[table.split == "validation"].sort_values(KEYS).reset_index(drop=True)
    prediction_path = evaluation_dir / f"{target_split}_predictions.parquet"
    predictions = pd.read_parquet(prediction_path).sort_values(KEYS).reset_index(drop=True)
    require(np.array_equal(predictions[KEYS + ["split"]], target[KEYS + ["split"]]), "평가 예측 표 키/split 불일치")
    close(predictions[["true_label", "pred_label", "error"] + CONFIDENCE + GEOMETRY],
          target[["true_label", "pred_label", "error"] + CONFIDENCE + GEOMETRY], "평가 표 원본 열", rtol=0, atol=0)
    metrics = pd.read_csv(evaluation_dir / "detection_metrics.csv").set_index("model")
    choices = json.loads((evaluation_dir / "model_selection.json").read_text())
    expected_models = {"baseline_msp", "lr_confidence", "lr_geometry", "lr_combined"}
    if not pilot:
        expected_models |= {"lgbm_confidence", "lgbm_geometry", "lgbm_combined"}
    require(set(metrics.index) == expected_models and len(metrics) == len(expected_models), "탐지 비교 조건 누락/중복")
    require(set(choices) == expected_models, "validation 선택 기록 모델 불일치")
    y, val_y = target.error.to_numpy(dtype=int), validation.error.to_numpy(dtype=int)
    metric_checks = []
    for name in sorted(expected_models):
        scores = predictions[f"score_{name}"].to_numpy()
        require(np.isfinite(scores).all() and (scores >= 0).all() and (scores <= 1).all(), f"{name} score 범위 오류")
        reported = metrics.loc[name]
        require(reported["evaluation_split"] == target_split, f"{name} 평가 split 기록 오류")
        require(int(reported["n_rows"]) == len(target) and int(reported["n_errors"]) == int(y.sum()), f"{name} 평가 모집단 오류")
        close(reported["error_prevalence"], y.mean(), f"{name} AP 무작위 기준")
        if name == "baseline_msp":
            val_scores = 1 - validation.msp.to_numpy()
            close(scores, 1 - target.msp.to_numpy(), "baseline MSP score")
        else:
            saved_model = joblib.load(evaluation_dir / "models" / f"{name}.joblib")
            expected_features = FEATURE_SETS[name.split("_", 1)[1]]
            require(saved_model["features"] == expected_features and choices[name]["features"] == expected_features,
                    f"{name} 입력 feature 목록 오류")
            require(choices[name]["fit_split"] == "train", f"{name} train fit 기록 오류")
            require(choices[name]["configuration"]["selection_metric"] == "validation AP", f"{name} validation 선택 기준 오류")
            val_scores = saved_model["model"].predict_proba(validation[expected_features])[:, 1]
            reproduced_test_scores = saved_model["model"].predict_proba(target[expected_features])[:, 1]
            close(scores, reproduced_test_scores, f"{name} 저장 모델 score 재현")
            close(saved_model["threshold"], reported["threshold_from_validation"], f"{name} 저장 모델 threshold")
        independent_threshold, validation_f1 = optimum_f1(val_y, val_scores)
        # JSON은 Python float를 그대로 복원한다. CSV reader의 마지막 자리 반올림으로
        # threshold와 정확히 같은 score의 포함 여부가 바뀌지 않도록 JSON을 기준으로 쓴다.
        threshold = float(choices[name]["threshold"])
        close(threshold, reported["threshold_from_validation"], f"{name} CSV threshold 기록")
        require(np.isclose(val_scores, threshold, rtol=1e-12, atol=1e-15).any(),
                f"{name} threshold가 validation score에서 선택되지 않음")
        # 수학적으로 같은 F1이 계산 순서에 따라 마지막 자릿수에서 달라질 수 있다.
        # 임계값 자체의 동일성보다 실제 선택된 임계값이 최적 F1을 달성하는지 검사한다.
        close(f1_score(val_y, val_scores >= threshold, zero_division=0), validation_f1,
              f"{name} 선택 threshold가 validation 최대 F1을 달성하는지 검증")
        close(threshold, choices[name]["threshold"], f"{name} 선택 기록 threshold")
        close(validation_f1, reported["validation_f1"], f"{name} validation F1")
        close(average_precision_score(val_y, val_scores), reported["validation_ap"], f"{name} validation AP")
        recomputed = {"ap": float(average_precision_score(y, scores)), "auroc": float(roc_auc_score(y, scores)),
                      "f1": float(f1_score(y, scores >= threshold, zero_division=0))}
        if name != "baseline_msp":
            recomputed["brier"] = float(brier_score_loss(y, scores))
        for key, value in recomputed.items():
            close(value, reported[key], f"{name} {key}")
        # numpy lexsort 대신 pandas의 다중 열 정렬로 독립적으로 순위를 계산한다.
        ordered = predictions[KEYS + ["error"]].assign(audit_score=scores).sort_values(
            ["audit_score"] + KEYS, ascending=[False, True, True, True], kind="mergesort")
        ranks = pd.Series(np.arange(1, len(ordered) + 1), index=ordered.index).sort_index().to_numpy()
        require(np.array_equal(ranks, predictions[f"rank_{name}"]), f"{name} 검토 순위 불일치")
        review_checks = []
        for percentage in [10, 20]:
            n_review = (percentage * len(y) + 99) // 100  # ceil를 정수 연산으로 계산
            numerator, denominator = int(ordered.iloc[:n_review].error.sum()), int(y.sum())
            values = {"n_reviewed": n_review, "n_captured_errors": numerator,
                      "n_total_errors": denominator, "capture_rate": numerator / denominator,
                      "actual_review_fraction": n_review / len(y), "random_expected_capture": n_review / len(y)}
            for key, value in values.items():
                close(value, reported[f"{key}_{percentage}"], f"{name} 검토 {percentage}% {key}")
            review_checks.append({"budget_percent": percentage, **values})
        metric_checks.append({"model": name, **recomputed, "validation_threshold": threshold,
                              "independent_optimal_threshold": independent_threshold,
                              "validation_f1": validation_f1, "reviews": review_checks})

    condition_metrics = pd.read_csv(evaluation_dir / "robustness_by_condition.csv")
    measured = table.groupby(["corruption", "severity"]).error.agg(n_rows="size", n_errors="sum", error_rate="mean").reset_index()
    condition_metrics = condition_metrics.sort_values(["corruption", "severity"]).reset_index(drop=True)
    require(np.array_equal(condition_metrics[["corruption", "severity"]], measured[["corruption", "severity"]]),
            "robustness 조건 키 오류")
    close(condition_metrics[["n_rows", "n_errors", "error_rate"]], measured[["n_rows", "n_errors", "error_rate"]],
          "robustness 조건 집계")
    result = {"passed": True, "pilot": pilot, **scope, "evaluation_split": target_split,
              "timestamp_kst": datetime.now(ZoneInfo("Asia/Seoul")).isoformat(),
              "n_rows": len(table), "n_objects": len(object_ids), "n_conditions": len(conditions),
              "duplicates": 0, "missing_values": 0, "split_object_intersections": 0, "splits": split_counts,
              "table_sha256": digest(table_path), "prediction_table_sha256": digest(prediction_path),
              "configuration_sha256": digest(configuration_path), "label_sha256": digest(data_dir / "label.npy"),
              "all_cache_table_and_logits_hashes_verified": True,
              "all_raw_file_shapes_verified": True, "raw_data_hashes_recomputed": False,
              "confidence_rows_recomputed": len(table), "confidence_max_abs_difference": confidence_max_difference,
              "cache_table_max_abs_difference": cache_table_max_difference,
              "geometry_method": "유형마다 SHA-256 이름으로 정한 1개 표본; np.cov + cdist 완전 거리행렬; 자기 인덱스만 제외",
              "geometry_samples": geometry_checks, "evaluation_rows": len(target), "models_checked": len(expected_models),
              "validation_thresholds_independently_recomputed": True,
              "model_predictions_reproduced": True, "metrics": metric_checks,
              "robustness_condition_counts_verified": True,
              "elapsed_seconds": time.perf_counter() - start}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(output)
    print(f"{'부분 결과 ' if scope['partial'] else ''}감사 통과: {len(table):,}행, {len(conditions)}조건, geometry {len(geometry_checks)}표본, "
          f"탐지기 {len(expected_models)}개; {output}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--expected-conditions", type=int, default=75,
                        help="기본 75조건. 부분 결과를 감사할 때만 실제 조건 수를 명시한다.")
    args = parser.parse_args()
    audit(args.table, args.data_dir, args.cache_dir, args.evaluation_dir, args.output,
          pilot=args.pilot, expected_conditions=args.expected_conditions)


if __name__ == "__main__":
    main()
