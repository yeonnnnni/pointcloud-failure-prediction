"""검증된 실측 표에서 발표용 그림과 수치 manifest를 만든다."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


CLASSES = [
    "airplane", "bathtub", "bed", "bench", "bookshelf", "bottle", "bowl",
    "car", "chair", "cone", "cup", "curtain", "desk", "door", "dresser",
    "flower_pot", "glass_box", "guitar", "keyboard", "lamp", "laptop",
    "mantel", "monitor", "night_stand", "person", "piano", "plant",
    "radio", "range_hood", "sink", "sofa", "stairs", "stool", "table",
    "tent", "toilet", "tv_stand", "vase", "wardrobe", "xbox",
]
MODEL_NAMES = {
    "baseline_msp": "1 − MSP",
    "lr_confidence": "LR · confidence",
    "lr_geometry": "LR · geometry",
    "lr_combined": "LR · confidence + geometry",
    "lgbm_confidence": "LightGBM · confidence",
    "lgbm_geometry": "LightGBM · geometry",
    "lgbm_combined": "LightGBM · confidence + geometry",
}
COLORS = {"baseline_msp": "#8793a2", "lgbm_confidence": "#2963a0", "lgbm_combined": "#db6b35"}


def configure_style():
    font = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({
        "font.size": 11, "axes.titlesize": 17, "axes.labelsize": 12,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.unicode_minus": False, "savefig.dpi": 180,
    })


def _native(value):
    if isinstance(value, dict):
        return {str(k): _native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _scope(frame):
    splits = "/".join(sorted(frame["split"].unique())) if "split" in frame else "평가"
    conditions = len(frame[["corruption", "severity"]].drop_duplicates())
    if "split" in frame and "test" not in set(frame["split"]):
        prefix = "파일럿 "
    else:
        prefix = "부분 결과 " if conditions < 75 else ""
    return f"{prefix}{splits} · {len(frame):,}행 · 물체 {frame.object_id.nunique():,}개 · {conditions}조건"


def _ordered(frame, model):
    column = f"rank_{model}"
    if column in frame:
        ranks = frame[column].to_numpy()
        if not np.array_equal(np.sort(ranks), np.arange(1, len(frame) + 1)):
            raise ValueError(f"평가 순위가 1..N 순열이 아닙니다: {column}")
        return frame.sort_values(column, kind="stable")
    return frame.sort_values(
        [f"score_{model}", "object_id", "corruption", "severity"],
        ascending=[False, True, True, True], kind="stable",
    )


def _save(fig, output, stem):
    fig.savefig(output / f"{stem}.png", bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _metrics(predictions):
    y = predictions.error.to_numpy()
    total_errors = int(y.sum())
    if not (0 < total_errors < len(y)):
        raise ValueError("평가에는 정답과 오답이 모두 있어야 합니다.")
    rows = []
    for model in MODEL_NAMES:
        score_col = f"score_{model}"
        if score_col not in predictions:
            continue
        score = predictions[score_col].to_numpy()
        if not np.isfinite(score).all():
            raise ValueError(f"비유한 예측 점수: {model}")
        ordered = _ordered(predictions, model)
        row = {"model": model, "ap": average_precision_score(y, score),
               "auroc": roc_auc_score(y, score), "total_errors": total_errors,
               "n_rows": len(y)}
        for pct in (10, 20):
            k = math.ceil(pct / 100 * len(y))
            captured = int(ordered.iloc[:k].error.sum())
            row[f"capture_{pct}"] = captured / total_errors
            row[f"captured_{pct}"] = captured
            row[f"reviewed_{pct}"] = k
        rows.append(row)
    return pd.DataFrame(rows)


def robustness_plot(table, output, manifest):
    by_condition = table.groupby(["corruption", "severity"]).agg(
        error_rate=("error", "mean"), n=("error", "size"), errors=("error", "sum")
    ).reset_index()
    by_condition.to_csv(output / "heatmap_values.csv", index=False)
    pivot = by_condition.pivot(index="corruption", columns="severity", values="error_rate").reindex(columns=range(1, 6))
    fig, ax = plt.subplots(figsize=(9.2, max(5.5, len(pivot) * .44 + 1.8)))
    im = ax.imshow(pivot.to_numpy() * 100, cmap="YlOrRd", vmin=0, vmax=100, aspect="auto")
    for i in range(len(pivot)):
        for j in range(5):
            v = pivot.iloc[i, j] * 100
            ax.text(j, i, f"{v:.1f}" if np.isfinite(v) else "미실행", ha="center", va="center", color="white" if v > 65 else "#182536", fontsize=10)
    ax.set_xticks(range(5), [str(x) for x in range(1, 6)])
    ax.set_yticks(range(len(pivot)), pivot.index)
    ax.set_xlabel("표시 level (LiDAR·Occlusion은 관측 각도)")
    ax.set_title(f"PointNet 손상 조건별 오분류율 (%)\n{_scope(table)}", pad=14)
    fig.colorbar(im, ax=ax, label="오분류율 (%)", fraction=.035, pad=.04)
    fig.tight_layout()
    _save(fig, output, "corruption_heatmap")
    manifest["corruption_heatmap"] = {"rows": len(table), "conditions": len(by_condition), "mean_error_rate": float(table.error.mean()), "highest": by_condition.nlargest(3, "error_rate").to_dict("records"), "lowest": by_condition.nsmallest(3, "error_rate").to_dict("records"), "scope": _scope(table)}


def comparison_plot(predictions, metrics, output, manifest):
    data = [[MODEL_NAMES[r.model], f"{r.ap:.4f}", f"{r.auroc:.4f}", f"{100*r.capture_10:.2f}%", f"{100*r.capture_20:.2f}%"] for r in metrics.itertuples()]
    fig, ax = plt.subplots(figsize=(12.8, 1.9 + len(data)*.54))
    ax.axis("off")
    tbl = ax.table(cellText=data, colLabels=["오분류 탐지 방법", "AP", "AUROC", "10% 검토 포착률", "20% 검토 포착률"], cellLoc="center", loc="center", colWidths=[.40, .12, .12, .18, .18])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 2.2)
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor("white")
        if row == 0:
            cell.set_facecolor("#17334f")
            cell.set_text_props(color="white", weight="bold")
        else:
            cell.set_facecolor("#fff0e7" if metrics.iloc[row-1].model == "lgbm_combined" else ("#edf3f8" if row % 2 else "#f8fafc"))
    ax.set_title(f"동일 held-out 물체에서 오분류 탐지 비교\n{_scope(predictions)} · 무작위 AP 기준 {predictions.error.mean():.4f}", pad=15)
    fig.tight_layout()
    _save(fig, output, "detection_comparison")
    manifest["detection_comparison"] = {"scope": _scope(predictions), "random_ap": float(predictions.error.mean()), "metrics": metrics.to_dict("records")}


def curves(predictions, output, manifest):
    n, errors = len(predictions), int(predictions.error.sum())
    models = [m for m in COLORS if f"score_{m}" in predictions]
    fig, ax = plt.subplots(figsize=(9.5, 6.1))
    budget_rows = []
    for model in models:
        y = _ordered(predictions, model).error.to_numpy()
        cumulative = np.r_[0, y.cumsum()] / errors
        grid = np.unique(np.r_[np.linspace(0, n, 301).astype(int), math.ceil(n*.1), math.ceil(n*.2)])
        ax.plot(100*grid/n, 100*cumulative[grid], label=MODEL_NAMES[model], color=COLORS[model], lw=2.5)
        for pct in (10, 20):
            k = math.ceil(pct/100*n)
            ax.scatter(100*k/n, cumulative[k]*100, color=COLORS[model], zorder=5)
            budget_rows.append({"model": model, "budget": pct/100, "reviewed": k, "n": n, "captured_errors": int(y[:k].sum()), "total_errors": errors, "capture_rate": cumulative[k]})
    ax.plot([0, 100], [0, 100], "--", color="#555555", alpha=.7, label="무작위 검토 기대값")
    for pct in (10, 20):
        ax.axvline(pct, lw=.9, color="#cccccc", linestyle=":")
    ax.set(xlim=(0, 50), ylim=(0, 100), xlabel="검토 비율 (%)", ylabel="전체 오분류 중 포착한 비율 (%)")
    ax.set_title(f"같은 검토 예산에서 발견한 오분류\n{_scope(predictions)} · 오분류 {errors:,}건")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=.15)
    fig.tight_layout()
    _save(fig, output, "review_budget")
    pd.DataFrame(budget_rows).to_csv(output / "review_budget_values.csv", index=False)
    manifest["review_budget"] = budget_rows

    fig, ax = plt.subplots(figsize=(9.5, 6))
    for model in models:
        y = _ordered(predictions, model).error.to_numpy()
        grid = np.unique(np.linspace(1, n, 400).astype(int))
        tail_errors = np.r_[0, y[::-1].cumsum()]
        ax.plot(100*grid/n, 100*tail_errors[grid]/grid, label=MODEL_NAMES[model], color=COLORS[model], lw=2.2)
    ax.axhline(errors/n*100, linestyle="--", color="#777777", label="전체 오분류율")
    ax.set(xlabel="남겨 둔 샘플 비율: coverage (%)", ylabel="남은 샘플의 오분류율: risk (%)", xlim=(0, 100))
    ax.set_ylim(bottom=0)
    ax.set_title(f"고위험 샘플을 제외한 뒤 남는 오류\n{_scope(predictions)}")
    ax.legend(fontsize=10)
    ax.grid(alpha=.15)
    fig.tight_layout()
    _save(fig, output, "risk_coverage")
    manifest["risk_coverage"] = {"scope": _scope(predictions), "full_risk": errors/n, "definition": "높은 위험도부터 제외하고 남은 샘플의 오류율"}


def confidence_plot(table, output, manifest):
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for error, name, color in [(0, "정답", "#2963a0"), (1, "오답", "#db6b35")]:
        values = table.loc[table.error == error, "msp"]
        ax.hist(values, bins=np.linspace(0, 1, 41), weights=np.full(len(values), 100/len(values)), histtype="step", lw=2.4, color=color, label=f"{name} ({len(values):,}행)")
    ax.set(xlabel="PointNet maximum softmax probability (MSP)", ylabel="각 집단 내 샘플 비율 (%)", xlim=(0, 1))
    ax.set_title(f"정답과 오답의 confidence 분포\n{_scope(table)}")
    ax.legend()
    fig.tight_layout()
    _save(fig, output, "confidence_distribution")
    manifest["confidence_distribution"] = {"scope": _scope(table), "high_confidence_errors_msp_ge_0_9": int(((table.error == 1) & (table.msp >= .9)).sum()), "total_errors": int(table.error.sum())}


def condition_deltas(predictions, output, manifest):
    columns = ["score_lgbm_confidence", "score_lgbm_combined"]
    if not all(c in predictions for c in columns):
        return
    rows = []
    for corruption, group in predictions.groupby("corruption"):
        if group.error.nunique() < 2:
            continue
        a = average_precision_score(group.error, group[columns[0]])
        b = average_precision_score(group.error, group[columns[1]])
        rows.append({"corruption": corruption, "n": len(group), "confidence_ap": a, "combined_ap": b, "delta_ap": b-a})
    frame = pd.DataFrame(rows).sort_values("delta_ap")
    frame.to_csv(output / "condition_delta_values.csv", index=False)
    fig, ax = plt.subplots(figsize=(9.5, max(4.5, len(frame)*.39+1.8)))
    ax.barh(frame.corruption, frame.delta_ap, color=["#db6b35" if v >= 0 else "#2963a0" for v in frame.delta_ap])
    ax.axvline(0, color="#555555", lw=1)
    ax.set_xlabel("AP 차이: LightGBM (confidence + geometry) − confidence")
    ax.set_title(f"손상 유형별 geometry 추가 효과\n{_scope(predictions)} · 실행된 level 통합")
    ax.grid(axis="x", alpha=.15)
    fig.tight_layout()
    _save(fig, output, "condition_deltas")
    manifest["condition_deltas"] = frame.to_dict("records")


def _point_file(data_dir, corruption, severity):
    filename = f"data_{corruption}_{int(severity)}.npy"
    candidates = [data_dir / filename, data_dir / "modelnet40_c" / filename, data_dir / "ModelNet40-C" / filename]
    candidates += list(data_dir.glob(f"*/{filename}"))
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(f"조건 파일을 찾지 못했습니다: {filename}")


def case_plots(predictions, data_dir, output, manifest):
    required = ["score_lgbm_confidence", "score_lgbm_combined"]
    if not all(c in predictions for c in required):
        return
    df = predictions.copy()
    for model in ("lgbm_confidence", "lgbm_combined"):
        ordered = _ordered(df, model)
        df.loc[ordered.index, f"rank_{model}"] = np.arange(1, len(df)+1)
    df["rank_gain"] = df.rank_lgbm_confidence - df.rank_lgbm_combined
    errors = df[df.error == 1]
    high = errors[errors.msp >= .9]
    if high.empty:
        high = errors
    best = high.sort_values(["rank_gain", "msp"], ascending=False).iloc[0]
    worst = errors.sort_values("rank_gain").iloc[0]
    choices = [
        ("geometry가 순위를 높인 오답" if best.rank_gain > 0 else "고신뢰 오답의 검토 순위", best),
        ("geometry가 순위를 낮춘 오답" if worst.rank_gain < 0 else "오답 중 가장 작은 순위 개선", worst),
    ]
    correct = df[df.error == 0]
    if not correct.empty:
        choices.append(("결합 모델이 높게 평가한 정답", correct.sort_values("rank_lgbm_combined").iloc[0]))
    fig = plt.figure(figsize=(6.5*len(choices), 7.5))
    cases = []
    for i, (label, row) in enumerate(choices):
        cloud = np.load(_point_file(data_dir, row.corruption, row.severity), mmap_mode="r")[int(row.object_id), :, :3]
        if cloud.shape[1] != 3 or not np.isfinite(cloud).all():
            raise ValueError("사례 좌표가 유효하지 않습니다.")
        ax = fig.add_subplot(1, len(choices), i+1, projection="3d")
        ax.scatter(cloud[:, 0], cloud[:, 1], cloud[:, 2], c=cloud[:, 2], cmap="viridis", s=3, alpha=.85, linewidths=0)
        center = (cloud.max(axis=0)+cloud.min(axis=0))/2
        radius = max(float(np.ptp(cloud, axis=0).max()/2), .01)
        ax.set_xlim(center[0]-radius, center[0]+radius)
        ax.set_ylim(center[1]-radius, center[1]+radius)
        ax.set_zlim(center[2]-radius, center[2]+radius)
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=22, azim=-58)
        ax.set_axis_off()
        ax.set_title(f"{label}\n{row.corruption} · level {int(row.severity)} · ID {int(row.object_id)}", fontsize=13, pad=10)
        rc, rg = int(row.rank_lgbm_confidence), int(row.rank_lgbm_combined)
        caption = f"정답: {CLASSES[int(row.true_label)]} / 예측: {CLASSES[int(row.pred_label)]}\nMSP {row.msp:.3f}\nconfidence 순위 {rc:,}/{len(df):,} (상위 {100*rc/len(df):.1f}%)\n결합 순위 {rg:,}/{len(df):,} (상위 {100*rg/len(df):.1f}%)"
        ax.text2D(.5, -.04, caption, transform=ax.transAxes, ha="center", va="top", fontsize=11)
        cases.append({"label": label, "object_id": int(row.object_id), "corruption": row.corruption, "severity": int(row.severity), "true_label": int(row.true_label), "true_class": CLASSES[int(row.true_label)], "pred_label": int(row.pred_label), "pred_class": CLASSES[int(row.pred_label)], "msp": float(row.msp), "rank_confidence": rc, "rank_combined": rg, "n": len(df), "point_count": len(cloud)})
    fig.suptitle(f"실제 손상 point cloud와 오분류 검토 순위\n{_scope(predictions)} · 순위가 작을수록 먼저 검토", fontsize=17, y=.98)
    fig.subplots_adjust(top=.79, bottom=.25, wspace=.03)
    _save(fig, output, "pointcloud_cases")
    manifest["pointcloud_cases"] = cases


def importance_plot(evaluation_dir, output, manifest):
    path = evaluation_dir / "permutation_importance.csv"
    if not path.exists():
        return
    frame = pd.read_csv(path).sort_values("ap_drop_mean")
    fig, ax = plt.subplots(figsize=(9.5, 6.1))
    ax.barh(frame.feature, frame.ap_drop_mean, xerr=frame.ap_drop_std, color="#2963a0", ecolor="#17334f", capsize=3)
    ax.axvline(0, color="#777777", lw=1)
    ax.set_xlabel("해당 feature를 섞었을 때 AP 감소 (반복 표준편차 표시)")
    ax.set_title(f"LightGBM 결합 모델의 permutation importance\n{manifest['evaluation_scope']}\n평가 표본 {int(frame.n_rows.iloc[0]):,}행 · {int(frame.n_repeats.iloc[0])}회 반복", fontsize=15)
    ax.grid(axis="x", alpha=.15)
    fig.tight_layout()
    _save(fig, output, "feature_importance")
    manifest["feature_importance"] = frame.to_dict("records")
    manifest["feature_importance_scope"] = manifest["evaluation_scope"]


def generate(table_path, evaluation_dir, data_dir, output):
    configure_style()
    output.mkdir(parents=True, exist_ok=True)
    table = pd.read_parquet(table_path)
    prediction_path = evaluation_dir / "test_predictions.parquet"
    if not prediction_path.exists():
        prediction_path = evaluation_dir / "validation_predictions.parquet"
    predictions = pd.read_parquet(prediction_path)
    keys = ["object_id", "corruption", "severity"]
    if table.duplicated(keys).any() or predictions.duplicated(keys).any():
        raise ValueError("그림 입력 표에 중복 키가 있습니다.")
    if not (table.error == (table.pred_label != table.true_label).astype(int)).all():
        raise ValueError("그림 입력 표의 오분류 타깃이 일치하지 않습니다.")
    metrics = _metrics(predictions)
    measured = pd.read_csv(evaluation_dir / "detection_metrics.csv").set_index("model")
    for row in metrics.itertuples():
        original = measured.loc[row.model]
        checks = {
            "ap": row.ap, "auroc": row.auroc,
            "capture_rate_10": row.capture_10, "capture_rate_20": row.capture_20,
            "n_captured_errors_10": row.captured_10,
            "n_captured_errors_20": row.captured_20,
        }
        for key, value in checks.items():
            if not np.isclose(original[key], value, atol=1e-12, rtol=1e-12):
                raise ValueError(f"원본 평가와 그림 숫자가 다릅니다: {row.model}/{key}")
    metrics.to_csv(output / "plot_metrics_check.csv", index=False)
    manifest = {"source_table": table_path.name, "evaluation_directory": evaluation_dir.name, "evaluation_scope": _scope(predictions), "table_scope": _scope(table)}
    robustness_plot(table, output, manifest)
    comparison_plot(predictions, metrics, output, manifest)
    curves(predictions, output, manifest)
    confidence_plot(table, output, manifest)
    condition_deltas(predictions, output, manifest)
    case_plots(predictions, data_dir, output, manifest)
    importance_plot(evaluation_dir, output, manifest)
    manifest["files"] = sorted(p.name for p in output.glob("*.png"))
    (output / "figure_manifest.json").write_text(json.dumps(_native(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="실측 결과의 발표 그림 생성")
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = generate(args.table, args.evaluation_dir, args.data_dir, args.output)
    print(json.dumps({"figures": result["files"], "scope": result["evaluation_scope"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
