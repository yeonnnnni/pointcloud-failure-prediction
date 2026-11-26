"""검증을 통과한 소형 실측 결과만 공개 전달 폴더로 모은다."""
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "artifacts/full/evaluation"
output = ROOT / "results/tables"
audit = json.loads((ROOT / "results/audit.json").read_text())
validation = json.loads((ROOT / "artifacts/full/validation.json").read_text())
if not audit.get("passed") or not validation.get("passed") or not validation.get("evaluation_passed"):
    raise RuntimeError("전체 실행·평가·독립 감사 통과 후 결과를 내보내야 함")
output.mkdir(parents=True, exist_ok=True)
names = [
    "detection_metrics.csv", "detection_metrics.json", "evaluation_summary.json",
    "table_validation.json", "model_selection.json", "robustness_by_condition.csv",
    "robustness_by_corruption.csv", "robustness_by_class.csv", "confidence_distribution.csv",
    "review_budget_curve.csv", "risk_coverage_curve.csv", "geometry_delta_by_condition.csv",
    "paired_bootstrap.json", "permutation_importance.csv",
]
for name in names:
    shutil.copy2(source / name, output / name)
for name in ["validation.json", "data_inventory.csv"]:
    shutil.copy2(ROOT / "artifacts/full" / name, output / name)
pilot = ROOT / "results/pilot"
pilot.mkdir(exist_ok=True)
for src in [ROOT / "artifacts/pilot/validation.json",
            ROOT / "artifacts/pilot/evaluation/detection_metrics.csv",
            ROOT / "artifacts/pilot/evaluation/evaluation_summary.json"]:
    shutil.copy2(src, pilot / src.name)
environment = json.loads((ROOT / "artifacts/full/environment.local.json").read_text())
public = {key: environment[key] for key in ["python", "torch", "cuda", "gpu", "batch_size", "geometry_workers", "level_mapping"]}
public["seed"] = 42
public["official_commit"] = json.loads((ROOT / "results/clean_metrics.json").read_text())["official_commit"]
(ROOT / "results/environment.json").write_text(json.dumps(public, indent=2, ensure_ascii=False) + "\n")
print(f"검증된 소형 결과 {len(names)+2}개와 파일럿 범위 기록을 results/에 복사했습니다.")
