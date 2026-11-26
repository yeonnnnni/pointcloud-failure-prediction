#!/usr/bin/env bash
# pfp 환경에서 실행. 중단 후 동일 명령으로 조건 캐시를 재사용한다.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs artifacts local
echo "$$" > local/run_all.pid
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MPLCONFIGDIR="$PWD/artifacts/matplotlib"
echo "작업 시작 PID=$$ $(date -Is)"

python -u -m pfp.pointnet --batch-size 128 --output results/clean_metrics.json 2>&1 | tee logs/clean.log

python -u - <<'PY'
from pathlib import Path
import time
paths = [Path('data/modelnet40_c') / f'data_{c}_{s}.npy'
         for c in ['gaussian', 'cutout'] for s in [1, 5]]
start = time.monotonic()
while any(not path.exists() for path in paths):
    if time.monotonic() - start > 3600:
        raise TimeoutError('파일럿 입력 다운로드 1시간 대기 초과')
    missing = [path.name for path in paths if not path.exists()]
    print(f'파일럿 입력 대기: {missing}', flush=True)
    time.sleep(15)
PY

python -u -m pfp.pipeline --mode pilot --output artifacts/pilot 2>&1 | tee logs/pilot.log
python -u -m pfp.pipeline --mode full --output artifacts/full --wait-data 7200 2>&1 | tee logs/full.log
python -u -m pfp.plots --table artifacts/full/table.parquet \
  --evaluation-dir artifacts/full/evaluation --data-dir data/modelnet40_c \
  --output results/figures 2>&1 | tee logs/plots.log
python -u -m pfp.audit --table artifacts/full/table.parquet \
  --data-dir data/modelnet40_c --cache-dir artifacts/full/conditions \
  --evaluation-dir artifacts/full/evaluation --output results/audit.json 2>&1 | tee logs/audit.log
echo "실험·그림·독립 검증 완료 $(date -Is)"
