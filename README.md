# PointNet 오분류 위험 예측

연구 질문은 **PointNet의 confidence에 입력 point cloud의 geometry를 추가하면 같은 검토 예산에서 오분류를 더 많이 포착할 수 있는가?**이다. 고정된 공식 PointNet으로 ModelNet40-C를 추론하고, 원본 물체 단위로 나눈 train/validation/test에서 오분류 탐지기를 비교한다.

실험 정의는 [EXPERIMENT_SPEC.md](doc/EXPERIMENT_SPEC.md), 진행·완료 여부는 [EXECUTION_TODO.md](doc/EXECUTION_TODO.md), 수업 요구사항 대조는 [REQUIREMENTS_REVIEW.md](doc/REQUIREMENTS_REVIEW.md)를 따른다. 2026-11-26에 **75/75개 조건·185,100행**의 실험과 독립 검증을 완료했다. [PRESENTATION_HANDOFF.md](PRESENTATION_HANDOFF.md)에 최종 그림·숫자·10분 발표 순서와 예상 질문을 정리했다.

동일 test 494개 물체·37,050행에서 LightGBM confidence→combined의 **AP는 0.7069→0.8055**, **10% 포착률은 28.51%→32.26%(+388건)**, **20%는 52.33%→57.10%(+493건)**였다. 원본 ID bootstrap 95% 구간은 세 차이 모두 0보다 크다. 조건별 AP는 41개 개선·34개 악화하므로 모든 손상에서 개선되었다는 의미는 아니다. 최종 자료는 `results/figures/`와 `results/tables/`, 과거 부분 기록은 `results/provisional74/`로 구분한다.

## 환경

```bash
conda create -n pfp python=3.11.18 pip -y
conda activate pfp
PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL= python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
PIP_CONFIG_FILE=/dev/null PIP_EXTRA_INDEX_URL= python -m pip install -r requirements-lock.txt --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cu128
```

본 서버의 Blackwell GPU를 지원하는 CUDA 12.8 wheel을 사용한다. Python은 3.11.18이며, `requirements-lock.txt`는 실제 `pfp` 환경의 `pip freeze`와 일치하는 Python 패키지 버전 기록이다. `requirements.txt`는 직접 의존성 목록이고 정확한 재현에는 lock 파일을 사용한다. GPU 접근이 제한된 실행 sandbox에서는 GPU 실행 권한이 있는 서버 셸에서 동일 명령을 실행해야 한다.

## 공식 코드와 입력 자산

```bash
mkdir -p vendor weights data
git clone https://github.com/jiachens/ModelNet40-C.git vendor/ModelNet40-C
git -C vendor/ModelNet40-C checkout fad786c430c007cfbb7b4fe90ed1c9873d960921
python scripts/download_modelnet40c.py --output data --workers 4
gdown 1eW26u0nm6HETwDSiCyCEoLLY3WnOVt73 -O weights/pointnet_model_best_test.pth
python scripts/prepare_clean.py --download
```

공식 PointNet source가 포함된 `vendor/ModelNet40-C/pointnet_pyt/pointnet/model.py`를 그대로 읽으므로 위 clone과 commit 고정이 필요하다. 다른 모델의 CUDA 확장은 설치하지 않는다. `vendor/`는 공개 레포에 포함하지 않으며 다시 clone하여 확보한다.

- [공식 코드·표준 PointNet checkpoint](https://github.com/jiachens/ModelNet40-C)
- [공식 손상 데이터](https://zenodo.org/records/6017834): `data/modelnet40_c/data_<corruption>_<level>.npy`, `label.npy`.
- Clean 경로는 `data/modelnet40_ply_hdf5_2048/`이며 `test_files.txt`, `ply_data_test0.h5`, `ply_data_test1.h5`를 사용한다. 공식 Stanford 서버가 응답하지 않아 동일 형식의 HDF5 미러를 사용했다. `prepare_clean.py --download`는 필요한 ZIP이 없을 때만 미러에서 받고, test 데이터·class 이름·원본 파일명 목록 6개만 SHA-256을 검증해 추출한다. 이미 검증된 파일은 재사용한다. 출처와 실측 clean accuracy는 [ASSET_PROVENANCE.md](doc/ASSET_PROVENANCE.md)에 기록했다.

Range 다운로드는 공식 ZIP의 파일별 CRC32와 크기를 검증한 뒤 저장한다. 전체 ZIP을 재구성하지 않으므로 게시된 ZIP MD5 검증과 구분한다. 다시 실행하면 검증된 파일을 재사용한다. `--pilot-only`로 파일럿과 원본 좌표를 먼저 확보할 수 있다.

데이터·checkpoint·원본 첨부 PDF·로그·대용량 캐시는 Git에서 제외한다. 개인 서버 절대 경로는 `local/`과 `artifacts/`에만 남긴다.

## 실행 순서

레포 루트에서 실행한다. GPU 번호는 필요에 따라 `--device cuda:0`을 바꾼다.

입력 자산을 준비한 뒤 다음 명령으로 clean 확인, 파일럿, 전체 실행, 그림, 독립 검증을 순서대로 실행한다. 손상 데이터 다운로드를 다른 셸에서 진행 중이라면 파일럿 입력을 최대 1시간, 전체 실행의 미수신 조건을 최대 누적 2시간 기다린다. 스크립트가 다운로드 자체를 시작하지는 않는다.

```bash
conda activate pfp
bash scripts/run_all.sh
```

서버에서 백그라운드로 실행하고 작업 ID와 통합 로그를 남길 때는 다음 명령을 사용한다. 같은 자산·캐시를 쓰는 작업을 중복 실행하지 않도록 기존 PID를 먼저 확인한다.

```bash
mkdir -p logs local
nohup bash scripts/run_all.sh > logs/run_all.log 2>&1 &
echo $! > local/run_all.pid
cat local/run_all.pid
tail -n 30 logs/run_all.log
```

`local/run_all.pid`는 위 실행에서 저장한 PID이며, `logs/run_all.log`는 통합 로그다. 단계별 로그는 `logs/clean.log`, `logs/pilot.log`, `logs/full.log`, `logs/plots.log`, `logs/audit.log`다. 셸을 종료해도 남겨 두려면 백그라운드 명령을 사용한다. 현재 실행 기록과 완료 상태는 TODO를 확인한다.

개별 단계를 직접 실행할 때는 다음 순서를 따른다.

```bash
conda activate pfp
python -m pytest -q

# 공식 clean 1024점 protocol 및 작은 실제 배치 wrapper 출력 일치 확인
python -m pfp.pointnet --batch-size 128 --output results/clean_metrics.json

# train/validation 공통 물체 100개 × Gaussian/Cutout level 1·5
python -m pfp.pipeline --mode pilot --output artifacts/pilot

# 파일럿 검증 성공 후 전체 75개 조건과 동일 test의 7가지 탐지 기준 평가
python -m pfp.pipeline --mode full --output artifacts/full

# 발표 그림 생성: 표의 AP·AUROC·포착 수를 독립 재집계하여 대조
python -m pfp.plots --table artifacts/full/table.parquet \
  --evaluation-dir artifacts/full/evaluation \
  --data-dir data/modelnet40_c --output results/figures

# 추론 캐시·지표·split을 독립적으로 대조
python -m pfp.audit --table artifacts/full/table.parquet \
  --data-dir data/modelnet40_c --cache-dir artifacts/full/conditions \
  --evaluation-dir artifacts/full/evaluation --output results/audit.json
```

원본 ID는 공식 배열 행 번호이며 `configs/object_splits.csv`에 seed 42로 저장된다. 같은 물체의 75개 손상은 하나의 split에만 속한다. Confidence 3개와 geometry 8개만 탐지기 입력으로 사용한다. 손상 이름·level·정답/예측 클래스·ID는 입력에 포함하지 않는다. Geometry는 모델 정규화 전 원본 손상 좌표에서 계산한다. LiDAR와 Occlusion의 level은 관측 각도이므로 강도 증가로 해석하지 않는다.

LR은 train에서만 scaling하고 C 3개 중 validation AP로 선택한다. LightGBM은 세 feature 조합 모두 같은 2개 후보와 validation AP early stopping을 사용한다. F1 임계값도 validation에서 결정한다. Test는 최종 성능 산출에 사용하며 post-hoc feature 해석을 재학습에 반영하지 않는다.

## 재개와 상태 확인

```bash
cat artifacts/full/status.json
tail -n 30 logs/full.log
python -m pfp.pipeline --mode full --output artifacts/full
```

각 조건은 `artifacts/<pilot|full>/conditions/<corruption>_<level>/`에 logits, 결합 표, 검증 metadata를 저장한다. 데이터·checkpoint·split·공식/프로젝트 코드 hash가 일치하고 내용 검증을 통과한 캐시만 재사용한다. 중단된 조건은 재계산한다. 파일럿과 전체 실행 사이 소스나 핵심 설정을 바꿨으면 파일럿도 다시 검증한다.

추론 캐시가 완성된 뒤 평가만 재실행할 때는 다음 명령을 사용한다.

```bash
python -m pfp.evaluation --table artifacts/full/table.parquet \
  --output artifacts/full/evaluation --bootstrap 300 --importance-samples 6000
```

## 지표와 결과 구조

- AP는 `average_precision_score`다. 무작위 AP 기준은 동일 test의 실제 오분류 비율이다.
- 예산 q에서는 `ceil(q × N)`개를 선택한다. 동점은 `object_id, corruption, severity` 오름차순이다. 포착률은 선택된 오분류 수/전체 test 오분류 수다.
- 핵심 차이는 LightGBM combined − LightGBM confidence이며, 포착률 차이는 percentage point(pp)로 보고한다.
- Brier는 학습형 탐지기의 보정하지 않은 `P(error=1)`에 적용한다. `1−MSP`는 오류 확률로 보정되지 않았으므로 Brier를 보고하지 않는다.
- Bootstrap은 같은 원본 물체의 모든 손상 행을 함께 재표집한다. 95% 구간은 고정된 탐지기의 test 표본 불확실성이며 재학습·분할 변동까지 포함하지 않는다.
- 검토 포착률은 사람이 해당 오류를 실제로 수정했다는 의미가 아니다. 적용 범위는 사용한 ModelNet40-C와 PointNet 체크포인트다.

`artifacts/full/evaluation/`에는 탐지 지표·robustness·곡선·model 선택 기록·test 점수·bootstrap·permutation importance가 저장된다. `results/`에는 공개 가능한 소형 표와 그림을 모은다. 파일럿 결과는 validation 동작 확인용이며 최종 test 결과와 분리한다.

## 개선 효과 중심 발표 그림

큰 글씨의 16:9 그림 8종을 추가했다. [핵심 개선](results/figures/presentation_main_gains.png), [같은 검토 개수의 결과](results/figures/presentation_review_outcomes.png), [실제 10% 진입 사례](results/figures/presentation_case_10.png), [geometry 설명](results/figures/presentation_geometry_explainer.png)을 먼저 사용하면 된다. 그림 선택 안내와 짧은 용어 설명은 [PRESENTATION_HANDOFF.md](PRESENTATION_HANDOFF.md)의 처음에 있다. 조건·사례의 사후 선택 여부와 전체 개선/악화 범위도 그림과 문서에 표시했다.

기존 최종 test 점수·원본 점군에서 재생성한다. 재학습이나 PointNet 재추론은 필요하지 않다.

```bash
conda activate pfp
python scripts/plot_presentation_highlights.py
python scripts/plot_geometry_explainer.py
```

첫 명령은 test ID·75조건·타깃·중복·점수 순위를 확인하고 AP·검토 포착 수·조건별 차이를 재계산해 기존 결과와 대조한다. 입력과 그림 코드의 SHA-256은 `results/figures/presentation_manifest.json`에 저장한다. 실험 당시의 코드 기록은 `results/reproducibility.json`에 보존한다.

## 발표 자료 내보내기

```bash
python scripts/export_results.py
python scripts/package_handoff.py
```

전체 독립 감사를 통과한 소형 결과를 `results/`로 모으고, 문서·그림·표만 들어 있는 `artifacts/PPT_HANDOFF_20261126.zip`을 만든다. 전체 실행의 명령·소요 시간·장애 복구·코드 hash는 [EXECUTION_REPORT.md](doc/EXECUTION_REPORT.md)에 기록했다.
