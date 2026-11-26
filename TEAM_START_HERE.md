# 팀원 시작 안내: 자료 받기부터 PPT 제작까지

**발표 자료를 읽고 PPT를 만드는 데에는 추가 데이터 다운로드, Python 설치, GPU가 필요하지 않다.** 이 저장소를 클론하거나 ZIP으로 받으면 시작 안내·상세 설명·인수인계 문서 3개, 최종 그림 16개와 결과표를 바로 볼 수 있다. Windows·macOS·Linux 어느 환경에서든 문서와 그림을 열람할 수 있다.

이정연이 실험 구현·실행·검증·해석을 완료했다. 이현택은 자료를 이해한 뒤 PPT를 만들고 **12/3에 10분 발표, 3분 질의응답**을 준비하면 된다. 아래의 **선택: 실험을 직접 재현하기**는 코드를 다시 실행하고 싶은 경우에만 해당한다.

## 먼저 확인: 추가로 무엇을 받아야 하나

**PPT 제작·발표 준비만 한다면 아래 추가 다운로드는 모두 생략한다.** 문서·그림·표와 모델 7개(약44.2MB)는 GitHub 저장소 전체 다운로드에 포함된다. 원본 데이터는 수 GB라 별도로 받는다. 아래 명령은 실험을 다시 실행할 때만, README대로 `pfp` 환경을 준비한 뒤 저장소 루트에서 실행한다.

| 대상 | 어디서·어떻게 받는가 | 최종 위치 |
|---|---|---|
| 발표 문서·최종 그림·표 | 이 저장소 `git clone` 또는 Code → Download ZIP | 루트의 MD, `results/figures/`, `results/tables/` |
| PointNet·탐지기 6개 | 클론에 포함. 파일 확인은 `python scripts/verify_shared_assets.py` | `weights/pointnet_model_best_test.pth`, `weights/detectors/` |
| 공식 PointNet 소스 — 재현할 때만 | 공식 저장소를 `vendor/ModelNet40-C`에 clone, 아래 6.3절의 고정 commit으로 checkout | `vendor/ModelNet40-C/` |
| ModelNet40-C — 재현할 때만 | `python scripts/download_modelnet40c.py --output data --workers 4` ([공식 Zenodo](https://zenodo.org/records/6017834)) | `data/modelnet40_c/`의 NPY 77개, 비압축 약2.28GB |
| Clean 평가 데이터 — 재현할 때만 | `python scripts/prepare_clean.py --download` ([사용한 HDF5 미러](https://drive.google.com/file/d/1BUP46fXOlLVeGKa7PT1CtBEi3s0HGHUi/view)) | `data/modelnet40_ply_hdf5_2048/`의 평가 파일6개 |
| 행 단위 예측표·실행 캐시 | 다운로드 대상이 아님. 아래 재현 명령으로 생성 | `artifacts/full/` |

다운로드가 끊기면 같은 명령을 다시 실행한다. 스크립트가 검증된 파일을 재사용한다. 수동 다운로드가 필요하면 ModelNet40-C ZIP 안의 `label.npy`, `data_original.npy`, `data_<corruption>_<level>.npy`가 **`data/modelnet40_c/` 바로 아래**에 오도록 압축을 푼다. Clean 미러 ZIP은 파일명을 **`data/modelnet40_ply_hdf5_2048_mirror.zip`**으로 맞춘 뒤 `python scripts/prepare_clean.py`를 실행하면 필요한6파일만 검증·추출한다. 상세 폴더 구조와 순서는 아래6절에 있다.

## 1. 자료 받기: 셋 중 하나 선택

### A. 설치 없이 GitHub에서 보기

1. [프로젝트 저장소](https://github.com/yeonnnnni/pointcloud-failure-prediction)를 연다.
2. 이 문서 다음으로 [TEAM_PROJECT_GUIDE.md](TEAM_PROJECT_GUIDE.md)를 누른다. GitHub에서 본문과 삽입 그림이 함께 보인다.
3. PPT에 쓸 그림은 [최종 그림 폴더](results/figures/)에서 PNG 파일을 열어 원본 파일로 저장한다. 파일 화면의 다운로드 버튼 또는 원본 파일 보기를 사용한다.

브라우저 화면을 캡처하기보다 PNG 원본을 넣으면 작은 글씨가 더 선명하다. 파일명이 같은 PDF는 확대해서 보거나 출력할 때 사용할 수 있다.

### B. Git으로 클론하기

Git이 설치되어 있다면 원하는 작업 폴더의 터미널에서 다음을 실행한다. Windows는 PowerShell이나 Git Bash, macOS·Linux는 터미널을 사용할 수 있다.

```bash
git clone https://github.com/yeonnnnni/pointcloud-failure-prediction.git
cd pointcloud-failure-prediction
```

`pointcloud-failure-prediction` 폴더 안에 이 문서와 `results` 폴더가 있으면 정상이다. 이미 클론한 폴더가 있다면 다시 클론하지 않고 아래의 **자료 업데이트**를 따르면 된다.

### C. Git 없이 ZIP으로 받기

1. [프로젝트 저장소](https://github.com/yeonnnnni/pointcloud-failure-prediction)에서 **Code → Download ZIP**을 선택한다.
2. 다운로드한 ZIP을 먼저 압축 해제한다. 기본 폴더명은 보통 `pointcloud-failure-prediction-main`이다.
3. 압축 해제한 폴더 안의 `TEAM_START_HERE.md`를 연다.

ZIP 안에서 MD 파일 하나만 꺼내지 말고 **폴더 전체를 같은 구조로 유지**한다. 문서가 `results/figures/`의 그림을 상대 경로로 읽기 때문이다.

별도로 받은 `PPT_HANDOFF_20261126.zip`도 먼저 압축 해제한 뒤 루트의 시작 안내를 열면 된다. 이 전달용 ZIP은 문서·그림·소형 표 중심의 묶음이며, 실험 코드 전체가 필요하면 저장소를 클론하거나 Code의 ZIP을 받는다.

## 2. 어디부터 읽으면 되는가

| 순서 | 파일·폴더 | 여기서 확인할 것 |
|---|---|---|
| 1 | [TEAM_START_HERE.md](TEAM_START_HERE.md) | 자료를 받는 방법, 읽을 순서, 업데이트 방법 |
| 2 | [TEAM_PROJECT_GUIDE.md](TEAM_PROJECT_GUIDE.md) | 선정 이유, 해결 방법과 설계 이유, 실험 성과, 실제 그림, 발표 구성 |
| 3 | [PRESENTATION_HANDOFF.md](PRESENTATION_HANDOFF.md) | 확정 숫자·그림 경로·해석·예상 질문을 빠르게 다시 확인 |
| 4 | [results/figures/](results/figures/) | PPT에 넣을 PNG 원본 16개와 같은 이름의 PDF |
| 5 | [results/tables/](results/tables/) | 숫자를 확인할 최종 CSV·JSON 결과표 |

상세 설명 문서의 Introduction·Method·Experiment는 각각 **5줄 요약**으로 시작한다. 먼저 이 요약으로 전체 내용을 잡고, 모르는 용어가 나오는 절의 표와 그림을 읽으면 된다. PointNet은 물체 종류를 고르는 고정 분류기이고, 이번에 학습한 LR·LightGBM은 **그 예측의 오분류 위험을 판단하는 별도 탐지기**라는 구분부터 이해하는 것이 좋다.

발표 준비에 바로 필요한 부분은 다음과 같다.

- [내용별 10분 발표 구성](TEAM_PROJECT_GUIDE.md#41-10분-안에-어떤-순서로-전달할까)
- [PPT에 반드시 포함할 내용 점검표](TEAM_PROJECT_GUIDE.md#42-반드시-들어가야-하는-내용)
- [그림 선택 안내](TEAM_PROJECT_GUIDE.md#43-그림은-무엇부터-고르면-좋은가)
- [채점 기준과 보여 줄 근거](TEAM_PROJECT_GUIDE.md#44-채점-기준에-맞춰-무엇을-보여-줄까)
- [예상 질문과 자주 생기는 오해](TEAM_PROJECT_GUIDE.md#45-발표-전에-확인할-질문과-오해)

발표는 **9분 30초 분량으로 리허설하고 30초 여유를 남기는 구성**이다. 위 자료는 내용과 시간 배분의 참고이며, 슬라이드별 대본이나 발표자 노트는 아니다.

## 3. 문서와 그림을 열기

### VS Code로 보는 경우

Markdown 미리보기 없이 읽으려면 [팀원 설명 PDF](TEAM_PROJECT_GUIDE.pdf) 또는 [발표 인수인계 PDF](PRESENTATION_HANDOFF.pdf)를 일반 PDF 뷰어로 열면 된다. 상세 설명 PDF에는 본문 그림11개가 포함되고, 인수인계 PDF는 원문처럼 표와 그림 링크 중심이다.

1. VS Code에서 **파일 → 폴더 열기**로 저장소 폴더 전체를 연다.
2. `TEAM_PROJECT_GUIDE.md`를 선택한다.
3. Markdown 미리보기를 연다. Windows·Linux는 `Ctrl+Shift+V`, macOS는 `Cmd+Shift+V`를 사용하거나, 명령 팔레트에서 `Markdown: Open Preview`를 선택한다.
4. 본문에 그림이 보이면 정상이다. 그림이 안 보이면 압축을 해제했는지, MD와 `results/figures/`가 원래 위치에 있는지 확인한다.

VS Code가 없다면 GitHub에서 읽어도 된다. MD는 일반 텍스트 파일이므로 메모장에서도 내용은 볼 수 있지만, 그림을 함께 보려면 Markdown 미리보기나 GitHub가 편하다.

### 그림을 PPT에 넣는 경우

`results/figures/`의 PNG를 PowerPoint의 **그림 삽입**으로 넣거나 슬라이드로 끌어다 놓는다. 추가 그림 8개는 16:9 비율이며, 기존 그림 8개는 내용에 맞춘 비율이다. 크기를 바꿀 때 가로세로 비율을 유지하고, 축·범례·표본 수·사후 선택 표시가 잘리지 않게 한다.

처음에는 아래 네 그림부터 고르면 된다.

| 용도 | 파일 |
|---|---|
| 핵심 AP와 10%·20% 포착률 개선 | [presentation_main_gains.png](results/figures/presentation_main_gains.png) |
| 같은 개수만 검토했을 때의 오류 포착 수 | [presentation_review_outcomes.png](results/figures/presentation_review_outcomes.png) |
| 실제 고신뢰 오답이 10% 검토 목록에 들어온 사례 | [presentation_case_10.png](results/figures/presentation_case_10.png) |
| Geometry의 bbox·PCA·kNN 개념 | [presentation_geometry_explainer.png](results/figures/presentation_geometry_explainer.png) |

그림 16개를 모두 본 발표에 넣을 필요는 없다. 전체 오류율 히트맵과 조건별 예외는 짧게 제시하고, 상세 비교·검증 그림은 질문에 대비한 보조 자료로 고를 수 있다. 최종 숫자와 해석은 상세 설명 문서를 따른다.

## 4. 숫자를 옮길 때 확인할 것

실험은 **75/75조건·185,100행** 전체를 완료했고, 탐지기 최종 비교는 같은 **test 494개 물체·37,050행**에서 계산했다. 이 test의 오분류는 10,331건이다.

| 핵심 비교 | LightGBM confidence | LightGBM combined | 차이 |
|---|---:|---:|---:|
| AP | 0.70691 | 0.80545 | +0.09854 |
| 10% 검토 포착률 | 28.51% | 32.26% | +3.76pp, 포착 수 순증 388건 |
| 20% 검토 포착률 | 52.33% | 57.10% | +4.77pp, 포착 수 순증 493건 |

- 최종 숫자는 [detection_metrics.csv](results/tables/detection_metrics.csv)와 [인수인계 문서](PRESENTATION_HANDOFF.md)의 표를 사용한다. 차이는 반올림 전 값으로 계산했으므로 표시된 숫자끼리 뺄 때 끝자리가 다를 수 있다.
- 포착률의 분모는 검토한 입력 수가 아니라 **test 전체 오분류 10,331건**이다. 두 방법은 동일한 **개수**를 검토하지만 선택한 입력은 다르다.
- +388/+493건은 오류 포착 수의 **순증**이다. 서로 다른 물체가 그만큼 늘었다거나, PointNet 분류 정확도가 개선됐다는 뜻이 아니다.
- `results/provisional74/`는 과거 **74조건 부분 결과**다. 최종 PPT에는 `results/figures/`와 `results/tables/`만 사용한다.
- 전체 혼합 조건에서는 개선됐지만, 개별 조건의 AP는 41개 개선·34개 악화했다. 모든 손상이나 실제 센서에서도 같은 효과가 입증됐다고 확대하지 않는다.

현재 파일의 검증 근거는 [전체 독립 감사](results/audit.json), [최종 표](results/tables/), [실행 완료 기록](doc/EXECUTION_TODO.md)에 있다. PPT를 만들기 위해 실험을 다시 실행할 필요는 없다.

## 5. 자료 업데이트

Git으로 클론했다면 저장소 폴더에서 다음을 실행한다.

```bash
git status --short
git pull --ff-only
git log -1 --oneline
```

첫 명령은 로컬에서 바뀐 파일이 있는지 확인한다. `git pull`이 로컬 수정 때문에 중단되면 그 수정 내용을 보존한 채 이정연에게 메시지와 파일명을 알려 주면 된다. 마지막 명령의 커밋 번호를 기록하면 어떤 버전으로 PPT를 만들었는지 확인하기 쉽다.

Code → Download ZIP으로 받았다면 `git pull`을 쓰는 대신 최신 ZIP을 다시 받아 **새 폴더에 압축 해제**한다. 직접 만든 PPT와 메모는 별도 작업 폴더에 보관하면 자료를 갱신하기 편하다. 이전 버전의 그림과 새 결과표를 섞지 않도록 같은 버전의 문서·그림을 함께 사용한다.

## 6. 선택: 실험을 직접 재현하기

이 절은 **이미 완성된 자료를 보는 작업과 별개**다. 실험을 다시 실행할 때는 데이터·공식 PointNet 소스·실행 환경이 필요하다. 검증한 환경은 Linux 서버의 `pfp` conda 환경과 NVIDIA CUDA GPU다. Windows·macOS에서도 문서·PNG·PDF는 그대로 볼 수 있지만, 아래 명령으로 해당 운영체제의 실험 실행까지 검증한 것은 아니다.

### 6.1 먼저 README의 환경 설치

[README의 환경 절](README.md#환경)에 있는 명령으로 Python 3.11.18의 `pfp` 환경과 CUDA용 PyTorch·패키지를 준비한다. 아래 명령은 **저장소 루트**에서 `conda activate pfp` 후 실행한다. GPU·CUDA 설정은 사용하는 서버와 맞아야 한다.

### 6.2 모델 가중치 배포 상태

PointNet checkpoint와 학습된 탐지기 6개는 저장소에 함께 제공한다. **정상적으로 클론하거나 최신 저장소 ZIP을 받았다면 가중치를 따로 다운로드하지 않아도 된다.** 발표 자료 열람에는 이 가중치들을 실행할 필요도 없다.

| 제공 파일 | 내용 |
|---|---|
| `weights/pointnet_model_best_test.pth` | 공식 PointNet 원본 checkpoint, 41,745,708 bytes |
| `weights/detectors/lr_confidence.joblib` 등 LR 3개 | Confidence·geometry·combined 탐지기와 학습된 표준화·임계값 |
| `weights/detectors/lgbm_confidence.joblib` 등 LightGBM 3개 | Confidence·geometry·combined 탐지기와 임계값 |

탐지기 6개의 파일 크기 합계는 2,449,329 bytes다. 파일 구성과 사용 방법은 [weights/README.md](weights/README.md), 해시는 [weights/manifest.json](weights/manifest.json)을 참고한다. 아래 명령으로 배포 파일의 크기·해시를 확인한다.

```bash
python scripts/verify_shared_assets.py
```

PointNet 파일이 없으면 `pfp` 환경에서 다음 복구 명령을 사용한다. 이미 올바른 파일이면 재사용한다. 검증에 실패한 파일은 별도로 옮긴 다음 복구 명령을 실행한다.

```bash
python scripts/prepare_pointnet.py --download
```

원래 공식 다운로드 주소와 체크포인트 SHA-256은 [자산 기록](doc/ASSET_PROVENANCE.md)에 있다. 탐지기를 처음부터 다시 학습하면 새 학습 산출물은 `artifacts/full/evaluation/models/`에 생성된다. 공개한 최종 탐지기 위치인 `weights/detectors/`와 구분하면 된다.

### 6.3 공식 소스와 데이터 준비

아래는 처음 준비할 때의 명령이다. 이미 `vendor/ModelNet40-C`가 있으면 다시 clone하지 않고 고정 commit을 확인한다.

```bash
conda activate pfp
mkdir -p vendor weights data
git clone https://github.com/jiachens/ModelNet40-C.git vendor/ModelNet40-C
git -C vendor/ModelNet40-C checkout fad786c430c007cfbb7b4fe90ed1c9873d960921
python scripts/download_modelnet40c.py --output data --workers 4
python scripts/prepare_clean.py --download
```

첫 Python 명령은 공식 ModelNet40-C의 손상 75개와 `label.npy`·`data_original.npy`를 `data/modelnet40_c/`에 준비한다. 두 번째는 clean 평가용 미러 ZIP에서 필요한 파일 6개만 검증해 `data/modelnet40_ply_hdf5_2048/`에 추출한다. 검증된 파일이 이미 있으면 재사용한다.

경로에서 주의할 점은 `--output data`가 최상위 `data/`를 가리킨다는 것이다. 여기에 `--output data/modelnet40_c`를 넣으면 의도하지 않은 중첩 폴더를 만들 수 있다. 기본 명령 그대로 실행하면 아래 구조가 된다.

```text
pointcloud-failure-prediction/
├── TEAM_START_HERE.md
├── TEAM_PROJECT_GUIDE.md
├── PRESENTATION_HANDOFF.md
├── configs/
│   └── object_splits.csv
├── results/
│   ├── figures/                  # 클론에 포함: 최종 PNG 16개와 PDF
│   └── tables/                   # 클론에 포함: 최종 소형 결과표
├── vendor/
│   └── ModelNet40-C/
│       └── pointnet_pyt/pointnet/model.py
├── weights/
│   ├── pointnet_model_best_test.pth  # 클론에 포함
│   └── detectors/                  # 클론에 포함: 최종 탐지기 6개
│       ├── lr_confidence.joblib
│       ├── lr_geometry.joblib
│       ├── lr_combined.joblib
│       ├── lgbm_confidence.joblib
│       ├── lgbm_geometry.joblib
│       └── lgbm_combined.joblib
└── data/
    ├── modelnet40_c/
    │   ├── label.npy
    │   ├── data_original.npy
    │   ├── data_gaussian_1.npy
    │   └── data_<corruption>_<level>.npy  # 손상 75개, level 1~5
    ├── modelnet40_ply_hdf5_2048_mirror.zip
    └── modelnet40_ply_hdf5_2048/
        ├── test_files.txt
        ├── shape_names.txt
        ├── ply_data_test0.h5
        ├── ply_data_test1.h5
        ├── ply_data_test_0_id2file.json
        └── ply_data_test_1_id2file.json
```

위는 **재현 자산을 준비한 뒤의 구조**다. 데이터와 `vendor`는 클론만으로 생기지 않으며, 공식 저장소와 데이터 서버에서 별도로 받는다. 데이터 다운로드 시간은 서버 상태에 따라 달라진다. 파일 크기·CRC·SHA와 출처는 [자산 기록](doc/ASSET_PROVENANCE.md)에서 확인할 수 있다.

### 6.4 실행과 재개

환경·공식 소스·PointNet checkpoint·데이터를 모두 준비한 뒤 실행한다.

```bash
conda activate pfp
bash scripts/run_all.sh
```

이 스크립트는 clean 점검→파일럿→75조건 실행→기본 그림→독립 검증 순서다. `run_all.sh` 자체는 데이터를 다운로드하지 않는다. 원래 결과와 비교하려면 재실행 전에 기존 `results/`를 별도 복사해 두면 된다. 실행하면서 같은 경로의 결과와 로그가 갱신된다.

진행 상황은 `artifacts/full/status.json`, 단계별 기록은 `logs/clean.log`, `logs/pilot.log`, `logs/full.log`, `logs/plots.log`, `logs/audit.log`에서 확인한다. 동일 설정의 완료 조건 캐시가 있으면 다시 실행할 때 검증 후 재사용한다. 백그라운드 실행·중단 후 재개·추가 발표 그림 재생성은 [README](README.md)의 명령을 따른다.

가중치만 받았다고 최종 test 예측표·모든 그림을 즉시 재생성할 수 있는 것은 아니다. 원본 점군과 필요한 추론·특징·평가 산출물이 함께 있어야 한다. 저장소에 제공된 그림을 PPT에 넣는 작업에는 이 과정이 필요하지 않다.

## 7. 질문할 때 함께 보내면 좋은 정보

내용 질문은 **문서 이름·절 번호·이해되지 않는 문장 또는 숫자**를 보내면 바로 같은 위치를 확인할 수 있다. 그림 질문은 **PNG 파일명과 어느 부분인지**를 알려 주면 된다. 숫자가 다른 경우에는 비교한 문서·표의 파일명을 모두 적어 준다.

실행 문제라면 운영체제, 실행한 명령, 오류 메시지, `git log -1 --oneline` 결과와 해당 로그 파일의 마지막 부분을 함께 전달한다. 로그인 정보나 토큰을 보낼 필요는 없다.

## 8. 팀원에게 보낼 메시지 초안

아래는 자료 전달용 메시지이며 발표 대본이 아니다.

> 현택아, PointNet 오분류 탐지 프로젝트의 전체 75조건 실험과 검증을 끝냈고 설명 문서·결과표·그림을 저장소에 정리했어.
>
> 저장소: https://github.com/yeonnnnni/pointcloud-failure-prediction
>
> GitHub에서 바로 읽어도 되고, 클론하거나 Code → Download ZIP으로 받아도 돼. PPT를 만드는 데에는 Python이나 GPU가 필요 없고 데이터도 추가로 받을 필요 없어.
>
> PointNet과 학습된 탐지기 가중치도 저장소에 넣었어. 원본 데이터와 대용량 캐시는 포함하지 않았고, 실험을 직접 재현하고 싶을 때만 시작 안내의 선택 절과 README를 따르면 돼.
>
> `TEAM_START_HERE.md` → `TEAM_PROJECT_GUIDE.md` → `PRESENTATION_HANDOFF.md` 순서로 읽어 줘. 상세 설명 문서에는 선정 이유·방법을 그렇게 설계한 이유·실험 결과와 실제 그림이 들어 있고, 각 큰 절 앞에 5줄 요약을 붙였어. 마지막에는 10분 발표 구성, 꼭 넣을 내용, 예상 질문도 정리했어.
>
> PPT에는 `results/figures/`의 PNG를 원본으로 넣으면 돼. 최종 그림은 16개고 전부 넣을 필요는 없어. 우선 핵심 개선, 같은 검토 개수, 실제 10% 사례, geometry 설명 그림을 보면 돼. 최종 숫자는 `results/tables/`와 인수인계 문서 기준이고, `results/provisional74/`는 과거 부분 결과라 사용하지 않으면 돼.
>
> 핵심은 같은 LightGBM에서 geometry를 더했을 때 AP가 0.70691→0.80545로 올랐고, 10%·20% 검토에서 오류 포착이 각각 388건·493건 순증했다는 거야. PointNet 자체의 분류 정확도가 높아진 결과는 아니라는 점만 구분해 줘.
>
> 12/3 발표 10분·질의응답 3분에 맞춰 준비하고, 이해가 안 되는 부분은 문서 절이나 그림 파일명을 알려 주면 같이 확인하자. 자료가 업데이트되면 클론한 폴더에서 `git pull --ff-only`로 받으면 돼.
