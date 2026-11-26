# 공유 모델 파일

최종 실험에 사용한 **PointNet 원본 checkpoint 1개와 학습한 오분류 탐지기 6개**를 저장소에 포함했다. Git clone 또는 GitHub의 Code → Download ZIP으로 함께 받는다. Git LFS를 추가로 설치할 필요는 없다. 문서·그림을 읽고 PPT를 만드는 데에는 이 모델들을 실행하지 않아도 된다.

| 위치 | 내용 | 파일 크기 |
|---|---|---:|
| `pointnet_model_best_test.pth` | 공식 PointNet 원본 checkpoint, optimizer 등 원래 항목을 그대로 유지 | 41,745,708 bytes |
| `detectors/lr_confidence.joblib` | Confidence 3개 입력의 LR | 2,059 bytes |
| `detectors/lr_geometry.joblib` | Geometry 8개 입력의 LR | 2,374 bytes |
| `detectors/lr_combined.joblib` | 11개 입력의 LR | 2,511 bytes |
| `detectors/lgbm_confidence.joblib` | Confidence 3개 입력의 LightGBM | 396,447 bytes |
| `detectors/lgbm_geometry.joblib` | Geometry 8개 입력의 LightGBM | 1,248,797 bytes |
| `detectors/lgbm_combined.joblib` | 11개 입력의 LightGBM | 797,141 bytes |

전체 모델은 **44,195,037 bytes, 약44.2MB**다. 파일별 SHA-256은 [manifest.json](manifest.json)에 기록했다. 원본 데이터와 행 단위 대용량 캐시는 포함하지 않는다. 발표 전용 전달 ZIP에도 모델 바이너리는 넣지 않으므로, 모델이 필요하면 저장소 전체를 받는다.

## 출처와 재현 설정

- PointNet: [ModelNet40-C 공식 코드](https://github.com/jiachens/ModelNet40-C), 고정 commit `fad786c430c007cfbb7b4fe90ed1c9873d960921`.
- 공식 checkpoint: [Google Drive 다운로드](https://drive.google.com/file/d/1eW26u0nm6HETwDSiCyCEoLLY3WnOVt73/view).
- 공식 저장소 라이선스 및 관련 고지는 [THIRD_PARTY_NOTICES.txt](../doc/THIRD_PARTY_NOTICES.txt)에 보존했다. 가중치는 공식 배포 파일을 변경하지 않은 복사본이며, 별도의 가중치 전용 라이선스 선언이 있다는 의미는 아니다.
- 탐지기 6개는 본 프로젝트 train 물체로 학습하고 validation에서 선택한 최종 모델의 복사본이다. Test로 다시 학습하지 않았다.
- 환경과 설정: [requirements-lock.txt](../requirements-lock.txt), [model_selection.json](../results/tables/model_selection.json).

## 무결성 확인과 PointNet 복구

저장소 루트에서 실행한다. 아래 검증 명령은 Python 표준 라이브러리만 사용하고 모델을 실행하지 않는다.

```bash
python scripts/verify_shared_assets.py
```

PointNet만 확인하려면 `python scripts/prepare_pointnet.py`를 사용한다. 파일을 잃어버렸다면 `pfp` 환경에서 다음을 실행한다.

```bash
python scripts/prepare_pointnet.py --download
```

이미 올바른 파일이면 다운로드하지 않는다. 크기나 SHA가 다른 파일이 있으면 자동으로 덮어쓰지 않으므로 그 파일을 별도로 옮긴 뒤 복구 명령을 실행한다. 직접 브라우저로 받았다면 파일명을 `pointnet_model_best_test.pth`로 맞춰 저장소의 `weights/` 바로 아래에 둔다.

## 탐지기를 읽는 방법

`joblib` 파일에는 `model`, `features`, `threshold`, `seed`, `configuration`이 들어 있다. LR의 `model`에는 train에서 학습한 scaler도 포함되어 있다. `features` 순서대로 입력 표를 선택하면 학습 때의 특징 순서가 유지된다. `threshold`는 validation에서 정한 F1용 값이며, 상위10%·20% 평가에서는 점수를 정렬해 개수로 선택했다.

다음은 이미 특징 표 `feature_table`을 준비한 경우의 코드 형태다. 특징 표는 실제 점군에서 계산해야 하며, 이 예제가 데이터 다운로드를 대신하지는 않는다.

```python
import joblib
bundle = joblib.load('weights/detectors/lgbm_combined.joblib')
risk = bundle['model'].predict_proba(feature_table[bundle['features']])[:, 1]
```

모델을 실행하려면 README의 `pfp` 환경을 사용한다. 탐지기 자체는 CPU에서 사용할 수 있으며 PointNet 재추론·전체 실험은 공식 소스와 데이터 준비가 추가로 필요하다. 자세한 시작 순서는 [TEAM_START_HERE.md](../TEAM_START_HERE.md)를 따른다.
