# 3D 포인트 클라우드 오분류 위험 예측 실험 명세

## 목표와 범위

**연구 질문.** PointNet의 confidence에 입력 point cloud의 기하 통계를 추가하면, 같은 검토 예산에서 오분류를 더 많이 포착할 수 있는가?

1. ModelNet40-C의 손상 유형과 level에 따라 PointNet의 오분류율이 어떻게 달라지는지 측정한다.
2. 같은 held-out 물체에서 confidence-only, geometry-only, confidence + geometry의 탐지 성능을 비교한다. 핵심 비교는 동일한 LightGBM에서 geometry를 추가했을 때의 차이다.
3. 어떤 feature가 탐지 점수와 관련되는지 분석하고, 검토 예산 10%와 20%에서 발견할 수 있는 오분류의 비율을 보고한다.

**일정과 담당.** 2026-11-26(KST) 안에 핵심 실험과 발표 인수인계 초안까지 만드는 것이 목표다. 이정연이 실험 코드, 실행, 결과 검증·해석과 인수인계를 맡는다. 이현택은 11/30까지 결과를 받아 PPT를 만들고 12/3에 발표한다. 발표 10분, 질의응답 3분이다.

## 1. 데이터와 분류기

| 항목 | 실행 기준 |
|---|---|
| 데이터 | ModelNet40-C 공식 배포본을 사용한다. 예상 규모는 40개 클래스, 원본 물체 2,468개, 손상 15종 × 5개 level, 총 185,100개 손상 샘플이다. 각 조건의 실제 배열 크기와 원본 물체 수를 기록한다. |
| Clean | 체크포인트에 맞는 공식 ModelNet40 clean 평가 데이터를 사용한다. Clean accuracy는 분류기 재현 점검에 사용한다. Clean과 손상 배열의 물체별 대응을 별도로 검증하기 전에는 행 번호로 둘을 결합하지 않는다. |
| 분류기 | ModelNet40-C 공식 표준 학습 PointNet 체크포인트 1개를 고정한다. `eval()`과 inference-only 모드를 사용하고 공식 설정·전처리를 따른다. Checkpoint를 strict load하고 예측 출력의 40개 클래스 차원과 유한값을 확인한다. |
| 실행 기록 | 공식 코드의 커밋, 사용한 clean·corruption 설정, 체크포인트 파일명과 SHA-256, 데이터 출처·경로, Python·PyTorch·CUDA와 실행 명령을 기록한다. 공식 표의 clean accuracy 90.7%와 평균 손상 오류율 28.3%는 동일 설정을 점검할 참고값이다. 큰 차이가 있으면 체크포인트, class order, 전처리와 평가 split을 확인한다. |

공식 배포 로더의 공통 `label.npy`와 조건별 배열의 대응을 확인한다. 공식 배열의 **원본 행 번호**를 `object_id`로 저장하며, 파일 순회나 배치 처리 과정에서 재번호를 매기지 않는다. 조인 키는 `(object_id, corruption, severity)`다. 저장 숫자와 논문의 표시 level 1~5가 다르면 대응표를 한 번 정의한다. LiDAR와 Occlusion의 다섯 level은 각각 다른 관측 각도이므로 손상의 순서형 강도로 해석하지 않는다.

- [ModelNet40-C 공식 저장소 및 체크포인트](https://github.com/jiachens/ModelNet40-C)
- [공식 데이터 로더](https://github.com/jiachens/ModelNet40-C/blob/master/dataloader.py)
- [ModelNet40-C 논문, §4 및 표 1~2](https://arxiv.org/pdf/2201.12296)

## 2. 샘플 표와 feature

한 행은 원본 물체 하나에 손상 조건 하나를 적용한 point cloud다. `error = int(pred_label != true_label)`이며 오분류가 양성이다. 각 행에는 조인 키, `split`, `true_label`, `pred_label`, `error`와 세 confidence feature 및 geometry feature를 남긴다. 40개 logits는 조건별로 캐시한다. 위험 예측기의 입력은 아래 feature 목록을 명시적으로 선택해 구성한다. 정답·예측 클래스 ID, 원본 ID, 조인 키, split, 손상 유형·level은 기록과 분석에 사용한다.

| Feature | 계산 방법 |
|---|---|
| `msp`, `entropy`, `margin` | 클래스 logits에 안정적인 softmax를 적용한다. `msp = p₁`, `entropy = −Σ p log p`, `margin = p₁ − p₂`다. `p₁`과 `p₂`는 가장 큰 확률 두 개다. |
| `pca_ratio_1`~`3` | 손상 좌표를 중심화해 공분산 고윳값을 내림차순으로 구하고 각각을 고윳값 합으로 나눈다. 수치 오차로 생긴 음의 고윳값은 0으로, 합이 사실상 0이면 세 비율을 0으로 기록한다. |
| `bbox_x`, `bbox_y`, `bbox_z` | 손상 좌표의 각 축에서 `max − min`을 계산한다. |
| `knn_dist_mean`, `knn_dist_std` | 점마다 자기 자신의 인덱스를 뺀 가장 가까운 `min(8, N−1)`개 이웃과의 유클리드 거리를 모아 평균·표준편차를 계산한다. 좌표가 같은 별개 점은 유지한다. `N < 2` 또는 비유한 좌표가 있으면 조건과 ID를 기록하고 원인을 확인한다. |

Geometry는 원본 손상 XYZ에서 계산한다. 모델용 추가 정규화나 재샘플링이 필요할 때는 추론 경로에만 적용한다. 조건별 점 수가 다를 수 있으므로 파일 단위로 읽고, 전체 좌표를 한 메모리 배열에 올리지 않는다. kNN은 `cKDTree` 등으로 계산하고 파일럿에서 실행 시간을 먼저 잰다.

## 3. 검증 설계와 실행 순서

1. **분할을 저장한다.** 원본 물체 ID를 가능한 한 클래스 비율을 유지하도록 train/validation/test 약 60/20/20으로 나눈다. 기본 seed는 `42`이며 실제 ID 목록과 정확한 split별 수를 저장한다. 같은 ID의 손상 버전 75개를 같은 split에 넣고 세 집합의 교집합이 비어 있는지 확인한다. 이 분할은 오분류 탐지기용이며 PointNet은 고정한다.
2. **파일럿을 실행한다.** 기본은 Gaussian과 Cutout의 표시 level 1·5, train/validation에 속한 공통 원본 ID 약 100개다. 추론 → geometry 추출 → 1:1 결합 → Logistic Regression → 점수 산출까지 실제 실행한다. 파일럿에서 양성 또는 음성이 부족하면 같은 train/validation 안에서 샘플을 늘린다. 행 수, 중복·누락, feature 유한값과 처리 시간까지 확인한다.
3. **전체 데이터로 확장한다.** 75개 조건을 순회하고 추론과 feature를 조건별로 저장한다. 검증을 통과한 조건만 완료로 표시한다. 재시작 시 데이터·체크포인트·전처리·feature 설정이 같은 유효한 캐시를 재사용한다. 전체 결합에는 중복 키·누락 행이 없어야 하며, ID당 75개 조건과 예상 전체 행 수를 확인한다.
4. **비교를 고정한다.** `1 − MSP`를 학습 없는 기준선으로 사용한다. Logistic Regression과 LightGBM 각각에서 confidence-only, geometry-only, confidence + geometry를 같은 분할로 비교한다. Logistic Regression의 scaling은 train에서 fit한다. LightGBM은 같은 설정 또는 같은 탐색 예산을 적용하고 validation에서 선택한다. feature 선택, 보정, threshold, early stopping은 train/validation에서 끝낸 뒤 동일 test에 적용한다.

## 4. 평가와 해석

| 구분 | 산출할 값 |
|---|---|
| PointNet robustness | Clean accuracy와 전체 손상 샘플의 15×5 오류율을 계산한다. 손상 유형별 accuracy·macro-F1은 해당 유형의 다섯 level을 합쳐 산출하고, level별 오류율을 별도로 기록한다. 클래스별 오류율과 정답·오답의 confidence 분포는 결과 해석에 활용한다. 표와 그림에 사용한 모집단과 행 수를 표시한다. |
| 오분류 탐지 | 같은 test에서 AP(`average_precision_score`)와 AUROC를 산출한다. Test의 오분류 비율을 AP의 무작위 기준으로 함께 표시한다. LightGBM의 결합 모델과 confidence-only 모델 간 차이를 보고한다. |
| 상위 10%·20% 검토 | 위험도가 큰 순서로 `ceil(q × N)`개를 선택한다. 동점은 공통의 결정적 규칙으로 처리한다. 선택된 오분류 수 ÷ test 전체 오분류 수로 포착률을 계산하고 분자·분모를 저장한다. 무작위 검토의 기대 포착률은 실제 선택 비율이다. |
| 제안서 보조 평가 | 오분류 탐지 F1의 임계값은 validation F1을 최대화하는 값으로 고정한다. Brier score는 학습형 탐지기의 `P(error=1)`에 적용하고 보정 여부를 명시한다. Risk–coverage 곡선은 고위험 샘플부터 제외한 뒤 남은 샘플의 비율과 그 안의 오분류율을 그린다. |
| Feature 해석 | 주요 수치가 나온 뒤 LightGBM의 SHAP 또는 held-out permutation importance로 관련 feature를 확인한다. 사용 표본과 방법을 적고 관찰된 연관성으로 해석한다. |

전체 성능을 먼저 확정한 다음 손상 유형이나 고신뢰 오답 구간을 분석한다. 평균 개선이 작으면 개선된 조건과 악화된 조건을 모두 확인한다. 차이가 작을 때 불확실성을 더 확인할 시간이 있으면 같은 `object_id`의 모든 손상 행을 함께 재표집하는 paired bootstrap을 사용한다. 다른 분류기나 미관측 손상 유형 실험은 핵심 결과의 검증과 발표 인수인계가 끝난 뒤 진행한다. 발표의 결론 범위는 ModelNet40-C와 사용한 PointNet 체크포인트다.
