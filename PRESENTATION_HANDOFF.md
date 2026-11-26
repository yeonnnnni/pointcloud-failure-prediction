# 발표 인수인계: PointNet 오분류 검토 우선순위

이정연은 구현·실행·검증·해석을, 이현택은 PPT 작성과 12/3 발표를 담당한다. 발표 10분과 질의응답 3분이며 리허설 목표는 9분 30초다. 11/30까지 전달할 문서·그림·코드 버전을 이 레포에 모은다.

**최종 상태(2026-11-26 KST): 75/75개 조건 전체 실험, 독립 감사, 기본 그림 8종 검수 완료. 개선 효과를 크게 보여 주는 추가 그림은 아래 안내를 따른다.** 이 문서의 최종 표는 동일 test에서 검증한 실측값이다. 파일럿·74개 조건 부분 기록은 최종 결과와 분리했다. 발표에 쓸 폴더는 `results/figures/`와 `results/tables/`다.

## 추가 그림: 개선 결과부터 보기

추가 그림은 16:9 비율로 만들었다. PNG를 PPT에 그대로 넣으면 되며, 대본이나 발표자 노트는 별도로 만들지 않았다. **핵심 개선·같은 검토량·실제 사례·geometry 설명** 네 종류를 우선 보고, 나머지는 설명이 더 필요할 때 선택한다. 전체 비교표와 기존 그림은 아래에 그대로 남아 있다.

| 용도 | 그림 파일 | 보여 주는 내용 |
|---|---|---|
| **핵심 개선 한눈에 보기** | [presentation_main_gains.png](results/figures/presentation_main_gains.png) | AP **+0.09854**, 10% 포착률 **+3.76pp**, 20% **+4.77pp**. 축을 0부터 표시했다. |
| **똑같은 개수만 검토하기** | [presentation_review_outcomes.png](results/figures/presentation_review_outcomes.png) | 검토 개수를 3,705건·7,410건으로 제한했을 때 오분류 포착 수 **+388건·+493건**. 막대 안의 정답/오답을 구분했다. |
| **10% 안으로 들어온 실제 오답** | [presentation_case_10.png](results/figures/presentation_case_10.png) | LiDAR 2, ID 2182. glass_box→desk, MSP **0.92592**. 순위 **28,309→3,047위**, 상위 **76.41→8.22%**. |
| **20% 안으로 들어온 실제 오답** | [presentation_case_20.png](results/figures/presentation_case_20.png) | LiDAR 2, ID 813. dresser→desk, MSP **0.91081**. 순위 **32,234→4,368위**, 상위 **87.00→11.79%**. 10% 안에는 들지 않는다. |
| **Geometry가 무엇인지** | [presentation_geometry_explainer.png](results/figures/presentation_geometry_explainer.png) | 실제 점군에서 크기(bbox), 퍼짐 방향(PCA), 이웃 간 거리(kNN)를 그림으로 설명한다. 성능 증거와는 구분한다. |
| 개선이 큰 조건 | [presentation_condition_highlights.png](results/figures/presentation_condition_highlights.png) | 조건 내부 AP 개선 상위 6개와 전체 75조건 분포. 최고는 Occlusion 2 **+0.06125**. 전체는 **41개 개선·34개 악화**다. |
| 추가 포착의 구성 | [presentation_review_turnover.png](results/figures/presentation_review_turnover.png) | 10%: 새 진입 **1,841−이탈 1,453=순증 388건**. 20%: **1,562−1,069=493건**. |
| 개선의 불확실성 | [presentation_gain_intervals.png](results/figures/presentation_gain_intervals.png) | AP·포착률 차이와 물체 단위 bootstrap 95% 구간. 세 구간이 모두 0보다 크다. |

색상은 **파랑=confidence**, **주황=confidence+geometry**다. 사례의 초록 음영은 먼저 검토하는 범위이며, 순위가 작을수록 더 일찍 검토한다. 각 그림에 실제 사용 범위를 표시했다. 결과 그림은 같은 파일명의 PDF도 제공한다.

처음 볼 때 필요한 용어만 정리하면, **point cloud**는 물체를 나타내는 3차원 점들의 집합, **confidence**는 PointNet의 예측 확신을 요약한 값, **geometry**는 점들의 크기·퍼짐·이웃 간격이다. **AP**는 오답을 앞쪽에 모으는 순위의 품질이고, **포착률**은 전체 오답 중 검토 목록에 포함된 비율이다. **pp**는 퍼센트포인트로, 28.51%→32.26%의 차이가 3.76pp다.

+388/+493은 **포착 수의 순증**이며 새로운 물체 수가 아니다. 1건은 물체 하나의 손상 버전 하나다. 두 방법의 검토 **개수**는 같지만 선택한 입력은 다르다. PointNet의 분류 결과는 고정되어 있으므로 이 숫자를 분류 정확도 개선이라고 쓰지 않는다.

조건 상위 6개와 실제 사례는 결과를 이해하기 위한 **사후 선택**이다. 전체 효과는 전체 test 비교로 판단한다. 조건별 AP 차이 평균 **+0.00661**과 여러 조건을 섞어 계산한 핵심 AP 차이 **+0.09854**는 집계 방식이 다르다. 사례를 골랐다고 모델을 다시 학습하거나 test에 맞춰 설정을 바꾸지는 않았다.

추가 그림의 수치는 [presentation_manifest.json](results/figures/presentation_manifest.json), 조건별 재계산은 [presentation_condition_values.csv](results/figures/presentation_condition_values.csv)에 있다.

## 발표 질문과 방법

“PointNet의 confidence에 입력 point cloud의 geometry를 추가하면, 같은 검토 예산에서 오분류를 더 많이 포착할 수 있는가?” 모델의 분류 결과를 바꾸는 실험이 아니라, 사람이 먼저 살펴볼 입력의 순서를 정하는 실험이다.

공식 표준 학습 PointNet 한 개를 고정하고 ModelNet40-C의 원본 물체 2,468개에 대한 15종 × 5개 level을 분석한다. 분류가 틀리면 `error=1`이다. Confidence 3개(MSP, entropy, margin)와 geometry 8개(PCA 비율 3개, bbox 3축, kNN 거리 평균·표준편차)를 추출한다. Geometry는 손상된 원본 XYZ에서 계산한다.

원본 물체 ID 단위 seed 42 split은 train 1,480개 / validation 494개 / test 494개다. 같은 물체의 손상 75개는 한 split에만 속한다. 클래스·ID·손상명·level·정답은 탐지 입력에서 제외한다. LR은 train에서만 scaling하고, LR·LightGBM의 설정과 F1 임계값은 validation에서 선택한다.

비교는 `1−MSP`, LR의 confidence/geometry/combined, LightGBM의 confidence/geometry/combined 총 7개다. 핵심은 **같은 LightGBM에서 geometry 추가 전후의 차이**다. AP는 오분류 탐지 순위의 품질이며 무작위 기준은 test의 오류 비율이다. 상위 10%·20% 포착률은 선택된 오류 수를 test 전체 오류 수로 나눈 값이다.

## 지금 확정된 재현 검증

| 항목 | 검증한 실측값 | 발표 해석 |
|---|---:|---|
| Clean accuracy | 2,238/2,468 = 90.6807% | 공식 90.7%와 반올림 수준에서 일치한다. |
| 공식 wrapper 출력 대조 | 실제 배치에서 최대 절대 차이 0 | 독립 추론 경로가 공식 연산과 일치한다. |
| 파일럿 | train 60개 + validation 40개, 4개 조건, 총 400행 | 추론부터 탐지 평가까지 실제 연결했다. |
| 파일럿 무결성 | 누락 0·중복 0·비유한값 0·split 교집합 0 | 같은 원본 물체가 학습과 평가에 섞이지 않는다. |

원본은 `results/clean_metrics.json`, `results/corruption_parity.json`, `results/pilot/validation.json`이다. 파일럿은 최종 test를 사용하지 않았다.

## 최종 그림·숫자: 전체 75개 조건 실측

**발표 결론:** 이번 PointNet과 ModelNet40-C에서 75개 손상 조건을 섞어 검토할 때, confidence에 geometry를 추가하면 같은 예산으로 더 많은 오분류를 포착했다. 모든 손상 조건에서 좋아진 것은 아니다.

전체 robustness는 **2,468개 물체 × 75개 조건 = 185,100행**을 사용한다. 탐지 성능은 **test 494개 물체 × 75개 조건 = 37,050행**에서 계산했고, 그중 오분류는 **10,331개(27.8839%)**다. AP의 무작위 기준은 **0.27884**다. 원본 ID 교집합·중복·누락·비유한 feature는 모두 0이다.

| 방법 | AP | AUROC | 10% 검토 포착률 | 20% 검토 포착률 | F1 | Brier |
|---|---:|---:|---:|---:|---:|---:|
| 1 − MSP | 0.7020 | 0.8691 | 28.32% | 51.93% | 0.6825 | — |
| LR · confidence | 0.7026 | 0.8697 | 28.29% | 52.08% | 0.6828 | 0.1271 |
| LR · geometry | 0.6173 | 0.7478 | 29.55% | 44.41% | 0.5158 | 0.1577 |
| LR · combined | 0.7776 | 0.8984 | 31.18% | 56.06% | 0.7179 | 0.1116 |
| LightGBM · confidence | 0.7069 | 0.8758 | 28.51% | 52.33% | 0.6854 | 0.1256 |
| LightGBM · geometry | 0.7022 | 0.8225 | 31.35% | 51.16% | 0.6151 | 0.1389 |
| LightGBM · combined | 0.8055 | 0.9135 | 32.26% | 57.10% | 0.7413 | 0.1043 |

F1의 임계값은 validation에서 정했고 test에서 조정하지 않았다. 학습형 탐지기의 Brier는 보정하지 않은 `P(error=1)`에 대한 값이다. `1−MSP`는 보정된 오류 확률이 아니므로 Brier를 보고하지 않는다. 원본 표는 [detection_metrics.csv](results/tables/detection_metrics.csv)다.

### 발표에서 꼭 읽을 핵심 비교

| 지표 | LightGBM confidence | LightGBM combined | 추가 효과 | 물체 단위 bootstrap 95% 구간 |
|---|---:|---:|---:|---:|
| AP | 0.70691 | 0.80545 | +0.09854 | [0.08410, 0.11304] |
| 10% 검토 (3,705개) | 2,945/10,331 = 28.51% | 3,333/10,331 = 32.26% | **+388건, +3.76pp** | [2.94, 4.60]pp |
| 20% 검토 (7,410개) | 5,406/10,331 = 52.33% | 5,899/10,331 = 57.10% | **+493건, +4.77pp** | [3.76, 6.05]pp |

표의 pp는 퍼센트포인트(%p)이며, 상대적인 증가율(%)과 구분한다.

수치 해석: 검토 개수를 각각 3,705건·7,410건으로 같게 제한했을 때, geometry 추가에 따른 오류 포착 수의 순증은 각각 388건·493건이다.

Bootstrap은 test 원본 물체 494개를 300회 재표집하면서 물체별 75개 행을 함께 뽑았다. 세 차이의 95% 구간이 모두 0보다 크다. 이는 고정된 탐지기와 이 test 표본의 불확실성이며, 다른 split·재학습·다른 모델의 변동까지 포함하지 않는다. 구간 원본은 [paired_bootstrap.json](results/tables/paired_bootstrap.json)이다.

### 핵심 그림 4개

PNG를 그대로 PPT에 넣으면 된다. 같은 파일명의 PDF도 확대·출력용으로 제공한다.

| 그림 | 파일 | 발표에서 읽을 숫자 | 해석 한 문장 |
|---|---|---|---|
| 손상별 오류율 히트맵 | [corruption_heatmap.png](results/figures/corruption_heatmap.png) | 전체 오류율 **52,400/185,100 = 28.3090%**, background level 5 **95.30%**, density_inc level 1 **10.29%** | 손상 유형에 따라 PointNet의 실패 빈도가 크게 다르다. |
| 탐지 성능 비교표 | [detection_comparison.png](results/figures/detection_comparison.png) | 같은 LightGBM AP **0.7069→0.8055**, AUROC **0.8758→0.9135** | 동일 알고리즘과 탐색 예산에서 geometry를 더한 조합의 전체 검토 순위가 좋아졌다. |
| 검토 예산 그래프 | [review_budget.png](results/figures/review_budget.png) | 10%에서 **28.51→32.26%**, 20%에서 **52.33→57.10%** 포착 | 같은 검토량으로 더 많은 기존 오류를 골랐으며, 실제 수정 효과를 측정한 것은 아니다. |
| 실제 point cloud 사례 | [pointcloud_cases.png](results/figures/pointcloud_cases.png) | 아래 사례표의 정답·예측·MSP·순위를 그대로 사용 | 개선 사례와 놓친 오류·오경보를 함께 보면 평균 개선의 한계가 드러난다. |

히트맵의 색 범위는 **0~100% 오류율**이고 각 칸은 2,468행이다. LiDAR와 Occlusion의 level은 관측 각도이므로 “level이 커질수록 손상이 심하다”고 설명하지 않는다.

| 실제 test 사례 | 정답 → PointNet 예측 | MSP | confidence 검토 순위 | combined 검토 순위 | 발표 해석 |
|---|---|---:|---:|---:|---|
| lidar level 2, ID 813 | dresser → desk | 0.91081 | 32,234/37,050 (상위 87.00%) | 4,368/37,050 (상위 11.79%) | 20% 검토에서는 combined만 이 오답을 포착한다. 10% 안에는 들지 않는다. |
| upsampling level 5, ID 1205 | plant → flower_pot | 0.93521 | 17,335/37,050 (상위 46.79%) | 33,973/37,050 (상위 91.70%) | Geometry를 추가했지만 검토 순위가 낮아진 실패 사례다. |
| background level 1, ID 908 | tv_stand → tv_stand | 0.08455 | 71/37,050 (상위 0.19%) | 115/37,050 (상위 0.31%) | 실제로 맞힌 샘플도 고위험으로 선택되는 오경보가 있다. |

첫 사례는 dresser를 desk로 분류한 입력이다. MSP가 **0.91081**인데 오답이고, combined를 쓰면 검토 순위가 32,234위에서 4,368위로 올라간다. 모든 고신뢰 오답을 잡는다는 뜻은 아니다. 사례는 설명을 위한 사후 선택이며 전체 효과는 위 test 표로 판단한다.

### 보조 그림과 해석 범위

| 그림 | 파일 | 핵심 수치·해석 |
|---|---|---|
| Risk–coverage | [risk_coverage.png](results/figures/risk_coverage.png) | 고위험 20% 제외 후 80%를 남겼을 때, 남은 오류율은 confidence **16.62%**, combined **14.95%**다. 실제 수정·재촬영 효과는 아니다. |
| Confidence 분포 | [confidence_distribution.png](results/figures/confidence_distribution.png) | 전체 오류 52,400개 중 MSP≥0.9인 오답은 **478개**다. 자신 있어도 틀리는 사례가 존재한다는 뜻이며 대부분의 오답이 고신뢰라는 뜻은 아니다. |
| 손상 유형별 AP 차이 | [condition_deltas.png](results/figures/condition_deltas.png) | 5개 level을 합친 15개 유형 중 **10개 유형 개선·5개 유형 악화**다. Cutout·density_inc·distortion·impulse·Gaussian에서는 소폭 하락했다. |
| Feature 관련성 | [feature_importance.png](results/figures/feature_importance.png) | Test 6,000행, permutation 5회. AP 감소량은 **margin 0.31849**, **knn_dist_std 0.16209**다. Geometry가 활용된 연관성이지 원인이라는 뜻은 아니다. |

더 세밀한 75개 조건별 AP는 **41개 개선·34개 악화**했다. 최대 개선은 Occlusion level 2 **+0.06125**, 최대 악화는 Cutout level 5 **−0.05419**다. 원본은 [geometry_delta_by_condition.csv](results/tables/geometry_delta_by_condition.csv)다.

전체 pooled AP 차이 **+0.09854**와 조건별 AP 차이의 단순평균 **+0.00661**은 서로 다른 지표다. 앞의 결과에는 여러 조건 사이의 위험 순서를 잘 정하는 효과도 포함된다. **“여러 손상 입력이 섞여 들어오는 상황에서 검토 우선순위가 좋아졌다”**고 설명하고, 모든 유형이나 단일 고정 손상에서 똑같이 개선됐다고 말하지 않는다.

### 공정한 비교와 재현 근거

LR은 각 조합에서 C={0.1, 1, 10}을 비교했고 모두 C=10이 선택됐다. LightGBM은 세 조합 모두 leaves={15, 31}, 최대 500 trees, learning_rate=0.05, validation AP early stopping 40이라는 동일 탐색 예산을 사용했다. 실제 선택값은 confidence **31 leaves·111 trees**, geometry **31 leaves·355 trees**, combined **15 leaves·443 trees**다. 따라서 “모든 설정이 완전히 동일”하다고 말하지 말고 **“같은 LightGBM과 같은 탐색 예산으로 validation에서 선택”**했다고 말한다.

- [audit.json](results/audit.json): 전체 185,100행의 logits·confidence·ID·split·타깃·검토 순위·7개 모델 지표 재계산을 통과했다. 유형별 raw geometry 15개 표본을 완전 거리행렬로 대조했고 최대 차이는 1.67×10⁻¹⁶이다.
- [data_download_validation.json](results/data_download_validation.json): 77개 파일의 크기·CRC32를 공식 ZIP의 메타데이터와 대조하고, 저장한 SHA-256을 전수 재검증했다. 전체 ZIP을 재구성하지 않았으므로 ZIP 전체 MD5 검증과는 구분한다.
- [clean_metrics.json](results/clean_metrics.json): clean accuracy **90.6807%**가 공식 90.7%와 일치한다. 평균 손상 오류율 **28.3090%**도 공식 28.3%와 일치한다.
- [data_inventory.csv](results/tables/data_inventory.csv): 조건별 점 수 **649~2,048개**, 원본 행 수·파일 SHA·계산 시간이다.
- [model_selection.json](results/tables/model_selection.json): 후보 설정과 validation 선택 기록이다. 코드와 환경 재현은 README를 따른다.

`results/provisional74/`는 서버 점검 중 만든 74개 조건 부분 기록이다. **최종 PPT의 숫자와 그림에는 이 절의 `results/tables/`, `results/figures/`만 사용한다.**

## 10분 발표 구성: 리허설 9분 30초

| 순서 | 시간 | 내용 | 확인할 핵심 |
|---|---:|---|---|
| 1 | 0:00~0:45 | 실제 손상 point cloud 사례·연구 질문 | 모델이 자신 있어도 틀릴 수 있으므로 먼저 볼 사례를 고른다. |
| 2 | 0:45~1:45 | 데이터 규모·원본 ID split | 75개 손상 버전이 같은 물체라는 점을 반영해 나눴다. |
| 3 | 1:45~3:00 | 고정 PointNet→11개 feature→탐지기 | 같은 LightGBM과 같은 탐색 예산으로 geometry의 추가 효과를 비교한다. |
| 4 | 3:00~4:15 | 손상별 오류율 히트맵·clean 재현 | 어떤 손상에서 문제가 커지는지 전체 185,100행에서 확인한다. |
| 5 | 4:15~5:45 | 7가지 탐지 성능 비교표 | AP·AUROC와 무작위 AP 기준을 제시하고 핵심 차이를 읽는다. |
| 6 | 5:45~7:15 | 검토 예산 그래프 | 10%·20%를 검토할 때 실제로 몇 개의 오류를 포착했는지 읽는다. |
| 7 | 7:15~8:30 | 중요 feature·개선/악화 사례 | geometry의 관련성과 조건별 예외를 함께 설명한다. |
| 8 | 8:30~9:30 | 결론·적용 범위·역할 | 이 benchmark와 checkpoint에서 확인한 효과와 남은 검증을 구분한다. |

남는 30초는 슬라이드 전환과 발화 속도 여유다. 위 시간표는 내용별 시간 배분 참고용이다. 추가 그림 중 핵심 개선·같은 검토량·실제 사례·geometry 설명을 우선 선택하고, risk–coverage·조건별 delta·bootstrap·세부 검증은 필요할 때 사용한다.

## 예상 질문과 답변

1. **같은 물체의 여러 손상을 왜 묶었나?** 같은 기하 구조가 반복되므로 행을 무작위로 나누면 같은 물체를 학습과 test에서 볼 수 있다. 원본 ID로 나눠 그 누수를 막았다.
2. **Geometry 추가 효과와 모델 복잡도 효과를 어떻게 구분했나?** 같은 LightGBM, 같은 split, 같은 후보 수와 validation 선택 기준으로 confidence-only와 combined를 비교했다. 선택된 tree 수와 leaf 수는 다르므로 모델 복잡도 효과를 완전히 분리한 실험은 아니다. 동일 탐색 예산에서 feature 정보를 추가한 효과를 비교했으며, LR도 세 feature 조합을 함께 보고한다.
3. **AP의 무작위 기준은 0.5인가?** 아니다. AP의 무작위 기준은 test에서 실제 오류가 차지하는 비율이며, AUROC의 무작위 기준이 0.5다.
4. **10% 포착률의 분모는 검토한 10%인가?** 아니다. 분모는 test 전체 오류 수다. 검토 개수는 `ceil(0.1×N)`이고 동점은 같은 원본 ID·손상·level 규칙으로 정한다.
5. **검토 포착률이 실제 오류 감소율인가?** 아니다. 고위험 목록에 원래의 오류가 얼마나 모였는지를 측정했다. 사람이 오류를 수정하거나 재촬영으로 성능이 회복되는지는 별도 검증이 필요하다.
6. **정답이나 손상 유형을 알아야 적용할 수 있나?** 탐지 학습의 target에는 정답이 필요하지만 추론 입력에는 XYZ와 PointNet의 confidence만 쓴다. 정답·예측 클래스 ID·손상 이름·level은 feature에서 제외했다.
7. **PointNet의 공식 test로 탐지기를 학습해도 되나?** PointNet은 이미 학습된 상태로 고정한다. 그 test 물체를 탐지기용 train/validation/test로 다시 분리하고, 탐지기가 학습하지 않은 물체에서 성능을 계산한다.
8. **LiDAR/Occlusion의 level 5가 더 심한 손상인가?** 다섯 level은 서로 다른 관측 각도이므로 순서형 강도로 해석하지 않는다.
9. **Feature importance가 원인인가?** 아니다. permutation에 따른 AP 변화로 모델이 활용한 연관성을 본다. feature 간 상관과 클래스별 모양 차이의 영향을 받을 수 있다.
10. **다른 분류기·실제 센서·처음 보는 손상에도 통하나?** 현재 결과의 범위는 사용한 ModelNet40-C와 PointNet이다. 새로운 물체를 평가했지만 손상 유형은 학습에서도 본 15종이다. 미관측 손상이나 다른 분류기로의 일반화는 검증하지 않았다.
11. **불확실성은 어떻게 확인했나?** 같은 원본 물체의 75개 행을 함께 뽑는 paired bootstrap을 사용한다. 고정된 탐지기의 test 표본 불확실성이며 새 split이나 재학습의 변동까지 포함하지 않는다.
12. **결과가 작거나 일부 악화되면?** 전체 평균 차이와 95% 구간을 먼저 말하고, 좋아진 조건과 나빠진 조건을 함께 제시한다. 모든 입력에 항상 유리하다고 주장하지 않는다.

## 채점 항목을 보여주는 위치

- 참신성 25점: 슬라이드 1의 제한된 검토 예산이라는 질문과 슬라이드 6의 포착률.
- 난이도 25점: 슬라이드 3의 추론·geometry·표 결합과 슬라이드 4의 전체 분석 규모.
- 연구 적절성 25점: 슬라이드 2의 ID split, 슬라이드 5의 공정한 비교, 부록의 독립 검증과 불확실성.
- 발표 25점: 사례→질문→방법→숫자→결론의 흐름, 9분 30초 리허설, 위 예상 질문 준비.

## 전달과 재현

코드 없이 발표 자료를 만들 때는 이 문서와 `results/figures/`의 PNG를 사용한다. `python scripts/package_handoff.py`로 만드는 `artifacts/PPT_HANDOFF_20261126.zip`에는 문서·그림·소형 결과표만 들어 있다. 원본 수업 PDF·데이터·가중치·대용량 캐시는 포함하지 않는다. 공개 코드와 결과는 [GitHub 저장소](https://github.com/yeonnnnni/pointcloud-failure-prediction)에서 확인한다. 재현할 코드 버전은 내려받은 저장소에서 `git rev-parse HEAD`로 확인한다.
