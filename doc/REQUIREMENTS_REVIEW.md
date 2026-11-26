# 제안서·수업 요구사항 대조

검토일: 2026-11-26(KST). `EXPERIMENT_SPEC.md`, `EXECUTION_TODO.md`, 로컬 제안서 PDF 3쪽, `Big_Data_Assignment1.pdf` 5쪽, 로컬 프로젝트 채점기준 PDF 1쪽을 모두 확인했다. 첨부 PDF 원본과 학번은 공개 결과물에 포함하지 않는다.

**실험 정의와 지표 계산은 `EXPERIMENT_SPEC.md`, 실행 순서와 완료 판정은 `EXECUTION_TODO.md`를 따른다. 제안서와 표현·범위가 다를 때는 현재 명세와 사용자의 최신 지시가 우선한다.** 아래 표는 수행해야 할 요구사항을 대조한 것으로, 실험 완료를 뜻하지 않는다. 실제 완료 상태는 TODO와 검증된 결과 파일로 확인한다.

## 핵심 연구 질문과 제안서 약속

핵심 질문은 “PointNet의 confidence에 입력 point cloud의 geometry를 추가하면, 같은 검토 예산에서 오분류를 더 많이 포착할 수 있는가?”이다. 제안서의 geometry 중심 질문을 confidence-only 대비 추가 가치가 있는지 묻는 공정한 비교로 구체화했다. 효과가 양수일 것을 전제로 하지 않으며, 개선이 작거나 음수이면 그 실측 결과를 발표한다.

| 근거 | 요구사항 | 명세에 따른 구현·검증 기준 | 발표·재현 산출물 |
|---|---|---|---|
| 제안서 1~2쪽 | 손상 유형·level에 따른 분류 실패 분석 | 공식 ModelNet40-C 15종 × 5개 level. 예상 원본 물체 2,468개, 손상 행 185,100개와 실제 배열 크기를 대조 | 15×5 오류율 히트맵, 조건별 수치와 표본 수 |
| 제안서 1~2쪽 | 공식 pretrained PointNet 사용 | 표준 checkpoint 고정, strict load, 40개 출력과 유한값 검사, `eval()`과 inference-only. 공식 코드 버전·checkpoint SHA-256·전처리 기록 | README 실행법, 로컬 자산·환경 기록 |
| 제안서 1·3쪽 | Clean 기준과 손상 robustness | Clean accuracy, 손상 유형별 5개 level 통합 accuracy·macro-F1, level별·클래스별 오류율, confidence 분포 | Robustness 표·히트맵·보조 그림 |
| 제안서 2쪽 | Geometry와 confidence 표 구성 | 원본 손상 XYZ에서 PCA 비율 3개·bbox 3개·kNN 거리 평균·표준편차. Confidence는 MSP·entropy·margin | 조건별 feature 캐시와 검증 보고 |
| 제안서 2쪽 | 위험 예측 입력의 정보 누출 방지 | 정답·예측 클래스 ID, object ID, split, 손상 종류·level은 feature 입력에서 제외. `error = int(pred_label != true_label)` | 명시적 feature 목록, 타깃·조인 검증 |
| 제안서 2~3쪽 | 같은 물체의 손상 버전 간 누출 방지 | seed 42, 원본 object ID 기준 약 60/20/20 split. 각 ID의 75개 변형은 같은 split. Scaling·선택·threshold는 train/validation에서만 결정 | 저장된 split ID, 교집합 0 확인, split별 행·양성·음성 수 |
| 제안서 2~3쪽 | Confidence/geometry 조합 비교 | `1 − MSP`, Logistic Regression 3조합, LightGBM 3조합을 동일 test에서 비교. 핵심 차이는 같은 LightGBM의 confidence + geometry − confidence-only | AP·AUROC 비교표, 추가 효과의 실측 차이 |
| 제안서 3쪽 | AUPRC·AUROC·F1·Brier·risk–coverage | AUPRC 표기는 명세상 AP(`average_precision_score`)로 고정. Validation F1 최대 threshold를 test에 적용. Brier는 학습형 탐지기의 `P(error=1)`에 계산하고 보정 여부 표시 | 탐지 성능표, 고정 threshold, 보조 지표·곡선 |
| 제안서 1·3쪽 | 같은 검토 예산에서 오류 탐지 효율 | 위험도순 `ceil(q × N)`개, 공통 결정적 동점 처리. q=10%·20%에서 선택 오분류 수 / test 전체 오분류 수 | 검토 예산 그래프, 분자·분모와 무작위 기대값 |
| 제안서 2~3쪽 | 실패 위험과 관련된 feature 해석 | 핵심 수치 확정 후 LightGBM SHAP 또는 held-out permutation importance. 방법·사용 표본 표시, 인과효과로 해석하지 않음 | Feature 중요도 표·그림과 해석 한 문장 |
| 제안서 3쪽 | 실제 검토 우선순위화의 가능성 제시 | 동일 손상 샘플의 정답, PointNet 예측·MSP, confidence-only와 결합 모델의 검토 순위를 함께 표시 | 실제 사례 그림, 포착 개선·악화 사례 설명 |

## 명세로 확정한 차이와 범위

| 제안서의 초기 표현 | 현재 실행 기준과 이유 |
|---|---|
| Geometry만으로 예측한 뒤 confidence를 추가하는 질문 | 세 조합을 모두 평가하되, 핵심 검증은 동일 LightGBM에서 confidence에 geometry를 추가한 효과로 고정한다. |
| kNN 거리 평균·분산, 공간적 분산, local density 등의 후보 feature | 명세의 8개 geometry feature만 사용한다. kNN은 표준편차이며, k=`min(8, N−1)`이다. 자기 자신의 인덱스만 제외하고 같은 좌표의 다른 점은 유지한다. |
| SHAP과 feature importance | 명세에 따라 SHAP 또는 held-out permutation importance를 허용한다. 무엇을 실제 사용했는지 결과에 쓴다. |
| Confidence-only threshold baseline | 핵심 순위 기준선은 학습 없는 `1 − MSP`이며, F1 threshold는 validation에서 결정한다. |
| PointNet++·DGCNN 추가 또는 미관측 손상 유형 일반화 | 제안서에서도 선택적 확장이다. 핵심 결과 검증과 발표 인수인계가 끝난 뒤에만 진행한다. |
| 제안서 작성 당시 팀 역할 분담 | 최신 지시에 따라 이정연이 실험 구현·실행·검증·해석·인수인계를, 이현택이 PPT 작성과 12/3 발표를 맡는다. |
| 모든 level을 심각도 증가로 설명할 가능성 | LiDAR·Occlusion의 level은 관측 각도 차이이므로 순서형 손상 강도로 해석하지 않는다. |
| Clean과 손상 결과 비교 | Clean은 재현 점검용이다. 개별 물체 대응을 별도로 검증하기 전에는 행 번호로 clean·corruption을 결합하지 않는다. |

## 수업 요구사항과 채점 항목

`Big_Data_Assignment1.pdf`의 슬라이드 11~12는 구체적인 데이터 기반 문제, 해결 필요성, 데이터 출처·규모·feature·타깃, 방법 선택의 이유, 지표와 검증 전략, 예상 인사이트·활용 가치를 요구한다. 공개 데이터와 전통적인 ML을 포함한 다양한 방법을 허용하므로 ModelNet40-C 및 Logistic Regression·LightGBM은 범위에 맞는다. 슬라이드 13~14는 최대 2인 팀과 균등한 기여를 요구하며, 두 사람이 감당할 만한 범위의 작업이어야 한다. 실제 역할과 수행 내용을 발표 인수인계에 적어 기여를 설명한다.

과제 안내 슬라이드 15의 발표 시점은 후속 공지 예정으로 되어 있다. 더 구체적인 채점기준 PDF는 발표 10분·질의응답 3분과 시간 준수를 명시한다. 이번 팀의 12/3 일정은 사용자 최신 지시를 따른다.

| 채점 항목 | 배점 | 발표에서 제시할 증거 |
|---|---:|---|
| 참신성 | 25 | 단순 정확도 비교를 넘어 같은 검토 예산으로 실패를 더 많이 찾을 수 있는지 명확하게 질문한다. 실제 실패 사례와 검토 우선순위화의 필요성을 연결한다. |
| 난이도 | 25 | 고정 PointNet 추론, 75개 조건, geometry 계산, 조인·캐시·재개, 두 탐지 모델과 세 feature 조합의 연결을 설명한다. 규모는 실제 완료 범위를 표시한다. |
| 연구 적절성 | 25 | 원본 ID 분할, test 분리, 동일 모델 간 비교, 실측 지표와 검증 결과, 개선·악화 양쪽 및 일반화 한계를 제시한다. |
| 발표 | 25 | 사례 → 질문 → 데이터·split → 방법 → 손상 오류 → 탐지 비교 → 검토 예산 → 결론 순서. 9분 30초 리허설을 목표로 10분을 지키며, 3분 질의응답을 준비한다. |

## 인수인계에서 빠지면 안 되는 내용

- 그림마다 파일 경로, 실제 모집단과 표본 수, 읽을 숫자, 해석 한 문장을 적는다.
- 핵심 LightGBM 비교의 AP 차이와 검토 10%·20% 포착률 차이는 같은 test에서 얻은 검증된 실측값으로만 채운다.
- “포착률”은 오류를 발견하는 비율이다. 검토 뒤 정답으로 수정됐다는 뜻이나 실제 오류 감소율로 바꾸어 말하지 않는다.
- 결론은 사용한 PointNet checkpoint와 ModelNet40-C 범위로 제한한다. 실제 센서·다른 분류기로의 일반화는 검증하지 않았다면 주장하지 않는다.
- 파일럿은 Gaussian·Cutout의 level 1·5와 train/validation 공통 ID에서 실행한다. 파일럿·부분 결과를 전체 최종 결과로 표시하지 않는다.
- 11/26 핵심 결과·초안, 11/30 자료 전달, 12/3 발표 일정 및 담당을 문서에 남긴다. 전체 실행이 진행 중이면 완료 조건 수·작업 ID·로그·ETA·재개 명령을 기록한다.

이 대조 문서는 수업 요구사항의 반영 여부를 판단하기 위한 기준이다. 최종 수치·그림·재현 명령은 검증 후 작성하는 `PRESENTATION_HANDOFF.md`, README, 결과 산출물에서 확인한다.
