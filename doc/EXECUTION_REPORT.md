# 전체 실험 실행·검증 기록

2026-11-26 KST에 공식 ModelNet40-C **75조건·185,100행**과 동일 test **37,050행**의 실험을 완료했다. 최종 결과는 `results/tables/`, 그림은 `results/figures/`, 발표 설명은 `PRESENTATION_HANDOFF.md`다. `results/provisional74/`는 서버 점검 중 만든 부분 기록으로 최종 표와 분리한다.

## 환경과 자산

레포 초기 상태는 `main`, 커밋 없음, 첨부 문서만 존재했다. 기존 자산을 탐색했으나 찾지 못해 공식 배포본과 공식 PointNet 코드를 확보했다. `pfp` conda 환경은 Python3.11.18, PyTorch2.9.1+cu128, CUDA12.8이며 실제 버전은 `requirements-lock.txt`, 공개 환경 요약은 `results/environment.json`에 있다. 서버 GPU는 RTX PRO6000 Blackwell4개이며 추론은 cuda:0 한 개를 사용했다. Geometry는12개 worker, PointNet batch128, LightGBM8개 CPU worker를 사용했다.

공식 코드 commit은 `fad786c430c007cfbb7b4fe90ed1c9873d960921`, checkpoint SHA-256은 `aeb4119c854f4e4c4f0983f008d51903caedf2414c9bf89aa81eb4a2381937b8`다. 개인 서버 절대 경로·PID 세부 환경은 Git에서 제외한 `artifacts/full/environment.local.json`에 있다. 자산 출처와 clean 미러 검증은 `ASSET_PROVENANCE.md`를 참고한다.

## 실제 실행 순서와 명령

```bash
conda activate pfp
python -m pytest -q
python -m pfp.pointnet --batch-size 128 --output results/clean_metrics.json
python -m pfp.pipeline --mode pilot --output artifacts/pilot
python -m pfp.pipeline --mode full --output artifacts/full --wait-data 7200
python -m pfp.plots --table artifacts/full/table.parquet --evaluation-dir artifacts/full/evaluation --data-dir data/modelnet40_c --output results/figures
python -m pfp.audit --table artifacts/full/table.parquet --data-dir data/modelnet40_c --cache-dir artifacts/full/conditions --evaluation-dir artifacts/full/evaluation --output results/audit.json
python scripts/verify_downloads.py
python scripts/export_results.py
python scripts/package_handoff.py
```

통합 작업은 `conda run --no-capture-output -n pfp bash scripts/run_all.sh`로 실행했고 실행 세션67933, 셸PID228502, 전체 pipeline PID228635를 기록했다. **해당 실험은 종료됐으며 남은 실험 조건은0개다.** PID는 당시 실행 기록이다. 현재 새 실행을 확인할 때는 새로 기록된 `local/run_all.pid`와 `artifacts/full/status.json`을 사용한다.

- 16:57: 파일럿100개 물체·400행 통과 후 전체 실행 시작.
- 17:08 이후: Zenodo 점검503으로 마지막 upsampling5가 지연. 검증된74조건은 캐시에 보존했다.
- 대기 중: 설정을 바꾸지 않고74조건 부분평가·그림·독립감사를 별도 산출했다.
- 17:31: 77파일 수신 및 크기·CRC·SHA 전수 검증 완료. 마지막 조건 처리 후75조건 전체 평가 완료.
- 이후: 전체 그림8종 생성·시각 검수·전수 지표 감사 통과, 소형 결과와 인수인계 정리.

## 실제 소요 시간

| 구간 | 측정값 | 해석 |
|---|---:|---|
| 75조건 PointNet 추론 시간 합 | 25.980초 | 모델 초기화·다운로드 제외, 조건별 실제 측정 합 |
| 75조건 geometry 시간 합 | 39.831초 | 원본 XYZ의 PCA·bbox·kNN 계산 |
| 전체 표 완성까지 | 2008.705초 | 미수신 조건 대기와 Zenodo 점검 포함 |
| 탐지 평가·bootstrap·importance | 8.000초 | 7개 비교·동일 test·bootstrap300회 포함 |
| 독립 감사 | 1.192초 | logits/표/모델/지표/geometry 대조 |

추론과 geometry는 조건별로 겹쳐 실행했으므로 위 두 합을 총 벽시계 시간으로 더하지 않는다. 원본 조건별 측정값은 `results/tables/data_inventory.csv`다. 다운로드와 환경 설치는 위 계산 시간과 구분한다.

## 검증 결과

- Clean accuracy:2,238/2,468=90.6807%. 평균 손상 오류율:52,400/185,100=28.3090%. 각각 공식90.7%,28.3%와 일치.
- train1,480개·111,000행·오류32,039개; validation494개·37,050행·오류10,030개; test494개·37,050행·오류10,331개.
- 원본 ID 교집합0, ID별75조건, 중복키0, 누락0, 비유한feature0.
- 185,100행 전체 logits/confidence/target 일치, 저장 탐지기7개 예측·순위·AP/AUROC/F1/분자분모 재현.
- 15개 손상 유형에서 원본geometry1표본씩 완전거리행렬로 독립 대조, 최대차이1.67×10⁻¹⁶.
- 파일럿4조건 캐시 재사용을 실제 실행해 같은 table SHA와 지표 확인.
- 단위검사19개 통과. 그림8종은 평가값 재집계 대조와 실제 이미지 열람을 모두 통과.

전체 표 SHA-256은 `57a084c6bdbe9bff3a3510957ade740fcb06b82f89fdfa090ba989d0c8efaa17`다. 프로젝트·공식 source hash와 test 예측 표 hash는 `results/reproducibility.json`에 있다.

## 발생한 문제와 복구

1. Sandbox GPU 조회는 실패했으나 서버 권한으로 실제 GPU4개를 확인했다. pfp GPU 실행은 정상 동작했다.
2. pip의 기존 추가 인덱스 DNS 실패는 해당 설치 명령에서 추가 인덱스를 비워 해결했다. 다른 환경은 변경하지 않았다.
3. 최초 pytest는 vendor의 미사용 Python2/PointNet++ 검사도 수집해 실패했다. `pytest.ini`에 본 프로젝트 `tests/`를 명시했고19개 검사를 통과했다.
4. 최초 nohup 작업은 실행 셸 종료 뒤 유지되지 않아 지속 실행 세션으로 바꾸고 PID·로그를 기록했다.
5. Zenodo 점검503은76파일을 보존하고4MiB 범위별재시도, 캐시ZIP인덱스, Retry-After 존중으로 복구했다. 마지막파일 CRC 검증 뒤 원자적으로 저장했다.
6. 실행 중 runner 파일 변경으로 후속 shell 명령 해석 오류(`-data: command not found`)가 발생했다. **추론과 전체 평가는 이미 정상 완료된 뒤였다.** 현재 스크립트 문법을 확인하고 그림·감사·내보내기를 위 개별 명령으로 재개해 마무리했다. 학습 결과는 바꾸지 않았다.
7. 서버 Git 인증 계정에는 원격 쓰기 권한이 없어403이 발생했다. 연결된 GitHub의 사용자 계정으로 공개 파일만 게시했고 원격 tree와 로컬 tree hash가 같은지 확인했다. 기존 로컬 커밋은 `local-implementation-20261126` 브랜치에 보존했다. 글로벌 인증 설정은 변경하지 않았다.

## 재개·재현

원본 자산과 코드가 같은 상태에서 전체 명령을 다시 실행하면 데이터·checkpoint·split·코드 hash와 내용 검증을 통과한 조건 캐시를 재사용한다. 캐시는 Git에 포함하지 않았으므로 새 서버에서는 README의 자산 준비를 먼저 수행한다. 최종 실험은 완료됐으므로 별도 대기 작업이나 남은 ETA는 없다. 첨부PDF·원본데이터·대용량캐시·개인서버경로·로그는 공개하지 않는다. 이후 팀원 전달 요청에 따라 최종 모델7개는 `weights/`에 공개하며 원래 파일의 SHA를 유지한다.
