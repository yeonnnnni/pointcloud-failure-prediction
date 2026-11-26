# 데이터와 PointNet 출처

실험 정의는 [EXPERIMENT_SPEC.md](EXPERIMENT_SPEC.md)를 따른다. 팀원 전달 요청에 따라 공식 PointNet 원본 checkpoint와 본 프로젝트 탐지기6개를 `weights/`에 포함했다. 원본 데이터와 공식 코드 clone은 Git에서 제외하며 재현할 때 별도로 받는다. 파일 크기와 hash는 `weights/manifest.json`, 시작 방법은 루트 `TEAM_START_HERE.md`에 있다.

## 공식 코드와 체크포인트

- 코드: [ModelNet40-C 공식 저장소](https://github.com/jiachens/ModelNet40-C)
- 고정 commit: `fad786c430c007cfbb7b4fe90ed1c9873d960921`
- 모델: 공식 `models/pointnet.py`와 `pointnet_pyt/pointnet/model.py`, `PointNetCls(k=40, feature_transform=True)`.
- 표준 학습 설정: `configs/dgcnn_pointnet_run_1.yaml`, 손상 평가 설정: `configs/corruption/pointnet.yaml`.
- [공식 PointNet 체크포인트](https://drive.google.com/file/d/1eW26u0nm6HETwDSiCyCEoLLY3WnOVt73/view), 로컬 상대 경로 `weights/pointnet_model_best_test.pth`.
- 체크포인트 SHA-256: `aeb4119c854f4e4c4f0983f008d51903caedf2414c9bf89aa81eb4a2381937b8`.
- 저장된 epoch은 404, 체크포인트 내부 clean test accuracy는 `0.906807131280389`다. 이는 이번 실행의 실측값과 구분한다.
- 공식 wrapper 소스 SHA-256: `598107018aed36c015f92e85e5eb5503d90ba2af32d451635b1d6e9918b72c41`.
- 공식 core 소스 SHA-256: `e5af6acf4dee77e98fe5f223878a230eea0e1bb80d974a2db119751ad5163c09`.

추론 래퍼는 공식 core를 그대로 읽는다. `module.` 접두사만 제거한 뒤 모든 가중치를 `strict=True`로 로드하고 `eval()`과 `torch.inference_mode()`를 사용한다. 공식 PointNet의 출력은 `log_softmax`이며 이를 클래스 점수로 캐시한다. 안정적인 softmax를 적용하면 원래 확률과 같다. 미사용 모델의 CUDA 확장을 피하기 위해 공식 wrapper의 class 본문만 별도로 실행하여 동일 batch의 출력과 대조한다.

## ModelNet40-C

- [공식 Zenodo record 6017834](https://zenodo.org/records/6017834), 파일 `modelnet40_c.zip`.
- ZIP 크기: 1,970,686,633 bytes. 배포자가 공개한 ZIP MD5: `c4a7fffaa52c80b33f7b3a0ac7782d3b`.
- `scripts/download_modelnet40c.py`는 같은 공식 ZIP의 HTTP Range로 필요한 파일부터 내려받는다. 각 파일의 비압축 크기와 ZIP CRC32를 검증하고 SHA-256을 저장한다. 전체 ZIP의 MD5 검증을 수행했다는 뜻은 아니다.
- 공식 로더는 `label.npy`와 `data_<corruption>_<severity>.npy`의 동일 행을 결합한다. 원본 행 번호가 `object_id`다.
- 파일 severity `1,2,3,4,5`는 발표 표시 level `1,2,3,4,5`와 동일하다.
- 공식 `eval_cor.sh`의 15종: `uniform`, `gaussian`, `background`, `impulse`, `upsampling`, `distortion_rbf`, `distortion_rbf_inv`, `density`, `density_inc`, `shear`, `rotation`, `cutout`, `distortion`, `occlusion`, `lidar`.
- 생성 스크립트에 있는 `scale`은 공식 15종 평가 목록에 포함되지 않는다. `data_original.npy`도 75개 손상 조건에서 제외한다.
- 손상 추론은 파일의 XYZ 전체를 사용하며 추가 중심화·정규화·재샘플링을 하지 않는다. Geometry 역시 원본 손상 XYZ에서 계산한다.
- LiDAR와 Occlusion의 level은 관측 각도이며 단조로운 손상 강도로 해석하지 않는다.

## Clean 데이터

공식 clean 설정은 `modelnet40_ply_hdf5_2048/test_files.txt`가 지정한 HDF5 순서와 각 물체의 첫 1024점이다. Clean과 corruption의 행 번호를 검증 없이 조인하지 않는다.

공식 Stanford 배포 서버가 응답하지 않아 [SpecEOT 연구 코드의 데이터 안내](https://github.com/a772316182/SpecEOT/blob/main/README.md)가 제공하는 [동일 형식의 HDF5 미러](https://drive.google.com/file/d/1BUP46fXOlLVeGKa7PT1CtBEi3s0HGHUi/view)를 사용했다. ZIP에는 ShapeNetPart도 포함되어 있으나 ModelNet40 test 파일과 class 이름만 추출했다.

- `ply_data_test0.h5`: 2,048개 × 2,048점 × 3축. SHA-256 `fb1631826ff4102aac03ee8a59ed309f8ce839a210b6da1a40a2e4d45bba9d56`.
- `ply_data_test1.h5`: 420개 × 2,048점 × 3축. SHA-256 `dc002c8e9e168cedd22afbdfa8f1fae19d0f2fafb6efa993e4c88fbcaee8f353`.
- 두 파일의 라벨을 순서대로 결합하면 공식 corruption `label.npy`의 2,468개 라벨과 완전히 일치한다.
- `shape_names.txt`의 클래스 순서가 모델의 40개 클래스 순서와 같고, `id2file`의 2,468개 원본 파일명이 모두 유일하며 클래스 라벨과 일치한다.
- 2026-11-26 실제 clean 추론: **2,238 / 2,468 정답, accuracy 90.680713128%**. 체크포인트에 기록된 정확도와 정확히 일치한다.
- 실제 clean 샘플 8개에서 공식 wrapper와 독립 래퍼의 40차원 출력은 최대 절대 차이 0, 예측은 모두 일치한다.
- 상세 실측값: `results/clean_metrics.json`. 추론은 `pfp` 환경의 PyTorch `2.9.1+cu128`, `cuda:0`에서 수행했다.

전체 2,468개 물체에 대해 공식 `data_original.npy`의 1,024점 각각이 같은 행 clean HDF5 2,048점 안에 정확히 존재하는지 cKDTree로 검증했다. 모든 최근접 거리가 **0**이었다. 라벨·파일명 검증과 함께 clean과 공식 원본 배열의 행 대응을 확인했으며 결과는 `results/data_alignment.json`이다. 재검증 명령은 `python scripts/verify_data_alignment.py`다. 실제 손상 데이터 Cutout level 1과 Gaussian level 5도 train/validation의 8개 물체씩 공식 wrapper와 출력 차이 0을 확인했고 `results/corruption_parity.json`에 저장했다.

## 전체 다운로드 완료 검증

2026-11-26 17:31 KST에 공식 손상 75조건과 original·label을 합친 **77개 파일**의 크기·ZIP CRC32·SHA-256 재검증이 통과했다. 15종 × 5개 level × 2,468개 물체 = **185,100행**이며, 손상 좌표는 모두 유한한 `float32`이고 조건별 점 수는 **649~2,048개**다. 손상 배열의 비압축 크기는 **2,253,283,728 bytes**이며 original·label 포함 **2,283,613,236 bytes**다.

마지막 `upsampling level 5` 다운로드 도중 Zenodo가 17:08 KST부터 서버 점검으로 HTTP 503을 반환했다. 검증된 76개 파일을 재사용하고 60초 간격 재시도 후 마지막 파일을 4 MiB 범위 조각으로 받아 검증했다. 전체 ZIP 파일을 다시 받지 않았으므로 전체 ZIP MD5 검증과 혼동하지 않는다. 상세 검증 결과는 `results/data_download_validation.json`, 재검증 명령은 `python scripts/verify_downloads.py`다.
