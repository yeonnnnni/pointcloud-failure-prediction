"""공식 배열 원본 행 ID, 고정 split과 데이터 무결성 검증."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .geometry import GEOMETRY

KEYS = ["object_id", "corruption", "severity"]
CONFIDENCE = ["msp", "entropy", "margin"]
CORRUPTIONS = ["uniform", "gaussian", "background", "impulse", "upsampling", "distortion",
               "distortion_rbf", "distortion_rbf_inv", "density", "density_inc", "shear",
               "rotation", "cutout", "occlusion", "lidar"]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def discover_conditions(data_dir, require_complete=True):
    conditions = []
    for path in Path(data_dir).glob("data_*.npy"):
        match = re.fullmatch(r"data_(.+)_(\d+)\.npy", path.name)
        if match and match[1] in CORRUPTIONS:
            conditions.append((match[1], int(match[2]), path))
    levels = sorted({level for _, level, _ in conditions})
    if not require_complete and set(levels).issubset(set(range(1, 6))):
        # 공식 배포 파일명은 1~5. 파일럿 우선 다운로드 시 1,5만 존재할 수 있다.
        offset = 0
    elif levels == list(range(5)):
        offset = 1
    elif levels == list(range(1, 6)):
        offset = 0
    else:
        raise ValueError(f"공식 5개 저장 level을 확인할 수 없음: {levels}")
    expected = {(name, level) for name in CORRUPTIONS for level in range(1, 6)}
    mapped = [(name, level + offset, path) for name, level, path in conditions]
    if require_complete and ({(name, level) for name, level, _ in mapped} != expected or len(mapped) != 75):
        raise ValueError(f"75개 조건 중 누락/중복: {len(mapped)}")
    return sorted(mapped), {str(level): level + offset for level in levels}


def load_or_create_splits(labels, path, seed=42):
    labels = np.asarray(labels).reshape(-1).astype(int)
    ids = np.arange(len(labels))
    path = Path(path)
    if path.exists():
        split = pd.read_csv(path)
        if not np.array_equal(split.object_id, ids) or not np.array_equal(split.true_label, labels):
            raise ValueError("저장 split과 현재 원본 ID/라벨이 일치하지 않음")
        meta = json.loads(path.with_suffix(".json").read_text())
        if meta["seed"] != seed:
            raise ValueError("기존 split과 요청 seed가 다름")
    else:
        train, remainder = train_test_split(ids, test_size=.4, stratify=labels, random_state=seed)
        validation, test = train_test_split(remainder, test_size=.5, stratify=labels[remainder], random_state=seed)
        split = pd.DataFrame({"object_id": ids, "true_label": labels, "split": "train"})
        split.loc[validation, "split"] = "validation"
        split.loc[test, "split"] = "test"
        path.parent.mkdir(parents=True, exist_ok=True)
        split.to_csv(path, index=False)
        write_json(path.with_suffix(".json"), {"seed": seed, "objects": len(labels),
                   "counts": split.split.value_counts().to_dict(), "method": "stratified 60/20/20"})
    if split.object_id.duplicated().any() or set(split.split) != {"train", "validation", "test"}:
        raise ValueError("split ID 중복 또는 split 누락")
    groups = {name: set(group.object_id) for name, group in split.groupby("split")}
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        if groups[a] & groups[b]:
            raise ValueError(f"split 교집합: {a}/{b}")
    return split


def pilot_ids(split, n=100, seed=42):
    rng = np.random.default_rng(seed)
    # 파일럿은 train과 validation만 사용한다. 작은 클래스별 최소 수 제약 없이 표본 추출.
    train = split.loc[split.split == "train", "object_id"].to_numpy()
    validation = split.loc[split.split == "validation", "object_id"].to_numpy()
    return np.sort(np.r_[rng.choice(train, int(n * .6), replace=False),
                         rng.choice(validation, n - int(n * .6), replace=False)])


def validate_table(table, splits, expected_ids, expected_conditions):
    ids = np.asarray(expected_ids)
    expected_conditions = set(expected_conditions)
    expected_count = len(ids) * len(expected_conditions)
    if len(table) != expected_count:
        raise ValueError(f"행 수 불일치 {len(table)} != {expected_count}")
    if table[KEYS].duplicated().any() or table.isna().any().any():
        raise ValueError("중복 키 또는 누락 값")
    if set(table.object_id) != set(ids):
        raise ValueError("원본 ID 누락/추가")
    if set(map(tuple, table[["corruption", "severity"]].drop_duplicates().values)) != expected_conditions:
        raise ValueError("조건 누락/추가")
    if not table.groupby("object_id").size().eq(len(expected_conditions)).all():
        raise ValueError("ID별 조건 수 불일치")
    reference = splits.set_index("object_id").loc[table.object_id]
    if not np.array_equal(table.split, reference.split) or not np.array_equal(table.true_label, reference.true_label):
        raise ValueError("split 또는 원본 라벨 매핑 불일치")
    if not np.array_equal(table.error, (table.pred_label != table.true_label).astype(int)):
        raise ValueError("오분류 타깃 생성 불일치")
    if not np.isfinite(table[CONFIDENCE + GEOMETRY].to_numpy()).all():
        raise ValueError("비유한 feature")
    if not table.true_label.between(0, 39).all() or not table.pred_label.between(0, 39).all():
        raise ValueError("40개 클래스 범위 위반")
    summary = {}
    for name, group in table.groupby("split"):
        summary[name] = {"objects": int(group.object_id.nunique()), "rows": len(group),
                         "positive": int(group.error.sum()), "negative": int((1-group.error).sum())}
    return {"passed": True, "rows": len(table), "objects": len(ids), "conditions": len(expected_conditions),
            "duplicates": 0, "missing": 0, "nonfinite": 0, "split_intersection": 0, "splits": summary}
