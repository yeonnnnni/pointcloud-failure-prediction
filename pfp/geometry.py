"""명세의 원본 XYZ 기하 통계. 중복 좌표를 보존한다."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.spatial import cKDTree

GEOMETRY = ["pca_ratio_1", "pca_ratio_2", "pca_ratio_3", "bbox_x", "bbox_y", "bbox_z",
            "knn_dist_mean", "knn_dist_std"]


def geometry_one(points):
    xyz = np.asarray(points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 2 or not np.isfinite(xyz).all():
        raise ValueError(f"유효하지 않은 point cloud: shape={xyz.shape}")
    centered = xyz - xyz.mean(axis=0)
    eigen = np.linalg.eigvalsh(centered.T @ centered / (len(xyz) - 1))[::-1].clip(0)
    total = eigen.sum()
    ratio = eigen / total if total > np.finfo(np.float64).eps else np.zeros(3)
    k = min(8, len(xyz) - 1)
    distance, index = cKDTree(xyz).query(xyz, k=k + 1, workers=1)
    # 중복 좌표일 때 자기 인덱스가 첫 번째가 아닐 수 있다.
    # 자기 자신만 inf로 바꾸고 가장 가까운 k개의 별개 점을 유지한다.
    distance[index == np.arange(len(xyz))[:, None]] = np.inf
    neighbors = np.partition(distance, kth=k - 1, axis=1)[:, :k]
    values = np.r_[ratio, np.ptp(xyz, axis=0), neighbors.mean(), neighbors.std(ddof=0)]
    if not np.isfinite(values).all():
        raise ValueError("비유한 geometry feature")
    return values


def extract_geometry(points, object_ids, workers=12):
    def one(object_id):
        try:
            return geometry_one(points[int(object_id)])
        except Exception as exc:
            raise ValueError(f"geometry 실패 object_id={object_id}: {exc}") from exc
    with ThreadPoolExecutor(max_workers=workers) as executor:
        return np.asarray(list(executor.map(one, object_ids)))
