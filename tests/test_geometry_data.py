import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist

from pfp.geometry import geometry_one
from pfp.data import load_or_create_splits, pilot_ids, validate_table


@pytest.mark.parametrize("n", [2, 5, 20])
def test_knn_duplicate_coordinates_preserved(n):
    xyz = np.random.default_rng(1).normal(size=(n, 3))
    xyz[:min(n, 10)] = 0
    actual = geometry_one(xyz)
    distances = cdist(xyz, xyz)
    np.fill_diagonal(distances, np.inf)
    neighbors = np.sort(distances, axis=1)[:, :min(8, n - 1)]
    np.testing.assert_allclose(actual[-2:], [neighbors.mean(), neighbors.std()])
    if np.all(xyz == 0):
        np.testing.assert_array_equal(actual[:3], [0, 0, 0])


def test_geometry_rejects_bad_input():
    with pytest.raises(ValueError):
        geometry_one([[0, 0, np.nan], [1, 1, 1]])
    with pytest.raises(ValueError):
        geometry_one([[0, 0, 0]])


def test_saved_split_is_disjoint_and_pilot_excludes_test(tmp_path):
    labels = np.repeat(np.arange(40), 20)
    path = tmp_path / "split.csv"
    split = load_or_create_splits(labels, path)
    pd.testing.assert_frame_equal(split, load_or_create_splits(labels, path))
    ids = pilot_ids(split)
    assert len(ids) == len(set(ids)) == 100
    assert set(split.set_index("object_id").loc[ids, "split"]) == {"train", "validation"}
    with pytest.raises(ValueError):
        load_or_create_splits(labels[::-1], path)
