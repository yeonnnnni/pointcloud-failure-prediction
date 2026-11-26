"""공식 corruption 원본 배열과 clean HDF5 물체 대응을 실제 좌표로 검증한다."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from scipy.spatial import cKDTree


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--clean-dir',default='data/modelnet40_ply_hdf5_2048')
    p.add_argument('--corruption-dir',default='data/modelnet40_c')
    p.add_argument('--output',default='results/data_alignment.json')
    a=p.parse_args()
    clean=Path(a.clean_dir);cor=Path(a.corruption_dir)
    points=[];labels=[]
    files=[Path(v.strip()).name for v in (clean/'test_files.txt').read_text().splitlines() if v.strip()]
    for name in files:
        with h5py.File(clean/name) as h:
            points.append(h['data'][:]);labels.append(h['label'][:].reshape(-1))
    x=np.concatenate(points);y=np.concatenate(labels)
    expected=np.load(cor/'label.npy').reshape(-1)
    if not np.array_equal(y,expected):raise ValueError('clean과 corruption 라벨 순서 불일치')
    names=json.loads((clean/'ply_data_test_0_id2file.json').read_text())+json.loads((clean/'ply_data_test_1_id2file.json').read_text())
    classes=(clean/'shape_names.txt').read_text().splitlines()
    if len(names)!=len(y) or len(set(names))!=len(y):raise ValueError('원본 파일명 중복/누락')
    if not all(classes[int(v)]==n.split('/')[0] for v,n in zip(y,names)):
        raise ValueError('원본 파일명 클래스와 라벨 불일치')
    original=np.load(cor/'data_original.npy',mmap_mode='r')
    if original.shape[0]!=len(x):raise ValueError('original 행 수 불일치')
    maxima=[]
    for i in range(len(x)):
        distances,_=cKDTree(x[i]).query(original[i],k=1,workers=1)
        maxima.append(float(distances.max()))
    if max(maxima)!=0:raise ValueError(f'원본 좌표가 clean 좌표의 부분집합이 아님: 최대거리 {max(maxima)}')
    result={'n_objects':len(y),'clean_points_per_object':int(x.shape[1]),'original_points_per_object':int(original.shape[1]),
            'labels_identical':True,'unique_original_filenames':len(set(names)),
            'class_name_matches':True,'all_original_rows_exact_subsets_of_same_clean_row':True,
            'max_nearest_neighbor_distance':max(maxima),
            'method':'clean 2048점의 cKDTree로 같은 행 original 1024점의 최근접 거리를 전체 2468개 물체에서 확인',
            'scope':'공식 배포 data_original.npy와 clean HDF5의 대응 검증; 손상 유형별 생성 방식은 공식 코드 참조'}
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
