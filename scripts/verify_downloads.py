"""공식 다운로드 manifest의 CRC·SHA·크기와 전체 75개 조건을 재검증한다."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo
import zlib
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',default='data')
    parser.add_argument('--output',default='results/data_download_validation.json')
    args=parser.parse_args()
    root=Path(args.data_root)
    manifest=json.loads((root/'modelnet40c_download_manifest.json').read_text())
    records=manifest['files']
    assert len(records)==77, f'전체77파일 필요: {len(records)}'
    assert len({r['file'] for r in records})==77, 'manifest 파일 중복'
    def check(record):
        path=root/record['file'];raw=path.read_bytes()
        assert len(raw)==record['bytes'], f'{path.name}: 크기 불일치'
        assert zlib.crc32(raw)&0xffffffff==record['crc32'], f'{path.name}: CRC 불일치'
        assert hashlib.sha256(raw).hexdigest()==record['sha256'], f'{path.name}: SHA 불일치'
        array=np.load(path,mmap_mode='r',allow_pickle=False)
        assert np.isfinite(array).all(), f'{path.name}: 비유한 값'
        return {'name':path.name,'bytes':len(raw),'shape':list(array.shape),'dtype':str(array.dtype)}
    with ThreadPoolExecutor(max_workers=4) as pool:
        verified=list(pool.map(check,records))
    corruption=[v for v in verified if v['name'] not in ['label.npy','data_original.npy']]
    assert len(corruption)==75
    types={v['name'].removeprefix('data_').rsplit('_',1)[0] for v in corruption}
    assert len(types)==15
    for name in types:
        assert {v['name'] for v in corruption if v['name'].removeprefix('data_').rsplit('_',1)[0]==name}=={f'data_{name}_{level}.npy' for level in range(1,6)}
    assert all(len(v['shape'])==3 and v['shape'][0]==2468 and v['shape'][2]==3 for v in corruption)
    labels=np.load(root/'modelnet40_c/label.npy').reshape(-1)
    assert len(labels)==2468 and set(labels)==set(range(40))
    report={'passed':True,'source_url':manifest['source'],'expected_archive_bytes':manifest['archive_bytes'],
            'published_archive_md5':manifest['published_archive_md5'],'whole_archive_md5_verified':False,
            'verification':'공식 ZIP 중앙디렉터리의 각 파일 크기·CRC32 및 저장한 SHA-256을 전체 파일에서 재검증',
            'transport':'공식 HTTP Range; 최초 파일 단위, upsampling5 복구는 4MiB 조각(최대4병렬); 429/503 Retry-After 존중',
            'verified_files':len(verified),'corruption_conditions':len(corruption),'corruption_types':sorted(types),
            'severity_levels':[1,2,3,4,5],'objects_per_condition':2468,'corruption_rows':75*2468,
            'num_classes':40,'min_points':min(v['shape'][1] for v in corruption),'max_points':max(v['shape'][1] for v in corruption),
            'corruption_dtypes':sorted({v['dtype'] for v in corruption}),
            'uncompressed_total_bytes':sum(v['bytes'] for v in verified),
            'uncompressed_corruption_bytes':sum(v['bytes'] for v in corruption),
            'finite_coordinates':True,'reused_files_in_final_download':sum(bool(r.get('reused')) for r in records),
            'zip_directory_tail_sha256':hashlib.sha256((root/'modelnet40c_zip_tail.bin').read_bytes()).hexdigest(),
            'verified_at_kst':datetime.now(ZoneInfo('Asia/Seoul')).isoformat(timespec='seconds')}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(report,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
