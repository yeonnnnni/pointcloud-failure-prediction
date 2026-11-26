"""공식 Zenodo ZIP의 각 파일을 HTTP Range로 받아 CRC 검증 후 저장한다."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import time
import zipfile
import zlib
import requests

URL = 'https://zenodo.org/api/records/6017834/files/modelnet40_c.zip/content'
SIZE = 1970686633
ARCHIVE_MD5 = 'c4a7fffaa52c80b33f7b3a0ac7782d3b'


def get_range(start, end):
    # 서버 점검(503)과 rate limit(429)은 Retry-After를 존중하며 기다린다.
    for attempt in range(30):
        try:
            response = requests.get(URL, headers={'Range': f'bytes={start}-{end}'}, timeout=(30, 60))
            if response.status_code in (429, 503):
                delay = 60.0
                retry_after = response.headers.get('Retry-After')
                if retry_after:
                    try:
                        delay = max(0.0, float(retry_after))
                    except ValueError:
                        from email.utils import parsedate_to_datetime
                        delay = max(0.0, parsedate_to_datetime(retry_after).timestamp() - time.time())
                print(f'HTTP {response.status_code}, 범위 {start}-{end}, {delay:.0f}초 후 재시도, 시도 {attempt+1}/30', flush=True)
                if attempt == 29:
                    response.raise_for_status()
                time.sleep(delay)
                continue
            response.raise_for_status()
            if response.status_code != 206 or len(response.content) != end - start + 1:
                raise ValueError(f'범위 응답 불일치: {response.status_code} {len(response.content)}')
            return response.content
        except (requests.RequestException, ValueError) as exc:
            print(f'범위 재시도 {start}-{end}: {type(exc).__name__}: {exc}, 시도 {attempt+1}/30', flush=True)
            if attempt == 29:
                raise
            time.sleep(min(60, 5 * (attempt + 1)))
    raise RuntimeError('범위 다운로드 재시도 횟수 초과')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='data')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--pilot-only', action='store_true')
    args = parser.parse_args()
    target = Path(args.output)
    target.mkdir(parents=True, exist_ok=True)
    tail_size = 65536
    tail_cache = target / 'modelnet40c_zip_tail.bin'
    if tail_cache.exists() and tail_cache.stat().st_size == tail_size:
        tail = tail_cache.read_bytes()
    else:
        tail = get_range(SIZE - tail_size, SIZE - 1)
        tail_cache.write_bytes(tail)
    with tempfile.TemporaryFile() as index:
        index.seek(SIZE - tail_size)
        index.write(tail)
        with zipfile.ZipFile(index) as archive:
            infos = archive.infolist()
    priorities = ['label.npy', 'data_gaussian_1.npy', 'data_gaussian_5.npy',
                  'data_cutout_1.npy', 'data_cutout_5.npy', 'data_original.npy']
    relevant = [i for i in infos if i.filename.startswith('modelnet40_c/') and i.filename.endswith('.npy')]
    relevant.sort(key=lambda i: priorities.index(Path(i.filename).name) if Path(i.filename).name in priorities else 99)
    if args.pilot_only:
        relevant = [i for i in relevant if Path(i.filename).name in priorities]

    def fetch(info):
        path = target / info.filename
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.stat().st_size == info.file_size:
            raw = path.read_bytes()
            if (zlib.crc32(raw) & 0xffffffff) == info.CRC:
                return {'file': info.filename, 'bytes': len(raw), 'crc32': info.CRC,
                        'sha256': hashlib.sha256(raw).hexdigest(), 'reused': True}
        # Header includes variable filename and extra fields, so read it first.
        header = get_range(info.header_offset, info.header_offset + 29)
        fields = struct.unpack('<IHHHHHIIIHH', header)
        if fields[0] != 0x04034b50:
            raise ValueError(f'ZIP header 오류: {info.filename}')
        offset = info.header_offset + 30 + fields[-2] + fields[-1]
        # 큰 단일 응답 정체에 대비해 4 MiB 조각을 독립 재시도한다.
        chunk_size = 4 * 1024 * 1024
        ranges = [(start, min(start + chunk_size - 1, offset + info.compress_size - 1))
                  for start in range(offset, offset + info.compress_size, chunk_size)]
        with ThreadPoolExecutor(max_workers=4) as chunks:
            compressed = b''.join(chunks.map(lambda pair: get_range(*pair), ranges))
        if info.compress_type == zipfile.ZIP_DEFLATED:
            raw = zlib.decompress(compressed, -15)
        elif info.compress_type == zipfile.ZIP_STORED:
            raw = compressed
        else:
            raise ValueError(f'지원하지 않는 압축: {info.compress_type}')
        if len(raw) != info.file_size or (zlib.crc32(raw) & 0xffffffff) != info.CRC:
            raise ValueError(f'크기/CRC 오류: {info.filename}')
        tmp = path.with_suffix('.npy.part')
        tmp.write_bytes(raw)
        tmp.replace(path)
        return {'file': info.filename, 'bytes': len(raw), 'crc32': info.CRC,
                'sha256': hashlib.sha256(raw).hexdigest(), 'reused': False}

    start = time.monotonic()
    manifest = {'source': URL, 'archive_bytes': SIZE, 'published_archive_md5': ARCHIVE_MD5,
                'verification': '각 파일의 ZIP CRC32와 비압축 크기 검증, SHA-256 기록; 전체 ZIP MD5는 별도',
                'files': []}
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(fetch, i): i for i in relevant}
        for future in as_completed(futures):
            record = future.result()
            manifest['files'].append(record)
            path = target / 'modelnet40c_download_manifest.json'
            path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
            print(f"{len(manifest['files'])}/{len(relevant)} {record['file']} {time.monotonic()-start:.1f}s", flush=True)

if __name__ == '__main__':
    main()
