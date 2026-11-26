"""공유 PointNet checkpoint를 SHA-256으로 확인하고 없으면 공식 파일을 받는다."""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DRIVE_ID = '1eW26u0nm6HETwDSiCyCEoLLY3WnOVt73'
EXPECTED_SHA256 = 'aeb4119c854f4e4c4f0983f008d51903caedf2414c9bf89aa81eb4a2381937b8'
EXPECTED_BYTES = 41745708


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check(path: Path) -> None:
    if path.stat().st_size != EXPECTED_BYTES or sha256(path) != EXPECTED_SHA256:
        raise ValueError('PointNet 파일이 실험에 사용한 원본과 다릅니다. 기존 파일을 별도 보관한 뒤 --download로 복구하세요.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'weights/pointnet_model_best_test.pth')
    parser.add_argument('--download', action='store_true', help='파일이 없을 때 공식 Google Drive에서 받기')
    args = parser.parse_args()
    if args.output.is_file():
        check(args.output)
        print('기존 PointNet checkpoint의 크기·SHA-256 일치: 추가 다운로드 없음')
        return
    if not args.download:
        raise FileNotFoundError('PointNet checkpoint가 없습니다. --download를 붙여 실행하세요.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.pointnet-download-', dir=args.output.parent) as work:
        temporary = Path(work)/'checkpoint.pth'
        subprocess.run([sys.executable, '-m', 'gdown', DRIVE_ID, '-O', str(temporary)], check=True)
        check(temporary)
        temporary.replace(args.output)
    print(f'공식 PointNet checkpoint 저장 및 SHA-256 검증 완료: {args.output.name}')


if __name__ == '__main__':
    main()
