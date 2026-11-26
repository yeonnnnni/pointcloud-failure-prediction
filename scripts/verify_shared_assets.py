"""클론한 발표 문서·그림과 공유 모델 7개의 누락 및 모델 SHA-256을 확인한다."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            value.update(chunk)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root
    manifest = json.loads((root/'weights/manifest.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        path = root/item['path']
        if not path.is_file() or path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
            raise RuntimeError(f'공유 모델 누락 또는 hash 불일치: {item["path"]}')
    for name in ('README.md', 'TEAM_START_HERE.md', 'TEAM_PROJECT_GUIDE.md', 'PRESENTATION_HANDOFF.md'):
        path = root/name
        if not path.is_file():
            raise RuntimeError(f'문서가 없습니다: {name}')
        for target in re.findall(r'!\[[^\]]*\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            if not target.startswith(('https:', 'http:')) and not (root/target).is_file():
                raise RuntimeError(f'본문 그림이 없습니다: {target}')
    png = list((root/'results/figures').glob('*.png'))
    pdf = list((root/'results/figures').glob('*.pdf'))
    if len(png) != 16 or len(pdf) != 16:
        raise RuntimeError(f'발표 그림 수가 다릅니다: PNG {len(png)}, PDF {len(pdf)}')
    print(json.dumps({'공유모델': len(manifest['files']), '모델_SHA256': '모두 일치',
                      '발표_PNG': len(png), '발표_PDF': len(pdf), '문서와_그림': '누락 없음'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
