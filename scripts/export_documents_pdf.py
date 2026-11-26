"""팀원 설명·인수인계 Markdown을 한글·표·그림과 링크가 보존된 A4 PDF로 변환한다."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path
import re
from urllib.parse import quote, unquote

import markdown
from bs4 import BeautifulSoup
from weasyprint import HTML

ROOT = Path(__file__).resolve().parents[1]
REPO = 'https://github.com/yeonnnnni/pointcloud-failure-prediction'
RAW = 'https://raw.githubusercontent.com/yeonnnnni/pointcloud-failure-prediction/main'
DOCUMENTS = ('TEAM_PROJECT_GUIDE.md', 'PRESENTATION_HANDOFF.md')

STYLE = '''
@page {
  size: A4; margin: 18mm 17mm 20mm;
  @top-left { content: string(running-title); color: #64748b; font-size: 8pt; }
  @top-right { content: "PointNet · ModelNet40-C"; color: #64748b; font-size: 8pt; }
  @bottom-left { content: "pointcloud-failure-prediction"; color: #64748b; font-size: 8pt; }
  @bottom-right { content: counter(page) " / " counter(pages); color: #64748b; font-size: 8pt; }
}
html { font-family: "Noto Sans CJK KR", "DejaVu Sans", sans-serif; color: #1e293b; }
body { font-size: 10pt; line-height: 1.62; margin: 0; }
h1 { font-size: 22pt; line-height: 1.35; color: #16324f; margin: 0 0 7mm;
     string-set: running-title attr(data-running-title); bookmark-level: 1; }
h1 span { white-space: nowrap; }
h2 { font-size: 16pt; line-height: 1.4; color: #16324f; margin: 8mm 0 4mm;
     padding-bottom: 2mm; border-bottom: 1.2pt solid #2963a0; bookmark-level: 2; }
.section-summary { break-inside: avoid; }
.keep-table { break-inside: avoid; }
h3 { font-size: 12.3pt; line-height: 1.45; color: #214a6e; margin: 6mm 0 3mm; bookmark-level: 3; }
h4 { font-size: 10.8pt; color: #214a6e; margin: 4mm 0 2mm; }
h1, h2, h3, h4 { break-after: avoid; }
p { margin: 2.4mm 0; orphans: 3; widows: 3; }
strong { color: #183a59; }
a { color: #215f91; text-decoration: none; overflow-wrap: anywhere; }
ul, ol { padding-left: 6mm; margin: 2.5mm 0 3mm; }
li { margin: 1.2mm 0; orphans: 2; widows: 2; }
blockquote { margin: 4mm 0; padding: 2mm 4mm; border-left: 3pt solid #d76526;
             background: #fdf4ed; break-inside: avoid; }
blockquote p { margin: 1mm 0; }
code { font-family: "DejaVu Sans Mono", "Noto Sans CJK KR", monospace; font-size: 8.6pt;
       background: #f0f4f7; padding: .3mm .6mm; overflow-wrap: anywhere; }
pre { padding: 3mm; background: #f0f4f7; border-left: 2pt solid #afc4d8;
      white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.5; break-inside: avoid; }
pre code { font-size: 8.3pt; background: none; padding: 0; }
table { width: 100%; border-collapse: collapse; table-layout: fixed; margin: 4mm 0;
        font-size: 8.4pt; line-height: 1.5; }
thead { display: table-header-group; }
tr { break-inside: avoid; }
th { background: #16324f; color: white; font-weight: bold; text-align: left; }
th, td { border: .5pt solid #d6dfe8; padding: 2mm 2.1mm; vertical-align: top;
         overflow-wrap: anywhere; }
th strong { color: white; }
tbody tr:nth-child(even) { background: #f2f6f9; }
table code { font-size: 7.7pt; background: transparent; padding: 0; }
figure { margin: 5mm 0 4mm; padding: 0; break-inside: avoid; }
figure img { display: block; width: 100%; height: auto; max-height: 110mm; object-fit: contain; }
figcaption { margin-top: 2mm; font-size: 8.5pt; line-height: 1.5; color: #475569; }
figcaption p { margin: 0; }
hr { border: 0; border-top: .6pt solid #d6dfe8; margin: 5mm 0; }
'''


def slug(value, separator):
    # GitHub식 한글 앵커를 보존한다. 문장부호 제거 후 각 공백을 '-'로 바꾼다.
    value = re.sub(r'[^\w\-\s]', '', value.strip().lower())
    return re.sub(r'\s', separator, value)


def prepare(source):
    text = source.read_text(encoding='utf-8')
    rendered = markdown.markdown(text, extensions=['tables', 'fenced_code', 'sane_lists', 'toc'],
                                 extension_configs={'toc': {'slugify': slug}})
    soup = BeautifulSoup(rendered, 'html.parser')
    h1 = soup.find('h1')
    h1['data-running-title'] = '팀원용 프로젝트 설명' if source.stem == 'TEAM_PROJECT_GUIDE' else '발표 인수인계'
    title = h1.get_text(' ', strip=True)
    h1.clear()
    for i, word in enumerate(title.split()):
        if i:
            h1.append(' ')
        span = soup.new_tag('span')
        span.string = word
        h1.append(span)
    # PDF 뷰어마다 한글 named destination을 다르게 처리하므로 내부 이름만 ASCII로 바꾼다.
    ids = {}
    for i, tag in enumerate(soup.find_all(id=True), start=1):
        ids[tag['id']] = f'section-{i}'
        tag['id'] = f'section-{i}'
    for link in soup.find_all('a', href=True):
        href = link['href']
        if href.startswith('#'):
            if unquote(href[1:]) not in ids:
                raise ValueError(f'존재하지 않는 문서 내 앵커: {href}')
            link['href'] = '#'+ids[unquote(href[1:])]
            continue
        if href.startswith(('https:', 'http:', 'mailto:')):
            continue
        path_part, _, fragment = href.partition('#')
        target = (source.parent/unquote(path_part)).resolve()
        if not target.exists():
            raise FileNotFoundError(target)
        relative = target.relative_to(ROOT).as_posix()
        encoded = quote(relative, safe='/')
        if target.suffix.lower() in ('.png', '.jpg', '.jpeg', '.svg'):
            link['href'] = f'{RAW}/{encoded}'
        else:
            kind = 'tree' if target.is_dir() else 'blob'
            link['href'] = f'{REPO}/{kind}/main/{encoded}'
        if fragment:
            link['href'] += '#'+quote(fragment, safe='-')
    images = []
    for img in list(soup.find_all('img')):
        path = (source.parent/img['src']).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        images.append(str(path.relative_to(ROOT)))
        img['src'] = path.as_uri()
        parent = img.parent
        if parent.name != 'p':
            raise ValueError('그림은 독립 문단에 있어야 합니다.')
        next_p = parent.find_next_sibling()
        figure = soup.new_tag('figure')
        parent.replace_with(figure)
        figure.append(img.extract())
        if next_p and next_p.name == 'p' and next_p.get_text(strip=True).startswith('그림 '):
            caption = soup.new_tag('figcaption')
            caption.append(next_p.extract())
            figure.append(caption)
    for heading in list(soup.find_all('h2')):
        summary_label = heading.find_next_sibling()
        summary_list = summary_label.find_next_sibling() if summary_label else None
        if summary_label and summary_label.get_text(strip=True) == '5줄 요약' and summary_list and summary_list.name == 'ol':
            block = soup.new_tag('div', attrs={'class': 'section-summary'})
            heading.insert_before(block)
            for element in (heading, summary_label, summary_list):
                block.append(element.extract())
    for li in soup.find_all('li'):
        first = li.contents[0] if li.contents else None
        if isinstance(first, str) and first.startswith('[ ] '):
            first.replace_with('☐ '+first[4:])
    for table in soup.find_all('table'):
        headers = [th.get_text(' ', strip=True) for th in table.find_all('th')]
        count = len(headers)
        if headers and len(table.find_all('tr')) <= 9 and (
                headers[0] in ('시간', '순서') or len(table.get_text()) <= 600):
            table['class'] = ['keep-table']
        widths = None
        if count == 8:
            widths = [27, 8, 8, 15, 15, 10, 9, 8]
        elif count == 7:
            widths = [29, 9, 9, 16, 16, 10, 11]
        elif count == 6:
            widths = [18, 20, 10, 17, 17, 18]
        elif count == 5:
            widths = [24, 18, 18, 18, 22]
        elif count == 4:
            widths = [19, 25, 27, 29]
        elif count == 3:
            widths = [22, 34, 44]
        elif count == 2:
            widths = [29, 71]
        if widths:
            group = soup.new_tag('colgroup')
            for width in widths:
                col = soup.new_tag('col')
                col['style'] = f'width:{width}%'
                group.append(col)
            table.insert(0, group)
    kind = 'guide' if source.stem == 'TEAM_PROJECT_GUIDE' else 'handoff'
    document = f'<!doctype html><html lang="ko"><head><meta charset="UTF-8"><title>{html.escape(title)}</title><style>{STYLE}</style></head><body class="{kind}">{soup}</body></html>'
    info = {'source': source.name, 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
            'images': images, 'table_count': len(soup.find_all('table')),
            'heading_count': len(soup.find_all(re.compile('^h[1-4]$'))),
            'external_links': len([a for a in soup.find_all('a', href=True) if not a['href'].startswith('#')]),
            'internal_links': len([a for a in soup.find_all('a', href=True) if a['href'].startswith('#')])}
    return document, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT)
    parser.add_argument('--report', type=Path, default=ROOT/'artifacts/pdf_export/manifest.json')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    report = []
    for name in DOCUMENTS:
        source = ROOT/name
        document, info = prepare(source)
        # HTML은 그림 검수·재현을 위한 로컬 캐시다.
        (args.report.parent/f'{source.stem}.html').write_text(document, encoding='utf-8')
        pdf = args.output_dir/source.with_suffix('.pdf').name
        rendered = HTML(string=document, base_url=str(ROOT)).render(
            optimize_images=True, jpeg_quality=88, dpi=160)
        rendered.write_pdf(pdf)
        info.update({'pdf': pdf.name, 'pages': len(rendered.pages), 'bytes': pdf.stat().st_size,
                     'pdf_sha256': hashlib.sha256(pdf.read_bytes()).hexdigest()})
        report.append(info)
        print(json.dumps({k:info[k] for k in ['pdf','pages','bytes','table_count','internal_links']}, ensure_ascii=False))
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
