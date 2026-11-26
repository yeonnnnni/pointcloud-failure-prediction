"""실제 평가용 점군 한 개로 세 기하 특징군을 설명하는 발표 그림을 만든다."""
from __future__ import annotations

import argparse
from itertools import product
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/pfp_geometry_explainer_mpl')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from pfp.geometry import GEOMETRY, geometry_one
from pfp.data import sha256

INK = '#172B43'
MUTED = '#556A7C'
CLOUD = '#587A8C'
AXIS_COLORS = ['#D76A46', '#2C9B8B', '#7862B4']
PERMUTATION = [0, 2, 1]
CORRUPTION_NAMES = {'gaussian': '가우시안 잡음', 'cutout': '부분 제거'}
FEATURE_NAMES = ['첫째 방향 퍼짐 비율', '둘째 방향 퍼짐 비율', '셋째 방향 퍼짐 비율',
                 '가로축 길이', '높이축 길이', '깊이축 길이', '이웃 거리 평균', '이웃 거리 표준편차']


def setup_font():
    candidates = ['/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
                  '/usr/share/fonts/truetype/noto/NotoSansKR-Regular.ttf']
    for path in candidates:
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            family = font_manager.FontProperties(fname=path).get_name()
            plt.rcParams.update({'font.family': family, 'axes.unicode_minus': False,
                                 'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
            return path
    raise FileNotFoundError('한국어 Noto 글꼴이 필요합니다.')


def draw_points(ax, xyz, *, alpha=0.68, size=7, color=CLOUD):
    view = xyz[:, PERMUTATION]
    ax.scatter(*view.T, c=color, s=size, alpha=alpha, edgecolors='none', depthshade=False)


def style_3d(ax, xyz, pad=0.18):
    view = xyz[:, PERMUTATION]
    center = (view.min(0) + view.max(0)) / 2
    radius = np.ptp(view, axis=0).max() * (0.5 + pad)
    for setter, value in zip([ax.set_xlim, ax.set_ylim, ax.set_zlim], center):
        setter(value - radius, value + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=27, azim=-54)
    ax.set_proj_type('ortho')
    ax.set_axis_off()
    ax.set_facecolor('white')


def plot(args):
    font_path = setup_font()
    source = ROOT / args.data_dir / f'data_{args.corruption}_{args.severity}.npy'
    splits = pd.read_csv(ROOT / 'configs/object_splits.csv')
    row = splits.loc[splits.object_id == args.object_id].iloc[0]
    if row['split'] != 'test':
        raise ValueError('이 그림은 저장된 test 물체만 사용합니다.')
    if int(row.true_label) != 0:
        raise ValueError('현재 설명 그림은 비행기 물체를 대상으로 합니다.')
    xyz = np.asarray(np.load(source, mmap_mode='r')[args.object_id], dtype=np.float64)
    # 좌표를 다시 정규화하거나 재샘플링하지 않는다.
    features = geometry_one(xyz)
    feature_map = dict(zip(GEOMETRY, map(float, features)))
    centered = xyz - xyz.mean(0)
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered / (len(xyz) - 1))
    eigenvalues, eigenvectors = eigenvalues[::-1].clip(0), eigenvectors[:, ::-1]
    for j in range(3):
        pivot = np.argmax(np.abs(eigenvectors[:, j]))
        if eigenvectors[pivot, j] < 0:
            eigenvectors[:, j] *= -1
    np.testing.assert_allclose(eigenvalues / eigenvalues.sum(), features[:3], rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(np.ptp(xyz, axis=0), features[3:6], rtol=0, atol=0)

    # 같은 물체의 오른쪽 날개 쪽 실제 점 하나를 결정적으로 선택한다.
    goal = xyz.mean(0).copy()
    goal[0] = np.quantile(xyz[:, 0], 0.78)
    anchor = int(np.argmin(np.linalg.norm(xyz - goal, axis=1)))
    all_distance, all_index = cKDTree(xyz).query(xyz, k=9, workers=1)
    self_mask = all_index == np.arange(len(xyz))[:, None]
    distances = np.where(self_mask, np.inf, all_distance)
    nearest = np.take_along_axis(all_index, np.argsort(distances, axis=1)[:, :8], axis=1)
    gathered = np.take_along_axis(distances, np.argsort(distances, axis=1)[:, :8], axis=1)
    np.testing.assert_allclose([gathered.mean(), gathered.std(ddof=0)], features[-2:], rtol=1e-13, atol=1e-13)
    neighbors = nearest[anchor]
    assert anchor not in neighbors and len(set(neighbors.tolist())) == 8
    cached_path = ROOT / 'artifacts/full/conditions' / f'{args.corruption}_{args.severity}' / 'table.parquet'
    cache_checked = False
    if cached_path.exists():
        cached = pd.read_parquet(cached_path).set_index('object_id').loc[args.object_id, GEOMETRY].astype(float)
        np.testing.assert_allclose(cached.to_numpy(), features, rtol=1e-12, atol=1e-12)
        cache_checked = True

    fig = plt.figure(figsize=(16, 9), facecolor='white')
    fig.text(.042, .929, '점군의 모양을 세 종류의 숫자로 요약합니다',
             color=INK, fontsize=28, weight='bold', ha='left')
    type_name = CORRUPTION_NAMES.get(args.corruption, args.corruption)
    fig.text(.043, .879,
             f'평가용 물체 {args.object_id} · 비행기 · {type_name} {args.severity}단계 · {len(xyz):,}개 점',
             color=MUTED, fontsize=17)
    fig.add_artist(plt.Line2D([.043, .958], [.849, .849], transform=fig.transFigure,
                             color='#DCE4E9', linewidth=1.3))
    starts = [.042, .359, .676]
    titles = ['① 전체 크기', '② 주된 방향과 퍼짐', '③ 점 사이의 간격']
    descriptions = ['상자로 감싸 세 축의 길이를 잽니다', '어느 방향으로 넓게 퍼졌는지 봅니다', '한 점에 가장 가까운 8개 점을 찾습니다']
    for x, title, description in zip(starts, titles, descriptions):
        fig.text(x, .803, title, fontsize=22, weight='bold', color=INK)
        fig.text(x, .762, description, fontsize=14.5, color=MUTED)
    for separator in [.337, .654]:
        fig.add_artist(plt.Line2D([separator, separator], [.18, .81], transform=fig.transFigure,
                                 color='#E5EBEF', linewidth=1))

    axes = [fig.add_axes([x - .033, .315, .358, .445], projection='3d') for x in starts]
    for ax in axes:
        ax.computed_zorder = False
    for ax in axes[:2]:
        style_3d(ax, xyz, pad=.04)
        draw_points(ax, xyz, size=8, alpha=.69)

    lo, hi = xyz.min(0), xyz.max(0)
    corners = np.array(list(product(*zip(lo, hi))))
    for a in range(8):
        for b in range(a + 1, 8):
            if np.count_nonzero(corners[a] != corners[b]) == 1:
                segment = np.stack([corners[a], corners[b]])[:, PERMUTATION]
                axes[0].plot(*segment.T, color='#B9C6CF', linewidth=.9, alpha=.9)
    origin = np.array([lo[0], lo[1], hi[2]])
    for dimension, color in enumerate(AXIS_COLORS):
        endpoint = origin.copy()
        endpoint[dimension] = hi[dimension] if origin[dimension] == lo[dimension] else lo[dimension]
        segment = np.stack([origin, endpoint])[:, PERMUTATION]
        axes[0].plot(*segment.T, color=color, linewidth=3.2)
    axes[0].text2D(.5, .02, '점들을 모두 감싸는 상자', transform=axes[0].transAxes,
                   ha='center', fontsize=13.5, color=MUTED)

    center = xyz.mean(0)
    arrow_vectors = []
    for j, color in enumerate(AXIS_COLORS):
        vector = eigenvectors[:, j] * (2.2 * np.sqrt(eigenvalues[j]))
        arrow_vectors.append(vector.tolist())
        start = center - vector * .45
        arrow = vector * 1.45
        axes[1].quiver(*start[PERMUTATION], *arrow[PERMUTATION], color=color,
                       linewidth=3.0, arrow_length_ratio=.13, normalize=False, zorder=5)
        endpoint = center + vector * (1.75 if j == 2 else 1.22)
        axes[1].text(*endpoint[PERMUTATION], str(j + 1), color=color, fontsize=17, weight='bold',
                     zorder=10, bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .88, 'pad': .6})
    axes[1].text2D(.5, .02, '화살표: 퍼짐을 설명하는 세 방향', transform=axes[1].transAxes,
                   ha='center', fontsize=13.5, color=MUTED)

    local = np.r_[anchor, neighbors]
    local_radius = gathered[anchor].max() * 1.75
    visible = np.linalg.norm(xyz - xyz[anchor], axis=1) <= local_radius
    nearby = xyz[visible]
    style_3d(axes[2], np.stack([xyz[anchor] - local_radius, xyz[anchor] + local_radius]), pad=-.05)
    draw_points(axes[2], nearby, color='#D1DCE3', alpha=.7, size=28)
    for neighbor in neighbors:
        segment = np.stack([xyz[anchor], xyz[neighbor]])[:, PERMUTATION]
        axes[2].plot(*segment.T, color='#2C9B8B', alpha=.75, linewidth=1.9)
    draw_points(axes[2], xyz[neighbors], size=95, color='#2C9B8B', alpha=1)
    draw_points(axes[2], xyz[[anchor]], size=165, color='#D76A46', alpha=1)
    axes[2].text2D(.5, .02, '주황색 한 점의 주변을 확대', transform=axes[2].transAxes,
                   ha='center', fontsize=13.5, color=MUTED)

    for x in starts:
        fig.add_artist(FancyBboxPatch((x - .004, .183), .29, .13,
                                      boxstyle='round,pad=0.008,rounding_size=0.012',
                                      transform=fig.transFigure, facecolor='#F3F7FA',
                                      edgecolor='none', zorder=-1))
    fig.text(starts[0] + .141, .285, '원본 좌표에서 잰 세 축의 길이',
             fontsize=13.5, color=MUTED, ha='center')
    for offset, label, value, color in zip([.035, .139, .245], ['가로', '높이', '깊이'], features[3:6], AXIS_COLORS):
        fig.text(starts[0] + offset, .248, label, fontsize=16, color=color, ha='center', weight='bold')
        fig.text(starts[0] + offset, .207, f'{value:.3f}', fontsize=21, color=INK, ha='center', weight='bold')
    fig.text(starts[1] + .141, .285, '전체 퍼짐 중 각 방향이 차지하는 비율',
             fontsize=13.5, color=MUTED, ha='center')
    for offset, label, value, color in zip([.035, .139, .245], ['첫째', '둘째', '셋째'], features[:3], AXIS_COLORS):
        fig.text(starts[1] + offset, .248, label, fontsize=16, color=color, ha='center', weight='bold')
        fig.text(starts[1] + offset, .207, f'{100 * value:.1f}%', fontsize=21, color=INK, ha='center', weight='bold')
    fig.text(starts[2] + .141, .285, f'모든 점의 이웃 거리 {gathered.size:,}개를 모아 계산',
             fontsize=13, color=MUTED, ha='center')
    for offset, label, value in zip([.067, .218], ['평균 거리', '거리 표준편차'], features[-2:]):
        fig.text(starts[2] + offset, .248, label, fontsize=15.5, color='#2C9B8B', ha='center', weight='bold')
        fig.text(starts[2] + offset, .207, f'{value:.4f}', fontsize=21, color=INK, ha='center', weight='bold')

    fig.text(.043, .121, '같은 입력의 전체 크기 · 방향별 퍼짐 · 국소적인 점 간격을 함께 수치화합니다.',
             fontsize=18, weight='bold', color=INK)
    fig.text(.043, .079, '원본 손상 좌표로 계산 · 추가 정규화 없음 · 자기 자신만 이웃에서 제외 · 길이·거리는 데이터 좌표 단위',
             fontsize=13.5, color=MUTED)
    fig.text(.043, .042, '이 그림은 특징의 의미를 설명하는 실제 예시이며, 탐지 성능을 입증하는 결과는 아닙니다.',
             fontsize=13.5, color=MUTED)

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix('.png'), dpi=300)
    fig.savefig(output.with_suffix('.pdf'))
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas_box = fig.bbox
    clipped = []
    for text in fig.texts:
        box = text.get_window_extent(renderer)
        if box.x0 < canvas_box.x0 - 1 or box.x1 > canvas_box.x1 + 1 or box.y0 < canvas_box.y0 - 1 or box.y1 > canvas_box.y1 + 1:
            clipped.append(text.get_text())
    assert not clipped, f'그림 밖으로 벗어난 설명: {clipped}'
    plt.close(fig)
    table = pd.DataFrame({'특징': FEATURE_NAMES, '코드의 특징 이름': GEOMETRY, '실측값': features})
    table.to_csv(output.with_name(output.name + '_values.csv'), index=False)
    report = {'용도': '실제 점군을 이용한 기하 특징 설명; 탐지 성능의 근거가 아님',
              '물체_ID': args.object_id, '분할': 'test', '정답_분류': '비행기',
              '손상_유형': args.corruption, '표시_손상_유형': type_name, '단계': args.severity,
              '점_개수': len(xyz), '이웃_개수': 8, '거리_전체_개수': int(gathered.size),
              '입력_파일': str(source.relative_to(ROOT)), '입력_파일_SHA256': sha256(source),
              '좌표_추가_정규화': False, '추가_재샘플링': False,
              '화면_축_순서': ['x', 'z', 'y'], '화면_축_설명': '높이축을 위로 보이게 표시한 것으로 계산 좌표는 바꾸지 않음',
              '특징_실측값': feature_map, '주성분_고윳값': eigenvalues.tolist(),
              '주성분_방향_열벡터': eigenvectors.tolist(),
              '확대_기준점_인덱스': anchor, '확대_이웃점_인덱스': neighbors.tolist(),
              '확대_이웃_거리': gathered[anchor].tolist(),
              '검증': {'원본_실험_함수와_일치': True, '독립_계산과_일치': True,
                       '최종_실험_캐시와_일치': cache_checked, '평가용_물체_확인': True,
                       '텍스트_캔버스_벗어남': len(clipped), '비율_합': float(features[:3].sum()),
                       '자기자신_제외와_이웃8개': True},
              '이미지_크기': [4800, 2700], '글꼴': 'Noto 한국어 지원 글꼴', '글꼴_파일명': Path(font_path).name}
    output.with_name(output.name + '_validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'그림': str(output.relative_to(ROOT)) + '.png', '물체': args.object_id,
                      '실측값': feature_map, '검증': report['검증']}, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--object-id', type=int, default=25)
    parser.add_argument('--corruption', choices=sorted(CORRUPTION_NAMES), default='gaussian')
    parser.add_argument('--severity', type=int, choices=range(1, 6), default=1)
    parser.add_argument('--data-dir', default='data/modelnet40_c')
    parser.add_argument('--output', default='results/figures/presentation_geometry_explainer')
    plot(parser.parse_args())


if __name__ == '__main__':
    main()
