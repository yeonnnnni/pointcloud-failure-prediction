"""최종 test 실측값을 재집계해 개선 중심 발표 그림을 생성한다."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from pfp.plots import CLASSES, configure_style

CONF = 'lgbm_confidence'
COMB = 'lgbm_combined'
BLUE, ORANGE, NAVY = '#2766A5', '#D76526', '#16324F'
GREEN, RED, GRAY = '#168171', '#BA4251', '#64748B'
SCOPE = '최종 test · 494개 물체 × 75조건 = 37,050건 · 오분류 10,331건'


def native(obj):
    if isinstance(obj, dict):
        return {str(k): native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [native(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def canvas(title, subtitle):
    fig = plt.figure(figsize=(16, 9), facecolor='white')
    fig.text(.055, .925, title, fontsize=28, weight='bold', color=NAVY)
    fig.text(.055, .868, subtitle, fontsize=15, color=GRAY)
    fig.text(.055, .037, SCOPE, fontsize=12, color=GRAY)
    return fig


def save(fig, output, name, manifest, detail):
    fig.savefig(output / f'{name}.png', dpi=180, facecolor='white')
    fig.savefig(output / f'{name}.pdf', facecolor='white')
    plt.close(fig)
    manifest['figures'][name] = detail


def measured(frame):
    result = {}
    total = int(frame.error.sum())
    for model in (CONF, COMB):
        ranks = frame[f'rank_{model}'].to_numpy()
        if not np.array_equal(np.sort(ranks), np.arange(1, len(frame)+1)):
            raise ValueError('순위가 1..N 순열이 아닙니다.')
        # 동점 처리까지 저장된 점수와 독립 대조한다.
        order = frame.sort_values([f'score_{model}', 'object_id', 'corruption', 'severity'],
                                 ascending=[False, True, True, True])
        if not np.array_equal(order[f'rank_{model}'], np.arange(1, len(frame)+1)):
            raise ValueError('점수와 순위가 다릅니다.')
        row = {'ap': average_precision_score(frame.error, frame[f'score_{model}'])}
        for pct in (10, 20):
            k = math.ceil(len(frame)*pct/100)
            caught = int(frame.loc[frame[f'rank_{model}'] <= k, 'error'].sum())
            row[f'caught_{pct}'] = caught
            row[f'capture_{pct}'] = caught / total
            row[f'reviewed_{pct}'] = k
        result[model] = row
    return result


def gains(frame, metrics, output, manifest):
    fig = canvas('같은 검토량으로 오분류를 더 많이 포착했다',
                 '동일 LightGBM에서 confidence 3개에 geometry 8개를 추가한 결과')
    c, g = metrics[CONF], metrics[COMB]
    cols = [('AP', 'ap', 1., 1.0, None), ('10% 검토 포착률', 'capture_10', 100., 100, 10),
            ('20% 검토 포착률', 'capture_20', 100., 100, 20)]
    for i, (title, key, scale, ymax, pct) in enumerate(cols):
        left = .08+i*.315
        ax = fig.add_axes([left, .245, .225, .47])
        values = [c[key]*scale, g[key]*scale]
        ax.bar([0, 1], values, color=[BLUE, ORANGE], width=.6)
        ax.set(ylim=(0, ymax), xticks=[0, 1], xticklabels=['confidence', '+ geometry'])
        ax.tick_params(axis='x', labelsize=14)
        ax.tick_params(axis='y', labelsize=12)
        ax.grid(axis='y', alpha=.12)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=21, weight='bold', pad=30, color=NAVY)
        for x, val in enumerate(values):
            ax.text(x, val+ymax*.025, f'{val:.4f}' if pct is None else f'{val:.2f}%',
                    ha='center', fontsize=22, weight='bold', color=[BLUE, ORANGE][x])
        delta = (g[key]-c[key])*scale
        fig.text(left+.112, .166, f'+{delta:.5f}' if pct is None else f'+{delta:.2f}pp',
                 ha='center', fontsize=27, weight='bold', color=ORANGE)
        text = ('오류 탐지 순위의 품질' if pct is None else
               f'오류 포착 +{g[f"caught_{pct}"]-c[f"caught_{pct}"]:,}건 (순증)')
        fig.text(left+.112, .118, text, ha='center', fontsize=17, color=NAVY)
    save(fig, output, 'presentation_main_gains', manifest, {'metrics': metrics,
         'note': 'AP는 0~1, 포착률은 0~100% 축. pp는 퍼센트포인트. 동일 알고리즘·동일 튜닝 예산.'})


def budget_outcomes(frame, metrics, output, manifest):
    fig = canvas('사람이 확인할 입력 수는 그대로, 그 안의 오답은 더 많게',
                 '막대의 전체 길이는 검토 건수 · 색 영역은 그중 실제 오분류인 건수')
    rows = []
    for i, pct in enumerate((10, 20)):
        ax = fig.add_axes([.13, .51-i*.32, .77, .21])
        k = metrics[CONF][f'reviewed_{pct}']
        caught = [metrics[m][f'caught_{pct}'] for m in (CONF, COMB)]
        for y, (n, col) in enumerate(zip(caught, [BLUE, ORANGE])):
            ax.barh(y, n, color=col, height=.48)
            ax.barh(y, k-n, left=n, color='#E7ECF2', height=.48)
            ax.text(n/2, y, f'오분류 {n:,}건', ha='center', va='center', color='white',
                    fontsize=20, weight='bold')
            ax.text(n+(k-n)/2, y, f'정답 {k-n:,}건', ha='center', va='center', fontsize=16, color=NAVY)
        ax.set(xlim=(0, k), yticks=[0, 1], yticklabels=['confidence', '+ geometry'])
        ax.invert_yaxis()
        ax.tick_params(axis='y', labelsize=16, length=0)
        ax.tick_params(axis='x', labelsize=12)
        ax.set_title(f'{pct}% 검토 = {k:,}건 확인  /  오류 포착 순증 +{caught[1]-caught[0]:,}건',
                     loc='left', color=NAVY, fontsize=22, pad=15)
        ax.spines[['left', 'bottom']].set_visible(False)
        rows.append({'budget_pct': pct, 'reviewed': k, 'caught_confidence': caught[0],
                     'caught_combined': caught[1], 'additional': caught[1]-caught[0]})
    fig.text(.13, .097, '포착률의 분모는 검토 건수가 아니라 test 전체 오분류 10,331건이다.', fontsize=16, color=GRAY)
    save(fig, output, 'presentation_review_outcomes', manifest, rows)


def turnover(frame, output, manifest):
    fig = canvas('새로 찾은 오류와 놓치게 된 오류를 함께 계산했다',
                 '검토 목록이 바뀌므로 추가 포착 건수는 새 진입 오류에서 이탈 오류를 뺀 순증가다.')
    summary = []
    for i, pct in enumerate((10, 20)):
        k = math.ceil(len(frame)*pct/100)
        errors = frame[frame.error == 1]
        a = errors[f'rank_{CONF}'] <= k
        b = errors[f'rank_{COMB}'] <= k
        common, new, lost = int((a & b).sum()), int((~a & b).sum()), int((a & ~b).sum())
        ax = fig.add_axes([.095+i*.47, .24, .36, .48])
        values = [new, -lost, new-lost]
        bars = ax.bar([0, 1, 2], values, color=[GREEN, RED, ORANGE], width=.57)
        ax.axhline(0, color=GRAY, lw=1)
        ax.set_xticks([0, 1, 2], ['새로 포착', '이탈', '순증가'], fontsize=17)
        ax.set_ylabel('오분류 건수', fontsize=15)
        ax.set_title(f'{pct}% 검토 예산', fontsize=23, color=NAVY, pad=20)
        maxy = max(abs(np.asarray(values)))
        ax.set_ylim(-maxy*1.3, maxy*1.3)
        for rect, val in zip(bars, values):
            ax.text(rect.get_x()+rect.get_width()/2, val+np.sign(val)*maxy*.07,
                    f'{val:+,}건', ha='center', va='bottom' if val>0 else 'top',
                    fontsize=22, weight='bold')
        fig.text(.275+i*.47, .13, f'두 방법이 공통으로 포착한 오류 {common:,}건',
                 ha='center', color=GRAY, fontsize=16)
        summary.append({'budget_pct': pct, 'common_errors': common, 'new_errors': new,
                        'lost_errors': lost, 'net_gain': new-lost})
    save(fig, output, 'presentation_review_turnover', manifest, summary)


def conditions(frame, output, manifest):
    rows = []
    for (corruption, severity), group in frame.groupby(['corruption', 'severity']):
        ac = average_precision_score(group.error, group[f'score_{CONF}'])
        ag = average_precision_score(group.error, group[f'score_{COMB}'])
        rows.append({'corruption': corruption, 'severity': severity, 'n_rows': len(group),
                     'ap_confidence': ac, 'ap_combined': ag, 'delta_ap': ag-ac})
    df = pd.DataFrame(rows)
    expected = pd.read_csv(ROOT/'results/tables/geometry_delta_by_condition.csv')
    merged = df.merge(expected, on=['corruption', 'severity'], suffixes=('', '_saved'), validate='one_to_one')
    np.testing.assert_allclose(merged.delta_ap, merged.delta_ap_saved, atol=1e-12, rtol=1e-12)
    top = df.nlargest(6, 'delta_ap').sort_values('delta_ap')
    fig = canvas('어떤 조건에서 개선이 컸나?',
                 '왼쪽은 조건 내부 AP 개선 상위 6개를 사후 선택 · 오른쪽은 전체 75개 조건의 변화')
    ax = fig.add_axes([.17, .23, .34, .51])
    for y, row in enumerate(top.itertuples()):
        ax.plot([row.ap_confidence, row.ap_combined], [y, y], color='#C6D0DC', lw=7, zorder=1)
        ax.scatter(row.ap_confidence, y, color=BLUE, s=100, zorder=2)
        ax.scatter(row.ap_combined, y, color=ORANGE, s=100, zorder=2)
        ax.text(row.ap_combined+.025, y, f'+{row.delta_ap:.4f}', va='center', fontsize=16,
                color=ORANGE, weight='bold')
    ax.set(yticks=range(len(top)), yticklabels=[f'{r.corruption} · {r.severity}' for r in top.itertuples()],
           xlim=(0, 1.04), ylim=(-.7, len(top)-.3), xlabel='각 조건 안에서의 AP (0~1)')
    ax.tick_params(axis='y', labelsize=15)
    ax.scatter([], [], color=BLUE, label='confidence')
    ax.scatter([], [], color=ORANGE, label='+ geometry')
    ax.legend(loc='upper left', bbox_to_anchor=(-.1, 1.18), ncol=2, frameon=False, fontsize=14)
    ax.grid(axis='x', alpha=.12)
    ax2 = fig.add_axes([.65, .23, .28, .51])
    ordered = df.sort_values('delta_ap')
    colors = np.where(ordered.delta_ap > 0, GREEN, RED)
    ax2.barh(np.arange(75), ordered.delta_ap, color=colors, height=.85)
    ax2.axvline(0, color=GRAY, lw=1)
    ax2.set(yticks=[], xlabel='AP 차이: combined − confidence', xlim=(-.07, .075), ylim=(-2, 77))
    positive = int((df.delta_ap > 0).sum())
    negative = int((df.delta_ap < 0).sum())
    ax2.set_title(f'{positive}개 개선 · {negative}개 악화', fontsize=22, color=NAVY, pad=20)
    fig.text(.095, .13, f'조건별 AP 차이의 평균 +{df.delta_ap.mean():.5f}  /  전체 조건을 섞은 핵심 AP 차이 +{manifest["metrics"][COMB]["ap"]-manifest["metrics"][CONF]["ap"]:.5f}',
             fontsize=17, color=NAVY)
    fig.text(.095, .083, '집계 방식이 달라 두 수치는 다르다. 특정 조건의 효과를 모든 조건에 일반화하지 않는다.', fontsize=14, color=GRAY)
    df.to_csv(output/'presentation_condition_values.csv', index=False)
    save(fig, output, 'presentation_condition_highlights', manifest,
         {'selection': '75조건에서 조건 내부 AP 차이 상위6개, 사후 선택', 'top6': top.to_dict('records'),
          'improved': positive, 'worsened': negative, 'mean_delta': df.delta_ap.mean(),
          'worst': df.nsmallest(1, 'delta_ap').to_dict('records')})


def uncertainty(metrics, output, manifest):
    data = json.loads((ROOT/'results/tables/paired_bootstrap.json').read_text())
    fig = canvas('주요 개선은 물체 단위 재표집에서도 양수였다',
                 '점은 실제 test의 차이 · 선은 paired object bootstrap 300회의 95% 구간')
    specs = [('AP 차이', 'ap', 'ap', 1, .14), ('10% 포착률 차이', 'capture_10', 'capture_rate_10', 100, 7),
             ('20% 포착률 차이', 'capture_20', 'capture_rate_20', 100, 7)]
    rows = []
    for i, (label, metric, bmetric, scale, xmax) in enumerate(specs):
        ax = fig.add_axes([.13, .62-i*.20, .72, .070])
        low = data['metrics'][bmetric]['ci95_low']*scale
        high = data['metrics'][bmetric]['ci95_high']*scale
        delta = (metrics[COMB][metric]-metrics[CONF][metric])*scale
        ax.errorbar(delta, 0, xerr=[[delta-low], [high-delta]], fmt='o', ms=13,
                    color=ORANGE, capsize=8, lw=4)
        ax.axvline(0, color=GRAY, linestyle='--', lw=1.5)
        ax.set(xlim=(-xmax*.05, xmax), ylim=(-1, 1), yticks=[])
        fig.text(.13, .755-i*.20, label, fontsize=20, color=NAVY)
        ax.spines[['left', 'bottom']].set_visible(False)
        ax.tick_params(axis='x', labelsize=12)
        suffix = '' if scale == 1 else 'pp'
        fig.text(.71, .755-i*.20, f'+{delta:.4f}' if scale == 1 else f'+{delta:.2f}pp',
                 fontsize=20, color=ORANGE, weight='bold')
        fig.text(.13, .712-i*.20, f'95% 구간 [{low:.4f}, {high:.4f}]' if scale == 1 else f'95% 구간 [{low:.2f}, {high:.2f}]pp',
                 fontsize=14, color=GRAY)
        rows.append({'metric': bmetric, 'observed_delta': delta/scale, 'ci95_low': low/scale,
                     'ci95_high': high/scale, 'display_unit': suffix or 'AP'})
    fig.text(.13, .095, '고정된 모델의 test 표본 불확실성이다. 재학습·다른 분할·다른 데이터의 변동은 포함하지 않는다.',
             fontsize=14, color=GRAY)
    save(fig, output, 'presentation_gain_intervals', manifest, rows)


def select_case(frame, pct):
    k = math.ceil(len(frame)*pct/100)
    candidates = frame[(frame.error == 1) & (frame[f'rank_{CONF}'] > k) & (frame[f'rank_{COMB}'] <= k)].copy()
    candidates['rank_gain'] = candidates[f'rank_{CONF}']-candidates[f'rank_{COMB}']
    if pct == 20:
        # 10% 그림과 중복 없이 20% 경계의 의미를 보여 준다.
        candidates = candidates[candidates[f'rank_{COMB}'] > math.ceil(len(frame)*.1)]
    high = candidates[candidates.msp >= .9]
    if not high.empty:
        candidates = high
    return candidates.sort_values(['rank_gain', 'msp', 'object_id'], ascending=[False, False, True]).iloc[0]


def point_case(frame, pct, data_dir, output, manifest):
    row = select_case(frame, pct)
    oid, severity = int(row.object_id), int(row.severity)
    original = np.load(data_dir/'data_original.npy', mmap_mode='r')[oid, :, :3]
    cloud = np.load(data_dir/f'data_{row.corruption}_{severity}.npy', mmap_mode='r')[oid, :, :3]
    if not np.isfinite(original).all() or not np.isfinite(cloud).all():
        raise ValueError('사례 좌표에 비유한값이 있습니다.')
    fig = canvas(f'놓쳤던 실제 오답이 {pct}% 검토 대상 안으로 들어왔다',
                 f'개선 사례 사후 선택 · {row.corruption} level {severity} · 원본 ID {oid} · 정답 {CLASSES[int(row.true_label)]} → PointNet 예측 {CLASSES[int(row.pred_label)]}')
    both = np.vstack([original, cloud])
    center = (both.min(axis=0)+both.max(axis=0))/2
    radius = np.ptp(both, axis=0).max()/2*1.05
    for left, points, label, color in [(.05, original, f'같은 물체의 기준 점군 · {len(original):,}점', BLUE),
                                       (.40, cloud, f'실제 손상 입력 · {len(cloud):,}점', ORANGE)]:
        ax = fig.add_axes([left, .37, .33, .43], projection='3d')
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=4, c=color, alpha=.72, linewidths=0)
        ax.set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[1]-radius, center[1]+radius),
               zlim=(center[2]-radius, center[2]+radius))
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=22, azim=-58)
        ax.set_axis_off()
        ax.set_title(label, fontsize=18, color=NAVY, pad=0)
    fig.text(.76, .69, 'PointNet의 최고 클래스 확률', fontsize=14, color=GRAY)
    fig.text(.76, .62, f'MSP {row.msp:.3f}', fontsize=30, weight='bold', color=NAVY)
    fig.text(.76, .55, '정답과 다른 클래스를 예측', fontsize=15, color=RED)
    fig.text(.76, .455, '점군은 같은 축 범위로 표시\n기준 점군의 예측은 표시하지 않음', fontsize=13, color=GRAY, linespacing=1.7)
    ax = fig.add_axes([.20, .15, .64, .16])
    ranks = [int(row[f'rank_{CONF}']), int(row[f'rank_{COMB}'])]
    percentages = np.asarray(ranks)/len(frame)*100
    ax.axvspan(0, pct, color='#E7F3EE', zorder=0)
    ax.axvline(pct, color=GREEN, linestyle='--', lw=1.8)
    for y, val, rank, color in zip([1, 0], percentages, ranks, [BLUE, ORANGE]):
        ax.plot([0, val], [y, y], color=color, alpha=.4, lw=3)
        ax.scatter(val, y, color=color, s=110, zorder=3)
        ax.text(val+2, y, f'{rank:,}위 (상위 {val:.2f}%)', fontsize=16, color=color, va='center', bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .9, 'pad': 1})
    ax.set(xlim=(0, 113), ylim=(-.65, 1.65), yticks=[1, 0], yticklabels=['confidence', '+ geometry'],
           xticks=[0, 10, 20, 40, 60, 80, 100], xlabel='전체 test에서의 검토 순위 비율 (%) · 작을수록 먼저 검토')
    ax.tick_params(axis='y', length=0, labelsize=17)
    ax.set_title(f'음영: 상위 {pct}% 검토 범위', loc='left', color=GREEN, fontsize=16, pad=8)
    ax.spines[['left', 'bottom']].set_visible(False)
    detail = {'selection': '해당 예산 새 진입 오답 중 MSP≥0.9 우선, 순위 상승 최대; 20%는 10% 범위 밖',
              'budget_pct': pct, 'object_id': oid, 'corruption': row.corruption, 'severity': severity,
              'true_label': int(row.true_label), 'true_class': CLASSES[int(row.true_label)],
              'pred_label': int(row.pred_label), 'pred_class': CLASSES[int(row.pred_label)],
              'msp': float(row.msp), 'rank_confidence': ranks[0], 'rank_combined': ranks[1],
              'original_points': len(original), 'corrupted_points': len(cloud), 'error': int(row.error),
              'score_confidence': float(row[f'score_{CONF}']), 'score_combined': float(row[f'score_{COMB}'])}
    save(fig, output, f'presentation_case_{pct}', manifest, detail)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, default=ROOT/'artifacts/full/evaluation/test_predictions.parquet')
    parser.add_argument('--data-dir', type=Path, default=ROOT/'data/modelnet40_c')
    parser.add_argument('--output', type=Path, default=ROOT/'results/figures')
    args = parser.parse_args()
    configure_style()
    args.output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(args.predictions)
    if len(frame) != 37050 or frame.object_id.nunique() != 494 or int(frame.error.sum()) != 10331:
        raise ValueError('최종75조건 test 범위와 다릅니다.')
    if set(frame.split) != {'test'} or len(frame[['corruption', 'severity']].drop_duplicates()) != 75:
        raise ValueError('최종 test 범위와 다릅니다.')
    if frame.duplicated(['object_id', 'corruption', 'severity']).any():
        raise ValueError('중복키가 있습니다.')
    if not frame.groupby('object_id').size().eq(75).all():
        raise ValueError('물체별75조건이 아닙니다.')
    if not frame.error.eq(frame.true_label.ne(frame.pred_label).astype(int)).all():
        raise ValueError('타깃과 클래스가 다릅니다.')
    split = pd.read_csv(ROOT/'configs/object_splits.csv')
    if set(frame.object_id) != set(split.loc[split.split == 'test', 'object_id']):
        raise ValueError('저장된 최종 test ID와 다릅니다.')
    metrics = measured(frame)
    saved = pd.read_csv(ROOT/'results/tables/detection_metrics.csv').set_index('model')
    for model, data in metrics.items():
        np.testing.assert_allclose(data['ap'], saved.loc[model, 'ap'], atol=1e-12, rtol=1e-12)
        for pct in (10, 20):
            np.testing.assert_allclose(data[f'capture_{pct}'], saved.loc[model, f'capture_rate_{pct}'], atol=1e-12, rtol=1e-12)
            assert data[f'caught_{pct}'] == saved.loc[model, f'n_captured_errors_{pct}']
    manifest = {'scope': SCOPE, 'source_predictions_sha256': hashlib.sha256(args.predictions.read_bytes()).hexdigest(),
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'metrics': metrics,
                'validation': '원본 test ID·75조건·타깃·중복·점수순위·기존 지표 대조 통과', 'figures': {}}
    gains(frame, metrics, args.output, manifest)
    budget_outcomes(frame, metrics, args.output, manifest)
    turnover(frame, args.output, manifest)
    conditions(frame, args.output, manifest)
    uncertainty(metrics, args.output, manifest)
    for pct in (10, 20):
        point_case(frame, pct, args.data_dir, args.output, manifest)
    (args.output/'presentation_manifest.json').write_text(json.dumps(native(manifest), ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'figures': list(manifest['figures']), 'validation': manifest['validation']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
