"""Descriptive reporting only; no policy, data selection or experiment rerun.

Added after execution to expand predeclared secondary metrics and separate two
overlapping plot labels. The original frozen five experiment sources stay intact.
"""
import json
from collections import defaultdict
from math import fsum
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parents[1]


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)+'\n')


def main():
    assert not (OUT/'completion_receipt.json').exists()
    rows = read(OUT/'metrics.json')
    ncell = {r: len(v['ids']) for r, v in read(ROOT/'output/disaster_route_loop_v1/public_geometry.json').items()}
    groups = defaultdict(list)
    keys = ('region', 'mode', 'noise', 'budget_s', 'policy')
    for r in rows:
        groups[tuple(r[k] for k in keys)].append(r)
    out = []
    for key, rs in sorted(groups.items()):
        arrivals = [r['first_target_arrival_s'] for r in rs if r['first_target_arrival_s'] is not None]
        recalls = [r['visited_target_recall'] for r in rs if r['visited_target_recall'] is not None]
        detections = [r['detected_target_recall'] for r in rs if r['detected_target_recall'] is not None]
        out.append({**dict(zip(keys, key)), 'runs': len(rs),
            'mean_cell_coverage_fraction': fsum(r['observations']/ncell[r['region']] for r in rs)/len(rs),
            'mean_visited_target_cells': fsum(r['visited_target_cells'] for r in rs)/len(rs),
            'mean_visited_target_recall_on_nonempty_worlds': fsum(recalls)/len(recalls) if recalls else None,
            'mean_detected_target_recall_on_nonempty_worlds': fsum(detections)/len(detections) if detections else None,
            'nonempty_worlds': len(recalls), 'target_arrival_failures': len(rs)-len(arrivals),
            'no_target_arrival_rate': (len(rs)-len(arrivals))/len(rs),
            'mean_first_target_arrival_s_on_successes': fsum(arrivals)/len(arrivals) if arrivals else None,
            'mean_first_target_arrival_or_budget_s': fsum(r['first_target_arrival_s'] if r['first_target_arrival_s'] is not None else r['budget_s'] for r in rs)/len(rs)})
    assert len(out) == 324 and all(r['runs'] == 20 for r in out)
    # Verify the supplementary aggregates by separate NumPy calculations.
    for r in out:
        rs = groups[tuple(r[k] for k in keys)]
        restricted = [x['first_target_arrival_s'] if x['first_target_arrival_s'] is not None else x['budget_s'] for x in rs]
        assert abs(r['mean_first_target_arrival_or_budget_s']-float(np.mean(restricted))) < 1e-7
        assert abs(r['mean_cell_coverage_fraction']-float(np.mean([x['observations']/ncell[x['region']] for x in rs]))) < 1e-12
        assert r['target_arrival_failures'] == sum(x['first_target_arrival_s'] is None for x in rs)
        assert r['nonempty_worlds'] == sum(x['true_target_cells'] > 0 for x in rs)
    save(OUT/'arrival_coverage_supplement.json', {
        'scope': 'Predeclared secondary arrival and coverage metrics;descriptive postprocessing,no policy/threshold change.',
        'failure_handling': 'Arrival null preserved in raw data;restricted arrival uses budget;conditional arrival separately reports failure count. Recall excludes only zero-target denominators and reports that count.',
        'condition_means': out})
    save(OUT/'secondary_reporting_verification.json', {'status': 'PASS', 'groups_checked': 324,
        'rows': 6480, 'scope': 'Metrics already independently checked against complete route histories by main audit;aggregation checked separately.'})
    colors = {'pre_static': '#94A3B8', 'post_static': '#334155', 'always': '#2563EB',
              'periodic_5': '#D97706', 'gain_5': '#059669', 'one_sd_5': '#9333EA'}
    labels = {'pre_static': 'Pre', 'post_static': 'Post', 'always': 'Always', 'periodic_5': 'Every5', 'gain_5': 'Mean', 'one_sd_5': '1SD'}
    offsets = {'periodic_5': (5, 10), 'gain_5': (5, -14)}
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.1), constrained_layout=True)
    for ax, noise in zip(axes, ('perfect', 'moderate', 'poor')):
        for policy in colors:
            r = next(x for x in read(OUT/'summary.json')['pooled_noise_means'] if x['noise'] == noise and x['policy'] == policy)
            ax.scatter(r['planning_calls'], r['detected_target_cells'], c=colors[policy], s=50)
            offset = offsets.get(policy, (4, 4))
            if noise == 'perfect' and policy in ('periodic_5', 'gain_5'):
                offset = (5, -12) if policy == 'periodic_5' else (5, 7)
            ax.annotate(labels[policy], (r['planning_calls'], r['detected_target_cells']),
                        xytext=offset, textcoords='offset points', fontsize=8,
                        arrowprops={'arrowstyle': '-', 'color': colors[policy], 'lw': .6})
        ax.grid(alpha=.2)
        ax.set(title=noise, xlabel='Paid planning calls,including rejections', xlim=(0, 43))
        ax.margins(y=.12)
    axes[0].set_ylabel('Mean correct synthetic target cells')
    fig.suptitle('SIMULATION | 1200 s | equal-weight mean across both geometries and3 priors\nMeans are descriptive;candidate computation is paid even when rejected')
    fig.savefig(OUT/'figures/planning_tradeoff.png', dpi=175)
    plt.close(fig)
    print('SECONDARY_METRICS_VERIFIED_AND_LABELS_SEPARATED')


if __name__ == '__main__':
    main()
