import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint

CONF = 0.95
MIOU = 0.35

city = pd.read_csv('./acdc_b0_b2_eval/csv/per_image_acdc.csv')
acdc = pd.read_csv('./cs_b0_b2_eval/csv/per_image_city.csv')
for df in [city, acdc]:
    df['backbone_short'] = df['student_backbone'].str.extract(r'mit-(b\d)')[0]
    df['seed'] = df['run_id'].str.extract(r'_s(\d+)').astype(int)

def compute_summary(df, dataset_label):
    rows = []
    for bb in ['b0', 'b1', 'b2']:
        base = {}
        for tm in ['sup', 'kd', 'skd']:
            sub = df[(df['seed'] == 42) & (df['backbone_short'] == bb) & (df['train_method'] == tm)].sort_values('image_id').copy()
            indicator = ((sub['student_msp'] >= CONF) & (sub['student_miou'] <= MIOU)).astype(int).values
            count = int(indicator.sum())
            n = int(len(indicator))
            lo, hi = proportion_confint(count, n, method='wilson')
            rows.append({
                'dataset': dataset_label,
                'backbone': bb,
                'method': tm,
                'count': count,
                'n': n,
                'rate': count / n,
                'ci_low': lo,
                'ci_high': hi,
            })
            base[tm] = indicator
        for tm in ['kd', 'skd']:
            x = base[tm]
            y = base['sup']
            offdiag_01 = int(((x == 0) & (y == 1)).sum())
            offdiag_10 = int(((x == 1) & (y == 0)).sum())
            p_value = mcnemar([[0, offdiag_01], [offdiag_10, 0]], exact=True).pvalue
            rows.append({
                'dataset': dataset_label,
                'backbone': bb,
                'method': f'{tm}_vs_sup',
                'p_value': p_value,
            })
    return pd.DataFrame(rows)

summary = pd.concat([
    compute_summary(city, 'Cityscapes'),
    compute_summary(acdc, 'ACDC'),
], ignore_index=True)
summary.to_csv('confident_failure_rate_summary.csv', index=False)

def p_to_stars(p):
    if pd.isna(p):
        return ''
    if p < 1e-3:
        return '***'
    if p < 1e-2:
        return '**'
    if p < 5e-2:
        return '*'
    return 'ns'

def plot_dataset(dataset, outpath):
    data = summary[(summary.dataset == dataset) & (summary.method.isin(['sup', 'kd', 'skd']))].copy()
    pvals = summary[(summary.dataset == dataset) & (summary.method.str.contains('_vs_sup'))].copy()

    methods = ['sup', 'kd', 'skd']
    labels = {
        'sup': 'Supervised student',
        'kd': 'KD student',
        'skd': 'SKD student',
    }

    x = np.arange(3)
    width = 0.24
    fig, ax = plt.subplots(figsize=(10, 5.6))

    for i, method in enumerate(methods):
        sub = data[data.method == method].set_index('backbone').loc[['b0', 'b1', 'b2']]
        offs = (i - 1) * width
        y = sub['rate'].values * 100
        yerr = np.vstack([
            (sub['rate'] - sub['ci_low']).values * 100,
            (sub['ci_high'] - sub['rate']).values * 100,
        ])
        ax.bar(x + offs, y, width=width, label=labels[method], capsize=4)
        # for xi, yi, count, n in zip(x + offs, y, sub['count'], sub['n']):
            # ax.text(xi, yi + max(0.7, yi * 0.03), f'{int(count)}/{int(n)}', ha='center', va='bottom', fontsize=9, rotation=90)

    for j, bb in enumerate(['b0', 'b1', 'b2']):
        for method in ['kd', 'skd']:
            p = float(pvals[(pvals.backbone == bb) & (pvals.method == f'{method}_vs_sup')]['p_value'].iloc[0])
            rate = float(data[(data.backbone == bb) & (data.method == method)]['rate'].iloc[0]) * 100
            xloc = j + (-1 + methods.index(method)) * width
            # ax.text(xloc, rate + 4.5, p_to_stars(p), ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(['MiT-B0', 'MiT-B1', 'MiT-B2'])
    ax.set_ylabel('Rate of confident, wrong frames (%)')
    ax.set_title(f'{dataset}: confident failure rate\nstudent MSP ≥ {CONF:.2f} and frame mIoU ≤ {MIOU:.2f}')
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    ax.set_ylim(0, max(data['ci_high'] * 100) + 2)
    ax.spines[['top', 'right']].set_visible(False)
    # ax.text(
    #     0.995,
    #     -0.18,
    #     'Error bars: Wilson 95% CI. Stars: exact McNemar test vs supervised student.',
    #     transform=ax.transAxes,
    #     ha='right',
    #     va='top',
    #     fontsize=9,
    # )
    plt.tight_layout()
    fig.savefig(outpath, dpi=220, bbox_inches='tight')
    plt.close(fig)

plot_dataset('Cityscapes', 'confident_failure_rate_city.png')
plot_dataset('ACDC', 'confident_failure_rate_acdc.png')
