#!/usr/bin/env python3
"""Validate Section 4 current-survivor features; NOT an online latency benchmark.

The PDF specifies predictor inputs but does not define its estimator. The fitted
stage-specific ridge model here is an explicit experimental implementation choice.
Labels use a fixed final oracle threshold, never the evolving search heap threshold.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ['eta_x_mean', 'eta_q', 'margin_tau_mean'] + [f'margin_u_bin{i}' for i in range(7)]
SETTINGS = {'glove-200-angular': (64, 512), 'landmark-dino-768-cosine': (160, 64)}


def refinement_value(survivors, predicted_rate, r, next_r, ready_dims,
                     exact_ns, projection_ns_per_dim, bound_ns_per_dim,
                     stage_ns_per_candidate=0., predictor_ns=0.):
    """Eq.25 decision with externally calibrated times, not guessed FLOP cycles.

    ready_dims is shared across lists for global, reset per list for local.
    stage cost includes residual/check/compaction; predictor overhead is explicit.
    Prediction never prunes: the deterministic bound must still be evaluated.
    """
    if next_r <= r or survivors < 0 or not 0 <= predicted_rate <= 1:
        raise ValueError('invalid refinement transition')
    costs = [exact_ns, projection_ns_per_dim, bound_ns_per_dim,
             stage_ns_per_candidate, predictor_ns]
    if not all(np.isfinite(costs)) or min(costs) < 0:
        raise ValueError('costs must be finite nonnegative calibrated times')
    projection_dims = max(0, next_r - max(r, ready_dims))
    refine_ns = (projection_dims * projection_ns_per_dim +
                 survivors * ((next_r - r) * bound_ns_per_dim + stage_ns_per_candidate) +
                 predictor_ns)
    saved_ns = survivors * predicted_rate * exact_ns
    return saved_ns - refine_ns


def features(rows):
    return np.asarray([[1.] + [float(row[key]) for key in FEATURES] for row in rows])


def fit(rows, ridge=0.001):
    x = features(rows)
    w = np.asarray([int(row['survivors']) for row in rows], dtype=float)
    y = np.asarray([int(row['true_next_pruned']) for row in rows]) / w
    w /= w.sum()
    penalty = np.eye(x.shape[1]) * ridge
    penalty[0, 0] = 0
    coefficients = np.linalg.solve(x.T @ (x * w[:, None]) + penalty, x.T @ (y * w))
    return coefficients, float(np.dot(w, y))


def predict(coefficients, rows):
    # UB^2 - LB^2 = 4*hq*hx. Candidates with relative margin >= 1
    # cannot be pruned at any deeper prefix while tau is fixed.
    eligible = 1 - np.asarray([float(row['margin_u_bin6']) for row in rows])
    return np.minimum(np.clip(features(rows) @ coefficients, 0, 1), eligible)


def validate_rows(rows):
    seen = set()
    groups = {}
    for row in rows:
        key = (int(row['q_id']), int(row['list_rank']), int(row['r']))
        if key in seen:
            raise ValueError(f'duplicate query/list/stage: {key}')
        seen.add(key)
        survivors = int(row['survivors'])
        gain = int(row['true_next_pruned'])
        if not 0 <= gain <= survivors or int(row['next_r']) <= int(row['r']):
            raise ValueError(f'invalid transition: {key}')
        values = [float(row[name]) for name in FEATURES]
        if not all(np.isfinite(values)):
            raise ValueError(f'nonfinite feature: {key}')
        if survivors and abs(sum(values[3:]) - 1) > 1e-6:
            raise ValueError(f'invalid margin histogram: {key}')
        groups.setdefault(key[:2], {})[key[2]] = row
    for stages in groups.values():
        for row in stages.values():
            following = stages.get(int(row['next_r']))
            if following is not None and int(row['survivors']) - int(row['true_next_pruned']) != int(following['survivors']):
                raise ValueError('survivor counts disagree across nested prefixes')


def analyze(path, output):
    with path.open() as stream:
        all_rows = list(csv.DictReader(stream))
    if not all_rows:
        raise ValueError(f'empty features: {path}')
    validate_rows(all_rows)
    query_ids = sorted({int(row['q_id']) for row in all_rows})
    if len(query_ids) < 5:
        raise ValueError('need at least five queries for independent validation')
    # Split entire queries, including all their visited lists and stages.
    test_ids = set(query_ids[::5])
    train_ids = set(query_ids) - test_ids
    assert not train_ids & test_ids
    stages = {}
    for row in all_rows:
        if int(row['survivors']) > 0:
            stages.setdefault((int(row['r']), int(row['next_r'])), []).append(row)
    report = {'source': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'oracle_final_tau': True, 'online_search_tested': False,
              'train_queries': sorted(train_ids), 'test_queries': sorted(test_ids),
              'features': FEATURES, 'stages': []}
    predictions = []
    model = {'features': ['intercept'] + FEATURES, 'ridge': 0.001, 'stages': {}}
    weighted_error = baseline_error = total_survivors = 0.
    for (r, next_r), rows in sorted(stages.items()):
        training = [row for row in rows if int(row['q_id']) in train_ids]
        testing = [row for row in rows if int(row['q_id']) in test_ids]
        if not training or not testing:
            report['stages'].append({'r': r, 'next_r': next_r, 'status': 'insufficient_nonempty_queries'})
            continue
        coefficients, baseline = fit(training)
        estimates = predict(coefficients, testing)
        weights = np.asarray([int(row['survivors']) for row in testing], dtype=float)
        actual = np.asarray([int(row['true_next_pruned']) for row in testing]) / weights
        error = np.dot(weights, np.abs(estimates - actual))
        base_error = np.dot(weights, np.abs(baseline - actual))
        weighted_error += error
        baseline_error += base_error
        total_survivors += weights.sum()
        report['stages'].append({'r': r, 'next_r': next_r, 'train_rows': len(training),
                                'test_rows': len(testing), 'survivors': int(weights.sum()),
                                'weighted_mae': float(error / weights.sum()),
                                'constant_baseline_mae': float(base_error / weights.sum())})
        model['stages'][f'{r}:{next_r}'] = coefficients.tolist()
        for row, estimate, truth in zip(testing, estimates, actual):
            predictions.append({'q_id': row['q_id'], 'list_rank': row['list_rank'],
                                'r': r, 'next_r': next_r, 'survivors': row['survivors'],
                                'predicted_rate': float(estimate), 'actual_rate': float(truth),
                                'constant_baseline_rate': baseline})
    if not total_survivors:
        raise ValueError('no nonempty held-out transitions; expand query sample')
    report.update(weighted_mae=weighted_error / total_survivors,
                  constant_baseline_mae=baseline_error / total_survivors)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    (output / 'model.json').write_text(json.dumps(model, indent=2) + '\n')
    with (output / 'heldout_predictions.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    print(f'{path.name}: heldout weighted MAE={report["weighted_mae"]:.4f}, '
          f'constant={report["constant_baseline_mae"]:.4f}', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, default=ROOT / 'datasets/cosine_supplement')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bin', type=Path, default=ROOT / 'build-perf/bin/query')
    parser.add_argument('--datasets', nargs='+', choices=list(SETTINGS), default=list(SETTINGS))
    parser.add_argument('--nq', type=int, default=64)
    parser.add_argument('--nlist', type=int, default=1024)
    parser.add_argument('--nprobe', type=int, help='override both datasets for small smoke tests')
    parser.add_argument('--threads', type=int, default=32)
    parser.add_argument('--analyze-only', action='store_true')
    args = parser.parse_args()
    if args.nq < 5 or args.threads < 1:
        parser.error('--nq must be >= 5 and --threads >= 1')
    args.output.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', OMP_WAIT_POLICY='PASSIVE')
    reports = []
    for ds in args.datasets:
        maximum, default_probe = SETTINGS[ds]
        for scope in ['global', 'per_list']:
            case = args.output / ds / scope
            case.mkdir(parents=True, exist_ok=True)
            csv_path = case / 'oracle.csv'
            feature_path = Path(str(csv_path) + '.predictor.csv')
            index = args.data_root / ds / 'index' / f'recall_shared_v2_nlist_{args.nlist}_seed0_unit_{scope}_pca_P{maximum}.index'
            if not args.analyze_only:
                if not index.is_file():
                    raise FileNotFoundError(index)
                command = [str(args.bin.resolve()), '--benchmarks_path', str(args.data_root.resolve()),
                           '--dataset', ds, '--input_format', 'fvecs', '--output_format', 'bin',
                           '--normalize', '--metric', 'l2', '--k', '1', '--nq', str(args.nq),
                           '--nlist', str(args.nlist), '--nprobes', str(args.nprobe or default_probe),
                           '--opt_levels', 'OPT_TRIANGLE', '--multipivot_modes', 'projection',
                           '--multipivot_scope', scope, '--multipivot_method', 'pca',
                           '--pivot_counts', str(maximum), '--signature_precision', 'float32',
                           '--load_index', str(index.resolve()), '--groundtruth_path',
                           str((args.data_root / ds / 'result/official_neighbors.i32bin').resolve()),
                           '--progressive_oracle', str(csv_path.resolve()),
                           '--progressive_oracle_only', '--oracle_predictor_features']
                (case / 'command.json').write_text(json.dumps(command, indent=2) + '\n')
                print(f'Collect {ds}/{scope}, nq={args.nq}', flush=True)
                with (case / 'collect.log').open('w') as log:
                    subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
            reports.append(analyze(feature_path, case / 'validation'))
    (args.output / 'summary.json').write_text(json.dumps(reports, indent=2) + '\n')


if __name__ == '__main__':
    main()
