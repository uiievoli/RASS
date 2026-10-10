import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

PATH = Path(__file__).resolve().parents[1] / 'test_paper_predictor.py'
spec = importlib.util.spec_from_file_location('predictor', PATH)
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class PredictorTests(unittest.TestCase):
    def row(self, gain=10):
        return dict(q_id='0', list_rank='0', r='0', next_r='2', survivors='100',
                    true_next_pruned=str(gain), eta_x_mean='1', eta_q='1',
                    margin_tau_mean='0.5', **{f'margin_u_bin{i}': str(int(i == 0)) for i in range(7)})

    def test_future_labels_are_not_features(self):
        a, b = self.row(10), self.row(90)
        b.update(q_id='999', list_rank='999', next_r='99')
        np.testing.assert_array_equal(p.features([a]), p.features([b]))

    def test_ub_caps_predicted_pruning(self):
        row = self.row()
        row['margin_u_bin0'], row['margin_u_bin6'] = '0.2', '0.8'
        coefficients = np.zeros(11)
        coefficients[0] = 1
        self.assertAlmostEqual(p.predict(coefficients, [row])[0], .2)

    def test_global_projection_reuse_changes_only_cost(self):
        # 100 survivors, 10 saved, exact=10ns, acquire 2 dims=40ns,
        # incremental bound=100ns; unshared unprofitable, shared profitable.
        args = dict(survivors=100, predicted_rate=.1, r=2, next_r=4,
                    exact_ns=10, projection_ns_per_dim=20, bound_ns_per_dim=.4)
        self.assertEqual(p.refinement_value(ready_dims=2, **args), -20)
        self.assertEqual(p.refinement_value(ready_dims=4, **args), 20)

    def test_counts_match_next_prefix(self):
        a = self.row()
        b = self.row()
        b.update(r='2', next_r='4', survivors='80')
        with self.assertRaises(ValueError):
            p.validate_rows([a, b])
        b['survivors'] = '90'
        p.validate_rows([a, b])

    def test_entire_queries_are_held_out(self):
        rows = []
        for qi in range(10):
            for rank in range(3):
                row = self.row(10 + qi)
                row.update(q_id=str(qi), list_rank=str(rank))
                rows.append(row)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / 'features.csv'
            import csv
            with source.open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            report = p.analyze(source, directory / 'output')
            self.assertFalse(set(report['train_queries']) & set(report['test_queries']))
            self.assertEqual(report['test_queries'], [0, 5])

    def test_current_eta_margin_do_not_determine_next_gain(self):
        # At r=0 both candidates have hx=hq=1, eta=1, LB=0, margin=1.
        # Resolving axis 0 yields LB^2=0 versus 4, hence different decisions.
        q = np.array([1., 0.])
        a, b = q.copy(), -q
        self.assertEqual(np.linalg.norm(a), np.linalg.norm(b))
        self.assertEqual((q[0] - a[0])**2, 0)
        self.assertEqual((q[0] - b[0])**2, 4)


if __name__ == '__main__':
    unittest.main()
