"""Small offline tests; native end-to-end checks are documented separately."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

SPEC = importlib.util.spec_from_file_location('mm_portable', Path(__file__).parents[1] / 'run_mm_portable.py')
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


class PortableMMTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'query.fvecs'
        self.records = np.zeros((20, 5), dtype='<i4')
        self.records[:, 0] = 4
        self.records[:, 1:] = np.arange(80, dtype='<f4').reshape(20, 4).view('<i4')
        self.records.tofile(self.source)

    def test_repeatable_sample_preserves_records(self):
        d, ids = RUNNER.sample_queries(self.source, self.root / 'a.fvecs', 8, 19)
        d2, ids2 = RUNNER.sample_queries(self.source, self.root / 'b.fvecs', 8, 19)
        self.assertEqual((d, ids), (d2, ids2))
        self.assertEqual(d, 4)
        self.assertEqual(ids, sorted(ids))
        self.assertEqual((self.root / 'a.fvecs').read_bytes(), self.records[ids].tobytes())

    def test_explicit_ids_keep_order_and_take_prefix(self):
        ids_path = self.root / 'ids.json'
        ids_path.write_text(json.dumps({'query_ids': [15, 2, 8]}))
        _, ids = RUNNER.sample_queries(self.source, self.root / 'out.fvecs', 2, 19, ids_path)
        self.assertEqual(ids, [15, 2])

    def test_invalid_and_duplicate_ids(self):
        ids_path = self.root / 'ids.json'
        for ids in [[2, 2], [20, 1], [-1, 2], [0], [0.5, 1]]:
            with self.subTest(ids=ids):
                ids_path.write_text(json.dumps(ids))
                with self.assertRaises(ValueError):
                    RUNNER.sample_queries(self.source, self.root / 'bad.fvecs', 2, 19, ids_path)

    def test_invalid_fvecs_and_too_many_queries(self):
        with self.assertRaises(ValueError):
            RUNNER.sample_queries(self.source, self.root / 'bad.fvecs', 21, 19)
        self.source.write_bytes(b'\x01')
        with self.assertRaises(ValueError):
            RUNNER.sample_queries(self.source, self.root / 'bad.fvecs', 1, 19)

    def test_no_query_overwrite(self):
        output = self.root / 'out.fvecs'
        RUNNER.sample_queries(self.source, output, 2, 19)
        with self.assertRaises(FileExistsError):
            RUNNER.sample_queries(self.source, output, 2, 19)


if __name__ == '__main__':
    unittest.main()
